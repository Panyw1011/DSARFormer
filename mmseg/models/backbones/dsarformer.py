import math
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import _load_checkpoint
from mmseg.utils import get_root_logger

from ..builder import BACKBONES
from .dsarb import ARAttention


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations = None,
        norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks//2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StackedMV2Block(nn.Module):
    def __init__(
            self,
            cfgs,
            stem,
            inp_channel=16,
            activation=nn.ReLU,
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
                activation()
            )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x

    
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x
    
class AR_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 kv_downsample_ratio=4,
                 kv_downsample_kernel=4,
                 topk=4):
        super().__init__()

        self.attn = ARAttention(dim=dim, num_heads=num_heads, n_win=7, qk_dim=dim,
                                        qk_scale=None, kv_per_win=-1, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode='identity',
                                        topk=topk, param_attention="qkvo", param_routing=False,
                                        diff_routing=False, soft_routing=False, side_dwconv=5,
                                        auto_pad=False, w_v=2, w_h=2)
    def forward(self, x):

        xx = self.attn(x)
        
        return xx

    
class Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True),
                 kv_downsample_ratio=4, kv_downsample_kernel=4, topk=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = AR_Attention(dim, key_dim=key_dim, num_heads=num_heads,
                                    attn_ratio=attn_ratio,
                                    activation=act_layer, norm_cfg=norm_cfg,
                                    kv_downsample_ratio=kv_downsample_ratio,
                                    kv_downsample_kernel=kv_downsample_kernel,
                                    topk=topk)
   
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None,
                 kv_downsample_kernels=4,
                 kv_downsample_ratios=4,
                 topk=4):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer,
                kv_downsample_ratio=kv_downsample_ratios,
                kv_downsample_kernel=kv_downsample_kernels,
                topk=topk))

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class ConvEncoder(nn.Module):
    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class DSARFormer(nn.Module):
    def __init__(self, 
                 cfgs,
                 channels,
                 emb_dims,
                 key_dims,
                 depths=[2,2],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2,4],
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 num_classes=1000,
                 init_cfg=None,
                 kv_downsample_kernels=[4, 2],
                 kv_downsample_ratios=[4, 2],
                 topks=[4, 2]):
        # topks=[8, 4]
        # topks=[2, 1]
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        if self.init_cfg is not None:
            self.pretrained = self.init_cfg['checkpoint']

        for i in range(len(cfgs)):
            smb = StackedMV2Block(cfgs=cfgs[i], stem=True if i == 0 else False, inp_channel=channels[i], norm_cfg=norm_cfg)
            setattr(self, f"smb{i + 1}", smb)

        for i in range(len(depths)):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[i])]  # stochastic depth decay rule
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=emb_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=norm_cfg,
                act_layer=act_layer,
                kv_downsample_kernels=kv_downsample_kernels[i],
                kv_downsample_ratios=kv_downsample_ratios[i],
                topk=topks[i])
            setattr(self, f"trans{i + 1}", trans)
        
        for i in range(len(depths)):
            convencoder = ConvEncoder(dim=emb_dims[i], hidden_dim=4*emb_dims[i], kernel_size=3)
            setattr(self, f"convencoder{i + 1}", convencoder)
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict_ema' in checkpoint:
                state_dict = checkpoint['state_dict_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)
    
    def forward(self, x):
        outputs = []
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)
        for i in range(num_smb_stage):
            
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            if i == 1:
                outputs.append(x)
            if num_trans_stage + i >= num_smb_stage:
                convencoder = getattr(self, f"convencoder{i + num_trans_stage - num_smb_stage + 1}")
                x = convencoder(x)
                trans = getattr(self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)
                outputs.append(x)
            

        return outputs


if __name__ == '__main__':
    model_cfgs = dict(
        cfg1=[
            [3, 1, 16, 1],  
            [3, 4, 16, 2],  
            [3, 3, 16, 1]],  
        cfg2=[
            [5, 3, 32, 2],  
            [5, 3, 32, 1]],  
        cfg3=[
            [3, 3, 64, 2],  
            [3, 3, 64, 1]],
        cfg4=[
            [5, 3, 128, 2]],
        cfg5=[
            [3, 6, 160, 2]],
        channels=[16, 16, 32, 64, 128, 160],
        num_heads=4,
        emb_dims=[64, 128, 160],
        key_dims=[12, 16,24],
        depths=[2, 2, 2],
        drop_path_rate=0.1,
        mlp_ratios=[2,4, 4],
        kv_downsample_kernels=[4, 2, 1],
        kv_downsample_ratios=[4, 2, 1],
        topks=[1, 4, 16]
    )
    model = DSARFormer(
        cfgs=[model_cfgs['cfg1'], model_cfgs['cfg2'], model_cfgs['cfg3'], model_cfgs['cfg4'], model_cfgs['cfg5']],
        channels=model_cfgs['channels'],
        key_dims=model_cfgs['key_dims'],
        emb_dims=model_cfgs['emb_dims'],
        depths=model_cfgs['depths'],
        num_heads=model_cfgs['num_heads'],
        mlp_ratios=model_cfgs['mlp_ratios'],
        drop_path_rate=model_cfgs['drop_path_rate'],
        kv_downsample_kernels=model_cfgs['kv_downsample_kernels'],
        kv_downsample_ratios=model_cfgs['kv_downsample_ratios'],
        topks=model_cfgs['topks'])

    input = torch.rand((1, 3, 512, 512))
    print(model)

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model.eval()
    flops = FlopCountAnalysis(model, input)
    print(flop_count_table(flops))

