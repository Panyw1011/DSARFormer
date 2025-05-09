from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, trunc_normal_

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Fusion_block(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            embed_dim: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(Fusion_block, self).__init__()
        self.norm_cfg = norm_cfg
        self.local_embedding = ConvModule(inp, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(oup, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
        self.co = ConvEncoder(dim=embed_dim, hidden_dim=4*embed_dim, kernel_size=3)

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        B, C_c, H_c, W_c = x_g.shape
      
        local_feat = self.local_embedding(x_l)
        local_feat = self.co(local_feat)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * sig_act
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


@HEADS.register_module()
class LightHead(BaseDecodeHead):
    def __init__(self, embed_dims, is_dw=False, **kwargs):
        super(LightHead, self).__init__(input_transform='multiple_select', **kwargs)

        head_channels = self.channels
        in_channels = self.in_channels    
        self.linear_fuse = ConvModule(
            in_channels=head_channels*2,
            out_channels=head_channels,
            kernel_size=1,
            stride=1,
            groups=head_channels if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        for i in range(len(embed_dims)):
            fuse = Fusion_block(in_channels[0] if i == 0 else embed_dims[i-1], in_channels[i+1], embed_dim=embed_dims[i], norm_cfg=self.norm_cfg)
            setattr(self, f"fuse{i + 1}", fuse)

        self.mlp = ConvModule(
            in_channels=embed_dims[0],
            out_channels=embed_dims[1],
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
            )

        self.embed_dims = embed_dims
        
    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  
        x_detail = xx[0]
        outputs = []
        for i in range(len(self.embed_dims)):
            fuse = getattr(self, f"fuse{i + 1}")
            x_detail = fuse(x_detail, xx[i+1])
            outputs.append(x_detail)
        outputs[0] = self.mlp(outputs[0])
        x_detail = torch.cat([outputs[0],outputs[1]], dim=1)
        _c = self.linear_fuse(x_detail)
        x = self.cls_seg(_c)
        return x
