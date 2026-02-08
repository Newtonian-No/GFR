# coding=utf-8
"""
回型Mamba模型 (Spiral Mamba)
扫描方式：从外向内，一层层收缩，类似于回型/螺旋形扫描
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .networks import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from mamba_ssm import Mamba

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


def generate_spiral_scan_indices(H, W):
    """
    生成回型扫描索引：从外向内，一层层收缩
    扫描顺序：从最外圈开始，顺时针方向扫描，然后进入内圈，重复直到中心
    
    例如对于4x4的图像，扫描顺序如下：
    外圈: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(3,3)->(3,2)->(3,1)->(3,0)->(2,0)->(1,0)
    内圈: (1,1)->(1,2)->(2,2)->(2,1)
    """
    indices = []
    visited = [[False] * W for _ in range(H)]
    
    top, bottom, left, right = 0, H - 1, 0, W - 1
    
    while top <= bottom and left <= right:
        # 上边：从左到右
        for j in range(left, right + 1):
            if not visited[top][j]:
                indices.append((top, j))
                visited[top][j] = True
        top += 1
        
        # 右边：从上到下
        for i in range(top, bottom + 1):
            if not visited[i][right]:
                indices.append((i, right))
                visited[i][right] = True
        right -= 1
        
        # 下边：从右到左
        if top <= bottom:
            for j in range(right, left - 1, -1):
                if not visited[bottom][j]:
                    indices.append((bottom, j))
                    visited[bottom][j] = True
            bottom -= 1
        
        # 左边：从下到上
        if left <= right:
            for i in range(bottom, top - 1, -1):
                if not visited[i][left]:
                    indices.append((i, left))
                    visited[i][left] = True
            left += 1
    
    # 转换为一维索引
    flat_indices = [i * W + j for (i, j) in indices]
    return flat_indices


def generate_spiral_scan_indices_ccw(H, W):
    """
    生成回型扫描索引（逆时针版本）：从外向内，一层层收缩
    扫描顺序：从最外圈开始，逆时针方向扫描
    """
    indices = []
    visited = [[False] * W for _ in range(H)]
    
    top, bottom, left, right = 0, H - 1, 0, W - 1
    
    while top <= bottom and left <= right:
        # 上边：从右到左
        for j in range(right, left - 1, -1):
            if not visited[top][j]:
                indices.append((top, j))
                visited[top][j] = True
        top += 1
        
        # 左边：从上到下
        for i in range(top, bottom + 1):
            if not visited[i][left]:
                indices.append((i, left))
                visited[i][left] = True
        left += 1
        
        # 下边：从左到右
        if top <= bottom:
            for j in range(left, right + 1):
                if not visited[bottom][j]:
                    indices.append((bottom, j))
                    visited[bottom][j] = True
            bottom -= 1
        
        # 右边：从下到上
        if left <= right:
            for i in range(bottom, top - 1, -1):
                if not visited[i][right]:
                    indices.append((i, right))
                    visited[i][right] = True
            right -= 1
    
    # 转换为一维索引
    flat_indices = [i * W + j for (i, j) in indices]
    return flat_indices


class SpiralMambaBlock(nn.Module):
    """
    回型Mamba块：从外向内，一层层收缩扫描
    使用两个方向的扫描：顺时针（外→内）和逆时针（外→内）
    以及它们的反向（内→外）
    """
    def __init__(self, config, fusion_type='concat'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.fusion_type = fusion_type
        
        # 顺时针扫描Mamba（外→内）
        self.mamba_cw = Mamba(
            d_model=config.hidden_size,
            d_state=16,
            d_conv=4,
            expand=1
        )
        
        # 逆时针扫描Mamba（外→内）
        self.mamba_ccw = Mamba(
            d_model=config.hidden_size,
            d_state=16,
            d_conv=4,
            expand=1
        )
        
        # 缓存扫描索引
        self._scan_indices_cache_cw = {}
        self._scan_indices_cache_ccw = {}
        
        # 根据融合方式设置不同的融合模块
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(2*config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif fusion_type == 'dot_product':
            self.fusion = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif fusion_type == 'attention':
            self.num_heads = 8
            self.head_dim = config.hidden_size // self.num_heads
            self.query = nn.Linear(config.hidden_size, config.hidden_size)
            self.key = nn.Linear(config.hidden_size, config.hidden_size)
            self.value = nn.Linear(config.hidden_size, config.hidden_size)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout = nn.Dropout(0.1)
            self.proj_dropout = nn.Dropout(0.1)
            self.softmax = nn.Softmax(dim=-1)

    def get_scan_indices_cw(self, H, W, device):
        """获取或生成顺时针扫描索引"""
        key = (H, W)
        if key not in self._scan_indices_cache_cw:
            indices = generate_spiral_scan_indices(H, W)
            self._scan_indices_cache_cw[key] = torch.tensor(indices, dtype=torch.long)
        return self._scan_indices_cache_cw[key].to(device)
    
    def get_scan_indices_ccw(self, H, W, device):
        """获取或生成逆时针扫描索引"""
        key = (H, W)
        if key not in self._scan_indices_cache_ccw:
            indices = generate_spiral_scan_indices_ccw(H, W)
            self._scan_indices_cache_ccw[key] = torch.tensor(indices, dtype=torch.long)
        return self._scan_indices_cache_ccw[key].to(device)

    def forward(self, x):
        """
        输入x形状: [B, L, C] (L=H*W)
        输出形状: [B, L, C]
        """
        residual = x
        B, L, C = x.shape
        H = int(L**0.5)
        W = H
        
        # 归一化
        x = self.norm(x)
        
        # 获取回型扫描索引
        scan_indices_cw = self.get_scan_indices_cw(H, W, x.device)
        scan_indices_ccw = self.get_scan_indices_ccw(H, W, x.device)
        
        reverse_indices_cw = torch.argsort(scan_indices_cw)
        reverse_indices_ccw = torch.argsort(scan_indices_ccw)
        
        # 顺时针扫描（外→内）
        x_cw = x[:, scan_indices_cw, :]  # [B, L, C]
        x_cw = self.mamba_cw(x_cw)
        x_cw = x_cw[:, reverse_indices_cw, :]  # 恢复原始顺序
        
        # 逆时针扫描（外→内）
        x_ccw = x[:, scan_indices_ccw, :]  # [B, L, C]
        x_ccw = self.mamba_ccw(x_ccw)
        x_ccw = x_ccw[:, reverse_indices_ccw, :]  # 恢复原始顺序
        
        # 融合
        if self.fusion_type == 'concat':
            fused = torch.cat([x_cw, x_ccw], dim=-1)
            output = self.fusion(fused)
        elif self.fusion_type == 'dot_product':
            fused = x_cw * x_ccw
            output = self.fusion(fused)
        elif self.fusion_type == 'attention':
            q = self.query(x_cw)
            k = self.key(x_ccw)
            v = self.value(x_ccw)
            
            q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_probs = self.softmax(attn_scores)
            attn_probs = self.attn_dropout(attn_probs)
            
            context = torch.matmul(attn_probs, v)
            context = context.permute(0, 2, 1, 3).contiguous().view(B, L, C)
            output = self.out_proj(context)
            output = self.proj_dropout(output)
        
        return output + residual


class SpiralMambaBlockFourDirection(nn.Module):
    """
    四向回型Mamba块：使用四个方向的回型扫描
    - 顺时针（外→内）
    - 逆时针（外→内）
    - 顺时针反向（内→外）
    - 逆时针反向（内→外）
    """
    def __init__(self, config, fusion_type='concat'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.fusion_type = fusion_type
        
        # 四个方向的Mamba
        self.mamba_cw_out2in = Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=1)
        self.mamba_ccw_out2in = Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=1)
        self.mamba_cw_in2out = Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=1)
        self.mamba_ccw_in2out = Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=1)
        
        # 缓存扫描索引
        self._scan_indices_cache_cw = {}
        self._scan_indices_cache_ccw = {}
        
        # 融合模块
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(4*config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif fusion_type == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(4) / 4)
            self.fusion = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

    def get_scan_indices_cw(self, H, W, device):
        key = (H, W)
        if key not in self._scan_indices_cache_cw:
            indices = generate_spiral_scan_indices(H, W)
            self._scan_indices_cache_cw[key] = torch.tensor(indices, dtype=torch.long)
        return self._scan_indices_cache_cw[key].to(device)
    
    def get_scan_indices_ccw(self, H, W, device):
        key = (H, W)
        if key not in self._scan_indices_cache_ccw:
            indices = generate_spiral_scan_indices_ccw(H, W)
            self._scan_indices_cache_ccw[key] = torch.tensor(indices, dtype=torch.long)
        return self._scan_indices_cache_ccw[key].to(device)

    def forward(self, x):
        residual = x
        B, L, C = x.shape
        H = int(L**0.5)
        W = H
        
        x = self.norm(x)
        
        scan_indices_cw = self.get_scan_indices_cw(H, W, x.device)
        scan_indices_ccw = self.get_scan_indices_ccw(H, W, x.device)
        
        reverse_indices_cw = torch.argsort(scan_indices_cw)
        reverse_indices_ccw = torch.argsort(scan_indices_ccw)
        
        # 顺时针（外→内）
        x_cw_out2in = x[:, scan_indices_cw, :]
        x_cw_out2in = self.mamba_cw_out2in(x_cw_out2in)
        x_cw_out2in = x_cw_out2in[:, reverse_indices_cw, :]
        
        # 逆时针（外→内）
        x_ccw_out2in = x[:, scan_indices_ccw, :]
        x_ccw_out2in = self.mamba_ccw_out2in(x_ccw_out2in)
        x_ccw_out2in = x_ccw_out2in[:, reverse_indices_ccw, :]
        
        # 顺时针（内→外）- 反向扫描
        x_cw_in2out = x[:, scan_indices_cw.flip(0), :]
        x_cw_in2out = self.mamba_cw_in2out(x_cw_in2out)
        x_cw_in2out = x_cw_in2out[:, torch.argsort(scan_indices_cw.flip(0)), :]
        
        # 逆时针（内→外）- 反向扫描
        x_ccw_in2out = x[:, scan_indices_ccw.flip(0), :]
        x_ccw_in2out = self.mamba_ccw_in2out(x_ccw_in2out)
        x_ccw_in2out = x_ccw_in2out[:, torch.argsort(scan_indices_ccw.flip(0)), :]
        
        # 融合
        if self.fusion_type == 'concat':
            fused = torch.cat([x_cw_out2in, x_ccw_out2in, x_cw_in2out, x_ccw_in2out], dim=-1)
            output = self.fusion(fused)
        elif self.fusion_type == 'weighted_sum':
            weights = torch.softmax(self.weights, dim=0)
            fused = (weights[0] * x_cw_out2in + weights[1] * x_ccw_out2in + 
                    weights[2] * x_cw_in2out + weights[3] * x_ccw_in2out)
            output = self.fusion(fused)
        
        return output + residual


class SpiralMambaTransformerBlock(nn.Module):
    def __init__(self, config, vis=False, use_four_direction=False):
        super().__init__()
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        
        if use_four_direction:
            self.mamba_block = SpiralMambaBlockFourDirection(config, fusion_type='concat')
        else:
            self.mamba_block = SpiralMambaBlock(config, fusion_type='dot_product')
        
        self.attn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)
        
        self.mlp_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Mlp(config)

    def forward(self, hidden_states):
        mamba_output = self.mamba_block(hidden_states)
        
        h = mamba_output
        x = self.attn_norm(h)
        x, weights = self.attn(x)
        x = x + h
        
        h = x
        x = self.mlp_norm(h)
        x = self.mlp(x)
        x = x + h
        
        return x, weights


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Encoder(nn.Module):
    def __init__(self, config, vis, use_four_direction=False):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = SpiralMambaTransformerBlock(config, vis, use_four_direction)
            self.layer.append(layer)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, use_four_direction=False):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, use_four_direction)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformerSpiralMamba(nn.Module):
    """回型Mamba视觉Transformer模型"""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, use_four_direction=False):
        super(VisionTransformerSpiralMamba, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, use_four_direction)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))
            
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                
                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
