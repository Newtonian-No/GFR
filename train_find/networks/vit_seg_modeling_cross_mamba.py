# coding=utf-8
"""
十字Mamba模型 (Cross Mamba)
扫描方式：从中央点开始向外围扩散，按十字形四个方向（上、下、左、右）同时扩展
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


def generate_cross_scan_indices(H, W):
    """
    生成十字扫描索引：从中心点开始向外围扩散
    扫描顺序：从中心开始，按照十字形（上、右、下、左）的顺序，一层一层向外扩展
    
    例如对于5x5的图像，扫描顺序如下：
    中心(2,2) -> 上(1,2) -> 右(2,3) -> 下(3,2) -> 左(2,1) -> 
    上上(0,2) -> 右右(2,4) -> 下下(4,2) -> 左左(2,0) -> ...
    然后是对角线方向依次填充
    """
    indices = []
    visited = set()
    
    center_h, center_w = H // 2, W // 2
    
    # 添加中心点
    indices.append((center_h, center_w))
    visited.add((center_h, center_w))
    
    # 从中心向外扩展
    max_radius = max(H, W)
    
    for radius in range(1, max_radius):
        # 按照十字形的四个方向扩展
        # 上方向
        for r in range(1, radius + 1):
            pos = (center_h - r, center_w)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
        
        # 右方向
        for r in range(1, radius + 1):
            pos = (center_h, center_w + r)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
        
        # 下方向
        for r in range(1, radius + 1):
            pos = (center_h + r, center_w)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
        
        # 左方向
        for r in range(1, radius + 1):
            pos = (center_h, center_w - r)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
        
        # 对角线方向 (右上、右下、左下、左上)
        for r in range(1, radius + 1):
            # 右上
            pos = (center_h - r, center_w + r)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
            
            # 右下
            pos = (center_h + r, center_w + r)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
            
            # 左下
            pos = (center_h + r, center_w - r)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
            
            # 左上
            pos = (center_h - r, center_w - r)
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                indices.append(pos)
                visited.add(pos)
        
        # 填充该层的其他位置（按照菱形/十字扩展的方式）
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                pos = (center_h + i, center_w + j)
                if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in visited:
                    indices.append(pos)
                    visited.add(pos)
    
    # 转换为一维索引
    flat_indices = [i * W + j for (i, j) in indices]
    return flat_indices


class CrossMambaBlock(nn.Module):
    """
    十字Mamba块：从中心向外扩散扫描
    使用两个方向的扫描：正向（中心→外围）和反向（外围→中心）
    """
    def __init__(self, config, fusion_type='concat'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.fusion_type = fusion_type
        
        # 正向扫描Mamba（中心→外围）
        self.mamba_forward = Mamba(
            d_model=config.hidden_size,
            d_state=16,
            d_conv=4,
            expand=1
        )
        
        # 反向扫描Mamba（外围→中心）
        self.mamba_backward = Mamba(
            d_model=config.hidden_size,
            d_state=16,
            d_conv=4,
            expand=1
        )
        
        # 缓存扫描索引
        self._scan_indices_cache = {}
        
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

    def get_scan_indices(self, H, W, device):
        """获取或生成扫描索引"""
        key = (H, W)
        if key not in self._scan_indices_cache:
            indices = generate_cross_scan_indices(H, W)
            self._scan_indices_cache[key] = torch.tensor(indices, dtype=torch.long)
        return self._scan_indices_cache[key].to(device)

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
        
        # 获取十字扫描索引
        scan_indices = self.get_scan_indices(H, W, x.device)
        reverse_indices = torch.argsort(scan_indices)
        
        # 正向扫描（中心→外围）
        x_forward = x[:, scan_indices, :]  # [B, L, C]
        x_forward = self.mamba_forward(x_forward)
        x_forward = x_forward[:, reverse_indices, :]  # 恢复原始顺序
        
        # 反向扫描（外围→中心）
        x_backward = x[:, scan_indices.flip(0), :]  # [B, L, C]
        x_backward = self.mamba_backward(x_backward)
        x_backward = x_backward[:, torch.argsort(scan_indices.flip(0)), :]  # 恢复原始顺序
        
        # 融合
        if self.fusion_type == 'concat':
            fused = torch.cat([x_forward, x_backward], dim=-1)
            output = self.fusion(fused)
        elif self.fusion_type == 'dot_product':
            fused = x_forward * x_backward
            output = self.fusion(fused)
        elif self.fusion_type == 'attention':
            q = self.query(x_forward)
            k = self.key(x_backward)
            v = self.value(x_backward)
            
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


class CrossMambaTransformerBlock(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.mamba_block = CrossMambaBlock(config, fusion_type='dot_product')
        
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
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = CrossMambaTransformerBlock(config, vis)
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
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

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


class VisionTransformerCrossMamba(nn.Module):
    """十字Mamba视觉Transformer模型"""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformerCrossMamba, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
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
