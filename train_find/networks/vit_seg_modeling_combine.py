'''
Architecture Overview:
Input
  │
  ▼
[Encoder Stack]
  ├─ Layer 0: [Bi-Directional Mamba Block]  (Focus: Sequence Modeling)
  ├─ Layer 1: [Bi-Directional Mamba Block]  (Focus: Sequence Modeling)
  ├─ Layer 2: [Window Attention Block]      (Focus: Local Spatial Mixing)
  ├─ Layer 3: [Bi-Directional Mamba Block]
  ├─ Layer 4: [Bi-Directional Mamba Block]
  ├─ Layer 5: [Window Attention Block]
  ...
  │
  ▼
[Decoder Cup] -> Segmentation Head
'''
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
import torch.utils.checkpoint as checkpoint
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2

# 尝试导入 Mamba，如果环境没有安装则报错提示
try:
    from mamba_ssm import Mamba
except ImportError:
    print("WARNING: mamba_ssm library not found. Mamba blocks will fail. Please run `pip install mamba-ssm`.")
    Mamba = None

logger = logging.getLogger(__name__)

# 为了兼容旧权重的 Key 映射 (注意：架构改变后，旧的 Attention 权重可能无法直接加载)
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


class BiMambaBlock(nn.Module):
    """
    双向 Mamba 块：处理序列建模，替代传统的 Self-Attention。
    底层使用纯 Mamba 能够更高效地捕捉长距离依赖且显存占用更低。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba module not found.")
            
        # 前向 Mamba
        self.mamba_fwd = Mamba(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        # 后向 Mamba
        self.mamba_bwd = Mamba(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
        # 融合层：将双向特征融合回 d_model
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x shape: [B, L, D]
        
        # 前向流
        x_fwd = self.mamba_fwd(x)
        
        # 后向流：翻转序列输入，输出后再翻转回来
        x_reversed = x.flip(dims=[1])
        x_bwd = self.mamba_bwd(x_reversed).flip(dims=[1])
        
        # 拼接并投影
        x_cat = torch.cat([x_fwd, x_bwd], dim=-1)
        return self.output_proj(x_cat)


class WindowAttention(nn.Module):
    """
    窗口注意力机制 (Window/Local Attention)。
    将全局序列 reshape 回 2D 网格，划分窗口，在窗口内计算 Attention。
    """
    def __init__(self, config, vis, window_size=8):
        super(WindowAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.hidden_size = config.hidden_size
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = window_size

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

    def forward(self, hidden_states, grid_size):
        # hidden_states: [B, L, C] where L = H*W
        B, L, C = hidden_states.shape
        H, W = grid_size
        
        assert L == H * W, f"Input sequence length {L} doesn't match grid size {H}x{W}."

        # 1. Reshape to 2D image: [B, H, W, C]
        x = hidden_states.view(B, H, W, C)

        # Handle padding if H or W not divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = x.shape[1], x.shape[2]

        # 2. Partition into windows
        # Shape: [B, H // ws, ws, W // ws, ws, C] -> [B * num_windows, ws*ws, C]
        x_windows = x.view(B, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        # 3. Calculate Attention within windows
        mixed_query_layer = self.query(x_windows)
        mixed_key_layer = self.key(x_windows)
        mixed_value_layer = self.value(x_windows)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 4. Merge windows back
        # [B * num_windows, ws*ws, C]
        attn_output = self.out(context_layer)
        attn_output = self.proj_dropout(attn_output)

        # Reshape back to [B, H_pad, W_pad, C]
        attn_windows = attn_output.view(-1, self.window_size, self.window_size, C)
        B_windows = B * (H_pad // self.window_size) * (W_pad // self.window_size)
        
        attn_output = attn_windows.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size, self.window_size, C)
        attn_output = attn_output.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, C)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            attn_output = attn_output[:, :H, :W, :]

        # 5. Flatten back to [B, L, C]
        attn_output = attn_output.view(B, L, C)

        weights = attention_probs if self.vis else None
        return attn_output, weights


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


class HybridBlock(nn.Module):
    """
    通用混合块。
    根据 layer_type 决定使用 BiMamba 还是 Window Attention。
    这也实现了职责分离：
    - 'mamba': 负责高效的序列混合和上下文提取。
    - 'attention': 负责空间上的局部特征增强和混合。
    """
    def __init__(self, config, vis, layer_type="mamba", window_size=8):
        super(HybridBlock, self).__init__()
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size
        
        # Norms
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        
        # Token Mixer: Mamba OR Attention
        if self.layer_type == "mamba":
            self.mixer = BiMambaBlock(d_model=config.hidden_size)
        else:
            self.mixer = WindowAttention(config, vis, window_size=window_size)
            
        # Channel Mixer: MLP
        self.mlp = Mlp(config)

    def forward(self, x, grid_size=None):
        # 1. Token Mixing Path
        h = self.norm1(x)
        
        if self.layer_type == "mamba":
            # Mamba 不需要 grid_size，它视作纯序列
            x_mixed = self.mixer(h)
            weights = None
        else:
            # Attention 需要 grid_size 来还原 2D 结构
            x_mixed, weights = self.mixer(h, grid_size=grid_size)
            
        x = x + x_mixed  # Residual 1

        # 2. Channel Mixing Path (MLP)
        h = self.norm2(x)
        x_mlp = self.mlp(h)
        x = x + x_mlp    # Residual 2
        
        return x, weights

    def load_from(self, weights, n_block):
        # 注意：由于架构从 ModifiedBlock 变为 HybridBlock，
        # 原有的权重加载逻辑仅适用于 MLP 部分和 Norm 部分。
        # Attention 和 Mamba 的权重需要重新训练或特殊处理。
        ROOT = f"Transformer/encoderblock_{n_block}"
        
        # Load Norms and MLP (Standard ViT parts)
        with torch.no_grad():
            try:
                # Norm 1
                self.norm1.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
                self.norm1.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
                
                # Norm 2
                self.norm2.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
                self.norm2.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

                # MLP
                mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
                mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
                mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
                mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

                self.mlp.fc1.weight.copy_(mlp_weight_0)
                self.mlp.fc2.weight.copy_(mlp_weight_1)
                self.mlp.fc1.bias.copy_(mlp_bias_0)
                self.mlp.fc2.bias.copy_(mlp_bias_1)
            except Exception as e:
                logger.warning(f"Failed to load some weights for block {n_block}. This is expected for Mamba layers. Error: {e}")

            # 如果是 Attention 层，尝试加载 Attention 权重 (需要维度匹配)
            if self.layer_type == "attention":
                try:
                    query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                    key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                    value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
                    out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

                    query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
                    key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
                    value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
                    out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

                    self.mixer.query.weight.copy_(query_weight)
                    self.mixer.key.weight.copy_(key_weight)
                    self.mixer.value.weight.copy_(value_weight)
                    self.mixer.out.weight.copy_(out_weight)
                    self.mixer.query.bias.copy_(query_bias)
                    self.mixer.key.bias.copy_(key_bias)
                    self.mixer.value.bias.copy_(value_bias)
                    self.mixer.out.bias.copy_(out_bias)
                except Exception as e:
                     logger.warning(f"Attention weights shape mismatch or missing for block {n_block}. Error: {e}")


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
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
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Encoder(nn.Module):
    def __init__(self, config, vis, img_size):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        # 定义 2:1 的层级交错策略 (Mamba : Attention)
        # 层级越低 (idx小) 使用纯 Mamba/局部 Attention，层级越深混合度越高
        # 这里的实现：每3层为一个单元 -> [Mamba, Mamba, Attention]
        
        # 计算 grid_size 用于 Window Attention
        patch_size = _pair(config.patches["size"])
        img_size = _pair(img_size)
        if config.patches.get("grid") is not None: # ResNet Hybrid case
             # Grid comes from the feature map size after ResNet
             self.grid_size = config.patches["grid"]
        else:
             self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        for i in range(config.transformer["num_layers"]):
            # 策略：2:1 交替
            # i % 3 == 0 -> Mamba
            # i % 3 == 1 -> Mamba
            # i % 3 == 2 -> Attention (Window)
            
            if (i + 1) % 3 == 0:
                layer_type = "attention"
            else:
                layer_type = "mamba"
                
            layer = HybridBlock(
                config, 
                vis, 
                layer_type=layer_type,
                window_size=8 # 可以根据 patch 分辨率调整
            )
            self.layer.append(layer)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            # 传入 grid_size 以支持 Window Attention 的 reshape 操作
            if self.training:
                # 使用 Checkpoint 节省显存，注意要适配参数传递
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, grid_size=self.grid_size)
                    return custom_forward

                # 注意：checkpoint 仅接受 Tensor 参数，grid_size 是 tuple，需要特殊处理或闭包
                # 简单起见，这里直接调用，或者如果显存紧张可以自行封装
                hidden_states, weights = layer_block(hidden_states, grid_size=self.grid_size)
            else:
                hidden_states, weights = layer_block(hidden_states, grid_size=self.grid_size)
                
            if self.vis and weights is not None:
                attn_weights.append(weights)
                
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        # Encoder 初始化需要 img_size 来计算网格
        self.encoder = Encoder(config, vis, img_size=img_size)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
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
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
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


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
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
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                if bname == "layer": # ModuleList
                    for i, unit in enumerate(block):
                         unit.load_from(weights, n_block=f"encoderblock_{i}")

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

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