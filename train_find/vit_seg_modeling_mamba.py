# coding=utf-8
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
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
# import vit_seg_configs as configs                  # <--- 新增
# from vit_seg_modeling_resnet_skip import ResNetV2  # <--- 新增
from mamba_ssm import Mamba  # 新增Mamba模块

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

# 新增Mamba Block定义 -------------------------------------------------- 结构：mamba+单个残差连接
    # 将每个样本的整个特征视为单个序列，然后使用Mamba模块对整个序列进行扫描处理，没有考虑二维空间结构
# class MambaBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.norm = LayerNorm(config.hidden_size, eps=1e-6) # layernorm层归一化
#         self.mamba = Mamba(
#             d_model=config.hidden_size,
#             d_state=16,  # 可根据需要调整
#             d_conv=4,
#             expand=1
#         )
        
#     def forward(self, x):
#         residual = x # 保存输入作为残差
#         x = self.norm(x)
#         x = self.mamba(x)
#         return x + residual # 返回残差连接结果

# 新增双向Mamba Block定义 --------------------------------------------------
'''
    # Mamba双向扫描方式：同时扫描B个样本的所有行（或列）数据，但将每一行（或列）视为独立序列处理。
    # 序列定义：
        # 水平：每个样本的每一行是一个独立序列
        # 垂直：每个样本的每一列是一个独立序列
    # 处理单位：行/列级别的更细粒度处理
    # 序列数量：
        # 水平扫描：B×H个序列
        # 垂直扫描：B×W个序列
    # 空间感知：明确考虑了特征图的二维结构和空间关系
    # 输入 --> 水平扫描 --> 垂直扫描 --> 特征融合 --> 输出
'''
# class MambaBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.norm = LayerNorm(config.hidden_size, eps=1e-6)

#         # 水平扫描Mamba
#         self.mamba_h = Mamba(
#             d_model=config.hidden_size,
#             d_state=16,
#             d_conv=4,
#             expand=1
#         )
        
#         # 垂直扫描Mamba
#         self.mamba_v = Mamba(
#             d_model=config.hidden_size,
#             d_state=16,
#             d_conv=4,
#             expand=1
#         )
        
#         # 交叉融合模块 concat
#         self.fusion = nn.Sequential(
#             nn.Linear(2*config.hidden_size, config.hidden_size),
#             nn.GELU(),
#             nn.Linear(config.hidden_size, config.hidden_size)
#         )



#     def forward(self, x): 
#         """
#         输入x形状: [B, L, C] (L=H*W)
#         输出形状: [B, L, C]
#         """
#         residual = x
#         B, L, C = x.shape
#         H = int(L**0.5) # L=H*W:将序列长度还原为二维坐标
#         W = H  # 假设输入是方形特征图
        
#         # 归一化
#         x = self.norm(x)
        
#         # 将一维序列
#         # 转换为二维特征图 [B, H, W, C]
#         x_2d = x.view(B, H, W, C)
        
#         # ========= 水平扫描 ========= 按行处理特征 (shape: [B*H, W, C])
#         '''
#         B*H为新的Batch大小: 表示处理B个样本中的每一行(共H行)
#         W为序列长度，每行有w个元素
#         C为特征维度
#         此处，mamba一次性处理了B*H个序列，将所有行的数据一次性输入到Mamba模块中
#         为什么这样可以实现水平扫描？
#             Mamba是一种单向序列模型，它从左到右扫描输入序列。通过上述变换：
#             1.每个样本的每一行变成了一个独立序列
#             2.Mamba模型对每一行进行从左到右的扫描处理: x_h = self.mamba_h(x_h)
#             3.每行的水平上下文信息被捕获到同一个序列处理中
#         '''
#         # 按行展开 [B*H, W, C] 
#         x_h = x_2d.permute(0, 2, 1, 3).contiguous().view(B*H, W, C) # 1.维度置换：[B, H, W, C] -> [B, W, H, C] 特征图90度的旋转 2.序列展开：[B*H, W, C]
#         # Mamba处理 [B*H, W, C]
#         x_h = self.mamba_h(x_h) # 对每个独立行序列执行从左到右的扫描
#         # 恢复形状 [B, H, W, C]
#         x_h = x_h.view(B, W, H, C).permute(0, 2, 1, 3)
        
#         # ========= 垂直扫描 =========
#         '''
#         共w列, 每列有h个元素
#         从上到下扫描每一列，捕获垂直方向上下文信息
#         '''
#         # 按列展开 [B*W, H, C]
#         x_v = x_2d.permute(0, 1, 3, 2).contiguous().view(B*W, H, C) # 1.维度置换：[B, H, W, C] -> [B, H, C, W] 2.序列展开：[B*W, H, C]
#         # Mamba处理 [B*W, H, C]
#         x_v = self.mamba_v(x_v)
#         # 恢复形状 [B, H, W, C]
#         x_v = x_v.view(B, H, W, C)
        
#         # ========= 特征融合 concat =========
#         # 拼接特征 [B, H, W, 2C]
#         fused = torch.cat([x_h, x_v], dim=-1)
#         # 融合降维 [B, H, W, C]
#         fused = self.fusion(fused)
        
#         # 展平回序列 [B, L, C]
#         output = fused.view(B, L, C)

        
#         return output + residual
    

class MambaBlock(nn.Module):
    def __init__(self, config, fusion_type='concat'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.fusion_type = fusion_type  # 融合类型: 'concat', 'dot_product', 'attention'

        # 水平扫描Mamba
        self.mamba_h = Mamba(
            d_model=config.hidden_size,
            d_state=16,
            d_conv=4,
            expand=1
        )
        
        # 垂直扫描Mamba
        self.mamba_v = Mamba(
            d_model=config.hidden_size,
            d_state=16,
            d_conv=4,
            expand=1
        )
        
        # 根据融合方式设置不同的融合模块
        if fusion_type == 'concat':
            # concat融合
            self.fusion = nn.Sequential(
                nn.Linear(2*config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif fusion_type == 'dot_product':
            # 点乘融合
            self.fusion = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        elif fusion_type == 'attention':
            # 多头注意力融合
            self.num_heads = 8  # 可以根据需要调整头数
            self.head_dim = config.hidden_size // self.num_heads
            
            # 定义query, key, value投影矩阵
            self.query = nn.Linear(config.hidden_size, config.hidden_size)
            self.key = nn.Linear(config.hidden_size, config.hidden_size)
            self.value = nn.Linear(config.hidden_size, config.hidden_size)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
            
            self.attn_dropout = nn.Dropout(0.1)
            self.proj_dropout = nn.Dropout(0.1)
            self.softmax = nn.Softmax(dim=-1)

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
        
        # 将一维序列转换为二维特征图 [B, H, W, C]
        x_2d = x.view(B, H, W, C)
        
        # 水平扫描处理
        x_h = x_2d.permute(0, 2, 1, 3).contiguous().view(B*H, W, C)
        x_h = self.mamba_h(x_h)
        x_h = x_h.view(B, W, H, C).permute(0, 2, 1, 3)
        
        # 垂直扫描处理
        x_v = x_2d.permute(0, 1, 3, 2).contiguous().view(B*W, H, C)
        x_v = self.mamba_v(x_v)
        x_v = x_v.view(B, H, W, C)
        
        # 根据不同的融合类型执行对应的融合操作
        if self.fusion_type == 'concat':
            # concat融合
            fused = torch.cat([x_h, x_v], dim=-1)
            fused = self.fusion(fused)
            
        elif self.fusion_type == 'dot_product':
            # 点乘融合
            fused = x_h * x_v  
            fused = self.fusion(fused) 
            
        elif self.fusion_type == 'attention':
            # 多头注意力融合
            # 先将特征图展平为序列形式 [B, H*W, C]
            h_flat = x_h.reshape(B, H*W, C)
            v_flat = x_v.reshape(B, H*W, C)
            
            # 投影得到查询、键、值
            q = self.query(h_flat)  # 以水平特征作为query
            k = self.key(v_flat)    # 以垂直特征作为key
            v = self.value(v_flat)  # 以垂直特征作为value
            
            # 重塑为多头形式
            q = q.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, H*W, head_dim]
            k = k.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # 计算注意力分数
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_probs = self.softmax(attn_scores)
            attn_probs = self.attn_dropout(attn_probs)
            
            # 加权聚合
            context = torch.matmul(attn_probs, v)  # [B, heads, H*W, head_dim]
            context = context.permute(0, 2, 1, 3).contiguous().view(B, H*W, C)
            
            # 投影输出
            fused = self.out_proj(context)
            fused = self.proj_dropout(fused)
            
            # 重塑回2D特征图形式
            fused = fused.view(B, H, W, C)
        
        # 展平回序列 [B, L, C]
        output = fused.view(B, L, C)
        
        return output + residual # 输出的是序列化后的结果
    
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


# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)

#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h

#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights

#     def load_from(self, weights, n_block):
#         ROOT = f"Transformer/encoderblock_{n_block}"
#         with torch.no_grad():
#             query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
#             out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

#             query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
#             key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
#             value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
#             out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

#             self.attn.query.weight.copy_(query_weight)
#             self.attn.key.weight.copy_(key_weight)
#             self.attn.value.weight.copy_(value_weight)
#             self.attn.out.weight.copy_(out_weight)
#             self.attn.query.bias.copy_(query_bias)
#             self.attn.key.bias.copy_(key_bias)
#             self.attn.value.bias.copy_(value_bias)
#             self.attn.out.bias.copy_(out_bias)

#             mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
#             mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
#             mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
#             mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

#             self.ffn.fc1.weight.copy_(mlp_weight_0)
#             self.ffn.fc2.weight.copy_(mlp_weight_1)
#             self.ffn.fc1.bias.copy_(mlp_bias_0)
#             self.ffn.fc2.bias.copy_(mlp_bias_1)

#             self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
#             self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
#             self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
#             self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = MambaBlock(config)
            self.layer.append(layer)  # 注意去掉了deepcopy

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)  # 移除attention weights
        encoded = self.encoder_norm(hidden_states)
        return encoded, []  # 保持输出格式兼容性


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # 保持接口兼容
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
            # 1. 加载Patch Embedding权重
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"]))
            
            # 2. 加载位置编码（重要！）
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            
            # 3. 加载ResNet部分权重（保持原有逻辑）
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                
                # 加载ResNet主体
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