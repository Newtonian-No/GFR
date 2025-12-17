import math
from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_collections

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    repeat = None

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

# ==============================================================================
# 1. 基础模块 (StdConv2d 修复版, ResNet Backbone)
# ==============================================================================

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        # 计算权重方差
        v = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = w / v.sqrt()
        # 修复 bias 问题：直接传递 self.bias，哪怕它是 None
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin, eps=1e-6)
        self.conv1 = StdConv2d(cin, cmid, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = StdConv2d(cmid, cmid, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv3 = StdConv2d(cmid, cout, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = StdConv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        out = self.relu(self.gn1(x))
        residual = x
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))
        return out + residual

class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ])) 

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
            ))), 
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
            ))), 
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
            ))), 
        ]))

    def forward(self, x):
        features = []
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]

# ==============================================================================
# 2. Mamba Block (关键修复: SS2D 逻辑)
# ==============================================================================

class SS2DMambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mamba = Mamba(d_model=config.hidden_size, d_state=16, d_conv=4, expand=2)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H # Square Grid assumption
        
        residual = x
        x = self.norm(x)
        
        # [B, L, C] -> [B, H, W, C]
        x_2d = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        # --- 1. Horizontal Forward ---
        # Scan along W: input (B*H, W, C)
        x_hf = rearrange(x_2d, 'b h w c -> (b h) w c')
        x_hf = self.mamba(x_hf)
        # Restore: (B*H, W, C) -> (B, H, W, C) -> Flatten
        x_hf = rearrange(x_hf, '(b h) w c -> b (h w) c', h=H)

        # --- 2. Horizontal Backward ---
        # Flip along W (dim=2), Scan, Restore
        x_hb = torch.flip(x_2d, dims=[2]) 
        x_hb = rearrange(x_hb, 'b h w c -> (b h) w c')
        x_hb = self.mamba(x_hb)
        x_hb = rearrange(x_hb, '(b h) w c -> b h w c', h=H)
        x_hb = torch.flip(x_hb, dims=[2]) 
        x_hb = rearrange(x_hb, 'b h w c -> b (h w) c')

        # --- 3. Vertical Forward (关键修复点) ---
        # Scan along H: input (B*W, H, C)
        x_vf = rearrange(x_2d, 'b h w c -> (b w) h c')
        x_vf = self.mamba(x_vf)
        # Restore: (B*W, H, C) -> (B, H, W, C). 
        # 注意: 必须传入 w=W，否则 Einops 无法解包 (b w)
        # 同时 Einops 会自动处理转置：Input是 (b,w,h)，Pattern RHS 是 b(h,w)，它会帮你把 h 和 w 的顺序理顺
        x_vf = rearrange(x_vf, '(b w) h c -> b (h w) c', w=W)

        # --- 4. Vertical Backward ---
        # Flip along H (dim=1)
        x_vb = torch.flip(x_2d, dims=[1])
        x_vb = rearrange(x_vb, 'b h w c -> (b w) h c')
        x_vb = self.mamba(x_vb)
        x_vb = rearrange(x_vb, '(b w) h c -> b h w c', w=W) # 同样需要 w=W
        x_vb = torch.flip(x_vb, dims=[1])
        x_vb = rearrange(x_vb, 'b h w c -> b (h w) c')

        out = x_hf + x_hb + x_vf + x_vb
        out = self.proj(out)
        out = self.dropout(out)
        
        return residual + out

# ==============================================================================
# 3. Decoder & Wrapper
# ==============================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = nn.Conv2d(config.hidden_size, head_channels, kernel_size=3, padding=1)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        
        if hasattr(config, 'skip_channels'):
            skip_channels = config.skip_channels
        else:
            skip_channels = [512, 256, 64]
            
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels) - 1): 
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            self.blocks.append(DecoderBlock(in_channels[i], out_channels[i], skip_ch))
        self.blocks.append(DecoderBlock(in_channels[-1], out_channels[-1], 0)) 

    def forward(self, hidden_states, features=None):
        B, L, C = hidden_states.shape
        H = int(math.sqrt(L))
        x = rearrange(hidden_states, 'b (h w) c -> b c h w', h=H, w=H)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < len(features)) else None
            x = decoder_block(x, skip=skip)
        return x

class MambaEncoder(nn.Module):
    def __init__(self, config):
        super(MambaEncoder, self).__init__()
        self.layer = nn.ModuleList([
            SS2DMambaBlock(config) for _ in range(config.transformer.num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Embeddings(nn.Module):
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (config.patches["size"][0], config.patches["size"][1])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        if config.patches.get("grid") is not None:
            backbone_width = config.resnet.width_factor * 64 * 16 
            self.patch_embeddings = nn.Conv2d(in_channels=backbone_width,
                                              out_channels=config.hidden_size,
                                              kernel_size=1, stride=1)
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        else:
            self.patch_embeddings = nn.Conv2d(in_channels=3,
                                              out_channels=config.hidden_size,
                                              kernel_size=patch_size, stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

# ==============================================================================
# 4. 主模型
# ==============================================================================

class SegmentationTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843):
        super(SegmentationTransformer, self).__init__()
        self.config = config
        self.num_classes = num_classes
        
        self.resnet = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = MambaEncoder(config)
        self.decoder = DecoderCup(config)
        self.segmentation_head = nn.Conv2d(config.decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        resnet_out, features = self.resnet(x)
        x_embed = self.embeddings(resnet_out)
        x_encoded = self.encoder(x_embed)
        x_out = self.decoder(x_encoded, features)
        logits = self.segmentation_head(x_out)
        return logits

    def load_from(self, weights):
        pass

# ==============================================================================
# 5. Configs
# ==============================================================================

def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_b16_config():
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    config.classifier = 'seg'
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'
    config.learning_rate = 3e-5
    return config

CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
}