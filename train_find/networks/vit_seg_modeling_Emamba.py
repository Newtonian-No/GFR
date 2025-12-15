import math
from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from einops import rearrange, repeat
except ImportError:
    print("Error: einops not found. Please install via 'pip install einops'")
    rearrange = None
    repeat = None

# 尝试导入 Mamba，如果环境没有配置好 mamba-ssm，请务必先安装
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Error: mamba_ssm not found. Please install via 'pip install mamba-ssm'")
    Mamba = None

# --------------------------------------------------------
# 1. ResNet V2 (Standard TransUNet implementation)
# --------------------------------------------------------

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v = w.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = w / v.sqrt()
        v = self.bias.var()
        b = self.bias / v.sqrt()
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""
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
    """Implementation of Pre-activation (v2) ResNet for TransUNet Backbone"""
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ])) # 1/2

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
            ))), # 1/4
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
            ))), # 1/8
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
            ))), # 1/16
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.shape
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]

# --------------------------------------------------------
# 2. SS2D Mamba Module (Replacing Transformer Layer)
# --------------------------------------------------------

class SS2DMambaBlock(nn.Module):
    """
    四向扫描 Mamba 模块 (SS2D: Cross-Scan Mechanism)
    Forward, Backward, Vertical-Forward, Vertical-Backward
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # 使用单个 Mamba 模块共享权重 (Parameter Efficient)
        # 如果追求极致性能，可以实例化4个不同的 Mamba
        self.mamba = Mamba(
            d_model=config.hidden_size,
            d_state=16,  # SSM 状态维度，通常 16
            d_conv=4,    # 局部卷积宽度
            expand=2     # 扩展因子
        )
        
        # 线性投影融合 (代替复杂的 Attention)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [B, L, C], where L = H*W
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = H
        
        residual = x
        x = self.norm(x)
        
        # 使用 einops 还原空间维度: [B, H, W, C]
        x_2d = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)

        # --- 1. Horizontal Forward (左->右) ---
        # 视为 (B*H) 个长度为 W 的序列
        x_hf = rearrange(x_2d, 'b h w c -> (b h) w c')
        x_hf = self.mamba(x_hf)
        x_hf = rearrange(x_hf, '(b h) w c -> b (h w) c', h=H)

        # --- 2. Horizontal Backward (右->左) ---
        # 先在 W 维度翻转
        x_hb = torch.flip(x_2d, dims=[2]) 
        x_hb = rearrange(x_hb, 'b h w c -> (b h) w c')
        x_hb = self.mamba(x_hb) # 共享权重
        x_hb = rearrange(x_hb, '(b h) w c -> b h w c', h=H)
        x_hb = torch.flip(x_hb, dims=[2]) # 翻转回来
        x_hb = rearrange(x_hb, 'b h w c -> b (h w) c')

        # --- 3. Vertical Forward (上->下) ---
        # 将列视为序列: (B*W) 个长度为 H 的序列
        x_vf = rearrange(x_2d, 'b h w c -> (b w) h c')
        x_vf = self.mamba(x_vf)
        x_vf = rearrange(x_vf, '(b w) h c -> b h w c', w=W) # 注意恢复顺序
        x_vf = rearrange(x_vf, 'b h w c -> b (h w) c')

        # --- 4. Vertical Backward (下->上) ---
        # 先在 H 维度翻转
        x_vb = torch.flip(x_2d, dims=[1])
        x_vb = rearrange(x_vb, 'b h w c -> (b w) h c')
        x_vb = self.mamba(x_vb)
        x_vb = rearrange(x_vb, '(b w) h c -> b h w c', w=W)
        x_vb = torch.flip(x_vb, dims=[1]) # 翻转回来
        x_vb = rearrange(x_vb, 'b h w c -> b (h w) c')

        # --- Fusion ---
        # 直接求和融合 (Element-wise Sum)，效率最高且不破坏线性度
        out = x_hf + x_hb + x_vf + x_vb
        
        out = self.proj(out)
        out = self.dropout(out)
        
        return residual + out

# --------------------------------------------------------
# 3. Encoder Wrapper (Maintains Interface)
# --------------------------------------------------------

class MambaEncoder(nn.Module):
    """
    替代原有的 Transformer Encoder，内部堆叠 Mamba Block
    """
    def __init__(self, config, vis):
        super(MambaEncoder, self).__init__()
        self.vis = vis
        # 这里对应原有 Transformer 的 layer 堆叠
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
    """Construct the embeddings from features"""
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = (img_size, img_size)
        patch_size = (config.patches["size"][0], config.patches["size"][1])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        if config.patches.get("grid") is not None:
            # Grid patches (usually from ResNet features)
            self.hybrid = True
            self.patch_embeddings = nn.Conv2d(in_channels=config.patches["grid"][0],
                                              out_channels=config.hidden_size,
                                              kernel_size=1, stride=1)
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        else:
            self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                              out_channels=config.hidden_size,
                                              kernel_size=patch_size, stride=patch_size)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if self.hybrid:
            x = self.patch_embeddings(x) # (B, Hidden, H/16, W/16)
        else:
            x = self.patch_embeddings(x)
        
        # Flatten: B C H W -> B (H W) C
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Transformer(nn.Module):
    """
    Wrapper class to keep name compatibility with TransUNet.
    Actually holds the Embeddings and MambaEncoder.
    """
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = MambaEncoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded

class VisionTransformer(nn.Module):
    """
    Main Backbone Class. 
    ResNet (Hybrid) + Mamba (Bottleneck)
    """
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        
        # 替换 Transformer 为 Mamba 结构
        self.transformer = Transformer(config, img_size, vis)
        
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.head_dist = nn.Linear(config.hidden_size, num_classes) if config.split == 'non-overlap' else None

    def forward(self, x, labels=None):
        x = self.transformer(x)
        logits = self.head(x[:, 0]) # 这一行在分割任务中其实不会被用到，主要是为了保持ViT分类接口
        return logits

    def load_from(self, weights):
        # 兼容性接口。
        # 注意：不要加载 ViT 的 npz 权重，因为我们把 Transformer 换成了 Mamba。
        # 这里的代码仅为了防止调用报错，实际不会加载不匹配的 key。
        with torch.no_grad():
            # 这是一个空的实现或者仅加载 ResNet 部分的逻辑
            pass

# --------------------------------------------------------
# 4. Decoder (CUP)
# --------------------------------------------------------

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
            # 简单的拼接
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
        
        # Decoder configs suitable for ResNet50 Backbone
        # ResNet skip channels: [0, 256, 128, 64] (feature map indices 3, 2, 1)
        decoder_channels = config.decoder_channels # [256, 128, 64, 16]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = config.n_skip_channels if hasattr(config, 'n_skip_channels') else [256, 128, 64]

        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels) - 1): # 3 blocks
            self.blocks.append(DecoderBlock(in_channels[i], out_channels[i], skip_channels[i]))
            
        self.blocks.append(DecoderBlock(in_channels[-1], out_channels[-1], 0)) # Last block no skip

    def forward(self, hidden_states, features=None):
        # hidden_states (Mamba Output): [B, L, C] -> Need reshape to [B, C, H, W]
        B, L, C = hidden_states.shape
        H = int(math.sqrt(L))
        x = rearrange(hidden_states, 'b (h w) c -> b c h w', h=H, w=H)
        
        x = self.conv_more(x)
        
        for i, decoder_block in enumerate(self.blocks):
            if features is not None and i < len(features):
                # features[::-1] used in main class usually
                skip = features[i] 
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

# --------------------------------------------------------
# 5. Main Model: SegmentationTransformer (The Entry Point)
# --------------------------------------------------------

class SegmentationTransformer(nn.Module):
    def __init__(self, config_vit, img_size=224, num_classes=21843):
        super(SegmentationTransformer, self).__init__()
        self.config = config_vit
        self.n_classes = num_classes
        
        # Hybrid Setup: ResNet Backbone
        self.resnet = ResNetV2(block_units=config_vit.resnet.num_layers, width_factor=config_vit.resnet.width_factor)
        
        # Bottleneck: Mamba (Hidden inside VisionTransformer wrapper)
        self.vit = VisionTransformer(config_vit, img_size=img_size, vis=True)
        
        # Decoder
        self.decoder = DecoderCup(config_vit)
        
        # Segmentation Head
        self.segmentation_head = nn.Conv2d(config_vit.decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # 1. ResNet Backbone
        x, features = self.resnet(x)
        
        # 2. Mamba Bottleneck (Uses the last feature map from ResNet)
        # ResNet V2 returns features list. Last one is input to Bottleneck.
        # VisionTransformer expects embedding input, so we pass the ResNet feature
        # which will be processed by Embeddings (hybrid=True) then MambaEncoder.
        x = self.vit.transformer.embeddings(x) 
        x = self.vit.transformer.encoder(x) # [B, L, C]
        
        # 3. Decoder
        # features[1:] drops the input image or earliest feature if not needed
        # Need to align with decoder skip connection logic.
        # TransUNet standard: features[::-1] but passed carefully.
        x = self.decoder(x, features[1:]) # Skip connection matching
        
        # 4. Head
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        # TransUNet standard loading function
        self.vit.load_from(weights) # Calls the empty function
        # NOTE: Ideally you should implement ResNet weight loading here from ImageNet
        # e.g. self.resnet.load_state_dict(..., strict=False)

# --------------------------------------------------------
# 6. Configurations (To maintain compatibility)
# --------------------------------------------------------

import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_b16_config():
    """Returns the ResNet-50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    config.n_skip_channels = [512, 256, 64] # Skip channels from ResNet50
    return config

# Config Dictionary
CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
}