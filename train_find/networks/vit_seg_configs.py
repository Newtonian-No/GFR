# import ml_collections

# def get_b16_config():
#     """Returns the ViT-B/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (16, 16)})
#     config.hidden_size = 768
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 3072
#     config.transformer.num_heads = 12
#     config.transformer.num_layers = 12
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1

#     config.classifier = 'seg'
#     config.representation_size = None
#     config.resnet_pretrained_path = None
#     config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
#     config.patch_size = 16

#     config.decoder_channels = (256, 128, 64, 16)
#     config.n_classes = 2
#     config.activation = 'softmax'
#     return config


# def get_testing():
#     """Returns a minimal configuration for testing."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (16, 16)})
#     config.hidden_size = 1
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 1
#     config.transformer.num_heads = 1
#     config.transformer.num_layers = 1
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.classifier = 'token'
#     config.representation_size = None
#     return config

# def get_r50_b16_config():
#     """Returns the Resnet50 + ViT-B/16 configuration."""
#     config = get_b16_config()
#     config.patches.grid = (16, 16)
#     config.resnet = ml_collections.ConfigDict()
#     config.resnet.num_layers = (3, 4, 9)
#     config.resnet.width_factor = 1

#     config.classifier = 'seg'
#     config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
#     config.decoder_channels = (256, 128, 64, 16)
#     config.skip_channels = [512, 256, 64, 16]
#     config.n_classes = 2
#     config.n_skip = 3
#     config.activation = 'softmax'

#     return config


# def get_b32_config():
#     """Returns the ViT-B/32 configuration."""
#     config = get_b16_config()
#     config.patches.size = (32, 32)
#     config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
#     return config


# def get_l16_config():
#     """Returns the ViT-L/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (16, 16)})
#     config.hidden_size = 1024
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 4096
#     config.transformer.num_heads = 16
#     config.transformer.num_layers = 24
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.representation_size = None

#     # custom
#     config.classifier = 'seg'
#     config.resnet_pretrained_path = None
#     config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
#     config.decoder_channels = (256, 128, 64, 16)
#     config.n_classes = 2
#     config.activation = 'softmax'
#     return config


# def get_r50_l16_config():
#     """Returns the Resnet50 + ViT-L/16 configuration. customized """
#     config = get_l16_config()
#     config.patches.grid = (16, 16)
#     config.resnet = ml_collections.ConfigDict()
#     config.resnet.num_layers = (3, 4, 9)
#     config.resnet.width_factor = 1

#     config.classifier = 'seg'
#     config.resnet_pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
#     config.decoder_channels = (256, 128, 64, 16)
#     config.skip_channels = [512, 256, 64, 16]
#     config.n_classes = 2
#     config.activation = 'softmax'
#     return config


# def get_l32_config():
#     """Returns the ViT-L/32 configuration."""
#     config = get_l16_config()
#     config.patches.size = (32, 32)
#     return config


# def get_h14_config():
#     """Returns the ViT-L/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (14, 14)})
#     config.hidden_size = 1280
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 5120
#     config.transformer.num_heads = 16
#     config.transformer.num_layers = 32
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.classifier = 'token'
#     config.representation_size = None

#     return config


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

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
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

    return config

# def get_r50_b16_config():
#     """返回适配Mamba的Resnet50+ViT-B/16配置"""
#     config = get_b16_config()
    
#     # 核心修改参数 -------------------------------
#     # Transformer相关参数调整
#     config.transformer.num_layers = 8           # 从12层减少到8层（关键修改）
#     config.transformer.num_heads = 0            # 禁用多头注意力
#     config.transformer.mlp_dim = 0              # 禁用MLP扩展
#     config.transformer.attention_dropout_rate = 0.0  # 移除注意力dropout
    
#     # 新增Mamba专用参数
#     config.mamba = ml_collections.ConfigDict()
#     config.mamba.d_state = 16                   # 状态扩展因子
#     config.mamba.d_conv = 4                     # 卷积核尺寸
#     config.mamba.expand = 2                     # 扩展比率
    
#     # 保持不变的参数 -------------------------------
#     config.patches.grid = (16, 16)
#     config.resnet = ml_collections.ConfigDict()
#     config.resnet.num_layers = (3, 4, 9)
#     config.resnet.width_factor = 1
#     config.classifier = 'seg'
#     config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
#     config.decoder_channels = (256, 128, 64, 16)
#     config.skip_channels = [512, 256, 64, 16]
#     config.n_classes = 2
#     config.n_skip = 3
#     config.activation = 'softmax'
    
#     return config



def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '/model/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'

    #新增学习率
    config.learning_rate = 0.0001

    return config

#目前使用这个配置：10/18 ryz
def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '/model/ViT-L_16.npz'
    # config.resnet_pretrained_path = '/model/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'

    config.learning_rate = 3e-5
    config.n_skip = 3
    
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config

