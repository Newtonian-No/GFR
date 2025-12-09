import torch
import torch.nn as nn
import numpy as np
import os

# --- 导入你的模型类和配置 ---
# 确保导入路径正确，并且你使用的是训练时实际使用的那个文件
from back.networks.vit_seg_modeling_bimambaattention import VisionTransformer as ViT_seg
from back.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg 

# 假设的配置参数 (请替换为你的实际训练参数)
VIT_NAME = 'R50-ViT-L_16'
IMG_SIZE = 224
NUM_CLASSES = 3
VIT_PATCHES_SIZE = 16
MODEL_PATH = '/home/kevin/Code/ROI/back/weights/results_single_gpu/epoch_last.pth' # <--- 请将此路径替换为你的 .pth 文件路径

def check_model_structure(model_path, vit_name, img_size, num_classes, vit_patches_size):
    """
    加载模型配置和权重，并打印关键结构信息。
    """
    print(f"--- 🚀 开始检查模型结构 ---")
    print(f"配置模型名称: {vit_name}, 图像大小: {img_size}, 类别数: {num_classes}")

    # --- 1. 加载配置 ---
    if vit_name not in CONFIGS_ViT_seg:
        print(f"错误: 配置名 {vit_name} 未在 CONFIGS_ViT_seg 中定义。")
        return

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.patches.size = (vit_patches_size, vit_patches_size)
    
    # R50-ViT 混合模型需要设置 grid
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size/vit_patches_size), int(img_size/vit_patches_size))
        
    # --- 2. 初始化模型 ---
    # 注意: 模型初始化时必须使用训练时的配置
    try:
        net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes, vis=False)
    except Exception as e:
        print(f"\n❌ 模型初始化失败！请检查配置或 Mamba 库安装。")
        print(f"错误信息: {e}")
        return

    print("\n✅ 模型初始化成功。")
    print(f"Encoder隐藏维度 (Hidden Size): {config_vit.hidden_size}")
    print(f"Encoder总层数 (Num Layers): {config_vit.transformer['num_layers']}")
    print("-" * 30)

    # --- 3. 打印 Encoder 层数结构 ---
    
    # 获取 Transformer 编码器层
    encoder_layers = net.transformer.encoder.layer 
    num_encoder_layers = len(encoder_layers)
    
    print(f"🔍 打印 Encoder 结构中的 {num_encoder_layers} 个 Block:")

    # 打印每层的类型，并确认 BiMamba 结构是否存在
    for i, block in enumerate(encoder_layers):
        block_type = type(block).__name__
        
        # 验证是否使用了 BiMamba attention 块 (即 ModifiedBlock)
        if 'ModifiedBlock' in block_type:
            has_bi_mamba = hasattr(block, 'bi_mamba')
            print(f"  Block {i:02d}: {block_type} (包含 BiMamba: {has_bi_mamba})")
        else:
            print(f"  Block {i:02d}: {block_type}")

    print("-" * 30)
    
    # --- 4. (可选) 尝试加载权重并检查 keys ---
    # 这可以帮助你诊断 size mismatch 错误
    print(f"💾 尝试加载权重文件: {model_path}...")
    if not os.path.exists(model_path):
        print(f"警告: 权重文件 {model_path} 不存在，跳过权重检查。")
        return
        
    try:
        # 只加载权重文件
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 如果模型有额外的键 (如 optimizer state)，可能需要清理
        if 'state_dict' in state_dict:
             state_dict = state_dict['state_dict']
             
        # 打印权重的总数量
        print(f"权重文件中的参数总数量 (Keys): {len(state_dict.keys())}")
        
        # 尝试加载权重并捕获错误
        net.load_state_dict(state_dict, strict=True)
        print("\n🎉 模型权重与当前结构 **严格匹配**，加载成功！")

    except RuntimeError as e:
        print("\n❌ 模型权重与当前结构 **不匹配** (RuntimeError)。")
        print("这通常是 **层数、隐藏维度** 或 **类别数** 不一致导致的。")
        print(f"错误信息 (导致 size mismatch 的原因): \n{e}")
        
    print(f"\n--- 检查结束 ---")
    
# --- 执行检查 ---
check_model_structure(MODEL_PATH, VIT_NAME, IMG_SIZE, NUM_CLASSES, VIT_PATCHES_SIZE)