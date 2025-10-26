import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import argparse
import logging
# 模型配置和模型主体
from .vit_seg_configs import get_r50_l16_config
from .vit_seg_modeling_mamba import VisionTransformer  # 主要模型类
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage

GLOBAL_MEAN = -60.0  # 示例值
GLOBAL_STD = 300.0   # 示例值
TARGET_SIZE = 256    # 假设模型要求 256x256 输入

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def resize_transform(target_size=256):
    # 将 NumPy 数组转为 Tensor，然后进行 Resize
    return transforms.Compose([
        # 1. Resize/Crop: 如果需要，将张量调整到目标大小
        # 注意: torch.nn.functional.interpolate 或自定义的 Resize function 
        # 必须在转换为 Tensor 后应用
        
        # 2. ToTensor: 假设您的 __getitem__ 已经返回 Tensor
        # 如果不是，这一步应该在 __getitem__ 中完成
    ])

class KidneyDataset(Dataset):
    def __init__(self, data_root_dir, transform=None):
        self.transform = transform
        self.data_paths = []
        
        # 遍历主文件夹下的所有 CT 文件目录
        for case_dir in glob.glob(os.path.join(data_root_dir, 's*')):
            image_path = os.path.join(case_dir, 'ct.nii.gz')
            
            # 找到 segmentations 文件夹
            seg_dir = os.path.join(case_dir, 'segmentations')
            left_kidney_path = os.path.join(seg_dir, 'kidney_left.nii.gz')
            right_kidney_path = os.path.join(seg_dir, 'kidney_right.nii.gz')

            if os.path.exists(image_path) and \
               os.path.exists(left_kidney_path) and \
               os.path.exists(right_kidney_path):
                
                self.data_paths.append({
                    'image': image_path,
                    'left': left_kidney_path,
                    'right': right_kidney_path
                })

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = self.data_paths[idx]
        
        # 1. 读取 CT 图像
        img_nii = nib.load(data['image'])
        img_data = img_nii.get_fdata().astype(np.float32)
        
        # 2. 读取标签 (Segmentation Masks)
        left_seg = nib.load(data['left']).get_fdata().astype(np.uint8)
        right_seg = nib.load(data['right']).get_fdata().astype(np.uint8)
        
        # 3. 合并标签：创建单一的标签图（背景为0，左肾为1，右肾为2）
        # 模型需要 K+1 个类别，这里是 3 个类别 (背景, 左肾, 右肾)
        segmentation_map = np.zeros_like(img_data, dtype=np.uint8)
        segmentation_map[left_seg > 0] = 1 # 左肾标签为 1
        segmentation_map[right_seg > 0] = 2 # 右肾标签为 2

        # 确保数据维度是 [D, H, W] (假设我们沿着 D 轴切片)
        # 您需要根据 nibabel 读取的实际轴顺序调整这里
        if img_data.shape[2] > img_data.shape[0]: 
             # 假设读取是 [H, W, D]，需要转为 [D, H, W]
             img_data = np.transpose(img_data, (2, 0, 1))
             segmentation_map = np.transpose(segmentation_map, (2, 0, 1))
        
        D, H, W = img_data.shape
        
        # 4. 预处理开始

        # 4.1. 强度归一化 (Z-Score)
        img_data = (img_data - GLOBAL_MEAN) / GLOBAL_STD
        
        # 4.2. 随机抽取一个 2D 切片 (沿着深度 D 轴)
        if D > 0:
            slice_idx = np.random.randint(D)
        else:
            # 避免空切片错误
            return self.__getitem__(np.random.randint(len(self)))

        image_slice = img_data[slice_idx, :, :] # 形状: [H, W]
        label_slice = segmentation_map[slice_idx, :, :] # 形状: [H, W]

        # 4.3. 获取当前切片尺寸
        current_H, current_W = image_slice.shape
        # 计算缩放因子
        zoom_factors = (TARGET_SIZE / current_H, TARGET_SIZE / current_W)

        # 图像缩放：使用三次样条插值 (order=3)
        image_slice = ndimage.zoom(image_slice, 
                                   zoom=zoom_factors, 
                                   order=3, 
                                   mode='nearest').astype(np.float32)
        
        # 标签缩放：使用最近邻插值 (order=0)，以保持类别值不变
        label_slice = ndimage.zoom(label_slice, 
                                   zoom=zoom_factors, 
                                   order=0, 
                                   mode='nearest').astype(np.uint8)


        # 5. 转换为 PyTorch Tensor (注意：现在是 2D 切片)
        # 形状: [H, W] -> [C, H, W]
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0) # [1, H, W]
        label_tensor = torch.from_numpy(label_slice).long()       # [H, W]
        
        # ... (如果 transform 存在，应用 transform) ...

        return image_tensor, label_tensor
    
def setup_model(num_classes):
    # 使用您提供的配置文件
    config = get_r50_l16_config()

    config.n_channels = 1

    # 类别数：背景 (0) + 左肾 (1) + 右肾 (2) = 3
    config.n_classes = num_classes 

    if not hasattr(config.patches, 'size'):
        # 确保 config.patches 是一个可配置的对象
        # 如果 config.patches 只是一个 ConfigDict，直接设置 size
        config.patches.size = (16, 16)
    elif config.patches.size[0] == 0:
        # 如果 size 属性存在但是 0，强制设置为 16x16
        config.patches.size = (16, 16)

    # 实例化模型 (假设 VisionTransformer 是您 ViT-Seg 的入口类)
    model = VisionTransformer(config, img_size=TARGET_SIZE, num_classes=num_classes)
    
    # 定义损失函数：交叉熵损失适用于多分类分割任务
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, train_loader, device, num_epochs=100):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # images: [B, 1, H, W] (输入)
            # labels: [B, H, W] (标签，包含类别索引 0, 1, 2)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # 前向传播 (Forward Pass)
            # outputs: [B, NumClasses, H, W] (模型输出)
            outputs = model(images)
            
            # -------------------------------------------------------------
            # 【适配修改】
            # 移除或修改此处的注释，确认维度匹配。
            # 您的模型输出是 [Batch, NumClasses, H, W] 的 4D 格式，
            # 这正是 nn.CrossEntropyLoss 所需的标准格式。
            # -------------------------------------------------------------
            
            # 计算损失
            # PyTorch 的 CrossEntropyLoss 期望：
            # outputs: (N, C, ...) 例如 (B, NumClasses, H, W)
            # labels: (N, ...) 例如 (B, H, W) 
            # 维度匹配，无需调整
            loss = criterion(outputs, labels)
            
            # 反向传播 (Backward Pass) 和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 可以在这里添加验证/保存模型权重的逻辑
        # ...

# 运行代码：
# python -m train_find.train_kidney_segmentor

if __name__ == '__main__':
    # 1. 定义数据路径和超参数
    DATA_ROOT = '/home/kevin/Code/ROI/train_find/extracted/' 
    NUM_CLASSES = 3  # 背景(0), 左肾(1), 右肾(2)
    BATCH_SIZE = 2
    NUM_EPOCHS = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. 数据加载
    full_dataset = KidneyDataset(data_root_dir=DATA_ROOT)
    
    # 简单的划分训练集和验证集 (实际项目中应更严谨)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # 验证集加载器

    # 3. 模型设置
    model, criterion, optimizer = setup_model(num_classes=NUM_CLASSES)

    # 4. 开始训练
    logging.info("Starting training...")
    train_model(model, criterion, optimizer, train_loader, device, num_epochs=NUM_EPOCHS)
    logging.info("Training complete.")

    # 5. 保存最终模型
    torch.save(model.state_dict(), 'kidney_segmenter.pth')