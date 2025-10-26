import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import argparse
import logging
import SimpleITK as sitk
# 模型配置和模型主体
# 假设这些文件 ('vit_seg_configs', 'vit_seg_modeling_mamba') 存在于您的项目中
from .vit_seg_configs import get_r50_l16_config
from .vit_seg_modeling_mamba import VisionTransformer
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage

# ==================== 全局配置 ====================
GLOBAL_MEAN = -60.0  # 示例值 (针对 CT 强度)
GLOBAL_STD = 300.0   # 示例值 (针对 CT 强度)
TARGET_SIZE = 256    # 假设模型要求 256x256 输入
NUM_CLASSES = 3      # 背景(0), 左肾(1), 右肾(2) (假设 CTx.nii.gz 包含了 3 个类别的标签)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ==================== DICOM 加载占位函数 ====================
# 【重要】您必须用实际的 DICOM 序列加载逻辑替换此函数。
# 这是一个占位函数，用于确保代码可以运行。
# 实际加载需要使用 pydicom, SimpleITK 等库。
def load_dicom_series_to_numpy(dicom_folder_path):
    """
    使用 SimpleITK 从 DICOM 文件夹加载 3D NumPy 数组。
    
    参数:
        dicom_folder_path: 包含 .dcm 文件的文件夹路径。
        
    返回:
        一个 3D NumPy 数组 (D, H, W)，或者 None。
        
    注意：SimpleITK 默认返回的维度通常是 (Z, Y, X) 或 (D, H, W)。
          这里没有进行方向或间距的标准化，仅加载体素数据。
    """
    logging.info(f"正在尝试加载 DICOM 序列: {dicom_folder_path}")
    
    try:
        # 1. 读取 DICOM 文件名序列
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetSeriesFileNames(dicom_folder_path)
        
        if not dicom_names:
            logging.error(f"在路径 {dicom_folder_path} 中未找到任何 DICOM 文件。")
            return None
        
        reader.SetFileNames(dicom_names)
        
        # 2. 读取图像体
        # SimpleITK.ReadImage() 会自动处理多文件序列并返回一个 3D/4D 图像
        image = reader.Execute()
        
        # 3. 将 SimpleITK Image 转换为 NumPy 数组
        # image.GetArrayFromImage() 的输出维度通常是 (D, H, W) 或 (Z, Y, X)
        image_data = sitk.GetArrayFromImage(image)
        
        # 可选：检查数据类型并转换为 float32 (与您的预处理兼容)
        image_data = image_data.astype(np.float32)

        logging.info(f"成功加载 DICOM 序列。形状: {image_data.shape}, 路径: {dicom_folder_path}")
        
        return image_data

    except Exception as e:
        logging.error(f"DICOM 序列加载失败: {e}")
        # 打印详细路径信息有助于调试
        logging.error(f"失败路径: {dicom_folder_path}")
        return None


# ==================== 数据集类 (适配新的数据结构) ====================
class KidneyDatasetDCM(Dataset):
    def __init__(self, data_root_dir, transform=None):
        self.transform = transform
        self.data_paths = []
        
        # 遍历主文件夹下的所有 CT 文件目录 (CT1, CT2, ...)
        # case_dir = /path/to/DATA_ROOT/CTx
        for case_dir in glob.glob(os.path.join(data_root_dir, 'CT*')):
            if not os.path.isdir(case_dir):
                continue
                
            # 1. 查找分割标签文件 (CTx.nii.gz)
            case_name = os.path.basename(case_dir) # 例如：CT1
            seg_nii_path = os.path.join(case_dir, case_name + '.nii.gz')

            # 2. 检查 DICOM 文件的存在性
            dicom_files = glob.glob(os.path.join(case_dir, '*.dcm'))

            # 检查条件
            if os.path.exists(seg_nii_path) and len(dicom_files) > 0:
                
                self.data_paths.append({
                    'image_dir': case_dir,     # DICOM 文件所在的文件夹
                    'segmentation': seg_nii_path # NIfTI 标签文件
                })
        
        logging.info(f"找到 {len(self.data_paths)} 个有效病例。")


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = self.data_paths[idx]
        
        # 1. 读取 CT 图像 (DICOM 序列)
        img_data = load_dicom_series_to_numpy(data['image_dir'])
        
        if img_data is None:
            # 如果加载失败，返回随机一个样本，以避免 Dataloader 崩溃
            logging.warning(f"无法加载 {data['image_dir']} 的 DICOM，尝试加载下一个。")
            return self.__getitem__(np.random.randint(len(self)))
        
        # 2. 读取标签 (Segmentation Mask)
        seg_nii = nib.load(data['segmentation'])
        segmentation_map = seg_nii.get_fdata().astype(np.uint8)
        
        # 3. 检查和调整维度 (DICOM 加载和 NIfTI 标签的对齐是关键)
        if img_data.shape != segmentation_map.shape:
            logging.warning(f"图像和标签维度不匹配: 图像 {img_data.shape}, 标签 {segmentation_map.shape}，将跳过。")
            return self.__getitem__(np.random.randint(len(self)))
        
        D, H, W = img_data.shape
        
        # 4. 预处理

        # 4.1. 强度归一化 (Z-Score)
        img_data = (img_data - GLOBAL_MEAN) / GLOBAL_STD
        
        # 4.2. 随机抽取一个 2D 切片 (沿着深度 D 轴)
        if D <= 0:
            return self.__getitem__(np.random.randint(len(self))) # D=0，跳过
            
        slice_idx = np.random.randint(D)

        image_slice = img_data[slice_idx, :, :]     # 形状: [H, W]
        label_slice = segmentation_map[slice_idx, :, :] # 形状: [H, W]

        # 4.3. 图像和标签缩放
        current_H, current_W = image_slice.shape
        zoom_factors = (TARGET_SIZE / current_H, TARGET_SIZE / current_W)

        # 图像缩放：三次样条插值
        image_slice = ndimage.zoom(image_slice, 
                                   zoom=zoom_factors, 
                                   order=3, 
                                   mode='nearest').astype(np.float32)
        
        # 标签缩放：最近邻插值
        label_slice = ndimage.zoom(label_slice, 
                                   zoom=zoom_factors, 
                                   order=0, 
                                   mode='nearest').astype(np.uint8)


        # 5. 转换为 PyTorch Tensor 
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0) # [1, H, W]
        label_tensor = torch.from_numpy(label_slice).long()       # [H, W]
        
        return image_tensor, label_tensor

    
# ==================== 模型设置和训练函数 (与原始代码保持一致) ====================
# 这部分与原始脚本完全相同

def setup_model(num_classes):
    config = get_r50_l16_config()
    config.n_channels = 1
    config.n_classes = num_classes 

    if not hasattr(config.patches, 'size') or config.patches.size[0] == 0:
        config.patches.size = (16, 16)

    model = VisionTransformer(config, img_size=TARGET_SIZE, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    config.learning_rate = 3e-4 # 假设从 config 获取学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, train_loader, device, num_epochs=100):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # 使用 images.size(0) 确保批量损失计算正确
            running_loss += loss.item() * images.size(0) 

        # 损失除以数据集大小
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')


# ==================== 运行代码 ====================

if __name__ == '__main__':
    # 1. 定义数据路径和超参数
    # 【请替换为您的 DICOM 数据根目录】
    DATA_ROOT = '/home/kevin/Code/ROI/train_find/CT' 
    
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. 数据加载 (使用适配后的数据集类)
    full_dataset = KidneyDatasetDCM(data_root_dir=DATA_ROOT)
    
    if len(full_dataset) == 0:
        logging.error("未找到任何有效的数据集，请检查 DATA_ROOT 和文件结构。")
    else:
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 3. 模型设置
        model, criterion, optimizer = setup_model(num_classes=NUM_CLASSES)

        # 4. 开始训练
        logging.info("Starting training...")
        train_model(model, criterion, optimizer, train_loader, device, num_epochs=NUM_EPOCHS)
        logging.info("Training complete.")

        # 5. 保存最终模型
        torch.save(model.state_dict(), '6_kidney_segmenter_dcm.pth')