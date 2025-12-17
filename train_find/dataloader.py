import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import nibabel as nib
from PIL import Image
from torchvision import transforms
from scipy.ndimage import zoom

class GFRDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        """
        root_dir: 数据集根目录, 例如 'Transunet/datasets/GFR/'
        split: 'train' 或 'test'
        img_size: 输入 TransUNet 的图像大小 (通常为 224)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # 构建路径
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')
        
        # 获取文件列表
        # 假设 dcm 文件和 nii 文件的主文件名是一致的，或者可以通过排序对应
        # 这里我们读取所有文件名并排序，确保一一对应
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.dcm')])
        self.label_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        
        assert len(self.image_files) == len(self.label_files), \
            f"图像数量 ({len(self.image_files)}) 与 标签数量 ({len(self.label_files)}) 不匹配！"

        # 基础预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 读取 DICOM 图像
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        dcm_data = pydicom.dcmread(img_path)
        image = dcm_data.pixel_array.astype(np.float32)
        
        # --- 修改部分 START: 对齐 predict.py 的归一化逻辑 ---
        # 原逻辑: (image - min) / (max - min)
        # 新逻辑: 截断在 max/2 处，并归一化
        max_value = np.max(image)
        if max_value > 0:
            image = np.clip(image, 0, max_value / 2)
            image /= (max_value / 2)
        # --- 修改部分 END ---
        
        # 2. 读取 NIfTI 标签
        lbl_path = os.path.join(self.labels_dir, self.label_files[idx])
        nii_data = nib.load(lbl_path)
        label = nii_data.get_fdata()
        
        # 确保 label 是 2D
        if label.ndim == 3:
            label = np.squeeze(label)
            
        # --- 修改部分 START: 对齐 predict.py 的 Resize 逻辑 ---
        # 原逻辑: 使用 PIL Resize
        # 新逻辑: 使用 scipy.ndimage.zoom
        
        x, y = image.shape
        if x != self.img_size or y != self.img_size:
            # 图像使用 order=3 (三次样条插值)，与 predict.py 一致
            image = zoom(image, (self.img_size / x, self.img_size / y), order=3)
            
            # 标签必须使用 order=0 (最近邻插值)，否则会产生小数，破坏类别标签
            label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
        # --- 修改部分 END ---
        
        # 4. 转换为 Tensor
        # image 增加通道维度 (H, W) -> (1, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)
        # label 保持 (H, W)，类型为 long
        label = torch.from_numpy(label).long()
        
        sample = {'image': image, 'label': label}
        return sample

# 测试数据加载部分
if __name__ == "__main__":
    # 模拟路径使用
    # dataset = GFRDataset(root_dir='Transunet/datasets/GFR/', split='train')
    # data = dataset[0]
    # print(data['image'].shape, data['label'].shape)
    pass