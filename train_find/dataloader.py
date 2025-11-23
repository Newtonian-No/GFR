import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import nibabel as nib
from PIL import Image
from torchvision import transforms

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
        
        # 简单的归一化 (Min-Max) 到 0-1 之间
        # 如果是CT数据可能需要根据 HU 值截断，这里做通用处理
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # 2. 读取 NIfTI 标签
        lbl_path = os.path.join(self.labels_dir, self.label_files[idx])
        nii_data = nib.load(lbl_path)
        label = nii_data.get_fdata() # 获取数组
        
        # 确保 label 是 2D (因为你说只有一张切片)
        if label.ndim == 3:
            # 有些 nii 保存单张切片时可能是 (H, W, 1)，需要 squeeze
            label = np.squeeze(label)
            
        # 3. 调整大小 (Resize)
        # TransUNet 需要特定尺寸 (通常 224x224)
        # 使用 PIL 进行 Resize 操作比较方便
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        lbl_pil = Image.fromarray(label.astype(np.uint8))
        
        img_pil = img_pil.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        lbl_pil = lbl_pil.resize((self.img_size, self.img_size), resample=Image.NEAREST) # 标签必须用最近邻插值
        
        # 4. 转换为 Tensor
        image = self.transform(img_pil) # 变为 (1, H, W)
        label = torch.from_numpy(np.array(lbl_pil)).long() # 变为 (H, W)
        
        # 你的标签说明：1 左肾, 2 右肾. 
        # TransUNet通常需要 CrossEntropyLoss，label 不需要 one-hot，只需 long 类型即可
        
        sample = {'image': image, 'label': label}
        return sample

# 测试数据加载部分
if __name__ == "__main__":
    # 模拟路径使用
    # dataset = GFRDataset(root_dir='Transunet/datasets/GFR/', split='train')
    # data = dataset[0]
    # print(data['image'].shape, data['label'].shape)
    pass