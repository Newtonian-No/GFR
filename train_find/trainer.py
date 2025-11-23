# train_kidney_segmentor.py (修改后的核心部分)

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import argparse
import logging
import sys # 移植：用于日志
import pydicom
from tqdm import tqdm # 移植：用于进度条
from tensorboardX import SummaryWriter # 移植：用于 TensorBoard
from scipy import ndimage
# 假设这两个文件/类在当前环境中是可用的
from .networks.vit_seg_configs import get_r50_l16_config
from .networks.vit_seg_modeling_mamba import VisionTransformer 
# from .utils import DiceLoss # 如果您想使用 DiceLoss，需要导入它

GLOBAL_MEAN = -60.0  # 示例值
GLOBAL_STD = 300.0   # 示例值
TARGET_SIZE = 256    # 假设模型要求 256x256 输入

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def read_dcm_to_numpy(dcm_path):
    """
    读取 DICOM 文件并转换为 numpy 数组 (内存中的 nii)。
    处理 RescaleSlope 和 RescaleIntercept 以获得正确的 HU 值。
    """
    try:
        ds = pydicom.dcmread(dcm_path)
        pixel_array = ds.pixel_array.astype(np.float32)

        # 转换为 HU 值 (Hounsfield Units)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        slope = getattr(ds, 'RescaleSlope', 1.0)
        
        image_hu = pixel_array * slope + intercept
        return image_hu
    except Exception as e:
        logging.error(f"Error reading DICOM {dcm_path}: {e}")
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.float32)


class KidneyDataset(Dataset):
    def __init__(self, data_root_dir, transform=None , positive_ratio=0.8 , data_structure='gfr'):
        """
        Args:
            data_root_dir: 数据集根目录
            transform: 数据增强
            positive_ratio: 仅对 'extracted' 模式有效，控制正样本比例
            data_structure: 'extracted' (3D NII) 或 'gfr' (2D DICOM)
        """
        self.transform = transform
        self.data_root_dir = data_root_dir
        self.data_structure = data_structure
        self.positive_ratio = positive_ratio
        
        # 核心存储列表，其中的每个元素代表一个可以直接输入网络的样本信息
        # 结构示例: {'type': '2d', 'img_path': '...', 'lbl_path': '...'} 
        # 或 {'type': '3d', 'img_path': '...', 'l_path': '...', 'r_path': '...', 'slice_idx': 55}
        self.sample_list = []

        logging.info(f"Initializing KidneyDataset with structure: {self.data_structure}")

        if self.data_structure == 'extracted':
            # 3D NII 模式：需要预扫描切片并支持随机抽取
            self._setup_extracted_data()
        elif self.data_structure == 'gfr':
            # 2D DCM 模式：无需抽取，直接加载所有对应文件
            self._setup_gfr_data()
        else:
            raise ValueError(f"Unknown data_structure: {data_structure}")

        logging.info(f"Dataset initialized. Total samples available: {len(self.sample_list)}")

    def _setup_gfr_data(self):
        """
        针对 GFR 数据的构建逻辑：
        1. 读取 images 下的 dcm 文件。
        2. 读取 labels 下的对应文件 (假设也是 dcm 或支持的格式)。
        3. 不进行随机抽取，所有文件都作为样本。
        """
        # 假设路径结构: data_root/images/*.dcm, data_root/labels/*.dcm
        # 注意：这里的 data_root_dir 应该是具体的 train 或 test 目录 (例如 .../datasets/GFR/train)
        images_dir = os.path.join(self.data_root_dir, 'images')
        labels_dir = os.path.join(self.data_root_dir, 'labels')

        if not os.path.exists(images_dir):
            logging.error(f"GFR images dir not found: {images_dir}")
            return

        # 获取所有 DCM 文件
        image_files = sorted(glob.glob(os.path.join(images_dir, '*.dcm')))
        
        for img_path in tqdm(image_files, desc="Loading GFR paths"):
            file_name = os.path.basename(img_path)
            # 假设 label 文件名与 image 文件名一致
            lbl_path = os.path.join(labels_dir, file_name)
            
            if os.path.exists(lbl_path):
                self.sample_list.append({
                    'type': '2d_gfr',
                    'image_path': img_path,
                    'label_path': lbl_path
                })
            else:
                # 尝试寻找同名的 .png 或 .nii.gz 作为 fallback (如果 label 不是 dcm)
                # 这里只演示 dcm 逻辑
                logging.warning(f"Label not found for {file_name}, skipping.")

                
    def _setup_extracted_data(self):
        """
        针对 extracted (3D NII) 数据的构建逻辑：
        1. 扫描 sXXX/ct.nii.gz 及其对应的分割文件。
        2. 预先读取 3D 标签，找出哪些切片包含肾脏（正样本）。
        3. 根据 positive_ratio 构建样本列表。
        """
        # extracted 数据通常位于 datasets/extracted/sXXX
        # 假设传入的 data_root_dir 已经是包含 sXXX 文件夹的父级目录
        
        case_dirs = glob.glob(os.path.join(self.data_root_dir, 's*'))
        
        # 临时存储用于构建的候选列表
        all_slices_info = []      # 所有的切片 [(path_dict, slice_idx), ...]
        positive_slices_info = [] # 仅包含目标的切片
        
        for case_dir in tqdm(case_dirs, desc="Scanning 3D Volumes"):
            image_path = os.path.join(case_dir, 'ct.nii.gz')
            left_path = os.path.join(case_dir, 'segmentations', 'kidney_left.nii.gz')
            right_path = os.path.join(case_dir, 'segmentations', 'kidney_right.nii.gz')

            if os.path.exists(image_path) and os.path.exists(left_path) and os.path.exists(right_path):
                # 需要快速读取标签来确定哪些切片有用
                # 优化：只读取 header 或者低分辨率加载，这里为准确性读取数据
                try:
                    # 仅加载标签数据以检查切片内容
                    l_data = nib.load(left_path).get_fdata().astype(np.uint8)
                    r_data = nib.load(right_path).get_fdata().astype(np.uint8)
                    
                    # 确保是 [D, H, W] 格式
                    if l_data.shape[2] > l_data.shape[0]: # 假设原始是 [H, W, D]
                         l_data = np.transpose(l_data, (2, 0, 1))
                         r_data = np.transpose(r_data, (2, 0, 1))
                    
                    depth = l_data.shape[0]
                    
                    paths = {
                        'image': image_path,
                        'left': left_path,
                        'right': right_path
                    }

                    for d in range(depth):
                        # 检查当前切片是否有左肾或右肾
                        has_kidney = np.any(l_data[d, ...]) or np.any(r_data[d, ...])
                        
                        item = {'paths': paths, 'slice_idx': d}
                        all_slices_info.append(item)
                        if has_kidney:
                            positive_slices_info.append(item)
                            
                except Exception as e:
                    logging.error(f"Error scanning {case_dir}: {e}")
        # 构建最终的 sample_list
        # 这里我们模拟一种"无限"或"基于Epoch"的采样策略
        # 但为了适配 PyTorch Dataset __len__，我们需要固定列表长度。
        # 策略：如果设定了 strict 模式，只取 positive；否则混合。
        # 这里简单起见，我们将所有切片加入列表。但在 __getitem__ 里可以不使用 positive_ratio (那是 sampler 的事)，
        # 或者我们在这里根据 ratio 重新采样。
        
        # 修改策略：为了简单且方便管理，我们将所有 slice 都作为潜在样本。
        # 如果需要由 Trainer 控制正负样本比例，建议使用 WeightedRandomSampler。
        # 但为了兼容原代码逻辑（在 Dataset 内部控制比例），我们在这里做一个技巧：
        
        # 仅仅为了让 __getitem__ 能够区分，我们存入所有切片。
        # 如果您希望 strictly 按照 positive_ratio 训练，建议在训练 loop 中处理，
        # 或者在这里仅保留 positive_slices_info。
        
        # 按照原代码意图，extracted 需要混合。
        # 为了高效，我们直接使用所有含有肾脏的切片 + 等量的背景切片 (或者全部切片)。
        # 此处逻辑：保存所有切片，标记是否为 positive，以便分析。
        
        for item in all_slices_info:
            self.sample_list.append({
                'type': '3d_extracted',
                'paths': item['paths'],
                'slice_idx': item['slice_idx']
            })

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        
        # -------------------------------------------------------
        # 逻辑分支 A: Extracted (3D Nii, 需要切片)
        # -------------------------------------------------------
        if sample['type'] == '3d_extracted':
            # 如果需要保持原有的 positive_ratio 随机性（即 idx 只是个触发器，实际数据随机取）
            # 可以在这里覆写 idx。但标准的 PyTorch 做法是 Dataset 返回确定的 idx 数据。
            # 假设我们严格遵循 idx：
            
            paths = sample['paths']
            slice_idx = sample['slice_idx']
            
            # 读取数据 (这部分 I/O 会比较频繁，生产环境建议缓存 Nifti 对象或使用 LMDB)
            img_obj = nib.load(paths['image'])
            img_vol = img_obj.get_fdata().astype(np.float32)
            
            l_vol = nib.load(paths['left']).get_fdata().astype(np.uint8)
            r_vol = nib.load(paths['right']).get_fdata().astype(np.uint8)
            
            # 维度校正 [H, W, D] -> [D, H, W]
            if img_vol.shape[2] > img_vol.shape[0]:
                img_vol = np.transpose(img_vol, (2, 0, 1))
                l_vol = np.transpose(l_vol, (2, 0, 1))
                r_vol = np.transpose(r_vol, (2, 0, 1))
            
            image_slice = img_vol[slice_idx, ...]
            
            # 合并 Mask: 左肾=1, 右肾=2
            mask_slice = np.zeros_like(image_slice, dtype=np.uint8)
            mask_slice[l_vol[slice_idx, ...] > 0] = 1
            mask_slice[r_vol[slice_idx, ...] > 0] = 2

        # -------------------------------------------------------
        # 逻辑分支 B: GFR (2D DCM)
        # -------------------------------------------------------
        elif sample['type'] == '2d_gfr':
            # 直接读取 DCM 转换为 Numpy
            image_slice = read_dcm_to_numpy(sample['image_path'])
            
            # 读取标签
            # 假设标签也是 dcm，像素值直接对应类别 (0, 1, 2)
            # 如果标签是 PNG，可以使用 PIL.Image.open 读取
            try:
                label_dcm = pydicom.dcmread(sample['label_path'])
                mask_slice = label_dcm.pixel_array.astype(np.uint8)
                
                # GFR 数据的 Mask 处理：
                # 确保值是 0, 1, 2。有些标注可能是 0, 255 (二值)。
                # 如果是二值且无法区分左右，默认设为 1 (左肾) 或者根据需求修改
                if np.max(mask_slice) > 2: 
                    # 简单的归一化处理，如果 mask 只有 0 和 255
                    mask_slice = (mask_slice > 0).astype(np.uint8) # 变为 0 和 1
                    # TODO: 如果 GFR 没有左右之分，这里可能需要全部设为 1
            except Exception as e:
                logging.error(f"Failed to load label {sample['label_path']}: {e}")
                mask_slice = np.zeros_like(image_slice, dtype=np.uint8)

        # -------------------------------------------------------
        # 通用后处理 (归一化 -> 缩放 -> Tensor)
        # -------------------------------------------------------
        
        # 1. 强度归一化
        image_slice = (image_slice - GLOBAL_MEAN) / GLOBAL_STD
        
        # 2. 缩放 (Zoom) 到 TARGET_SIZE
        h, w = image_slice.shape
        if h != TARGET_SIZE or w != TARGET_SIZE:
            scale_h = TARGET_SIZE / h
            scale_w = TARGET_SIZE / w
            image_slice = ndimage.zoom(image_slice, (scale_h, scale_w), order=3)
            mask_slice = ndimage.zoom(mask_slice, (scale_h, scale_w), order=0, prefilter=False) # 最近邻插值

        # 3. 转 Tensor
        image_tensor = torch.from_numpy(image_slice).unsqueeze(0).float() # [1, H, W]
        label_tensor = torch.from_numpy(mask_slice).long()                # [H, W]

        return image_tensor, label_tensor

    def _build_slice_indices(self):
        """扫描所有 3D 体积，记录包含肾脏（正样本）和所有切片的索引。"""
        self.positive_slices = []
        self.all_slices = []

        for file_idx, data in enumerate(tqdm(self.data_paths, desc="Indexing slices")):
            # 1. 读取 CT 图像和标签
            img_nii = nib.load(data['image'])
            img_data = img_nii.get_fdata()
            left_seg = nib.load(data['left']).get_fdata()
            right_seg = nib.load(data['right']).get_fdata()
            
            # 2. 合并标签
            segmentation_map = np.zeros_like(img_data, dtype=np.uint8)
            segmentation_map[left_seg > 0] = 1 
            segmentation_map[right_seg > 0] = 2 

            # 3. 维度调整 (保持与 __getitem__ 中的逻辑一致)
            if img_data.shape[2] > img_data.shape[0]: 
                 # 假设读取是 [H, W, D]，转为 [D, H, W]
                 segmentation_map = np.transpose(segmentation_map, (2, 0, 1))
            
            D = segmentation_map.shape[0]
            
            # 4. 遍历所有切片
            for slice_idx in range(D):
                self.all_slices.append((file_idx, slice_idx))
                
                # 检查切片中是否有肾脏 (类别 1 或 2)
                slice_data = segmentation_map[slice_idx, :, :]
                if np.any(slice_data == 1) or np.any(slice_data == 2):
                    self.positive_slices.append((file_idx, slice_idx))
        
        logging.info(f"Total volumes: {len(self.data_paths)}")
        logging.info(f"Total 2D slices: {len(self.all_slices)}")
        logging.info(f"Total positive slices (containing kidney): {len(self.positive_slices)}")

    
    
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

# -------------------------------------------------------------
# 【训练函数】
# -------------------------------------------------------------

def train_model(args, model, criterion, optimizer, train_loader, device, snapshot_path):
    """
    移植了 TransUNet 训练逻辑的训练函数。
    
    参数:
        args: 包含所有训练参数的对象 (需要从 main 中传入)
        model, criterion, optimizer: 模型、损失函数、优化器
        train_loader: 数据加载器
        device: 设备 (cuda/cpu)
        snapshot_path: 模型保存路径
    """
    
    # 移植：日志文件配置（与 snapshot_path 绑定）
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # 确保模型在正确的设备上并设置为训练模式
    model.to(device)
    model.train()
    
    # 移植：TensorBoard Writer
    writer = SummaryWriter(snapshot_path + '/log')
    
    # 训练参数
    iter_num = 0
    base_lr = args.base_lr # 假设您会通过 argparse 传入 base_lr
    num_epochs = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader) 

    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    
    iterator = tqdm(range(num_epochs), ncols=70) # 移植：使用 tqdm
    
    # 假设您的模型使用 nn.CrossEntropyLoss
    # 如果您想添加 Dice Loss，需要像 TransUNet 脚本那样定义它：
    # dice_loss = DiceLoss(args.num_classes) 

    for epoch_num in iterator:
        for i_batch, (images, labels) in enumerate(train_loader):
            # images: [B, 1, H, W], labels: [B, H, W]
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) # outputs: [B, NumClasses, H, W]
            
            # 计算损失 (使用 CrossEntropyLoss)
            loss = criterion(outputs, labels)
            # 如果您添加了 Dice Loss: loss = 0.5 * loss + 0.5 * dice_loss(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 移植：Poly 学习率衰减
            # 注意: 如果您使用 AdamW，可能需要调整 Poly 衰减的参数或使用 PyTorch 自带的 Scheduler
            # 这里沿用 TransUNet 的 Poly 策略：
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                
            iter_num = iter_num + 1
            
            # 移植：TensorBoard 记录
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            # 移植：可视化逻辑（以 20 迭代为间隔）
            if iter_num % 20 == 0:
                # 图像：从 [1, H, W] 归一化后显示
                image = images[0, 0:1, :, :] 
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                # 预测结果：需要 argmax 获得类别图，然后乘以 50 增强对比度
                predicted_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', predicted_labels[0, ...] * 50, iter_num)
                
                # 真实标签：需要 unsqueeze 维度，然后乘以 50
                labs = labels[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 移植：基于 Epoch 的模型保存逻辑
        save_interval = 20 # 可以调整保存频率
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= num_epochs - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# -------------------------------------------------------------
# 【修改后的 __main__ 运行块】
# -------------------------------------------------------------

if __name__ == '__main__':
    # 0. 移植：添加 argparse 以支持训练器所需的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='base learning rate for AdamW')
    parser.add_argument('--exp_name', type=str, default='KidneySeg_ViTMamba', help='Experiment name for snapshot folder')
    parser.add_argument('--data_root', type=str, default='/home/kevin/Code/ROI/train_find/extracted/', help='Root directory for NII data')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes: 3 (BG, L-Kidney, R-Kidney)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    args = parser.parse_args()

    # 1. 定义数据路径和超参数 (使用 args)
    NUM_CLASSES = args.num_classes 
    
    # 移植：创建快照路径
    snapshot_path = os.path.join('./model_snapshots', args.exp_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. 数据加载
    full_dataset = KidneyDataset(data_root_dir=args.data_root, positive_ratio=0.8)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. 模型设置
    model, criterion, optimizer = setup_model(num_classes=NUM_CLASSES)

    # 4. 开始训练
    logging.info("Starting training...")
    # 移植：传入 args 和 snapshot_path
    train_model(args, model, criterion, optimizer, train_loader, device, snapshot_path)
    logging.info("Training complete.")

    # 5. 保存最终模型 (这里可以保留，也可以在 train_model 中处理)
    torch.save(model.state_dict(), 'kidney_segmenter_final.pth')