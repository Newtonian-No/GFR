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
from tqdm import tqdm # 移植：用于进度条
from tensorboardX import SummaryWriter # 移植：用于 TensorBoard

# 假设这两个文件/类在当前环境中是可用的
from .vit_seg_configs import get_r50_l16_config
from .vit_seg_modeling_mamba import VisionTransformer 
# from .utils import DiceLoss # 如果您想使用 DiceLoss，需要导入它

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
    full_dataset = KidneyDataset(data_root_dir=args.data_root)
    
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
    # torch.save(model.state_dict(), 'kidney_segmenter_final.pth')