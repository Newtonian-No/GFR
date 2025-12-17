import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# --- 关键修改 1: 导入路径适配 ---
# 确保项目根目录在 path 中，防止 ModuleNotFoundError
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    # 假设你的 dataset 文件名为 dataloader.py
    from dataloader import GFRDataset
    # 假设上一轮重构的模型保存为 networks/vit_seg_modeling_Emamba.py
    from networks.vit_seg_modeling_Emamba import SegmentationTransformer, CONFIGS
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 'dataloader.py' 和 'networks/vit_seg_modeling_Emamba.py' 存在于正确位置。")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet+Mamba Segmentation")
    
    # 路径参数
    parser.add_argument('--root_path', type=str, default='/home/cu01/Code/GFR/train_find/datasets/GFR/', help='dataset root path')
    parser.add_argument('--output_dir', type=str, default='results/', help='output dir for logs and checkpoints')
    
    # 训练超参数
    parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network') # 通常分割是2类(背景+前景)
    parser.add_argument('--save_interval', type=int, default=20, help='save model every X epochs')
    
    # 硬件参数
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    
    args = parser.parse_args()
    return args

def set_logging(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Training started. Logs saved to {log_file}")

def trainer(args, model, snapshot_path):
    # 1. 数据加载
    trainloader = DataLoader(
        args.db_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True # 建议开启，防止最后一个batch尺寸不一导致某些计算错误
    )
    
    logging.info(f"Train dataset length: {len(args.db_train)}")
    logging.info(f"Batch size: {args.batch_size}")
    
    # 2. 优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    logging.info(f"Start training loop...")
    
    # 调试标志位：只在第一个 Batch 打印形状
    debug_printed = False

    for epoch_num in range(args.max_epochs):
        epoch_loss = 0.0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            # 移动到 GPU
            image_batch = image_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)
            
            # --- 调试代码：验证形状避免通道错误 ---
            if not debug_printed:
                logging.info(f"DEBUG: Input Image Shape: {image_batch.shape}")
                logging.info(f"DEBUG: Input Label Shape: {label_batch.shape}")
                # 检查输入是否为3通道，如果不是，模型内部会处理，但最好在这里知道
                if image_batch.shape[1] == 1:
                    logging.info("DEBUG: Detected 1-channel input (will be repeated to 3 internally).")
                debug_printed = True
            # ------------------------------------
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(image_batch)
            
            # Loss Calculation
            # CrossEntropy 要求 outputs: [B, C, H, W], label: [B, H, W] (long)
            loss = criterion(outputs, label_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i_batch % 20 == 0:
                logging.info(f'Epoch: {epoch_num}/{args.max_epochs} | Batch: {i_batch}/{len(trainloader)} | Loss: {loss.item():.6f}')

        # Epoch 结束
        avg_loss = epoch_loss / len(trainloader)
        logging.info(f"==> Epoch {epoch_num} finished. Average Loss: {avg_loss:.6f}")
        
        # 保存模型
        if (epoch_num + 1) % args.save_interval == 0: 
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved model to {save_mode_path}")
        
        if (epoch_num + 1) == args.max_epochs: 
            save_mode_path = os.path.join(snapshot_path, 'epoch_last.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved last model to {save_mode_path}")

    return "Training Finished!"

if __name__ == "__main__":
    args = get_args()
    
    # --- 1. GPU 设置 ---
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        logging.info("Using GPU: 0")
    else:
        logging.error("No GPU detected! Exiting.")
        sys.exit(1)
    
    # --- 2. 随机种子 ---
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # --- 3. 日志与路径 ---
    set_logging(args)
    snapshot_path = args.output_dir

    # --- 4. 数据集 ---
    logging.info(f"Loading data from {args.root_path}")
    # 请确保 GFRDataset 返回的 image 是 tensor, shape [C, H, W]
    db_train = GFRDataset(root_dir=args.root_path, split='train', img_size=args.img_size)
    args.db_train = db_train

    # --- 5. 模型初始化 (关键修改) ---
    logging.info("Initializing SegmentationTransformer (R50 + Mamba)...")
    
    # 获取配置
    config_name = 'R50-ViT-B_16'
    if config_name not in CONFIGS:
        raise ValueError(f"Config {config_name} not found in CONFIGS!")
        
    config = CONFIGS[config_name]
    
    # 覆盖配置中的 num_classes
    config.n_classes = args.num_classes
    
    # 实例化模型
    # 注意：使用上一轮定义的 SegmentationTransformer
    net = SegmentationTransformer(config, img_size=args.img_size, num_classes=args.num_classes)
    
    # 移动到 GPU
    net = net.cuda()
    
    # 预训练权重加载提示
    # 如果你有 ImageNet 的预训练权重 (R50+ViT-B_16.npz)，可以在这里编写加载逻辑
    # 目前保持从头训练 (Random Init)，用于验证网络结构正确性
    logging.info("Model initialized. (Training from scratch/random init)")

    # --- 6. 开始训练 ---
    try:
        trainer(args, net, snapshot_path)
    except Exception as e:
        logging.error(f"Training interrupted by error: {e}")
        raise e