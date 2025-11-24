# --- START OF FILE train_find/train_single.py ---

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datetime import datetime

# 引入自定义的数据集和模型
# 依然保持相对导入，所以必须用 -m 运行
from .dataloader import GFRDataset
from .networks.vit_seg_configs import get_r50_l16_config
from .networks.vit_seg_modeling_bimambaattention import VisionTransformer as TransUnet

def get_args():
    parser = argparse.ArgumentParser(description="Train TransUNet Single GPU")
    parser.add_argument('--root_path', type=str, default='/home/cu01/Code/GFR/train_find/datasets/GFR/', help='dataset root path')
    parser.add_argument('--output_dir', type=str, default='results_single/', help='output dir')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--num_classes', type=int, default=3, help='output channel')
    parser.add_argument('--save_interval', type=int, default=20, help='save model every X epochs')
    parser.add_argument('--deterministic', type=int, default=1, help='deterministic training')
    return parser.parse_args()

def set_logging(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f'log_single_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s'))
    logging.getLogger('').addHandler(console)
    logging.info(f"Single GPU Training started. Logs: {log_file}")

def trainer(args, model, device):
    # 1. 数据加载 (单卡模式不需要 DistributedSampler)
    logging.info(f"Loading data from {args.root_path}")
    db_train = GFRDataset(root_dir=args.root_path, split='train', img_size=args.img_size)
    
    # 直接在 DataLoader 中 shuffle=True
    trainloader = DataLoader(
        db_train, 
        batch_size=args.batch_size, 
        shuffle=True,  # <--- 关键点：单卡训练开启 Shuffle
        num_workers=4, 
        pin_memory=True
    )
    
    logging.info(f"Train dataset length: {len(db_train)}")
    
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch_num in range(args.max_epochs):
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(image_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i_batch % 20 == 0:
                logging.info(f'Epoch: {epoch_num}/{args.max_epochs} | Batch: {i_batch}/{len(trainloader)} | Loss: {loss.item():.6f}')

        avg_loss = epoch_loss / len(trainloader)
        logging.info(f"==> Epoch {epoch_num} finished. Avg Loss: {avg_loss:.6f}")
        
        # 保存模型
        if (epoch_num + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'epoch_{epoch_num + 1}.pth')
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved model to {save_path}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'epoch_last.pth'))
    return "Finished"

if __name__ == "__main__":
    args = get_args()
    
    # 设定使用的 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0") # 默认使用可见的第0号卡
        logging.info("Using GPU: cuda:0")
    else:
        device = torch.device("cpu")
        logging.warning("Using CPU!")

    # 随机种子
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

    set_logging(args)

    # 模型初始化
    config = get_r50_l16_config()
    config.n_classes = args.num_classes
    config.n_skip = 3
    if 'R50' in config.transformer.get('name', 'R50-ViT-B_16'):
         config.patches.grid = (int(args.img_size / 16), int(args.img_size / 16))

    logging.info("Initializing Model...")
    net = TransUnet(config, img_size=args.img_size, num_classes=args.num_classes)
    net.to(device)

    # 开始训练
    trainer(args, net, device)