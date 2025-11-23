import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# 引入自定义的数据集和模型
# 请确保 train_find 文件夹在 python 的搜索路径中，或者该脚本在 train_find 的上一级目录运行
from .dataloader import GFRDataset
from .networks.vit_seg_configs import get_r50_l16_config
from .networks.vit_seg_modeling_bimambaattention import VisionTransformer as TransUnet

def get_args():
    parser = argparse.ArgumentParser(description="Train TransUNet with BiMamba")
    
    # 路径参数
    parser.add_argument('--root_path', type=str, default='/home/kevin/Code/ROI/train_find/datasets/GFR/', help='dataset root path')
    parser.add_argument('--output_dir', type=str, default='results/', help='output dir for logs and checkpoints')
    
    # 训练超参数
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
    parser.add_argument('--save_interval', type=int, default=20, help='save model every X epochs')
    
    # 硬件参数
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')

    args = parser.parse_args()
    return args

def set_logging(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Training started. Logs saved to {log_file}")
    logging.info(f"Args: {args}")

def trainer(args, model, snapshot_path):
    # 1. 数据加载
    logging.info(f"Loading data from {args.root_path}")
    db_train = GFRDataset(root_dir=args.root_path, split='train', img_size=args.img_size)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    logging.info(f"Train dataset length: {len(db_train)}")
    
    # 2. 优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)  # 用于学习率衰减或其他调度
    
    logging.info(f"Start training: Total epochs: {args.max_epochs}, Batch size: {args.batch_size}")

    for epoch_num in range(args.max_epochs):
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            optimizer.zero_grad()
            outputs = model(image_batch)
            
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            iter_num += 1
            
            # 简单的进度打印，每20个batch打印一次
            if i_batch % 20 == 0:
                logging.info(f'Epoch: {epoch_num}/{args.max_epochs} | Batch: {i_batch}/{len(trainloader)} | Loss: {loss.item():.6f}')

        avg_loss = epoch_loss / len(trainloader)
        logging.info(f"==> Epoch {epoch_num} finished. Average Loss: {avg_loss:.6f}")
        
        # 保存模型逻辑
        if (epoch_num + 1) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved model to {save_mode_path}")
            
        # 保存最新模型
        if (epoch_num + 1) == args.max_epochs:
            save_mode_path = os.path.join(snapshot_path, 'epoch_last.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved last model to {save_mode_path}")

    return "Training Finished!"

if __name__ == "__main__":
    args = get_args()
    
    # 随机种子设置
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

    # 配置日志和快照目录
    set_logging(args)
    snapshot_path = args.output_dir
    
    # 模型配置初始化
    config = get_r50_l16_config()
    config.n_classes = args.num_classes
    config.n_skip = 3 # R50通常用3个skip connection
    
    # 这里的 patches.grid 可能需要在 config 中确认，如果不一致会导致 position embedding 报错
    if 'R50' in config.transformer.get('name', 'R50-ViT-B_16'):
         config.patches.grid = (int(args.img_size / 16), int(args.img_size / 16))

    # 模型初始化
    logging.info("Initializing Model...")
    net = TransUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    
    # 如果有预训练权重，可以在这里加载
    # net.load_from(torch.load('/models/R50-ViT-B_16.npz')) 

    trainer(args, net, snapshot_path)