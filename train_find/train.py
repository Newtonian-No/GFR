# --- START OF FILE train.py ---
import faulthandler
faulthandler.enable()
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime

# 引入自定义的数据集和模型
# 注意：必须以模块方式运行 (-m) 才能解析相对导入
from .dataloader import GFRDataset
from .networks.vit_seg_configs import get_r50_l16_config
from .networks.vit_seg_modeling_combine import VisionTransformer as TransUnet

def get_args():
    parser = argparse.ArgumentParser(description="Train TransUNet with BiMamba")
    
    # 路径参数
    parser.add_argument('--root_path', type=str, default='/home/cu01/Code/GFR/train_find/datasets/GFR/', help='dataset root path')
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
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    # local_rank 参数在使用 torch.distributed.launch 时会自动传入，但在 torchrun 环境变量模式下不一定需要，保留以防万一
    parser.add_argument('--local-rank', type=int, default=-1, help='local rank for distributed training')

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
    
    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Training started. Logs saved to {log_file}")
    logging.info(f"Args: {args}")

def trainer(args, model, snapshot_path, train_sampler):
    
    # 1. 数据加载
    # 注意：sampler 已经在 main 中创建并传入，这里只需要创建 DataLoader
    # DataLoader 中的 shuffle 必须为 False，因为 sampler 已经处理了 shuffle
    trainloader = DataLoader(
        args.db_train, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True, 
        sampler=train_sampler
    )
    
    if args.rank == 0:
        logging.info(f"Train dataset length: {len(args.db_train)}")
    
    # 2. 优化器与损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    scaler = GradScaler()

    model.train()
    
    if args.rank == 0:
        logging.info(f"Start training: Total epochs: {args.max_epochs}, Batch size per GPU: {args.batch_size}")

    for epoch_num in range(args.max_epochs):
        # [重要修正] DDP 必须在每个 epoch 开始前调用 set_epoch
        train_sampler.set_epoch(epoch_num)
        
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # 使用 non_blocking=True 可以稍微加速数据传输
            image_batch, label_batch = image_batch.cuda(args.gpu, non_blocking=True), label_batch.cuda(args.gpu, non_blocking=True)
            
            with autocast():
                # 注意：这里我们使用 autocast 来执行 FP16 
                # 大部分操作，可以大幅降低内存占用
                outputs = model(image_batch)
                loss = criterion(outputs, label_batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            outputs = model(image_batch)
            
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 只在 Rank 0 打印日志
            if args.rank == 0 and i_batch % 20 == 0:
                logging.info(f'Epoch: {epoch_num}/{args.max_epochs} | Batch: {i_batch}/{len(trainloader)} | Loss: {loss.item():.6f}')

        # 计算平均 Loss (这里的 Loss 是当前 GPU 的平均，严格来说应该 reduce 所有 GPU 的 loss 再平均，但作为监控通常不需要那么精确)
        avg_loss = epoch_loss / len(trainloader)
        
        if args.rank == 0:
            logging.info(f"==> Epoch {epoch_num} finished. Average Loss: {avg_loss:.6f}")
        
        # 保存模型逻辑 (只在 Rank 0 保存)
        if args.rank == 0:
            # 间隔保存
            if (epoch_num + 1) % args.save_interval == 0: 
                save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}.pth')
                torch.save(model.module.state_dict(), save_mode_path)
                logging.info(f"Saved model to {save_mode_path}")
            
            # 保存最新模型
            if (epoch_num + 1) == args.max_epochs: 
                save_mode_path = os.path.join(snapshot_path, 'epoch_last.pth')
                torch.save(model.module.state_dict(), save_mode_path)
                logging.info(f"Saved last model to {save_mode_path}")

    return "Training Finished!"

if __name__ == "__main__":
    args = get_args()
    
    # --- 1. DDP 初始化设置 ---
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    dist.barrier() # 等待所有进程初始化完毕
    
    # --- 2. 随机种子设置 ---
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

    # --- 3. 日志配置 (仅 Rank 0) ---
    if args.rank == 0:
        set_logging(args)
    else:
        # 非 Rank 0 进程不输出 INFO 日志，避免刷屏
        logging.basicConfig(level=logging.WARNING)
        
    snapshot_path = args.output_dir

    # --- 4. 数据集初始化 ---
    # 数据集初始化放在 main 中，方便创建 sampler
    if args.rank == 0:
        logging.info(f"Loading data from {args.root_path}")
    
    db_train = GFRDataset(root_dir=args.root_path, split='train', img_size=args.img_size)
    args.db_train = db_train # 传递给 trainer
    train_sampler = DistributedSampler(db_train, shuffle=True) # shuffle=True 是默认的，但显式写出来更好

    # --- 5. 模型配置与初始化 ---
    config = get_r50_l16_config()
    config.n_classes = args.num_classes
    config.n_skip = 3
    
    if 'R50' in config.transformer.get('name', 'R50-ViT-B_16'):
         config.patches.grid = (int(args.img_size / 16), int(args.img_size / 16))

    if args.rank == 0:
        logging.info("Initializing Model...")
        
    net = TransUnet(config, img_size=args.img_size, num_classes=args.num_classes)
    
    # SyncBN (可选): 如果显存允许且 batch size 较小，建议开启 SyncBN 以获得更好的统计量
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    net.to(args.gpu)
    
    # DDP 包装
    find_unused_parameters=True 
    net = DDP(net, device_ids=[args.gpu]) # , find_unused_parameters=True)

    if args.rank == 0:
        logging.info(f"Model wrapped in DDP. Starting training loop...")

    # --- 6. 开始训练 ---
    trainer(args, net, snapshot_path, train_sampler)

    # 清理进程组
    dist.destroy_process_group()