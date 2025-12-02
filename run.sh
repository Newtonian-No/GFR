#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
export PYTHONPATH=$(pwd):$PYTHONPATH

# === 新增以下环境变量 ===
# 禁用 P2P 通信（解决段错误的核心）
export NCCL_P2P_DISABLE=1
# 禁用 InfiniBand（如果不是跨节点训练，通常不需要 IB）
export NCCL_IB_DISABLE=1
# 如果你是通过 Ethernet 联网，可能需要强制指定网卡（可选，如果上面两个不行再加）
# export NCCL_SOCKET_IFNAME=eth0 

# 开启调试日志，如果还报错可以看到更详细原因
export NCCL_DEBUG=INFO

torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 \
    -m train_find.train \
    --root_path '/home/cu01/Code/GFR/train_find/datasets/GFR/' \
    --batch_size 1 \
    --max_epochs 100 \
    --output_dir './results_ddp'
