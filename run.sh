#!/bin/bash

# 设置使用的 GPU ID，例如使用 0号和 1号 GPU
export CUDA_VISIBLE_DEVICES=0,1

# 这里的 nproc_per_node 应该等于上面可见 GPU 的数量
NUM_GPUS=2

# 设置 PYTHONPATH 确保 python 能找到 train_find 包
# $(pwd) 表示当前目录，假设你在 train_find 的上一级目录运行此脚本
export PYTHONPATH=$(pwd):$PYTHONPATH

# 运行命令
# --nproc_per_node: 单机多卡数量
# --master_port: 防止端口冲突，可以随机指定一个
# -m train_find.train: 以模块方式运行 train_find 包下的 train.py
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    -m train_find.train \
    --root_path '/home/cu01/Code/GFR/train_find/datasets/GFR/' \
    --batch_size 8 \
    --max_epochs 100 \
    --output_dir './results_ddp'