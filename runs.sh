#!/bin/bash

# 指定使用哪一张显卡，这里使用 id 为 0 的显卡
export CUDA_VISIBLE_DEVICES=0

# 设置 Python 路径，确保能找到包
export PYTHONPATH=$(pwd):$PYTHONPATH

# 如果之前设置过 NCCL 相关的环境变量，建议 unset 掉，或者是覆盖掉，避免单卡也去检测网络
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE
unset NCCL_SOCKET_IFNAME

# 运行单卡脚本
# -m train_find.train_single 表示运行 train_find 包下的 train_single.py 模块
python -m train_find.train_single \
    --root_path '/home/cu01/Code/GFR/train_find/datasets/GFR/' \
    --batch_size 4 \
    --max_epochs 100 \
    --output_dir './results_single'