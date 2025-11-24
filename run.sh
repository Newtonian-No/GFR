torchrun --nproc_per_node=2 --nnodes=1 --master_port=29500 \
python -m train_find.train --max_epochs=100 --batch_size=8