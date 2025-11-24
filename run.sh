torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=1234 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
python -m train_find.train --max_epochs=100 --batch_size=8