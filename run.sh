export CUDA_VISIBLE_DEVICES=0


ROOT_PATH="/root/autodl-tmp/train_find/datasets/GFR/"
OUTPUT_DIR="results_e"
MAX_EPOCHS=160
BATCH_SIZE=16

echo "Starting single-GPU training on Device: $CUDA_VISIBLE_DEVICES"

python3 -m train_find.strain \
    --root_path "$ROOT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --img_size 224 \
    --num_classes 3 \
    --save_interval 20 \
    --deterministic 1

echo "Training script finished."
