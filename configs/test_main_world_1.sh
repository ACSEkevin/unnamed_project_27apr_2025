# test training script
# PYTORCH_MPS_HIGH_WATERMARK_RATIO unlocks the constraint of GPU memory limit in MPS (at least) device

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python main.py \
    --device mps \
    --epoch 2 \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 0. \
    --weight_decay 1e-4 \
    --clip_max_norm 0.5 \
    --focal_alpha 0. \
    --batch_size 2 \
    --num_iters 3 \
    --random_drop 0.1 \
    --fp_ratio 0.2 \
    --fp_max_score 0.5 \
    --eval_freq 1 \
    --log_dir ./logs \
    --dataset_file "test" \
    --num_workers 2 \
    --update_track_pos
    --init_state ./detr-r50-e632da11.pth