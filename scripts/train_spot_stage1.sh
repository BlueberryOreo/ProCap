#!/bin/bash

echo "pid: $$"

source setup.sh

CUDA_VISIBLE_DEVICES=7 python3 train_stage1.py --name train_image_text_4layer_spot \
    --image_text_folder data/spot_for_mmvid/train \
    --dataset h5_text --beit --num_workers 32 \
    --num_visuals 0 --fullvc --batch_size 8 \
    --text_seq_len 24 \
    --use_html --log_every 200 \
    --sample_every 5000 --n_sample 4 --n_per_sample 1 \
    --num_targets 4 --frame_num 4 --deterministic --filtered --filter_file_path ./filter_files/spot_similarity_scores.json --max_k 2 \
    --image_size 224 --dropout_vc 0.4 \
    --dist_url tcp://localhost:10010 --vae_path ./pretrained_vqgan/spot_epoch=000042.ckpt --rel_no_fully_masked \
    --mask_predict_steps 10 20 30 --mask_predict_steps1 20 --vision_layers 4 \
