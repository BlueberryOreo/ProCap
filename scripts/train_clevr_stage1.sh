#!/bin/bash

echo "pid: $$"

source setup.sh

python3 train_stage1.py --name train_image_text_12layer_clevr \
    --image_text_folder data/clevr_for_mmvid/train \
    --dataset h5_text --beit \
    --num_visuals 0 --fullvc --batch_size 8 \
    --text_seq_len 24 \
    --use_html --log_every 200 \
    --sample_every 5000 --n_sample 4 --n_per_sample 1 \
    --num_targets 4 --frame_num 4 --max_k 2 --filtered --filter_file_path ./filter_files/clevr_similarity_scores.json \
    --image_size 224 --dropout_vc 0.4 \
    --dist_url tcp://localhost:10010 --vae_path ./pretrained_vqgan/clevr_epoch=000035.ckpt --rel_no_fully_masked \
    --mask_predict_steps 10 20 30 --mask_predict_steps1 20 --vision_layers 12 \
