#!/bin/bash

echo $$

# set pretrained model path
symlink_path="/path/to/stage1/weight/dalle.pt"
config="./config/mmvid_edit_config.yaml"
seed=42
GPUS=(6 7)
PORT=9433

source setup.sh

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --num_processes ${#GPUS[@]} --main_process_port $PORT --mixed_precision no \
  train_stage2.py \
  --config $config \
  --save_model model \
  --save_mode best \
  --res_root_dir ./logs \
  --no_pin_memory \
  --seed $seed \
  --pretrained_model $symlink_path \

