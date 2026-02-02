#!/bin/bash

echo $$

# set pretrained model path
resume_path="/path/to/pretrained/model/model.chkpt"
config="./config/mmvid_edit_config.yaml"
seed=42
PORT=9433

source setup.sh

TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch --num_processes 1 --main_process_port $PORT --mixed_precision no \
  train_stage2.py \
  --config $config \
  --save_model model \
  --save_mode best \
  --res_root_dir ./logs \
  --no_pin_memory \
  --seed $seed \
  --resume $resume_path \
  --eval
