#!/bin/bash

poetry run python blip2_fusion_train.py \
--mode test \
--dataset mustard \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name mustard_blip2_fuser \
--val_batch_size 4 \
--eval_steps 10 \
--epochs 5 \
--max_length 512
