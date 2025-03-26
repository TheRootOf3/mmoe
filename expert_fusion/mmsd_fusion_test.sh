#!/bin/bash

poetry run python blip2_fusion_train.py \
--mode test \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name mmsd_blip2_fuser \
--val_batch_size 4 \
--eval_steps 10 \
--epochs 5 \
--max_length 512
