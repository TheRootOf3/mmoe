#!/bin/bash

poetry run python blip2_fusion_train.py \
--mode test \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name urfunny_blip2_fuser_focal_loss \
--val_batch_size 4 \
--eval_steps 10 \
--epochs 5 \
--max_length 512
