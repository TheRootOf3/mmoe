#!/bin/bash
#SBATCH -w gpu-vm
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=priority
#SBATCH --job-name=multimodal-blip-fuser-mmsd

poetry run python blip2_fusion_train.py \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./mmsd_blip2_fuser \
--batch_size 32 \
--eval_steps 300 \
--epochs 10 \
--max_length 512 \
--lr 2e-4 \
