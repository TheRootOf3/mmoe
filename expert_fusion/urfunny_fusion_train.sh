#!/bin/bash
#SBATCH -w gpu-vm
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=priority
#SBATCH --job-name=multimodal-blip-fuser-urfunny


poetry run python blip2_fusion_train.py \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./urfunny_blip2_fuser_focal_loss \
--batch_size 16 \
--eval_steps 210 \
--epochs 50 \
--max_length 512
