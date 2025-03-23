#!/bin/bash
#SBATCH -w gpu-vm
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=priority
#SBATCH --job-name=multimodal-blip-fuser-mustard


poetry run python blip2_fusion_train.py \
--dataset mustard \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./mustard_blip2_fuser \
--batch_size 4 \
--val_batch_size 4 \
--eval_steps 50 \
--lr 3e-4 \
--epochs 50 \
--max_length 512
