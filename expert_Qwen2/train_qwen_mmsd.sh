#!/bin/bash
#SBATCH -w gpu-vm
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=priority
#SBATCH --job-name=multimodal-qwen2-mmsd

poetry run python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512 \
--load_model_name none;

poetry run python train.py \
--dataset mmsd \
--train_path  ../mmsd_data/data_split_output/mmsd_AS_dataset_train_cogvlm2_qwen2.json \
--val_path  ../mmsd_data/data_split_output/mmsd_AS_dataset_val_cogvlm2_qwen2.json \
--test_path  ../mmsd_data/data_split_output/mmsd_AS_dataset_test_cogvlm2_qwen2.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_AS_model \
--load_model_name ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

poetry run python train.py \
--dataset mmsd \
--train_path  ../mmsd_data/data_split_output/mmsd_R_dataset_train_cogvlm2_qwen2.json \
--val_path  ../mmsd_data/data_split_output/mmsd_R_dataset_val_cogvlm2_qwen2.json \
--test_path  ../mmsd_data/data_split_output/mmsd_R_dataset_test_cogvlm2_qwen2.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_R_model \
--load_model_name ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

poetry run python train.py \
--dataset mmsd \
--train_path  ../mmsd_data/data_split_output/mmsd_U_dataset_train_cogvlm2_qwen2.json \
--val_path  ../mmsd_data/data_split_output/mmsd_U_dataset_val_cogvlm2_qwen2.json \
--test_path  ../mmsd_data/data_split_output/mmsd_U_dataset_test_cogvlm2_qwen2.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_U_model \
--load_model_name ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--device 0 \
--seed 32 \
--max_length 512;