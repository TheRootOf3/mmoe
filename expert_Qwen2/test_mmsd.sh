#!/bin/bash
#SBATCH -w gpu-vm
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --partition=priority
#SBATCH --job-name=multimodal-qwen2-mmsd-test


poetry run python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_AS_model \
--save_path ./0.5_qwen_mmsd_AS_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;

poetry run python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_R_model \
--save_path ./0.5_qwen_mmsd_R_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;


poetry run python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_U_model \
--save_path ./0.5_qwen_mmsd_U_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;


poetry run python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_baseline_model \
--save_path ./0.5_qwen_mmsd_baseline_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;
