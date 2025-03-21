python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name 1.5_qwen_mustard_U_model \
--save_path ./qwen_mustard_test_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;

python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name 1.5_qwen_mustard_AS_model \
--save_path ./qwen_mustard_test_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;

python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name 1.5_qwen_mustard_R_model \
--save_path ./qwen_mustard_test_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;

python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name 1.5_qwen_mustard_baseline_model \
--save_path ./qwen_mustard_U_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;
