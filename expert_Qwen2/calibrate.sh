SEED=32

python3 calibrate_confidence.py \
--dataset funny \
--train_path ../urfunny_data/data_split_output/urfunny_R_dataset_train_cogvlm2_qwen2.json \
--val_path  ../urfunny_data/data_split_output/urfunny_R_dataset_val_cogvlm2_qwen2.json \
--test_path  ../urfunny_data/data_split_output/urfunny_R_dataset_test_cogvlm2_qwen2.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path "./0.5_qwen_urfunny_AS_model_${SEED}_calibrated" \
--load_model_name "./0.5_qwen_urfunny_AS_model_${SEED}" \
--eval_steps 10 \
--device 0;
