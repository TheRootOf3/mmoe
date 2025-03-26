SEED=32

for INTERACTION_TYPE in R U AS
do
    poetry run python3 calibrate_confidence.py \
    --dataset funny \
    --train_path ../urfunny_data/data_split_output/urfunny_${INTERACTION_TYPE}_dataset_train_cogvlm2_qwen2.json \
    --val_path  ../urfunny_data/data_split_output/urfunny_${INTERACTION_TYPE}_dataset_val_cogvlm2_qwen2.json \
    --test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
    --image_data_path ../urfunny_data/data_raw/images \
    --save_path "./0.5_qwen_urfunny_${INTERACTION_TYPE}_model_${SEED}" \
    --load_model_name "./0.5_qwen_urfunny_${INTERACTION_TYPE}_model_${SEED}" \
    --eval_steps 10 \
    --batch_size 1 \
    --device 0;
done