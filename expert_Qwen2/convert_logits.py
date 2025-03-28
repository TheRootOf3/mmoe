import json
import os

import jsonlines

dataset_name = "urfunny"
model_size = "0.5"
model_seed = "32"

inference_results = [
    f"{dataset_name}_baseline",
    f"{dataset_name}_AS",
    f"{dataset_name}_R",
    f"{dataset_name}_U",
]
model_directories = [
    f"{model_size}_qwen_{dataset_name}_baseline_model_{model_seed}",
    f"{model_size}_qwen_{dataset_name}_AS_model_{model_seed}",
    f"{model_size}_qwen_{dataset_name}_R_model_{model_seed}",
    f"{model_size}_qwen_{dataset_name}_U_model_{model_seed}",
]


for inference_result, model_directory in zip(inference_results, model_directories):
    overall_dataset = []

    gth = {}
    with open(
        f"../{dataset_name}_data/data_raw/{dataset_name}_dataset_test.json", "r"
    ) as f:
        dataset = json.load(f)
        for image_id, data in dataset.items():
            if dataset_name == "mustard":
                gth[image_id] = data["sarcasm"]
            else:
                gth[image_id] = data["label"]

    with open(f"./{model_directory}/test_yesno_logits.json", "r") as f:
        inference_output = json.load(f)
        for image_id, logits in inference_output.items():
            overall_dataset.append(
                {"image_id": image_id, "logits": logits, "target": gth[image_id]}
            )

    # create expert_inference_output expert_blip2 directory if it does not exist
    if not os.path.exists(
        f"../{dataset_name}_data/expert_inference_output_{model_seed}/expert_qwen-{model_size}b"
    ):
        os.makedirs(
            f"../{dataset_name}_data/expert_inference_output_{model_seed}/expert_qwen-{model_size}b"
        )

    with jsonlines.open(
        f"../{dataset_name}_data/expert_inference_output_{model_seed}/expert_qwen-{model_size}b/{inference_result}_logits.jsonl",
        "w",
    ) as f:
        f.write_all(overall_dataset)

    # baseline model is not calibrated, so skip copying the calibration file
    if "baseline" not in inference_result:
        calibration_dict = {}

        with open(f"./{model_directory}/calibration.json", "r") as f:
            calibration_dict = json.load(f)

        with open(
            f"../{dataset_name}_data/expert_inference_output_{model_seed}/expert_qwen-{model_size}b/{inference_result}_calibration.json",
            "w",
        ) as f:
            json.dump(calibration_dict, f)
