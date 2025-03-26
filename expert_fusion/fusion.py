import json
import os
from collections import defaultdict
import time

import jsonlines
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tabulate
from matplotlib import pyplot as plt

INTERACTION_TYPE_DICT = {"R": 0, "U": 1, "AS": 2}


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=np.float16)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


def load_weights(weights_file):
    with jsonlines.open(weights_file, "r") as f:
        return {line["image_id"]: line["logits"] for line in f}


def load_and_transform_data(dataset_name, file_dir, subset_names, weights=None):
    results = defaultdict(
        lambda: {"logits": defaultdict(list), "target": None, "weights": {}}
    )
    for name in subset_names:
        file_path = os.path.join(file_dir, f"{dataset_name}_{name}_logits.jsonl")
        with jsonlines.open(file_path, "r") as f:
            for line in f:
                data_id = line["image_id"]
                results[data_id]["logits"][name] = line["logits"]
                results[data_id]["target"] = (
                    results[data_id]["target"] or line["target"]
                )
                assert (
                    results[data_id]["target"] == line["target"]
                ), "Targets do not match across subsets for the same data."
                if weights and data_id in weights:
                    results[data_id]["weights"][name] = weights[data_id][name]
    return results


def calculate_metrics(gths, preds):
    return {
        "f1": f1_score(gths, preds),
        "precision": precision_score(gths, preds),
        "recall": recall_score(gths, preds),
        "accuracy": accuracy_score(gths, preds),
    }


def get_predictions(results, fusion_strategy, *args):
    gths, preds = [], []
    for data in results.values():
        predicted_label = fusion_strategy(
            data["logits"], data["weights"], data["target"], *args
        )
        gths.append(data["target"])
        preds.append(predicted_label)
    return calculate_metrics(gths, preds)


def weighted_softmax_rus_fusion(logits, weights, *args):
    softmax_weights = {
        "R": np.exp(weights["R"])
        / (np.exp(weights["R"]) + np.exp(weights["U"]) + np.exp(weights["AS"])),
        "U": np.exp(weights["U"])
        / (np.exp(weights["R"]) + np.exp(weights["U"]) + np.exp(weights["AS"])),
        "AS": np.exp(weights["AS"])
        / (np.exp(weights["R"]) + np.exp(weights["U"]) + np.exp(weights["AS"])),
    }
    softmax_logits = {
        name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()
    }
    weighted_logits = sum(
        softmax_weights[name] * np.array(logit)
        for name, logit in softmax_logits.items()
    )
    return np.argmax(weighted_logits)


def weighted_softmax_temperature_rus_fusion(logits, weights, t, *args):
    softmax_weights = {
        "R": np.exp(weights["R"] / t)
        / (
            np.exp(weights["R"] / t)
            + np.exp(weights["U"] / t)
            + np.exp(weights["AS"] / t)
        ),
        "U": np.exp(weights["U"] / t)
        / (
            np.exp(weights["R"] / t)
            + np.exp(weights["U"] / t)
            + np.exp(weights["AS"] / t)
        ),
        "AS": np.exp(weights["AS"] / t)
        / (
            np.exp(weights["R"] / t)
            + np.exp(weights["U"] / t)
            + np.exp(weights["AS"] / t)
        ),
    }
    softmax_weights_control = {
        "R": np.exp(weights["R"])
        / (np.exp(weights["R"]) + np.exp(weights["U"]) + np.exp(weights["AS"])),
        "U": np.exp(weights["U"])
        / (np.exp(weights["R"]) + np.exp(weights["U"]) + np.exp(weights["AS"])),
        "AS": np.exp(weights["AS"])
        / (np.exp(weights["R"]) + np.exp(weights["U"]) + np.exp(weights["AS"])),
    }

    softmax_logits = {
        name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()
    }
    weighted_logits = sum(
        softmax_weights[name] * np.array(logit)
        for name, logit in softmax_logits.items()
    )
    return np.argmax(weighted_logits)


def simple_average(logits, *args):
    avg_logits = np.mean([logits[name] for name in logits], axis=0)
    return np.argmax(avg_logits)


def simple_average_sigmoid(logits, *args):
    avg_logits = np.mean([sigmoid(np.array(logits[name])) for name in logits], axis=0)
    # avg_logits_normal = np.mean([logits[name] for name in logits], axis=0)
    # print([sigmoid(np.array(logits[name])) for name in logits])
    # print([logits[name] for name in logits])
    # print(avg_logits, avg_logits_normal)
    return np.argmax(avg_logits)


def weighted_average(logits, weights, *args):
    weighted_logits = sum(weights[name] * np.array(logits[name]) for name in logits)
    return np.argmax(weighted_logits)


def max_fusion(logits, *args):
    max_logits = np.max([logits[name] for name in logits], axis=0)
    return np.argmax(max_logits)


def softmax_fusion(logits, *args):
    softmaxed_probs = np.mean(
        [np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits], axis=0
    )
    return np.argmax(softmaxed_probs)


def test_confidence_average(logits, weights, target, interaction_type, *args):
    # print(logits)
    # print(target)
    # print(interaction_type)
    # print([np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits])
    # time.sleep(1)

    # computing the confidence as the difference between softmax probabilities
    softmax_outputs_per_expert = np.array(
        [np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits]
    )
    confidence_per_expert = softmax_outputs_per_expert.max(
        axis=1
    ) - softmax_outputs_per_expert.min(axis=1)

    softmaxed_confidence_per_expert = np.exp(confidence_per_expert) / np.sum(
        np.exp(confidence_per_expert)
    )

    # print(confidence_per_expert)
    # print(softmax_outputs_per_expert)
    interaction_type_confidence = {v: k for k, v in INTERACTION_TYPE_DICT.items()}[
        np.argmax(confidence_per_expert)
    ]
    interaction_type_model = {v: k for k, v in INTERACTION_TYPE_DICT.items()}[
        np.argmax(weights)
    ]
    # print(
    #     f"confidence_predicted_interaction: {interaction_type_confidence}, model_predicted_interaction: {interaction_type_model} real_interaction: {interaction_type}"
    # )

    # weight the softmax probabilities by the confidence
    weighted_softmaxed_probs = softmaxed_confidence_per_expert @ np.array(
        [np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits]
    )

    # weight logits by confidence and then apply softmax
    # weighted_logits = softmaxed_confidence_per_expert @ np.array(
    #     [logits[name] for name in logits]
    # )
    # weighted_softmaxed_probs = np.exp(weighted_logits) / np.sum(
    #     np.exp(weighted_logits)
    # )
    return np.argmax(weighted_softmaxed_probs)


def cascaded_fusion(logits, threshold, *args):
    softmaxed_probs = {
        name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()
    }
    if max(softmaxed_probs["R"]) > threshold and max(softmaxed_probs["U"]) > threshold:
        return np.argmax(
            softmaxed_probs["R"]
            if max(softmaxed_probs["R"]) > max(softmaxed_probs["U"])
            else softmaxed_probs["U"]
        )
    return np.argmax(softmaxed_probs["AS"])


def get_oracle_prediction(dataset_name, logits):
    fusion_type_dict = {}
    for interaction_type in INTERACTION_TYPE_DICT.keys():
        with open(
            f"../{dataset_name}_data/data_split_output/{dataset_name}_{interaction_type}_dataset_test_cogvlm2_qwen2.json",
            "r",
        ) as f:
            dataset = json.load(f)
        for image_id in dataset:
            fusion_type_dict[image_id] = interaction_type

    gths, preds = [], []
    for data_id, data in logits.items():
        interaction_type = fusion_type_dict[data_id]
        pred = np.argmax(data["logits"][interaction_type])
        gths.append(data["target"])
        preds.append(pred)
    return calculate_metrics(gths, preds)


def get_prediction_analysis(dataset_name, results, fusion_strategy, *args):
    fusion_type_dict = {}
    for interaction_type in INTERACTION_TYPE_DICT.keys():
        with open(
            f"../{dataset_name}_data/data_split_output/{dataset_name}_{interaction_type}_dataset_test_cogvlm2_qwen2.json",
            "r",
        ) as f:
            dataset = json.load(f)
        for image_id in dataset:
            fusion_type_dict[image_id] = interaction_type

    gths, preds, meta = [], [], []
    for data_id, data in results.items():
        interaction_type = fusion_type_dict[data_id]
        predicted_label = fusion_strategy(
            data["logits"],
            data["weights"],
            data["target"],
            interaction_type,
            *args,
        )

        # computing the confidence as the difference between softmax probabilities
        softmax_outputs_per_expert = np.array(
            [
                np.exp(data["logits"][name]) / np.sum(np.exp(data["logits"][name]))
                for name in INTERACTION_TYPE_DICT.keys()
            ]
        )
        confidence_per_expert = softmax_outputs_per_expert.max(
            axis=1
        ) - softmax_outputs_per_expert.min(axis=1)

        softmaxed_confidence_per_expert = np.exp(confidence_per_expert) / np.sum(
            np.exp(confidence_per_expert)
        )

        gths.append(data["target"])
        preds.append(predicted_label)
        meta.append(
            {
                "real_interaction_type": INTERACTION_TYPE_DICT[interaction_type],
                "softmaxed_confidence_per_expert": softmaxed_confidence_per_expert.tolist(),
                "weight_per_expert": list(data["weights"].values()),
            }
        )
    return calculate_metrics(gths, preds), meta


def main():
    dataset_name = "mmsd"
    model_name = "qwen-0.5b"

    with open(
        f"../{dataset_name}_data/data_split_output/{dataset_name}_AS_dataset_test_cogvlm2_qwen2.json",
        "r",
    ) as f:
        dataset = json.load(f)
    AS_test_data_ids = list(dataset.keys())

    with open(
        f"../{dataset_name}_data/data_split_output/{dataset_name}_R_dataset_test_cogvlm2_qwen2.json",
        "r",
    ) as f:
        dataset = json.load(f)
    R_test_data_ids = list(dataset.keys())

    with open(
        f"../{dataset_name}_data/data_split_output/{dataset_name}_U_dataset_test_cogvlm2_qwen2.json",
        "r",
    ) as f:
        dataset = json.load(f)
    U_test_data_ids = list(dataset.keys())

    file_dir = f"../{dataset_name}_data/expert_inference_output/expert_{model_name}"
    # file_dir = f"../{dataset_name}_data/new_expert_inference_output/expert_{model_name}"
    weights_file = f"../{dataset_name}_data/expert_inference_output/expert_{model_name}/{dataset_name}_rus_logits.jsonl"
    # weights_file = "./urfunny_blip2_fuser_focal_loss/test_rus_logits.jsonl"
    # weights_file = "./mmsd_blip2_fuser/test_rus_logits.jsonl"
    # weights_file = "./mustard_blip2_fuser/test_rus_logits.jsonl"
    subset_names = INTERACTION_TYPE_DICT.keys()

    weights = load_weights(weights_file) if os.path.exists(weights_file) else None
    results = load_and_transform_data(dataset_name, file_dir, subset_names, weights)

    baseline_results = load_and_transform_data(dataset_name, file_dir, ["baseline"], {})
    results_log = {}

    results_log["Baseline Interaction Type Accuracy"] = (
        get_predictions(baseline_results, lambda x, _, __: np.argmax(x["baseline"])),
    )
    results_log["[test] confidence average"], meta_interaction_confidence = (
        get_prediction_analysis(dataset_name, results, test_confidence_average)
    )

    if weights:
        results_log["RUS Fusion"] = get_predictions(
            results, weighted_softmax_rus_fusion
        )
        results_log["RUS Fusion t=1e-1"] = get_predictions(
            results, lambda x, y, _: weighted_softmax_temperature_rus_fusion(x, y, 1e-1)
        )
        # results_log["RUS Fusion t=100"] = get_predictions(
        #     results, lambda x, y: weighted_softmax_temperature_rus_fusion(x, y, 100)
        # )

    # tmp_res = {}
    # temps_exponents = list(range(1, 21))
    # for temp_exp in temps_exponents:
    #     tmp_res[f"temp={temp_exp / temps_exponents[-1]}"] = get_predictions(
    #         results,
    #         lambda x, y: weighted_softmax_temperature_rus_fusion(
    #             x, y, temp_exp / temps_exponents[-1]
    #         ),
    #     )

    # plt.plot(
    #     [x / temps_exponents[-1] for x in temps_exponents],
    #     [res["accuracy"] for res in tmp_res.values()],
    #     "-o",
    # )
    # plt.axhline(results_log["RUS Fusion"]["accuracy"], color="r", linestyle="--")
    # plt.grid(alpha=0.6, zorder=1)
    # plt.show()

    results_log["Oracle Prediction"] = get_oracle_prediction(dataset_name, results)
    results_log["Simple Average Fusion"] = get_predictions(results, simple_average)
    results_log["Simple Average Sigmoid Fusion"] = get_predictions(
        results, simple_average_sigmoid
    )
    results_log["Max Fusion"] = get_predictions(results, max_fusion)
    results_log["Softmax Fusion"] = get_predictions(results, softmax_fusion)

    subpart_results = {"AS": {}, "R": {}, "U": {}}
    subpart_baseline_results = {"AS": {}, "R": {}, "U": {}}
    for result in results:
        if result in AS_test_data_ids:
            subpart_results["AS"][result] = results[result]
            subpart_baseline_results["AS"][result] = baseline_results[result]
        elif result in R_test_data_ids:
            subpart_results["R"][result] = results[result]
            subpart_baseline_results["R"][result] = baseline_results[result]
        elif result in U_test_data_ids:
            subpart_results["U"][result] = results[result]
            subpart_baseline_results["U"][result] = baseline_results[result]

    for interaction_type in subset_names:
        results_log[
            f"{interaction_type} expert results on the {interaction_type} test set"
        ] = (
            get_predictions(
                subpart_results[interaction_type],
                lambda x, _, __: np.argmax(x[interaction_type]),
            ),
        )

        results_log[f"Baseline results on the {interaction_type} test set"] = (
            get_predictions(
                subpart_baseline_results[interaction_type],
                lambda x, _, __: np.argmax(x["baseline"]),
            ),
        )

        results_log[f"{interaction_type} expert results on the whole test set"] = (
            get_predictions(results, lambda x, _, __: np.argmax(x[interaction_type]))
        )

    # Prepare table rows; extract metrics even if they are wrapped in a tuple.
    table = []
    for method, metrics in results_log.items():
        # If metrics is a tuple, take the first element.
        if isinstance(metrics, tuple):
            metrics = metrics[0]
        # Append a row: method, accuracy, f1, precision, recall.
        table.append(
            [
                method,
                100 * metrics.get("accuracy", None),
                100 * metrics.get("f1", None),
                100 * metrics.get("precision", None),
                100 * metrics.get("recall", None),
            ]
        )

    # Define headers for each metric.
    headers = ["Method", "Accuracy", "F1", "Precision", "Recall"]

    # Print the table in GitHub-flavored markdown style.
    print(
        tabulate.tabulate(
            table,
            headers=headers,
            tablefmt="github",
            floatfmt=".2f",
        )
    )


if __name__ == "__main__":
    main()
