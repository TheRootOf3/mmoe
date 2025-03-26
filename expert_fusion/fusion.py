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


def analyse_confidence(dataset_name, results):
    fusion_type_dict = {}
    for interaction_type in INTERACTION_TYPE_DICT.keys():
        with open(
            f"../{dataset_name}_data/data_split_output/{dataset_name}_{interaction_type}_dataset_test_cogvlm2_qwen2.json",
            "r",
        ) as f:
            dataset = json.load(f)
        for image_id in dataset:
            fusion_type_dict[image_id] = interaction_type

    labels, confidence, targets, interaction_types = [], [], [], []
    for data_id, data in results.items():
        interaction_type = INTERACTION_TYPE_DICT[fusion_type_dict[data_id]]

        logits_R = data["logits"]["R"]
        logits_U = data["logits"]["U"]
        logits_AS = data["logits"]["AS"]

        predicted_label_R = np.argmax(logits_R)
        predicted_label_U = np.argmax(logits_U)
        predicted_label_AS = np.argmax(logits_AS)

        confidence_R = np.exp(logits_R[predicted_label_R]) / np.sum(np.exp(logits_R))
        confidence_U = np.exp(logits_U[predicted_label_U]) / np.sum(np.exp(logits_U))
        confidence_AS = np.exp(logits_AS[predicted_label_AS]) / np.sum(
            np.exp(logits_AS)
        )

        labels.append([predicted_label_R, predicted_label_U, predicted_label_AS])
        confidence.append([confidence_R, confidence_U, confidence_AS])
        targets.append(data["target"])
        interaction_types.append(interaction_type)

    return (
        np.array(labels),
        np.array(confidence),
        np.array(targets),
        np.array(interaction_types),
    )


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

    (
        analysis_labels,
        analysis_confidence,
        analysis_targets,
        analysis_interaction_type,
    ) = analyse_confidence(dataset_name, results)

    for expert_name, expert_id in INTERACTION_TYPE_DICT.items():
        print(f"Interaction type: {expert_name}")

        bin_idx = (
            np.digitize(
                analysis_confidence[:, expert_id],
                bins=[x / 10 for x in range(11)],
            )
            - 1  # digitize returns 1-indexed bins, account for it to count from 0
        )

        bin_ids = []
        bin_accs = []
        bin_confs = []
        running_ece = 0
        for bin_id in range(10):
            bin_data_idx = bin_idx == bin_id

            if np.sum(bin_data_idx) == 0:
                continue

            bin_acc = np.sum(
                analysis_labels[:, expert_id][bin_data_idx]
                == analysis_targets[bin_data_idx]
            ) / len(analysis_targets[bin_data_idx])
            bin_conf = np.sum(analysis_confidence[:, expert_id][bin_data_idx]) / len(
                analysis_confidence[:, expert_id][bin_data_idx]
            )
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_ids.append(bin_id)
            running_ece += (
                abs(bin_acc - bin_conf)
                * len(analysis_targets[bin_data_idx])
                / len(analysis_targets)
            )
            print(
                f"bin_id: {bin_id}, bin_acc: {bin_acc}, perfect_acc: {bin_id / 10 +0.05 :.2f}, bin_conf:{bin_conf}, perfect_conf:{bin_id / 10 +0.05 :.2f}"
            )

        print(f"Expected Calibration Error (ECE): {running_ece}")

        # overall accuracy
        overall_acc = np.sum(analysis_labels[:, expert_id] == analysis_targets) / len(
            analysis_targets
        )
        print(f"overall_acc: {overall_acc}")

        plt.hist(
            analysis_confidence[:, expert_id],
            bins=[x / 10 for x in range(5, 11)],
            zorder=2,
            edgecolor="darkblue",
        )
        plt.grid(alpha=0.6, zorder=1)
        plt.xticks(ticks=[x / 10 for x in range(5, 11)])
        plt.xlim(0.5, 1)
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.title(f"Confidence Histogram for {expert_name}")
        plt.savefig(f"confidence_histogram_{expert_name}.png")
        plt.clf()

        plt.bar(
            [x / 10 + 0.05 for x in bin_ids],
            bin_accs,
            width=0.1,
            align="center",
            alpha=0.7,
            zorder=3,
            color="blue",
            edgecolor="darkblue",
            label="actual accuracy",
        )
        plt.bar(
            [x / 10 + 0.05 for x in bin_ids],
            [x / 10 + 0.05 for x in bin_ids],
            width=0.1,
            align="center",
            alpha=0.3,
            color="red",
            zorder=2,
            edgecolor="darkred",
            label="perfect calibration",
        )
        plt.legend()
        plt.grid(alpha=0.6, zorder=1)
        plt.xticks(ticks=[x / 10 for x in range(5, 11)])
        plt.xlim(0.5, 1)
        plt.ylim(0, 1)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title(f"Reliability Diagram\nAccuracy vs Confidence for {expert_name}")
        plt.savefig(f"reliability_diagram_{expert_name}.png")
        plt.clf()

    # what is the average confidence for each interaction type when the model is correct?

    numbers_to_plot = []
    for expert_name, expert_id in INTERACTION_TYPE_DICT.items():
        for interaction_type_id in INTERACTION_TYPE_DICT.values():
            correct_idx = (analysis_labels[:, expert_id] == analysis_targets) & (
                analysis_interaction_type == interaction_type_id
            )
            avg_conf = np.mean(analysis_confidence[:, expert_id][correct_idx])
            acc = np.mean(
                analysis_labels[:, expert_id][
                    analysis_interaction_type == interaction_type_id
                ]
                == analysis_targets[analysis_interaction_type == interaction_type_id]
            )
            numbers_to_plot.append((interaction_type_id, expert_id, avg_conf, acc))
            print(
                f"Expert: {expert_name}, Interaction Type: {interaction_type_id}, Average Confidence: {avg_conf}, Average Accuracy: {acc}"
            )

    numbers_to_plot = np.array(numbers_to_plot)
    plt.bar(
        [x - 0.1 for x in range(3)],
        numbers_to_plot[numbers_to_plot[:, 1] == 0, 2],
        width=0.1,
        color="blue",
        edgecolor="darkblue",
        alpha=0.7,
        label="expert R",
        zorder=2,
    )
    plt.bar(
        range(3),
        numbers_to_plot[numbers_to_plot[:, 1] == 1, 2],
        width=0.1,
        color="red",
        edgecolor="darkred",
        alpha=0.7,
        label="expert U",
        zorder=2,
    )
    plt.bar(
        [x + 0.1 for x in range(3)],
        numbers_to_plot[numbers_to_plot[:, 1] == 2, 2],
        width=0.1,
        color="green",
        edgecolor="darkgreen",
        alpha=0.7,
        label="expert AS",
        zorder=2,
    )

    plt.scatter(
        [x - 0.1 for x in range(3)],
        numbers_to_plot[numbers_to_plot[:, 1] == 0, 3],
        # width=0.1,
        color="black",
        marker="_",
        s=100,
        # edgecolor="darkblue",
        alpha=0.7,
        # label="expert R",
        zorder=2,
    )
    plt.scatter(
        range(3),
        numbers_to_plot[numbers_to_plot[:, 1] == 1, 3],
        # width=0.1,
        color="black",
        marker="_",
        s=100,
        # edgecolor="darkred",
        alpha=0.7,
        # label="expert U",
        zorder=3,
    )
    plt.scatter(
        [x + 0.1 for x in range(3)],
        numbers_to_plot[numbers_to_plot[:, 1] == 2, 3],
        # width=0.1,
        color="black",
        marker="_",
        s=100,
        # edgecolor="darkgreen",
        alpha=0.7,
        # label="expert AS",
        zorder=2,
    )

    plt.grid(alpha=0.6, zorder=1)
    plt.xticks(ticks=[0, 1, 2], labels=["Redundancy", "Uniqueness", "Synthesis"])
    plt.ylim(0, 1.2)
    plt.xlabel("Interaction Type")
    plt.ylabel("Average Confidence")
    plt.title("Average Confidence for each Interaction Type when the Model is Correct")
    plt.legend()
    plt.savefig("average_confidence_correct.png")
    plt.clf()

    # calulcate interaction stats:
    # confidence_confusion_matrix = np.zeros((3, 3))
    # weights_confusion_matrix = np.zeros((3, 3))
    # for meta in meta_interaction_confidence:
    #     real_interaction = meta["real_interaction_type"]
    #     confidence = np.argmax(meta["softmaxed_confidence_per_expert"])
    #     weight = np.argmax(meta["weight_per_expert"])
    #     confidence_confusion_matrix[real_interaction, confidence] += 1
    #     weights_confusion_matrix[real_interaction, weight] += 1

    # print("confidence_confusion_matrix")
    # print(confidence_confusion_matrix)
    # print("weights_confusion_matrix")
    # print(weights_confusion_matrix)

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
