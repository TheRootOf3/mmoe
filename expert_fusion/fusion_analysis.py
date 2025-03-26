import json
import os
from collections import defaultdict

import jsonlines
import numpy as np
from matplotlib import pyplot as plt

INTERACTION_TYPE_DICT = {"R": 0, "U": 1, "AS": 2}


def load_weights(weights_file):
    with jsonlines.open(weights_file, "r") as f:
        return {line["image_id"]: line["logits"] for line in f}


def load_and_transform_data(dataset_name, file_dir, subset_names, weights=None):
    results = defaultdict(
        lambda: {"logits": defaultdict(list), "target": None, "weights": {}}
    )
    calibration_dict = {}
    for name in subset_names:
        file_path = os.path.join(file_dir, f"{dataset_name}_{name}_logits.jsonl")
        calibration_file_path = os.path.join(
            file_dir, f"{dataset_name}_{name}_calibration.json"
        )
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

        with open(calibration_file_path, "r") as f:
            calibration_dict[name] = json.load(f)

    return results, calibration_dict


def analyse_confidence(dataset_name, results, calibration_dict=None):
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

        # not to confuse: logits_R means logits of the R expert model on all samples, not only the samples that are of R interaction type
        if calibration_dict is not None:
            logits_R = [
                x / calibration_dict["R"]["softmax_temperature"]
                for x in data["logits"]["R"]
            ]
            logits_U = [
                x / calibration_dict["U"]["softmax_temperature"]
                for x in data["logits"]["U"]
            ]
            logits_AS = [
                x / calibration_dict["AS"]["softmax_temperature"]
                for x in data["logits"]["AS"]
            ]
        else:
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
    dataset_name = "urfunny"
    model_name = "qwen-0.5b"
    seed = 32

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

    file_dir = f"../{dataset_name}_data/expert_inference_output_32/expert_{model_name}"
    # file_dir = f"../{dataset_name}_data/new_expert_inference_output/expert_{model_name}"
    weights_file = f"../{dataset_name}_data/expert_inference_output_32/expert_{model_name}/{dataset_name}_rus_logits.jsonl"
    # weights_file = "./urfunny_blip2_fuser_focal_loss/test_rus_logits.jsonl"
    # weights_file = "./mmsd_blip2_fuser/test_rus_logits.jsonl"
    # weights_file = "./mustard_blip2_fuser/test_rus_logits.jsonl"
    subset_names = INTERACTION_TYPE_DICT.keys()

    weights = load_weights(weights_file) if os.path.exists(weights_file) else None
    results, calibration_dict = load_and_transform_data(
        dataset_name, file_dir, subset_names, weights
    )

    # calibration_dict = None

    (
        analysis_labels,
        analysis_confidence,
        analysis_targets,
        analysis_interaction_type,
    ) = analyse_confidence(dataset_name, results, calibration_dict)

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
                f"bin_id: {bin_id}, bin_acc: {bin_acc:.2f}, perfect_acc: {bin_id / 10 +0.05 :.2f}, bin_conf:{bin_conf:.2f}, perfect_conf:{bin_id / 10 +0.05 :.2f}"
            )

        print(f"Expected Calibration Error (ECE): {running_ece:.2f}")

        # overall accuracy
        overall_acc = np.sum(analysis_labels[:, expert_id] == analysis_targets) / len(
            analysis_targets
        )
        print(f"overall_acc: {overall_acc:.2f}")

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
        plt.savefig(
            f"confidence_histogram_{expert_name}{'_calibrated' if calibration_dict is not None else ''}.png"
        )
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
        plt.savefig(
            f"reliability_diagram_{expert_name}{'_calibrated' if calibration_dict is not None else ''}.png"
        )
        plt.clf()

    # what is the average confidence for each interaction type when the model is correct?

    numbers_to_plot = []
    for expert_name, expert_id in INTERACTION_TYPE_DICT.items():
        for interaction_type_name, interaction_type_id in INTERACTION_TYPE_DICT.items():
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
                f"Expert: {expert_name}, Interaction Type: {interaction_type_name}, Average Confidence: {avg_conf:.2f}, Average Accuracy: {acc:.2f}"
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
    plt.savefig(
        f"average_confidence_correct{'_calibrated' if calibration_dict is not None else ''}.png"
    )
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


if __name__ == "__main__":
    main()
