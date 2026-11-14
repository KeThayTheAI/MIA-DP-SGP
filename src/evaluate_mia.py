import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Xac dinh thu muc goc cua project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config, load_datasets
from models.target_model import MLP
from models.attack_classifier import AttackClassifier


# ==========================================
# 1. SELECT DEVICE
# ==========================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==========================================
# 2. LOAD MODEL CHECKPOINT
# ==========================================

def load_model_checkpoint(model, checkpoint_path, device):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found: {}".format(checkpoint_path)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


# ==========================================
# 3. EXTRACT TARGET MODEL OUTPUTS
# ==========================================

def get_target_probabilities(model, dataset, batch_size, device):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_probabilities = []

    model.eval()

    with torch.no_grad():

        for images, digit_labels in dataloader:

            images = images.to(device)

            # Target Model produces 10 logits
            logits = model(images)

            # Convert logits into 10 probabilities
            probabilities = torch.softmax(logits, dim=1)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    return np.concatenate(all_probabilities, axis=0)


# ==========================================
# 4. CALCULATE ROC-AUC
# ==========================================

def calculate_auc(labels, scores):

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("Both membership classes are required.")

    # Sort scores in ascending order
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Assign average ranks to tied scores
    ranks = np.zeros(len(scores), dtype=np.float64)

    start = 0

    while start < len(scores):

        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (start + 1 + end) / 2

        ranks[sorted_indices[start:end]] = average_rank

        start = end

    positive_rank_sum = np.sum(ranks[labels == 1])

    n_positive = len(positive_scores)
    n_negative = len(negative_scores)

    auc = (
        positive_rank_sum
        - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)

    return float(auc)


# ==========================================
# 5. CALCULATE ATTACK METRICS
# ==========================================

def calculate_metrics(labels, predictions, scores):

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    accuracy = (tp + tn) / len(labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    auc = calculate_auc(labels, scores)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn
        }
    }


# ==========================================
# 6. EVALUATE ONE TARGET MODEL
# ==========================================

def evaluate_one_target(
    target_name,
    target_checkpoint_path,
    attack_model,
    member_dataset,
    nonmember_dataset,
    batch_size,
    device,
    experiment_id,
    metrics_path
):

    print("\n" + "=" * 55)
    print("Evaluating:", target_name)
    print("=" * 55)

    # --------------------------------------
    # Load Target Model
    # --------------------------------------

    target_model = MLP().to(device)

    target_model = load_model_checkpoint(
        target_model,
        target_checkpoint_path,
        device
    )

    print("Target Model loaded successfully.")

    # --------------------------------------
    # Extract Member probabilities
    # --------------------------------------

    print("Extracting Member outputs...")

    member_probabilities = get_target_probabilities(
        target_model,
        member_dataset,
        batch_size,
        device
    )

    # --------------------------------------
    # Extract Non-member probabilities
    # --------------------------------------

    print("Extracting Non-member outputs...")

    nonmember_probabilities = get_target_probabilities(
        target_model,
        nonmember_dataset,
        batch_size,
        device
    )

    # --------------------------------------
    # Create attack dataset
    # --------------------------------------

    X = np.concatenate(
        [member_probabilities, nonmember_probabilities],
        axis=0
    )

    y = np.concatenate([
        np.ones(len(member_probabilities), dtype=np.int64),
        np.zeros(len(nonmember_probabilities), dtype=np.int64)
    ])

    print("\nTarget Attack Data:")
    print("Features shape:", X.shape)
    print("Members:", int(np.sum(y == 1)))
    print("Non-members:", int(np.sum(y == 0)))

    # --------------------------------------
    # Run fixed Attack Classifier
    # --------------------------------------

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(device)

    attack_model.eval()

    with torch.no_grad():

        attack_logits = attack_model(X_tensor)

        attack_scores = torch.sigmoid(
            attack_logits
        ).cpu().numpy()
    member_scores = attack_scores[:len(member_probabilities)]
    nonmember_scores = attack_scores[len(member_probabilities):]

    print("\nAttack Score Statistics:")
    print("Member mean:", member_scores.mean())
    print("Member min:", member_scores.min())
    print("Member max:", member_scores.max())

    print("Non-member mean:", nonmember_scores.mean())
    print("Non-member min:", nonmember_scores.min())
    print("Non-member max:", nonmember_scores.max())
    predictions = (
        attack_scores >= 0.5
    ).astype(np.int64)

    # --------------------------------------
    # Calculate MIA metrics
    # --------------------------------------

    nn_metrics = calculate_metrics(
        y,
        predictions,
        attack_scores
    )

    metrics = {
        "experiment_id": experiment_id,
        "target_model": target_name,
        "member_samples": len(member_probabilities),
        "nonmember_samples": len(nonmember_probabilities),
        "total_samples": len(y),

        "neural_network_attack": {
            **nn_metrics,
            "threshold": 0.5
        }
    }

    # --------------------------------------
    # Save results
    # --------------------------------------

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(metrics_path, "w", encoding="utf-8") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # --------------------------------------
    # Display results
    # --------------------------------------

    print("\n--- MIA Results ---")

    print("Accuracy:  {:.4f} ({:.2f}%)".format(
        nn_metrics["accuracy"],
        nn_metrics["accuracy"] * 100
    ))

    print("Precision: {:.4f}".format(
        nn_metrics["precision"]
    ))

    print("Recall:    {:.4f}".format(
        nn_metrics["recall"]
    ))

    print("F1-score:  {:.4f}".format(
        nn_metrics["f1_score"]
    ))

    print("ROC-AUC:   {:.4f}".format(
        nn_metrics["roc_auc"]
    ))

    print("\nResults saved to:", metrics_path)


# ==========================================
# 7. MAIN EVALUATION
# ==========================================

def evaluate_mia():

    print("========== EVALUATE MIA ==========")

    # --------------------------------------
    # Load config and datasets
    # --------------------------------------

    config = load_config()
    data = load_datasets()

    experiment_config = config["experiment"]

    experiment_id = experiment_config["id"]

    experiment_dir = (
        PROJECT_ROOT
        / experiment_config["output_dir"]
        / experiment_id
    )

    attacker_dir = experiment_dir / "attacker"

    attack_checkpoint_path = (
        attacker_dir
        / "checkpoints"
        / "attack_classifier.pth"
    )

    batch_size = config["training"]["batch_size"]

    device = get_device()

    print("Experiment:", experiment_id)
    print("Device:", device)

    # --------------------------------------
    # Load Attack Classifier ONCE
    # --------------------------------------

    if not attack_checkpoint_path.exists():
        raise FileNotFoundError(
            "Attack checkpoint not found: {}".format(
                attack_checkpoint_path
            )
        )

    attack_checkpoint = torch.load(
        attack_checkpoint_path,
        map_location=device
    )

    input_size = attack_checkpoint.get(
        "input_size",
        10
    )

    attack_model = AttackClassifier(
        input_size=input_size
    ).to(device)

    attack_model.load_state_dict(
        attack_checkpoint["model_state_dict"]
    )

    attack_model.eval()

    print("Attack Classifier loaded successfully.")

    # --------------------------------------
    # Define common Target datasets
    # --------------------------------------

    member_dataset = data["target_train"]
    nonmember_dataset = data["target_nonmember"]

    # --------------------------------------
    # Evaluate Non-DP Target
    # --------------------------------------

    non_dp_dir = experiment_dir / "non_dp"

    non_dp_checkpoint = (
        non_dp_dir
        / "checkpoints"
        / "target_model.pth"
    )

    non_dp_metrics_path = (
        non_dp_dir
        / "results"
        / "mia_target_metrics.json"
    )

    evaluate_one_target(
        target_name="Non-DP",
        target_checkpoint_path=non_dp_checkpoint,
        attack_model=attack_model,
        member_dataset=member_dataset,
        nonmember_dataset=nonmember_dataset,
        batch_size=batch_size,
        device=device,
        experiment_id=experiment_id,
        metrics_path=non_dp_metrics_path
    )

    # --------------------------------------
    # Evaluate DP Target Models
    # --------------------------------------

    noise_multipliers = config["dp"]["noise_multipliers"]

    for noise_multiplier in noise_multipliers:

        noise_name = str(noise_multiplier).rstrip("0").rstrip(".")

        noise_dir_name = "noise_{}".format(
            noise_name.replace(".", "_")
        )

        dp_dir = experiment_dir / "dp" / noise_dir_name

        dp_checkpoint = (
            dp_dir
            / "checkpoints"
            / "target_model.pth"
        )

        dp_metrics_path = (
            dp_dir
            / "results"
            / "mia_metrics.json"
        )

        evaluate_one_target(
            target_name="DP (noise={})".format(
                noise_multiplier
            ),
            target_checkpoint_path=dp_checkpoint,
            attack_model=attack_model,
            member_dataset=member_dataset,
            nonmember_dataset=nonmember_dataset,
            batch_size=batch_size,
            device=device,
            experiment_id=experiment_id,
            metrics_path=dp_metrics_path
        )

    print("\n========== ALL MIA EVALUATIONS COMPLETED ==========")


# ==========================================
# 8. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    evaluate_mia()