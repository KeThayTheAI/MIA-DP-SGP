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


# 1. Chon thiet bi
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# 2. Lay probability va digit label tu mot dataset
def get_predictions(model, dataset, batch_size, device):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_probabilities = []
    all_digit_labels = []

    model.eval()

    with torch.no_grad():
        for images, digit_labels in dataloader:
            images = images.to(device)

            # Lay logits tu model
            logits = model(images)

            # Chuyen logits thanh xac suat
            probabilities = torch.softmax(logits, dim=1)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_digit_labels.append(
                digit_labels.numpy()
            )

    probabilities = np.concatenate(all_probabilities, axis=0)
    digit_labels = np.concatenate(all_digit_labels, axis=0)

    return probabilities, digit_labels


# 3. Chuong trinh chinh
def main():
    config = load_config()
    data = load_datasets()
    experiment_config = config["experiment"]

    experiment_dir = (
        PROJECT_ROOT
        / experiment_config["output_dir"]
        / experiment_config["id"]
    )

    expected_models = config["dataset"]["num_shadow_models"]
    actual_models = len(data["shadow_models"])

    if actual_models != expected_models:
        raise ValueError(
            "Expected {} Shadow Models, but received {}.".format(
                expected_models,
                actual_models
            )
        )

    print("Number of Shadow Models:", actual_models)

    batch_size = config["training"]["batch_size"]
    device = get_device()

    checkpoint_dir = experiment_dir / "checkpoints"
    results_dir = experiment_dir / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Danh sach chua du lieu Attack
    all_features = []
    all_membership_labels = []
    all_digit_labels = []
    all_shadow_model_ids = []

    print("Device:", device)
    print("\n========== PREPARING ATTACK DATA ==========")

    # Xu ly lan luot cac Shadow Models
    for shadow in data["shadow_models"]:
        model_id = shadow["model_id"]
        expected_train_size = config["dataset"]["shadow_train_size"]
        expected_nonmember_size = config["dataset"]["shadow_nonmember_size"]

        actual_train_size = len(shadow["train"])
        actual_nonmember_size = len(shadow["nonmember"])

        if actual_train_size != expected_train_size:
            raise ValueError(
                "Shadow {} has incorrect member size.".format(model_id)
            )

        if actual_nonmember_size != expected_nonmember_size:
            raise ValueError(
                "Shadow {} has incorrect non-member size.".format(model_id)
            )

        print(f"\nProcessing Shadow Model {model_id}...")

        # 1. Tao model co cung kien truc
        model = MLP().to(device)

        # 2. Tai trong so da huan luyen
        checkpoint_path = (
            checkpoint_dir / f"shadow_model_{model_id}.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found: {}".format(checkpoint_path)
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        # Kiem tra checkpoint co dung Shadow Model khong
        if checkpoint["model_id"] != model_id:
            raise ValueError(
                "Checkpoint model ID does not match Shadow Model ID."
            )

        # Kiem tra checkpoint co thuoc Experiment hien tai khong
        checkpoint_experiment_id = (
            checkpoint["config"]["experiment"]["id"]
        )

        if checkpoint_experiment_id != experiment_config["id"]:
            raise ValueError(
                "Checkpoint belongs to another experiment."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # 3. Lay du doan tren Member
        member_probabilities, member_digit_labels = get_predictions(
            model,
            shadow["train"],
            batch_size,
            device
        )

        # 4. Lay du doan tren Non-member
        nonmember_probabilities, nonmember_digit_labels = get_predictions(
            model,
            shadow["nonmember"],
            batch_size,
            device
        )

        # 5. Tao Membership Labels
        member_labels = np.ones(
            len(member_probabilities),
            dtype=np.int64
        )

        nonmember_labels = np.zeros(
            len(nonmember_probabilities),
            dtype=np.int64
        )

        # 6. Ghep Member va Non-member
        features = np.concatenate(
            [member_probabilities, nonmember_probabilities],
            axis=0
        )

        membership_labels = np.concatenate(
            [member_labels, nonmember_labels],
            axis=0
        )

        digit_labels = np.concatenate(
            [member_digit_labels, nonmember_digit_labels],
            axis=0
        )

        model_ids = np.full(
            len(membership_labels),
            model_id,
            dtype=np.int64
        )

        # 7. Them vao danh sach chung
        all_features.append(features)
        all_membership_labels.append(membership_labels)
        all_digit_labels.append(digit_labels)
        all_shadow_model_ids.append(model_ids)

        print("Member samples:", len(member_probabilities))
        print("Non-member samples:", len(nonmember_probabilities))
        print("Feature shape:", features.shape)

        del model

    # 8. Ghep du lieu tu ca cac Shadow Models
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_membership_labels, axis=0)
    digit_y = np.concatenate(all_digit_labels, axis=0)
    shadow_ids = np.concatenate(all_shadow_model_ids, axis=0)
    # Kiem tra kich thuoc Attack Dataset
    expected_feature_size = config["model"]["num_classes"]

    if X.shape[1] != expected_feature_size:
        raise ValueError(
            "Expected {} features, but received {}.".format(
                expected_feature_size,
                X.shape[1]
            )
        )

    if not (len(X) == len(y) == len(digit_y) == len(shadow_ids)):
        raise ValueError("Attack Dataset arrays have inconsistent sizes.")

    if np.sum(y == 1) != expected_models * config["dataset"]["shadow_train_size"]:
        raise ValueError("Incorrect total Member samples.")

    if np.sum(y == 0) != expected_models * config["dataset"]["shadow_nonmember_size"]:
        raise ValueError("Incorrect total Non-member samples.")
    # 9. Kiem tra ket qua
    print("\n========== ATTACK DATASET SUMMARY ==========")
    print("Features shape:", X.shape)
    print("Membership labels shape:", y.shape)
    print("Digit labels shape:", digit_y.shape)

    print("Member samples:", int(np.sum(y == 1)))
    print("Non-member samples:", int(np.sum(y == 0)))

    print("Shadow model IDs:", np.unique(shadow_ids))

    # 10. Luu Attack Dataset
    output_path = results_dir / "attack_dataset.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        digit_labels=digit_y,
        shadow_model_ids=shadow_ids
    )

    print("\nSaved Attack Dataset:", output_path)
    print("========== ATTACK DATA PREPARATION COMPLETED ==========")


if __name__ == "__main__":
    main()
