import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Xac dinh thu muc goc cua project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# De Python tim duoc cac module trong project
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config, load_datasets
from models.target_model import MLP


# 1. Chon thiet bi huan luyen
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# 2. Thiet lap random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 3. Huan luyen mot Epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# 4. Danh gia mo hinh
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# 5. Chuong trinh chinh
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
    seed = config["dataset"]["random_seed"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    device = get_device()

    print("Device:", device)
    print("\n========== SHADOW MODEL TRAINING ==========")

    # DataLoader danh gia chung
    utility_loader = DataLoader(
        data["utility_test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()

    # Thu muc luu checkpoint va metrics
    checkpoint_dir = experiment_dir / "checkpoints"
    results_dir = experiment_dir / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # Huan luyen lan luot cac Shadow Models
    for shadow in data["shadow_models"]:
        model_id = shadow["model_id"]

        # Moi Shadow Model co random seed rieng
        set_seed(seed + model_id)

        print(f"\n========== SHADOW MODEL {model_id} ==========")

        # Lay du lieu Member cua Shadow Model hien tai
        train_dataset = shadow["train"]
        nonmember_dataset = shadow["nonmember"]
        if len(train_dataset) != config["dataset"]["shadow_train_size"]:
            raise ValueError(
                "Shadow {} has incorrect training size.".format(model_id)
            )

        if len(nonmember_dataset) != config["dataset"]["shadow_nonmember_size"]:
            raise ValueError(
                "Shadow {} has incorrect non-member size.".format(model_id)
            )

        # Chi dung Member de huan luyen
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # Khoi tao mot MLP moi
        model = MLP().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        # Training
        for epoch in range(epochs):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {train_loss:.4f} "
                f"Accuracy: {train_accuracy:.4f}"
            )

        # Danh gia tren Utility Test
        test_loss, test_accuracy = evaluate(
            model,
            utility_loader,
            criterion,
            device
        )

        print("\nFinal Evaluation:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        # Luu checkpoint rieng
        checkpoint_path = (
            checkpoint_dir / f"shadow_model_{model_id}.pth"
        )

        torch.save(
            {
                "model_id": model_id,
                "model_state_dict": model.state_dict(),
                "test_accuracy": test_accuracy,
                "config": config
            },
            checkpoint_path
        )

        print("Saved:", checkpoint_path)

        # Luu thong tin danh gia
        shadow_metrics = {
            "experiment_id": config["experiment"]["id"],
            "model_id": model_id,
            "train_samples": len(train_dataset),
            "nonmember_samples": len(nonmember_dataset),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_accuracy_percent": test_accuracy * 100,
            "device": str(device)
        }

        all_metrics.append(shadow_metrics)

        # Giai phong bien model truoc khi sang model tiep theo
        del model
        del optimizer

    # Luu ket qua cua cac Shadow Models
    metrics_path = results_dir / "shadow_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    print("\n========== ALL SHADOW MODELS COMPLETED ==========")
    print("Number of models:", len(all_metrics))
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
