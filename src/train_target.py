import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Để Python tìm được các module trong project
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config, load_datasets, setup_experiment
from models.target_model import MLP


# 1. Chọn thiết bị huấn luyện
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# 2. Thiết lập random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 3. Huấn luyện một Epoch
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


# 4. Đánh giá mô hình
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


# 5. Chương trình chính
def main():
    config = load_config()
    data = load_datasets()

    experiment_config = config["experiment"]

    experiment_dir = (
        PROJECT_ROOT
        / experiment_config["output_dir"]
        / experiment_config["id"]
    )

    seed = config["dataset"]["random_seed"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    set_seed(seed)

    device = get_device()
    print("Device:", device)

    # Tạo DataLoader
    train_loader = DataLoader(
        data["target_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        data["utility_test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Khởi tạo mô hình
    model = MLP().to(device)

    # Loss Function và Optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Huấn luyện
    print("\n========== TARGET MODEL TRAINING ==========")

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

    # Đánh giá cuối cùng trên MNIST Test
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )
    # Danh gia lai tren tap Train
    train_eval_loss, train_accuracy = evaluate(
        model,
        train_loader,
        criterion,
        device
    )

    print("\n========== FINAL EVALUATION ==========")
    print(f"Train Loss: {train_eval_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Accuracy (%): {train_accuracy * 100:.2f}%")


    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

   
    # Tạo thư mục lưu kết quả Non-DP
    non_dp_dir = experiment_dir / "non_dp"

    checkpoint_dir = non_dp_dir / "checkpoints"
    results_dir = non_dp_dir / "results"


    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Lưu trọng số mô hình
    checkpoint_path = checkpoint_dir / "target_model.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "config": config
        },
        checkpoint_path
    )

    # Lưu metrics
    metrics = {
        "model": "Target Model",
        "experiment_id": config["experiment"]["id"],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(data["target_train"]),
        "test_samples": len(data["utility_test"]),

        "train_loss": train_eval_loss,
        "train_accuracy": train_accuracy,
        "train_accuracy_percent": train_accuracy * 100,

        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
        "device": str(device)
    }

    metrics_path = results_dir / "target_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== SAVED FILES ==========")
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()