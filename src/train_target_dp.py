import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

# Xac dinh thu muc goc cua project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# De Python tim duoc cac module trong project
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config, load_datasets
from models.target_model import MLP


def get_device():
    # Dung CPU de tranh van de tuong thich giua Opacus va Apple MPS.
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_noise_tag(noise_multiplier):
    # Vi du: 0.5 -> noise_0_5; 1.0 -> noise_1
    value = format(noise_multiplier, "g").replace(".", "_")
    return f"noise_{value}"


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        # Poisson sampling cua Opacus co the tao batch rong.
        if images.size(0) == 0:
            continue

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

    if total == 0:
        raise RuntimeError("No samples were processed in this epoch.")

    return total_loss / total, correct / total


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

    if total == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    return total_loss / total, correct / total


def train_one_dp_target(
    config,
    data,
    experiment_dir,
    noise_multiplier,
    device,
):
    # Reset seed de cac lan chay co cung diem khoi tao va thu tu lay mau.
    seed = config["dataset"]["random_seed"]
    set_seed(seed)

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    max_grad_norm = config["dp"]["max_grad_norm"]
    delta = config["dp"]["delta"]

    train_loader = DataLoader(
        data["target_train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        data["utility_test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Tao mot Target Model moi cho moi noise_multiplier.
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    print("\n" + "=" * 60)
    print(f"DP TARGET TRAINING - noise_multiplier={noise_multiplier}")
    print("=" * 60)
    print("Device:", device)
    print("Noise multiplier:", noise_multiplier)
    print("Max grad norm:", max_grad_norm)
    print("Delta:", delta)

    history = []
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_accuracy_percent": train_accuracy * 100,
            }
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_accuracy:.4f}"
        )

    # Tinh epsilon sau khi hoan tat huan luyen.
    epsilon = privacy_engine.get_epsilon(delta=delta)

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    # Danh gia lai tren tap train goc, khong dung DPDataLoader.
    train_eval_loader = DataLoader(
        data["target_train"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_eval_loss, train_accuracy = evaluate(
        model, train_eval_loader, criterion, device
    )

    # Moi noise_multiplier co mot thu muc rieng, khong ghi de nhau.
    noise_tag = make_noise_tag(noise_multiplier)
    dp_run_dir = experiment_dir / "dp" / noise_tag
    checkpoint_dir = dp_run_dir / "checkpoints"
    results_dir = dp_run_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "target_model.pth"
    torch.save(
        {
            "model_state_dict": model._module.state_dict(),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "config": config,
        },
        checkpoint_path,
    )

    metrics = {
        "model": "DP Target Model",
        "experiment_id": config["experiment"]["id"],
        "noise_tag": noise_tag,
        "noise_multiplier": noise_multiplier,
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
        "privacy": {
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "accountant": "rdp",
        },
        "history": history,
        "device": str(device),
    }

    metrics_path = results_dir / "target_dp_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== FINAL EVALUATION ==========")
    print(f"Train Loss: {train_eval_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("\n========== PRIVACY BUDGET ==========")
    print(f"Epsilon: {epsilon:.4f}")
    print(f"Delta: {delta}")
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)

    return {
        "noise_multiplier": noise_multiplier,
        "epsilon": epsilon,
        "delta": delta,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
    }


def main():
    config = load_config()
    data = load_datasets()

    experiment_config = config["experiment"]
    experiment_dir = (
        PROJECT_ROOT
        / experiment_config["output_dir"]
        / experiment_config["id"]
    )

    # Danh sach noise_multiplier duoc khai bao trong config.yaml.
    noise_multipliers = config["dp"]["noise_multipliers"]
    if not noise_multipliers:
        raise ValueError("dp.noise_multipliers must not be empty.")
    if any(value <= 0 for value in noise_multipliers):
        raise ValueError("Every noise_multiplier must be greater than 0.")
    if len(set(noise_multipliers)) != len(noise_multipliers):
        raise ValueError("dp.noise_multipliers contains duplicate values.")

    delta = config["dp"]["delta"]
    if not 0 < delta < 1:
        raise ValueError("dp.delta must be between 0 and 1.")

    device = get_device()
    print("Device:", device)
    print("Noise multipliers:", noise_multipliers)
    print("Delta:", delta)

    all_results = []
    for index, noise_multiplier in enumerate(noise_multipliers, start=1):
        print(f"\nStarting DP experiment {index}/{len(noise_multipliers)}")
        result = train_one_dp_target(
            config=config,
            data=data,
            experiment_dir=experiment_dir,
            noise_multiplier=float(noise_multiplier),
            device=device,
        )
        all_results.append(result)

    # Luu bang tong hop de tien so sanh cac muc DP.
    summary_dir = experiment_dir / "dp" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "dp_experiments_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print("\n" + "=" * 60)
    print("ALL DP EXPERIMENTS COMPLETED")
    print("Summary:", summary_path)
    print("=" * 60)
    for result in all_results:
        print(
            f"noise={result['noise_multiplier']}, "
            f"epsilon={result['epsilon']:.4f}, "
            f"test_accuracy={result['test_accuracy'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()