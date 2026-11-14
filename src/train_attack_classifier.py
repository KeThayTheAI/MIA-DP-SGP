import json
import sys
from pathlib import Path

import numpy as np
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Xac dinh thu muc goc cua project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# De Python tim duoc cac module trong project
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_config
from models.attack_classifier import AttackClassifier


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# 2. SELECT DEVICE
# ==========================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==========================================
# 3. LOAD ATTACK DATASET
# ==========================================



def load_attack_data(config, data_path):

    if not data_path.exists():
        raise FileNotFoundError(
            "Attack dataset not found: {}".format(data_path)
        )

    data = np.load(data_path, allow_pickle=False)

    X = data["X"]
    y = data["y"]
    shadow_ids = data["shadow_model_ids"]

    # Kiem tra kich thuoc du lieu
    if X.ndim != 2 or X.shape[1] != 10:
        raise ValueError(
            "Attack feature dimension must be 10."
        )

    if len(X) != len(y) or len(X) != len(shadow_ids):
        raise ValueError(
            "Attack dataset arrays have inconsistent lengths."
        )

    # Kiem tra du so luong Shadow Models
    expected_models = config["dataset"]["num_shadow_models"]

    unique_ids = np.sort(np.unique(shadow_ids))
    expected_ids = np.arange(1, expected_models + 1)

    if not np.array_equal(unique_ids, expected_ids):
        raise ValueError(
            "Shadow Model IDs in attack dataset do not match config."
        )

    # Su dung toan bo du lieu tu tat ca Shadow Models
    X_train = X
    y_train = y

    print("\n========== ATTACK DATASET ==========")
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Shadow Model IDs:", unique_ids)

    print("\n========== TRAINING DATA ==========")
    print("Training samples:", len(X_train))
    print("Training members:", int(np.sum(y_train == 1)))
    print("Training non-members:", int(np.sum(y_train == 0)))

    if len(X_train) == 0:
        raise ValueError("Attack dataset is empty.")

    return X_train, y_train



# ==========================================
# 4. EVALUATE CLASSIFIER
# ==========================================

def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels in dataloader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).float()

            total_loss += loss.item() * labels.size(0)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    return average_loss, accuracy, all_predictions, all_labels


# ==========================================
# 5. MAIN TRAINING FUNCTION
# ==========================================

def train_attack_classifier():

    print("========== TRAIN ATTACK CLASSIFIER ==========")

    # Load configuration
    config = load_config()

    # Xac dinh Experiment hien tai
    experiment_config = config["experiment"]

    experiment_dir = (
        PROJECT_ROOT
        / experiment_config["output_dir"]
        / experiment_config["id"]
    )

    # Xac dinh cac duong dan cua Attacker
    attacker_dir = experiment_dir / "attacker"

    data_path = attacker_dir / "results" / "attack_dataset.npz"

    checkpoint_path = (
        attacker_dir
        / "checkpoints"
        / "attack_classifier.pth"
    )

    metrics_path = (
        attacker_dir
        / "results"
        / "attack_classifier_metrics.json"
    )

    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    # Load dataset
    
    X_train, y_train = load_attack_data(config, data_path)

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Create TensorDataset
    train_dataset = TensorDataset(X_train, y_train)

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    # Select device
    device = get_device()

    print("\nDevice:", device)

    # Initialize Attack Classifier
    input_size = 10
    model = AttackClassifier(input_size=input_size).to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = []

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(features)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            # Calculate training accuracy
            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).float()

            total_loss += loss.item() * labels.size(0)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        train_loss = total_loss / total_samples

        train_accuracy = total_correct / total_samples

        # Validation
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        })

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy * 100:.2f}%"
        )


    # ======================================
    # 6. SAVE CHECKPOINT
    # ======================================

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "experiment_id": experiment_config["id"],
        "config": config,
        "history": history
    }, checkpoint_path)

    # ======================================
    # 7. SAVE METRICS
    # ======================================

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    
    metrics = {
        "experiment_id": experiment_config["id"],
        "input_size": input_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "shadow_models_used": config["dataset"]["num_shadow_models"],
        "train_samples": len(X_train),
        "train_members": int(torch.sum(y_train == 1).item()),
        "train_nonmembers": int(torch.sum(y_train == 0).item()),
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_train_accuracy_percent": history[-1]["train_accuracy"] * 100
    }


    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\n========== TRAINING COMPLETED ==========")

    print("Experiment:", experiment_config["id"])
    print("Checkpoint:", checkpoint_path)
    print("Metrics:", metrics_path)

    print(
    f"Final Train Accuracy: "
    f"{history[-1]['train_accuracy'] * 100:.2f}%"
    )


# ==========================================
# 8. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    train_attack_classifier()