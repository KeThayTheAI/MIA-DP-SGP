
import yaml
import numpy as np
import json
import shutil

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Subset


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
# Default paths (will be updated by setup_experiment)
SPLIT_PATH = PROJECT_ROOT / "results" / "splits.npz"
SPLIT_CONFIG_PATH = PROJECT_ROOT / "results" / "split_config.json"

# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

def load_config():
    """Load project configuration from config.yaml."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Cannot find config.yaml at: {}".format(CONFIG_PATH)
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML dictionary.")

    return config

"""Tạo thư mục experiments/exp_001/results.
    Tạo thư mục experiments/exp_001/checkpoints.
    Lưu bản sao cấu hình vào exp_001/config.yaml.
    Chuyển đường dẫn lưu split sang thư mục experiment.
    Nếu là exp_001, sao chép split baseline cũ sang vị trí mới, không xóa hay sửa file gốc."""

def setup_experiment(config):
    """Create experiment directories and configure experiment paths."""

    global SPLIT_PATH, SPLIT_CONFIG_PATH

    experiment_config = config.get("experiment")

    if not isinstance(experiment_config, dict):
        raise ValueError("Missing experiment section in config.yaml.")

    experiment_id = experiment_config.get("id")
    output_dir = experiment_config.get("output_dir")

    if not experiment_id or Path(experiment_id).name != experiment_id:
        raise ValueError("Invalid experiment ID.")

    if not output_dir:
        raise ValueError("Missing experiment output_dir.")

    experiment_dir = PROJECT_ROOT / output_dir / experiment_id

    experiment_results = experiment_dir / "results"
    experiment_checkpoints = experiment_dir / "checkpoints"

    experiment_results.mkdir(parents=True, exist_ok=True)
    experiment_checkpoints.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration used for this experiment.
    config_snapshot = experiment_dir / "config.yaml"

    if not config_snapshot.exists():
        shutil.copy2(CONFIG_PATH, config_snapshot)

    # New experiment-specific split paths.
    SPLIT_PATH = experiment_results / "splits.npz"
    SPLIT_CONFIG_PATH = experiment_results / "split_config.json"

    # Preserve the original baseline split when migrating to exp_001.
    legacy_split = PROJECT_ROOT / "results" / "splits.npz"
    legacy_split_config = PROJECT_ROOT / "results" / "split_config.json"

    if (
        experiment_id == "exp_001"
        and not SPLIT_PATH.exists()
        and legacy_split.exists()
    ):
        shutil.copy2(legacy_split, SPLIT_PATH)

        if legacy_split_config.exists():
            shutil.copy2(legacy_split_config, SPLIT_CONFIG_PATH)

        print("Existing baseline splits copied to exp_001.")

    print("Current Experiment:", experiment_id)
    print("Experiment Directory:", experiment_dir)

    return experiment_dir

# ============================================================
# 3. VALIDATE CONFIGURATION
# ============================================================

def validate_config(config, train_size):
    """Validate dataset configuration before splitting."""

    dataset_config = config.get("dataset")

    if not isinstance(dataset_config, dict):
        raise ValueError("Missing dataset section in config.yaml.")

    required_keys = [
        "target_pool_size",
        "shadow_pool_size",
        "target_train_size",
        "target_nonmember_size",
        "num_shadow_models",
        "shadow_train_size",
        "shadow_nonmember_size",
        "random_seed"
    ]

    for key in required_keys:
        if key not in dataset_config:
            raise ValueError("Missing dataset config: {}".format(key))

    values = {}

    for key in required_keys:
        try:
            values[key] = int(dataset_config[key])
        except (TypeError, ValueError):
            raise ValueError("{} must be an integer.".format(key))

    for key in required_keys[:-1]:
        if values[key] <= 0:
            raise ValueError("{} must be greater than 0.".format(key))

    target_pool_size = values["target_pool_size"]
    shadow_pool_size = values["shadow_pool_size"]

    target_used = (
        values["target_train_size"]
        + values["target_nonmember_size"]
    )

    shadow_used = (
        values["num_shadow_models"]
        * (
            values["shadow_train_size"]
            + values["shadow_nonmember_size"]
        )
    )

    if target_used > target_pool_size:
        raise ValueError(
            "Target data exceeds target_pool_size."
        )

    if shadow_used > shadow_pool_size:
        raise ValueError(
            "Shadow data exceeds shadow_pool_size."
        )

    if target_pool_size + shadow_pool_size > train_size:
        raise ValueError(
            "Target Pool + Shadow Pool exceeds MNIST train size."
        )

    return values


# ============================================================
# 4. LOAD MNIST
# ============================================================

def load_mnist(data_dir):
    """Download/load MNIST train and test datasets."""

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# ============================================================
# 5. STRATIFIED SAMPLING
# ============================================================

def stratified_take(indices, labels, sample_size, rng):
    """
    Select exactly sample_size indices.

    Samples are selected proportionally from each class.
    """

    indices = np.asarray(indices, dtype=np.int64)

    if sample_size < 0 or sample_size > len(indices):
        raise ValueError("Invalid stratified sample size.")

    if sample_size == 0:
        return np.array([], dtype=np.int64)

    classes = np.unique(labels[indices])

    class_groups = {}

    for class_id in classes:
        class_indices = indices[labels[indices] == class_id]
        class_indices = rng.permutation(class_indices)

        class_groups[class_id] = class_indices

    # Calculate proportional allocation.
    exact_counts = {
        class_id: (
            len(class_groups[class_id])
            * sample_size
            / len(indices)
        )
        for class_id in classes
    }

    allocations = {
        class_id: int(np.floor(exact_counts[class_id]))
        for class_id in classes
    }

    remaining = sample_size - sum(allocations.values())

    # Distribute remaining samples according to largest remainders.
    sorted_classes = sorted(
        classes,
        key=lambda class_id: (
            exact_counts[class_id] - allocations[class_id]
        ),
        reverse=True
    )

    for class_id in sorted_classes:
        if remaining == 0:
            break

        if allocations[class_id] < len(class_groups[class_id]):
            allocations[class_id] += 1
            remaining -= 1

    selected = []

    for class_id in classes:
        count = allocations[class_id]

        selected.extend(
            class_groups[class_id][:count].tolist()
        )

    return rng.permutation(
        np.asarray(selected, dtype=np.int64)
    )


# ============================================================
# 6. CREATE DATA SPLITS
# ============================================================

def create_splits(labels, settings):
    """Create Target, Shadow and unused dataset indices."""

    rng = np.random.default_rng(settings["random_seed"])

    all_indices = np.arange(len(labels))

    # --------------------------------------------------------
    # Target Pool
    # --------------------------------------------------------

    target_pool = stratified_take(
        all_indices,
        labels,
        settings["target_pool_size"],
        rng
    )

    remaining_indices = np.setdiff1d(
        all_indices,
        target_pool
    )

    # --------------------------------------------------------
    # Shadow Pool
    # --------------------------------------------------------

    shadow_pool = stratified_take(
        remaining_indices,
        labels,
        settings["shadow_pool_size"],
        rng
    )

    unused_indices = np.setdiff1d(
        remaining_indices,
        shadow_pool
    )

    # --------------------------------------------------------
    # Target Model
    # --------------------------------------------------------

    target_train_indices = stratified_take(
        target_pool,
        labels,
        settings["target_train_size"],
        rng
    )

    target_remaining = np.setdiff1d(
        target_pool,
        target_train_indices
    )

    target_nonmember_indices = stratified_take(
        target_remaining,
        labels,
        settings["target_nonmember_size"],
        rng
    )

    # --------------------------------------------------------
    # Shadow Models
    # --------------------------------------------------------

    shadow_models = []

    remaining_shadow = shadow_pool.copy()

    for model_id in range(settings["num_shadow_models"]):

        model_pool_size = (
            settings["shadow_train_size"]
            + settings["shadow_nonmember_size"]
        )

        model_pool = stratified_take(
            remaining_shadow,
            labels,
            model_pool_size,
            rng
        )

        remaining_shadow = np.setdiff1d(
            remaining_shadow,
            model_pool
        )

        shadow_train_indices = stratified_take(
            model_pool,
            labels,
            settings["shadow_train_size"],
            rng
        )

        model_remaining = np.setdiff1d(
            model_pool,
            shadow_train_indices
        )

        shadow_nonmember_indices = stratified_take(
            model_remaining,
            labels,
            settings["shadow_nonmember_size"],
            rng
        )

        shadow_models.append({
            "model_id": model_id + 1,
            "train_indices": shadow_train_indices,
            "nonmember_indices": shadow_nonmember_indices
        })

    splits = {
        "target_pool_indices": target_pool,
        "shadow_pool_indices": shadow_pool,
        "target_train_indices": target_train_indices,
        "target_nonmember_indices": target_nonmember_indices,
        "unused_indices": unused_indices
    }

    for shadow in shadow_models:
        model_id = shadow["model_id"]

        splits["shadow_{}_train".format(model_id)] = (
            shadow["train_indices"]
        )

        splits["shadow_{}_nonmember".format(model_id)] = (
            shadow["nonmember_indices"]
        )

    return splits


# ============================================================
# 7. SAVE / LOAD SPLITS
# ============================================================

def save_splits(splits):
    """Save all dataset indices for reproducibility."""

    SPLIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        SPLIT_PATH,
        **splits
    )

    print("Dataset splits saved to:", SPLIT_PATH)


def load_saved_splits():
    """Load previously saved dataset indices."""

    with np.load(SPLIT_PATH, allow_pickle=False) as data:
        return {
            key: data[key].copy()
            for key in data.files
        }

#Luu cau hinh hien tại vao JSON
def save_split_config(settings):
    """Save the configuration used to create dataset splits."""

    SPLIT_CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(SPLIT_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(
            settings,
            file,
            indent=4
        )

    print("Split configuration saved to:", SPLIT_CONFIG_PATH)


def load_saved_split_config():
    """Load the configuration used by existing dataset splits."""

    if not SPLIT_CONFIG_PATH.exists():
        return None

    with open(SPLIT_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 8. CHECK SPLITS
# ============================================================

def check_splits(splits, settings, labels):
    """Check sizes, overlaps and class distributions."""

    target_train = splits["target_train_indices"]
    target_nonmember = splits["target_nonmember_indices"]

    shadow_train_sets = []
    shadow_nonmember_sets = []

    # --------------------------------------------------------
    # Check Target
    # --------------------------------------------------------

    assert len(target_train) == settings["target_train_size"]
    assert len(target_nonmember) == settings["target_nonmember_size"]

    assert set(target_train).isdisjoint(
        set(target_nonmember)
    )

    # --------------------------------------------------------
    # Check Shadow Models
    # --------------------------------------------------------

    for model_id in range(1, settings["num_shadow_models"] + 1):

        train_indices = splits[
            "shadow_{}_train".format(model_id)
        ]

        nonmember_indices = splits[
            "shadow_{}_nonmember".format(model_id)
        ]

        assert len(train_indices) == settings["shadow_train_size"]

        assert len(nonmember_indices) == (
            settings["shadow_nonmember_size"]
        )

        assert set(train_indices).isdisjoint(
            set(nonmember_indices)
        )

        shadow_train_sets.append(set(train_indices))
        shadow_nonmember_sets.append(set(nonmember_indices))

        # Check Target vs Shadow.
        assert set(target_train).isdisjoint(set(train_indices))
        assert set(target_train).isdisjoint(set(nonmember_indices))

        assert set(target_nonmember).isdisjoint(set(train_indices))
        assert set(target_nonmember).isdisjoint(set(nonmember_indices))

    # --------------------------------------------------------
    # Check different Shadow Models
    # --------------------------------------------------------

    all_shadow_sets = []

    for train_set, nonmember_set in zip(
        shadow_train_sets,
        shadow_nonmember_sets
    ):
        all_shadow_sets.append(train_set)
        all_shadow_sets.append(nonmember_set)

    for i in range(len(all_shadow_sets)):
        for j in range(i + 1, len(all_shadow_sets)):
            assert all_shadow_sets[i].isdisjoint(
                all_shadow_sets[j]
            )

    # --------------------------------------------------------
    # Check pool consistency
    # --------------------------------------------------------

    target_pool = set(splits["target_pool_indices"])
    shadow_pool = set(splits["shadow_pool_indices"])

    assert target_pool.isdisjoint(shadow_pool)

    assert set(target_train).issubset(target_pool)
    assert set(target_nonmember).issubset(target_pool)

    # --------------------------------------------------------
    # Check all indices
    # --------------------------------------------------------

    all_used = (
        target_pool
        | shadow_pool
        | set(splits["unused_indices"])
    )

    assert len(all_used) == len(labels)
    assert all_used == set(range(len(labels)))

    # --------------------------------------------------------
    # Check class distribution
    # --------------------------------------------------------

    print("\n========== CLASS DISTRIBUTION ==========")

    for name, key in [
        ("Target Member", "target_train_indices"),
        ("Target Non-member", "target_nonmember_indices")
    ]:
        counts = np.bincount(
            labels[splits[key]],
            minlength=10
        )

        print("{}: {}".format(name, counts.tolist()))

    print("\nAll dataset split checks passed!")


# ============================================================
# 9. BUILD PYTORCH DATASETS
# ============================================================

def build_datasets(train_dataset, test_dataset, splits, settings):
    """Convert saved indices into PyTorch Subsets."""

    target_train = Subset(
        train_dataset,
        splits["target_train_indices"].tolist()
    )

    target_nonmember = Subset(
        train_dataset,
        splits["target_nonmember_indices"].tolist()
    )

    shadow_models = []

    for model_id in range(1, settings["num_shadow_models"] + 1):

        shadow_models.append({
            "model_id": model_id,

            "train": Subset(
                train_dataset,
                splits[
                    "shadow_{}_train".format(model_id)
                ].tolist()
            ),

            "nonmember": Subset(
                train_dataset,
                splits[
                    "shadow_{}_nonmember".format(model_id)
                ].tolist()
            )
        })

    utility_test = test_dataset

    return {
        "target_train": target_train,
        "target_nonmember": target_nonmember,
        "shadow_models": shadow_models,
        "utility_test": utility_test,
        "unused_indices": splits["unused_indices"]
    }


# ============================================================
# 10. MAIN DATASET PIPELINE
# ============================================================

def load_datasets():
    """Load MNIST and prepare all datasets."""

    config = load_config()
    setup_experiment(config)
    dataset_config = config["dataset"]

    data_dir = PROJECT_ROOT / dataset_config["data_dir"]

    train_dataset, test_dataset = load_mnist(data_dir)

    labels = np.asarray(
        train_dataset.targets,
        dtype=np.int64
    )

    settings = validate_config(
        config,
        len(train_dataset)
    )

    
    # --------------------------------------------------------
    # Load existing splits or create new ones
    # --------------------------------------------------------

    needs_save_config = False

    if SPLIT_PATH.exists():

        print("Loading existing dataset splits...")

        splits = load_saved_splits()

        # Check expected keys.
        expected_shadow_keys = []

        for model_id in range(1, settings["num_shadow_models"] + 1):
            expected_shadow_keys.extend([
                "shadow_{}_train".format(model_id),
                "shadow_{}_nonmember".format(model_id)
            ])

        required_keys = [
            "target_pool_indices",
            "shadow_pool_indices",
            "target_train_indices",
            "target_nonmember_indices",
            "unused_indices"
        ] + expected_shadow_keys

        if set(splits.keys()) != set(required_keys):
            raise ValueError(
                "Saved splits do not match current configuration. "
                "Back up and regenerate results/splits.npz."
            )

        # Check saved configuration.
        saved_settings = load_saved_split_config()

        if saved_settings is not None:

            if saved_settings != settings:
                raise ValueError(
                    "Dataset configuration has changed. "
                    "Existing splits cannot be reused. "
                    "Back up and regenerate splits.npz."
                )

            print("Saved split configuration matches current config.")

        else:
            # Legacy split: validate first, then save its configuration.
            print(
                "Warning: Existing splits have no configuration metadata. "
                "They will be adopted if all checks pass."
            )

            needs_save_config = True

    else:

        print("Creating new dataset splits...")

        splits = create_splits(
            labels,
            settings
        )

        save_splits(splits)

        needs_save_config = True

    # --------------------------------------------------------
    # Validate splits
    # --------------------------------------------------------

    check_splits(
        splits,
        settings,
        labels
    )

    # Save metadata only after all split checks pass.
    if needs_save_config:
        save_split_config(settings)


    # --------------------------------------------------------
    # Build PyTorch datasets
    # --------------------------------------------------------

    datasets_dict = build_datasets(
        train_dataset,
        test_dataset,
        splits,
        settings
    )

    return datasets_dict


# ============================================================
# 11. DISPLAY DATASET INFORMATION
# ============================================================

def print_dataset_info(data):

    print("\n========== TARGET DATA ==========")

    print(
        "Target Member:",
        len(data["target_train"])
    )

    print(
        "Target Non-member:",
        len(data["target_nonmember"])
    )

    print(
        "Utility Test:",
        len(data["utility_test"])
    )

    print("\n========== SHADOW DATA ==========")

    for shadow in data["shadow_models"]:

        print(
            "Shadow Model {}: {} Members, {} Non-members".format(
                shadow["model_id"],
                len(shadow["train"]),
                len(shadow["nonmember"])
            )
        )

    print("\n========== UNUSED DATA ==========")

    print(
        "Unused training images:",
        len(data["unused_indices"])
    )


# ============================================================
# 12. RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    data = load_datasets()

    print_dataset_info(data)

    print("\nAll dataset checks passed!")