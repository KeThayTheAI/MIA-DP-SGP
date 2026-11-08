
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Subset


def load_datasets(root="data/MNIST", seed=42):
    """
    Download MNIST and split it into:
    - Target training data
    - Shadow training data
    - Shadow non-member data
    - Target non-member data
    """

    # Convert images into PyTorch tensors
    transform = transforms.ToTensor()

    # Download MNIST training and test sets
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    # Shuffle training indices reproducibly
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_dataset))

    # Split 60,000 training images into two pools
    target_indices = indices[:30000]
    shadow_indices = indices[30000:]

    # Split shadow pool into members and non-members
    shadow_train_indices = shadow_indices[:15000]
    shadow_nonmember_indices = shadow_indices[15000:]

    # Create dataset subsets
    target_train = Subset(
        train_dataset,
        target_indices.tolist()
    )

    shadow_train = Subset(
        train_dataset,
        shadow_train_indices.tolist()
    )

    shadow_nonmember = Subset(
        train_dataset,
        shadow_nonmember_indices.tolist()
    )

    # Official MNIST test set:
    # All images are non-members of the target training set
    target_nonmember = test_dataset

    return {
        "target_train": target_train,
        "target_nonmember": target_nonmember,
        "shadow_train": shadow_train,
        "shadow_nonmember": shadow_nonmember
    }


if __name__ == "__main__":
    datasets_dict = load_datasets()

    print("\n===== MNIST DATASET SUMMARY =====")

    for name, dataset in datasets_dict.items():
        print(f"{name}: {len(dataset)} images")

    # Inspect one sample
    image, label = datasets_dict["target_train"][0]

    print("\n===== SAMPLE INFORMATION =====")
    print("Image shape:", image.shape)
    print("Image type:", image.dtype)
    print("Label:", label)
