import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split

SEED = 6304

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

CHECKPOINT_DIR = "task4/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_CHECKPOINT_PATH = (
    "task4/checkpoints/gcsc_best.pth"
)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(
        num_ops=2,
        magnitude=9
    ),

    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.ToTensor()
])

full_train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=None
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=eval_transform
)

labels = np.array(
    full_train_dataset.targets
)

indices = np.arange(
    len(full_train_dataset)
)

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    stratify=labels,
    random_state=SEED
)

train_base = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=train_transform
)

val_base = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=eval_transform
)

train_dataset = Subset(
    train_base,
    train_indices
)

val_dataset = Subset(
    val_base,
    val_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

model = models.resnet18(
    weights=None
)

model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

model.maxpool = nn.Identity()

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy

best_val_accuracy = 0.0


for epoch in range(100):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    scheduler.step()

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )

    val_accuracy = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch + 1:03d}/100 | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            },
            BEST_CHECKPOINT_PATH
        )

        print(
            f"Saved best checkpoint "
            f"(Val Acc: {val_accuracy:.4f})"
        )

checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_accuracy = evaluate(
    model,
    test_loader
)

print("\nTraining complete.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Final CIFAR-10 test accuracy: "
    f"{test_accuracy:.4f}"
)