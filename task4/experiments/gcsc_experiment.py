import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split


SEED = 6304
BATCH_SIZE = 128
EPOCHS = 50

DATA_DIR = "common/datasets/CIFAR10"
RESULTS_DIR = "task4/results"
CHECKPOINT_DIR = "task4/models"

BEST_CHECKPOINT_PATH = (
    "task4/models/gcsc_best.pth"
)

NEAR_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel",
]

FAR_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_array(filename, array):

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    np.save(
        path,
        array
    )

    print(
        f"Saved {filename}: {array.shape}"
    )


set_seed(SEED)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


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
    root=DATA_DIR,
    train=True,
    download=True,
    transform=None
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
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
    root=DATA_DIR,
    train=True,
    download=False,
    transform=train_transform
)

val_base = datasets.CIFAR10(
    root=DATA_DIR,
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def extract_outputs(model, loader):

    model.eval()

    all_features = []
    all_logits = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)

            features = torch.flatten(
                x,
                1
            )

            logits = model.fc(
                features
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_features.append(
                features.cpu().numpy()
            )

            all_logits.append(
                logits.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    return (
        np.concatenate(all_features),
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_predictions)
    )


best_val_accuracy = 0.0


print("\nStarting GCSC training...\n")


for epoch in range(EPOCHS):

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

        predictions = logits.argmax(
            dim=1
        )

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
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
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

model.eval()


test_accuracy = evaluate(
    model,
    test_loader
)


print("\nTraining complete.")

print(
    f"Best validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

print(
    f"Final CIFAR-10 test accuracy: "
    f"{test_accuracy:.4f}"
)


print("\nExtracting GCSC Part 6 outputs...")


cifar10_train_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=eval_transform
)


cifar10_test_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=eval_transform
)


train_eval_dataset = Subset(
    cifar10_train_eval,
    train_indices
)

val_eval_dataset = Subset(
    cifar10_train_eval,
    val_indices
)


cifar100_dataset = datasets.CIFAR100(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform
)


near_indices = []
far_indices = []


for index, label in enumerate(
    cifar100_dataset.targets
):

    class_name = cifar100_dataset.classes[label]

    if class_name in NEAR_CLASSES:
        near_indices.append(index)

    if class_name in FAR_CLASSES:
        far_indices.append(index)


near_dataset = Subset(
    cifar100_dataset,
    near_indices
)

far_dataset = Subset(
    cifar100_dataset,
    far_indices
)


train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_eval_loader = DataLoader(
    val_eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_eval_loader = DataLoader(
    cifar10_test_eval,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

near_loader = DataLoader(
    near_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

far_loader = DataLoader(
    far_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Extracting validation outputs...")

_, val_logits, _, _ = extract_outputs(
    model,
    val_eval_loader
)


print("Extracting known test outputs...")

_, test_logits, _, _ = extract_outputs(
    model,
    test_eval_loader
)


print("Extracting near outputs...")

_, near_logits, near_labels, near_predictions = extract_outputs(
    model,
    near_loader
)


print("Extracting far outputs...")

_, far_logits, far_labels, far_predictions = extract_outputs(
    model,
    far_loader
)


save_array(
    "gcsc_validation_logits.npy",
    val_logits
)

save_array(
    "gcsc_known_test_logits.npy",
    test_logits
)

save_array(
    "gcsc_near_logits.npy",
    near_logits
)

save_array(
    "gcsc_far_logits.npy",
    far_logits
)

save_array(
    "gcsc_near_true_classes.npy",
    near_labels
)

save_array(
    "gcsc_near_predicted_classes.npy",
    near_predictions
)

save_array(
    "gcsc_far_true_classes.npy",
    far_labels
)

save_array(
    "gcsc_far_predicted_classes.npy",
    far_predictions
)


print("\nAll GCSC Part 6 outputs saved successfully.")