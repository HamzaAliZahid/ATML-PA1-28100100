import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


SEED = 6304
BATCH_SIZE = 128
EPOCHS = 50

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DATA_DIR = "common/datasets/CIFAR10"
RESULTS_DIR = "task4/results"
CHECKPOINT_DIR = "task4/models"

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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset_aug = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform,
)

train_dataset_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=eval_transform,
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform,
)


targets = np.array(train_dataset_aug.targets)
indices = np.arange(len(targets))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    random_state=SEED,
    stratify=targets,
)


print("\nDataset split:")
print("Training:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_dataset))


train_subset = Subset(
    train_dataset_aug,
    train_indices
)

val_subset = Subset(
    train_dataset_eval,
    val_indices
)


train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
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
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)


def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    average_loss = total_loss / total_examples

    return average_loss, accuracy


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

            logits = model.fc(features)

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
        np.concatenate(all_predictions),
    )


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


best_val_accuracy = -1.0
best_epoch = -1

history = []

print("\nStarting Vanilla training...\n")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    total_examples = 0

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

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_examples += batch_size

    train_loss = running_loss / total_examples

    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.6f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_accuracy,
                "seed": SEED,
            },
            os.path.join(
                CHECKPOINT_DIR,
                "vanilla_best.pth"
            ),
        )

        print(
            f"  -> Saved best checkpoint "
            f"(Val Acc: {best_val_accuracy * 100:.2f}%)"
        )

    scheduler.step()


checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "vanilla_best.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nBest checkpoint:")
print("Epoch:", checkpoint["epoch"])
print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print("\nFinal Vanilla CIFAR-10 Test Result")
print("----------------------------------")
print(
    f"Test Loss: {test_loss:.4f}"
)
print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


results = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "test_accuracy": test_accuracy,
}


np.save(
    os.path.join(
        RESULTS_DIR,
        "vanilla_history.npy"
    ),
    np.array(
        [
            [
                h["epoch"],
                h["train_loss"],
                h["val_loss"],
                h["val_accuracy"],
                h["learning_rate"],
            ]
            for h in history
        ]
    ),
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "vanilla_results.npy"
    ),
    results,
    allow_pickle=True,
)


print("\nExtracting Part 6 outputs...")


cifar10_train_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=eval_transform,
)


cifar10_test_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=eval_transform,
)


train_subset_eval = Subset(
    cifar10_train_eval,
    train_indices
)

val_subset_eval = Subset(
    cifar10_train_eval,
    val_indices
)


near_class_indices = []
far_class_indices = []

for i, label in enumerate(
    cifar100.targets
) if False else []:
    pass


cifar100_dataset = datasets.CIFAR100(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform,
)


for index, label in enumerate(
    cifar100_dataset.targets
):

    class_name = cifar100_dataset.classes[label]

    if class_name in NEAR_CLASSES:
        near_class_indices.append(index)

    if class_name in FAR_CLASSES:
        far_class_indices.append(index)


near_dataset = Subset(
    cifar100_dataset,
    near_class_indices
)

far_dataset = Subset(
    cifar100_dataset,
    far_class_indices
)


train_eval_loader = DataLoader(
    train_subset_eval,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

val_eval_loader = DataLoader(
    val_subset_eval,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_eval_loader = DataLoader(
    cifar10_test_eval,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

near_loader = DataLoader(
    near_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

far_loader = DataLoader(
    far_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print("\nExtracting CIFAR-10 training features...")

train_features, train_logits, train_labels, train_predictions = (
    extract_outputs(
        model,
        train_eval_loader
    )
)


print("Extracting CIFAR-10 validation outputs...")

val_features, val_logits, val_labels, val_predictions = (
    extract_outputs(
        model,
        val_eval_loader
    )
)


print("Extracting CIFAR-10 test outputs...")

test_features, test_logits, test_labels, test_predictions = (
    extract_outputs(
        model,
        test_eval_loader
    )
)


print("Extracting CIFAR-100 near outputs...")

near_features, near_logits, near_labels, near_predictions = (
    extract_outputs(
        model,
        near_loader
    )
)


print("Extracting CIFAR-100 far outputs...")

far_features, far_logits, far_labels, far_predictions = (
    extract_outputs(
        model,
        far_loader
    )
)


save_array(
    "vanilla_train_features.npy",
    train_features
)

save_array(
    "vanilla_train_labels.npy",
    train_labels
)

save_array(
    "vanilla_validation_features.npy",
    val_features
)

save_array(
    "vanilla_validation_logits.npy",
    val_logits
)

save_array(
    "vanilla_known_test_features.npy",
    test_features
)

save_array(
    "vanilla_known_test_logits.npy",
    test_logits
)

save_array(
    "vanilla_near_features.npy",
    near_features
)

save_array(
    "vanilla_near_logits.npy",
    near_logits
)

save_array(
    "vanilla_far_features.npy",
    far_features
)

save_array(
    "vanilla_far_logits.npy",
    far_logits
)

save_array(
    "near_true_classes.npy",
    near_labels
)

save_array(
    "near_predicted_classes.npy",
    near_predictions
)

save_array(
    "far_true_classes.npy",
    far_labels
)

save_array(
    "far_predicted_classes.npy",
    far_predictions
)


print("\nAll Vanilla Part 6 outputs saved successfully.")