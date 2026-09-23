import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

SEED = 6304

BATCH_SIZE_PER_DOMAIN = 8
BATCH_SIZE = 24

NUM_CLASSES = 7
NUM_EPOCHS = 30
PATIENCE = 5

LR = 1e-4
WEIGHT_DECAY = 1e-4

DATA_ROOT = "task2/data/PACS"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def load_domain(domain, transform):
    path = os.path.join(DATA_ROOT, domain)

    return datasets.ImageFolder(
        root=path,
        transform=transform
    )

def make_split(dataset):
    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        stratify=labels,
        random_state=SEED
    )

    return train_indices, val_indices

def build_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model

def set_bn_eval(module):

    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def train_one_epoch(
    model,
    loaders,
    optimizer,
    criterion
):

    model.train()

    model.apply(set_bn_eval)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterators = {
        domain: iter(loader)
        for domain, loader in loaders.items()
    }

    num_batches = max(
        len(loader)
        for loader in loaders.values()
    )

    for _ in range(num_batches):

        images_list = []
        labels_list = []

        for domain, loader in loaders.items():

            try:
                images, labels = next(iterators[domain])

            except StopIteration:
                iterators[domain] = iter(loader)
                images, labels = next(iterators[domain])

            images_list.append(images)
            labels_list.append(labels)

        images = torch.cat(images_list, dim=0).to(DEVICE)
        labels = torch.cat(labels_list, dim=0).to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )

def evaluate(model, loader):

    model.eval()

    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = total_correct / total_samples

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        tp = sum(
            1
            for p, y in zip(all_predictions, all_labels)
            if p == class_id and y == class_id
        )

        fp = sum(
            1
            for p, y in zip(all_predictions, all_labels)
            if p == class_id and y != class_id
        )

        fn = sum(
            1
            for p, y in zip(all_predictions, all_labels)
            if p != class_id and y == class_id
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return accuracy, macro_f1

def main():

    source_domains = [
        "photo",
        "art_painting",
        "cartoon"
    ]

    train_datasets = {}
    val_datasets = {}

    train_loaders = {}
    val_loaders = {}

    for domain in source_domains:

        dataset_train = load_domain(
            domain,
            train_transform
        )

        dataset_val = load_domain(
            domain,
            val_transform
        )

        train_indices, val_indices = make_split(
            dataset_train
        )

        train_datasets[domain] = Subset(
            dataset_train,
            train_indices
        )

        val_datasets[domain] = Subset(
            dataset_val,
            val_indices
        )

        train_loaders[domain] = DataLoader(
            train_datasets[domain],
            batch_size=BATCH_SIZE_PER_DOMAIN,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loaders[domain] = DataLoader(
            val_datasets[domain],
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(
            domain,
            "train:",
            len(train_datasets[domain]),
            "val:",
            len(val_datasets[domain])
        )

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_mean_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    os.makedirs("task2/checkpoints", exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loaders,
            optimizer,
            criterion
        )

        source_results = {}

        for domain in source_domains:

            acc, f1 = evaluate(
                model,
                val_loaders[domain]
            )

            source_results[domain] = {
                "accuracy": acc,
                "macro_f1": f1
            }

        mean_f1 = np.mean([
            source_results[d]["macro_f1"]
            for d in source_domains
        ])

        print(
            f"\nEpoch {epoch:02d} | "
            f"Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f}"
        )

        for domain in source_domains:

            print(
                f"  {domain:13s} "
                f"Acc={source_results[domain]['accuracy']:.4f} "
                f"F1={source_results[domain]['macro_f1']:.4f}"
            )

        print(
            f"  Mean Macro-F1 = {mean_f1:.4f}"
        )

        if mean_f1 > best_mean_f1:

            best_mean_f1 = mean_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_source_macro_f1": mean_f1
                },
                "task2/checkpoints/erm_best.pt"
            )

            print("  -> Best checkpoint saved")

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best mean source Macro-F1: "
        f"{best_mean_f1:.4f}"
    )