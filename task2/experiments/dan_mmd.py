import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


SEED = 6304
NUM_CLASSES = 7

SOURCE_BATCH_SIZE = 8
TARGET_BATCH_SIZE = 24

NUM_EPOCHS = 30
PATIENCE = 5

LR = 1e-4
WEIGHT_DECAY = 1e-4
LAMBDA_MMD = 1.0

DATA_ROOT = "task2/data/PACS"
CHECKPOINT = "task2/checkpoints/dan_best.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def load_domain(domain, transform):
    return datasets.ImageFolder(
        root=os.path.join(DATA_ROOT, domain),
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


class ResNet18DAN(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.classifier = nn.Linear(
            backbone.fc.in_features,
            NUM_CLASSES
        )

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)

        return features, logits


def set_bn_eval(module):
    if isinstance(
        module,
        nn.modules.batchnorm._BatchNorm
    ):
        module.eval()


def gaussian_kernel(x, y, bandwidth):
    xx = torch.sum(x * x, dim=1, keepdim=True)
    yy = torch.sum(y * y, dim=1, keepdim=True)

    distance = (
        xx
        + yy.t()
        - 2 * torch.mm(x, y.t())
    )

    return torch.exp(
        -distance / (2 * bandwidth)
    )


def mmd_loss(source_features, target_features):

    combined = torch.cat(
        [source_features, target_features],
        dim=0
    )

    with torch.no_grad():
        distances = torch.cdist(
            combined,
            combined
        ).pow(2)

        median_distance = torch.median(
            distances
        ).clamp_min(1e-6)

    bandwidths = [
        0.5 * median_distance,
        1.0 * median_distance,
        2.0 * median_distance
    ]

    loss = 0.0

    for bandwidth in bandwidths:

        k_ss = gaussian_kernel(
            source_features,
            source_features,
            bandwidth
        )

        k_tt = gaussian_kernel(
            target_features,
            target_features,
            bandwidth
        )

        k_st = gaussian_kernel(
            source_features,
            target_features,
            bandwidth
        )

        loss += (
            k_ss.mean()
            + k_tt.mean()
            - 2 * k_st.mean()
        )

    return loss / len(bandwidths)


def evaluate(model, loader):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(DEVICE)

            _, logits = model(images)

            predictions.extend(
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            labels.extend(
                targets.numpy()
            )

    accuracy = np.mean(
        np.array(predictions)
        == np.array(labels)
    )

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        tp = sum(
            p == class_id and y == class_id
            for p, y in zip(predictions, labels)
        )

        fp = sum(
            p == class_id and y != class_id
            for p, y in zip(predictions, labels)
        )

        fn = sum(
            p != class_id and y == class_id
            for p, y in zip(predictions, labels)
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0
        )

        f1_scores.append(f1)

    return accuracy, np.mean(f1_scores)


def main():

    source_domains = [
        "photo",
        "art_painting",
        "cartoon"
    ]

    train_loaders = {}
    val_loaders = {}

    for domain in source_domains:

        train_dataset = load_domain(
            domain,
            train_transform
        )

        val_dataset = load_domain(
            domain,
            val_transform
        )

        train_indices, val_indices = make_split(
            train_dataset
        )

        train_subset = Subset(
            train_dataset,
            train_indices
        )

        val_subset = Subset(
            val_dataset,
            val_indices
        )

        train_loaders[domain] = DataLoader(
            train_subset,
            batch_size=SOURCE_BATCH_SIZE,
            shuffle=True
        )

        val_loaders[domain] = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False
        )

    target_dataset = load_domain(
        "sketch",
        train_transform
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=TARGET_BATCH_SIZE,
        shuffle=True
    )

    model = ResNet18DAN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    os.makedirs(
        "task2/checkpoints",
        exist_ok=True
    )

    best_f1 = -1
    no_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()
        model.apply(set_bn_eval)

        source_iterators = {
            domain: iter(loader)
            for domain, loader in train_loaders.items()
        }

        target_iterator = iter(target_loader)

        num_batches = max(
            len(loader)
            for loader in train_loaders.values()
        )

        for _ in range(num_batches):

            source_images = []
            source_labels = []

            for domain in source_domains:

                try:
                    images, labels = next(
                        source_iterators[domain]
                    )

                except StopIteration:

                    source_iterators[domain] = iter(
                        train_loaders[domain]
                    )

                    images, labels = next(
                        source_iterators[domain]
                    )

                source_images.append(images)
                source_labels.append(labels)

            try:
                target_images, _ = next(
                    target_iterator
                )

            except StopIteration:

                target_iterator = iter(
                    target_loader
                )

                target_images, _ = next(
                    target_iterator
                )

            source_images = torch.cat(
                source_images,
                dim=0
            ).to(DEVICE)

            source_labels = torch.cat(
                source_labels,
                dim=0
            ).to(DEVICE)

            target_images = target_images.to(DEVICE)

            source_features, source_logits = model(
                source_images
            )

            target_features, _ = model(
                target_images
            )

            classification_loss = criterion(
                source_logits,
                source_labels
            )

            alignment_loss = mmd_loss(
                source_features,
                target_features
            )

            loss = (
                classification_loss
                + LAMBDA_MMD * alignment_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        results = {}

        for domain in source_domains:

            acc, f1 = evaluate(
                model,
                val_loaders[domain]
            )

            results[domain] = (acc, f1)

        mean_f1 = np.mean([
            results[d][1]
            for d in source_domains
        ])

        print(
            f"Epoch {epoch:02d} | "
            f"Photo F1={results['photo'][1]:.4f} | "
            f"Art F1={results['art_painting'][1]:.4f} | "
            f"Cartoon F1={results['cartoon'][1]:.4f} | "
            f"Mean F1={mean_f1:.4f}"
        )

        if mean_f1 > best_f1:

            best_f1 = mean_f1
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_source_macro_f1": mean_f1
                },
                CHECKPOINT
            )

            print("Best checkpoint saved.")

        else:

            no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping.")
            break