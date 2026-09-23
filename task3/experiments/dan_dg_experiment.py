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
BATCH_SIZE_PER_DOMAIN = 8
BATCH_SIZE = 24
EPOCHS = 50
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4
LAMBDA_DG = 1.0

DATA_ROOT = "task3/data/PACS"
CHECKPOINT_DIR = "task3/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dan_dg.pt")


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_domain(domain, transform):
    path = os.path.join(DATA_ROOT, domain)
    return datasets.ImageFolder(path, transform=transform)


def make_split(dataset):
    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    return train_idx, val_idx


photo = load_domain("photo", train_transform)
art = load_domain("art_painting", train_transform)
cartoon = load_domain("cartoon", train_transform)

photo_train, photo_val = make_split(photo)
art_train, art_val = make_split(art)
cartoon_train, cartoon_val = make_split(cartoon)


photo_train_loader = DataLoader(
    Subset(photo, photo_train),
    batch_size=BATCH_SIZE_PER_DOMAIN,
    shuffle=True,
    drop_last=True
)

art_train_loader = DataLoader(
    Subset(art, art_train),
    batch_size=BATCH_SIZE_PER_DOMAIN,
    shuffle=True,
    drop_last=True
)

cartoon_train_loader = DataLoader(
    Subset(cartoon, cartoon_train),
    batch_size=BATCH_SIZE_PER_DOMAIN,
    shuffle=True,
    drop_last=True,
    num_workers=2
)


def build_backbone():
    backbone = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    features = nn.Sequential(
        *list(backbone.children())[:-1]
    )

    classifier = nn.Linear(
        backbone.fc.in_features,
        NUM_CLASSES
    )

    return features, classifier


class DAN_DG(nn.Module):
    def __init__(self):
        super().__init__()

        self.features, self.classifier = build_backbone()

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)

        return features, logits


def mmd_loss(x, y):
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)

    bandwidth = torch.median(
        torch.cat([
            xx.flatten(),
            yy.flatten(),
            xy.flatten()
        ])
    )

    bandwidth = bandwidth.detach().clamp_min(1e-6)

    kernels = []

    for scale in [0.5, 1.0, 2.0]:
        sigma = bandwidth * scale

        kernels.append(
            torch.exp(-xx / (2 * sigma))
        )

        kernels.append(
            torch.exp(-yy / (2 * sigma))
        )

        kernels.append(
            torch.exp(-xy / (2 * sigma))
        )

    k_xx = sum(kernels[0::3]) / 3
    k_yy = sum(kernels[1::3]) / 3
    k_xy = sum(kernels[2::3]) / 3

    return (
        k_xx.mean()
        + k_yy.mean()
        - 2 * k_xy.mean()
    )


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            _, logits = model(images)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


val_photo = load_domain("photo", val_transform)
val_art = load_domain("art_painting", val_transform)
val_cartoon = load_domain("cartoon", val_transform)

photo_val_loader = DataLoader(
    Subset(val_photo, photo_val),
    batch_size=32,
    shuffle=False
)

art_val_loader = DataLoader(
    Subset(val_art, art_val),
    batch_size=32,
    shuffle=False
)

cartoon_val_loader = DataLoader(
    Subset(val_cartoon, cartoon_val),
    batch_size=32,
    shuffle=False
)


model = DAN_DG().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

best_score = -float("inf")
patience_counter = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for photo_batch, art_batch, cartoon_batch in zip(
        photo_train_loader,
        art_train_loader,
        cartoon_train_loader
    ):
        photo_images, photo_labels = photo_batch
        art_images, art_labels = art_batch
        cartoon_images, cartoon_labels = cartoon_batch

        photo_images = photo_images.to(device)
        photo_labels = photo_labels.to(device)

        art_images = art_images.to(device)
        art_labels = art_labels.to(device)

        cartoon_images = cartoon_images.to(device)
        cartoon_labels = cartoon_labels.to(device)

        optimizer.zero_grad()

        photo_features, photo_logits = model(photo_images)
        art_features, art_logits = model(art_images)
        cartoon_features, cartoon_logits = model(cartoon_images)

        classification_loss = (
            criterion(photo_logits, photo_labels)
            + criterion(art_logits, art_labels)
            + criterion(cartoon_logits, cartoon_labels)
        ) / 3

        mmd_photo_art = mmd_loss(
            photo_features,
            art_features
        )

        mmd_photo_cartoon = mmd_loss(
            photo_features,
            cartoon_features
        )

        mmd_art_cartoon = mmd_loss(
            art_features,
            cartoon_features
        )

        mmd = (
            mmd_photo_art
            + mmd_photo_cartoon
            + mmd_art_cartoon
        ) / 3

        loss = classification_loss + LAMBDA_DG * mmd

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    photo_acc = evaluate(
        model,
        photo_val_loader,
        device
    )

    art_acc = evaluate(
        model,
        art_val_loader,
        device
    )

    cartoon_acc = evaluate(
        model,
        cartoon_val_loader,
        device
    )

    mean_acc = (
        photo_acc
        + art_acc
        + cartoon_acc
    ) / 3

    print(
        f"Epoch {epoch + 1}: "
        f"Loss={total_loss / len(photo_train_loader):.4f} "
        f"Photo={photo_acc:.4f} "
        f"Art={art_acc:.4f} "
        f"Cartoon={cartoon_acc:.4f} "
        f"Mean={mean_acc:.4f}"
    )

    if mean_acc > best_score:
        best_score = mean_acc
        patience_counter = 0

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )
    else:
        patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

print("Best mean source validation accuracy:", best_score)
print("Saved:", CHECKPOINT_PATH)