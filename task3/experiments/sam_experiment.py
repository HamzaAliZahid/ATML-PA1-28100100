import os
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split


SEED = 6304
NUM_CLASSES = 7

BATCH_SIZE_PER_DOMAIN = 8
VAL_BATCH_SIZE = 32

EPOCHS = 50
PATIENCE = 5

LR = 1e-4
WEIGHT_DECAY = 1e-4
RHO = 0.05

DATA_ROOT = "common/datasets/PACS"
CHECKPOINT_DIR = "task3/models"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "sam.pt"
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

    return datasets.ImageFolder(
        root=path,
        transform=transform
    )


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


photo_train_dataset = load_domain(
    "photo",
    train_transform
)

art_train_dataset = load_domain(
    "art_painting",
    train_transform
)

cartoon_train_dataset = load_domain(
    "cartoon",
    train_transform
)


photo_train_idx, photo_val_idx = make_split(
    photo_train_dataset
)

art_train_idx, art_val_idx = make_split(
    art_train_dataset
)

cartoon_train_idx, cartoon_val_idx = make_split(
    cartoon_train_dataset
)


photo_train_loader = DataLoader(
    Subset(
        photo_train_dataset,
        photo_train_idx
    ),
    batch_size=BATCH_SIZE_PER_DOMAIN,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


art_train_loader = DataLoader(
    Subset(
        art_train_dataset,
        art_train_idx
    ),
    batch_size=BATCH_SIZE_PER_DOMAIN,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


cartoon_train_loader = DataLoader(
    Subset(
        cartoon_train_dataset,
        cartoon_train_idx
    ),
    batch_size=BATCH_SIZE_PER_DOMAIN,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


photo_val_dataset = load_domain(
    "photo",
    val_transform
)

art_val_dataset = load_domain(
    "art_painting",
    val_transform
)

cartoon_val_dataset = load_domain(
    "cartoon",
    val_transform
)


photo_val_loader = DataLoader(
    Subset(
        photo_val_dataset,
        photo_val_idx
    ),
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


art_val_loader = DataLoader(
    Subset(
        art_val_dataset,
        art_val_idx
    ),
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


cartoon_val_loader = DataLoader(
    Subset(
        cartoon_val_dataset,
        cartoon_val_idx
    ),
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0
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


class SAMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features, self.classifier = build_backbone()

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(
            features,
            start_dim=1
        )

        logits = self.classifier(features)

        return logits


def compute_loss(
    model,
    images,
    labels,
    criterion
):
    logits = model(images)

    loss = criterion(
        logits,
        labels
    )

    return loss


def get_grad_norm(model):
    gradients = []

    for parameter in model.parameters():

        if parameter.grad is not None:
            gradients.append(
                parameter.grad.norm(
                    p=2
                )
            )

    if len(gradients) == 0:
        return torch.tensor(
            0.0,
            device=device
        )

    return torch.norm(
        torch.stack(gradients),
        p=2
    )


def sam_step(
    model,
    optimizer,
    images,
    labels,
    criterion,
    rho
):
    optimizer.zero_grad()

    loss = compute_loss(
        model,
        images,
        labels,
        criterion
    )

    loss.backward()

    grad_norm = get_grad_norm(model)

    scale = rho / (
        grad_norm + 1e-12
    )

    perturbations = []

    with torch.no_grad():

        for parameter in model.parameters():

            if parameter.grad is None:
                continue

            epsilon = (
                parameter.grad * scale
            )

            parameter.add_(epsilon)

            perturbations.append(
                (parameter, epsilon)
            )

    optimizer.zero_grad()

    perturbed_loss = compute_loss(
        model,
        images,
        labels,
        criterion
    )

    perturbed_loss.backward()

    with torch.no_grad():

        for parameter, epsilon in perturbations:
            parameter.sub_(epsilon)

    optimizer.step()

    return loss.item()


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

    if total == 0:
        return 0.0

    return correct / total


model = SAMModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


best_score = -float("inf")
patience_counter = 0


print("Starting SAM training...")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    steps = 0

    for (
        photo_batch,
        art_batch,
        cartoon_batch
    ) in zip(
        photo_train_loader,
        art_train_loader,
        cartoon_train_loader
    ):

        photo_images, photo_labels = photo_batch
        art_images, art_labels = art_batch
        cartoon_images, cartoon_labels = cartoon_batch

        images = torch.cat(
            [
                photo_images,
                art_images,
                cartoon_images
            ],
            dim=0
        )

        labels = torch.cat(
            [
                photo_labels,
                art_labels,
                cartoon_labels
            ],
            dim=0
        )

        images = images.to(device)
        labels = labels.to(device)

        loss = sam_step(
            model=model,
            optimizer=optimizer,
            images=images,
            labels=labels,
            criterion=criterion,
            rho=RHO
        )

        total_loss += loss
        steps += 1

    photo_acc = evaluate(
        model,
        photo_val_loader
    )

    art_acc = evaluate(
        model,
        art_val_loader
    )

    cartoon_acc = evaluate(
        model,
        cartoon_val_loader
    )

    mean_acc = (
        photo_acc
        + art_acc
        + cartoon_acc
    ) / 3.0

    average_loss = (
        total_loss / steps
        if steps > 0
        else 0.0
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Loss={average_loss:.4f} | "
        f"Photo={photo_acc:.4f} | "
        f"Art={art_acc:.4f} | "
        f"Cartoon={cartoon_acc:.4f} | "
        f"Mean={mean_acc:.4f}"
    )

    if mean_acc > best_score:

        best_score = mean_acc
        patience_counter = 0

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print(
            f"  Saved best model: {CHECKPOINT_PATH}"
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping")
            break


print()
print(
    "Best mean source validation accuracy:",
    f"{best_score:.4f}"
)

print(
    "Saved:",
    CHECKPOINT_PATH
)