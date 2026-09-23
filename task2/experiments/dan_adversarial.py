import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


SEED = 6304
SOURCE_BATCH_SIZE = 8
TARGET_BATCH_SIZE = 24

NUM_CLASSES = 7
MAX_EPOCHS = 30
PATIENCE = 5

LR = 1e-4
WEIGHT_DECAY = 1e-4

DATA_ROOT = "common/datasets/PACS"
CHECKPOINT_PATH = "task2/models/dann_best.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_domain(domain, transform):
    path = os.path.join(DATA_ROOT, domain)
    return datasets.ImageFolder(path, transform=transform)


def make_split(dataset, seed=SEED):
    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=seed
    )

    return train_idx, val_idx

def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)

class DANN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        feature_dim = backbone.fc.in_features

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0):

        features = self.features(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        class_logits = self.classifier(features)

        reversed_features = gradient_reverse(
            features,
            alpha
        )

        domain_logits = self.domain_classifier(
            reversed_features
        )

        return (
            features,
            class_logits,
            domain_logits
        )

def get_alpha(progress):
    return 2.0 / (
        1.0 + np.exp(-10.0 * progress)
    ) - 1.0

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            _, class_logits, _ = model(
                images,
                alpha=0.0
            )

            predictions = class_logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = correct / total

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        tp = sum(
            1
            for p, y in zip(
                all_predictions,
                all_labels
            )
            if p == class_id and y == class_id
        )

        fp = sum(
            1
            for p, y in zip(
                all_predictions,
                all_labels
            )
            if p == class_id and y != class_id
        )

        fn = sum(
            1
            for p, y in zip(
                all_predictions,
                all_labels
            )
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

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return accuracy, macro_f1

def train_one_epoch(
    model,
    photo_loader,
    art_loader,
    cartoon_loader,
    sketch_loader,
    optimizer,
    epoch,
    total_epochs
):

    model.train()
    model.apply(set_bn_eval)

    photo_iter = iter(photo_loader)
    art_iter = iter(art_loader)
    cartoon_iter = iter(cartoon_loader)
    sketch_iter = iter(sketch_loader)

    num_steps = max(
        len(photo_loader),
        len(art_loader),
        len(cartoon_loader)
    )

    total_loss = 0.0

    for step in range(num_steps):

        try:
            photo_images, photo_labels = next(
                photo_iter
            )
        except StopIteration:
            photo_iter = iter(photo_loader)
            photo_images, photo_labels = next(
                photo_iter
            )

        try:
            art_images, art_labels = next(
                art_iter
            )
        except StopIteration:
            art_iter = iter(art_loader)
            art_images, art_labels = next(
                art_iter
            )

        try:
            cartoon_images, cartoon_labels = next(
                cartoon_iter
            )
        except StopIteration:
            cartoon_iter = iter(cartoon_loader)
            cartoon_images, cartoon_labels = next(
                cartoon_iter
            )

        try:
            sketch_images, _ = next(
                sketch_iter
            )
        except StopIteration:
            sketch_iter = iter(sketch_loader)
            sketch_images, _ = next(
                sketch_iter
            )

        source_images = torch.cat(
            [
                photo_images,
                art_images,
                cartoon_images
            ],
            dim=0
        ).to(device)

        source_labels = torch.cat(
            [
                photo_labels,
                art_labels,
                cartoon_labels
            ],
            dim=0
        ).to(device)

        target_images = sketch_images.to(device)

        source_domain_labels = torch.zeros(
            source_images.size(0),
            dtype=torch.long,
            device=device
        )

        target_domain_labels = torch.ones(
            target_images.size(0),
            dtype=torch.long,
            device=device
        )

        images = torch.cat(
            [
                source_images,
                target_images
            ],
            dim=0
        )

        domain_labels = torch.cat(
            [
                source_domain_labels,
                target_domain_labels
            ],
            dim=0
        )

        progress = (
            epoch * num_steps + step
        ) / (
            total_epochs * num_steps
        )

        alpha = get_alpha(progress)

        optimizer.zero_grad()

        _, source_class_logits, source_domain_logits = model(
            source_images,
            alpha=alpha
        )

        _, target_class_logits, target_domain_logits = model(
            target_images,
            alpha=alpha
        )

        classification_loss = F.cross_entropy(
            source_class_logits,
            source_labels
        )

        domain_logits = torch.cat(
            [
                source_domain_logits,
                target_domain_logits
            ],
            dim=0
        )

        domain_loss = F.cross_entropy(
            domain_logits,
            domain_labels
        )

        loss = (
            classification_loss
            + domain_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_steps

def main():

    set_seed(SEED)

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True
    )

    photo_train = load_domain(
        "photo",
        train_transform
    )

    photo_val = load_domain(
        "photo",
        val_transform
    )

    art_train = load_domain(
        "art_painting",
        train_transform
    )

    art_val = load_domain(
        "art_painting",
        val_transform
    )

    cartoon_train = load_domain(
        "cartoon",
        train_transform
    )

    cartoon_val = load_domain(
        "cartoon",
        val_transform
    )

    sketch = load_domain(
        "sketch",
        val_transform
    )

    photo_train_idx, photo_val_idx = make_split(
        photo_train
    )

    art_train_idx, art_val_idx = make_split(
        art_train
    )

    cartoon_train_idx, cartoon_val_idx = make_split(
        cartoon_train
    )

    photo_train = Subset(
        photo_train,
        photo_train_idx
    )

    photo_val = Subset(
        photo_val,
        photo_val_idx
    )

    art_train = Subset(
        art_train,
        art_train_idx
    )

    art_val = Subset(
        art_val,
        art_val_idx
    )

    cartoon_train = Subset(
        cartoon_train,
        cartoon_train_idx
    )

    cartoon_val = Subset(
        cartoon_val,
        cartoon_val_idx
    )

    photo_loader = DataLoader(
        photo_train,
        batch_size=SOURCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    art_loader = DataLoader(
        art_train,
        batch_size=SOURCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    cartoon_loader = DataLoader(
        cartoon_train,
        batch_size=SOURCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    sketch_loader = DataLoader(
        sketch,
        batch_size=TARGET_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    photo_val_loader = DataLoader(
        photo_val,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    art_val_loader = DataLoader(
        art_val,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    cartoon_val_loader = DataLoader(
        cartoon_val,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = DANN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):

        loss = train_one_epoch(
            model,
            photo_loader,
            art_loader,
            cartoon_loader,
            sketch_loader,
            optimizer,
            epoch,
            MAX_EPOCHS
        )

        photo_acc, photo_f1 = evaluate(
            model,
            photo_val_loader
        )

        art_acc, art_f1 = evaluate(
            model,
            art_val_loader
        )

        cartoon_acc, cartoon_f1 = evaluate(
            model,
            cartoon_val_loader
        )

        mean_f1 = np.mean(
            [
                photo_f1,
                art_f1,
                cartoon_f1
            ]
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss {loss:.4f} | "
            f"Photo F1 {photo_f1:.4f} | "
            f"Art F1 {art_f1:.4f} | "
            f"Cartoon F1 {cartoon_f1:.4f} | "
            f"Mean F1 {mean_f1:.4f}"
        )

        if mean_f1 > best_f1:

            best_f1 = mean_f1
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping.")
                break

    print(
        f"Best source validation macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Saved checkpoint to: "
        f"{CHECKPOINT_PATH}"
    )

if __name__ == "__main__":
    main()