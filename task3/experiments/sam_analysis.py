import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


SEED = 6304
NUM_CLASSES = 7
BATCH_SIZE_PER_DOMAIN = 8
EPOCHS = 50
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4

RHO_VALUES = [0.01, 0.05, 0.1]

DATA_ROOT = "common/datasets/PACS"

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]

TARGET_DOMAIN = "sketch"

OUTPUT_DIR = "task3/results/sam_rho"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


source_datasets = []

for domain in SOURCE_DOMAINS:

    dataset = datasets.ImageFolder(
        os.path.join(DATA_ROOT, domain),
        transform=transform
    )

    source_datasets.append(dataset)

    print(
        domain,
        "images:",
        len(dataset)
    )


source_loaders = [
    DataLoader(
        dataset,
        batch_size=BATCH_SIZE_PER_DOMAIN,
        shuffle=True
    )
    for dataset in source_datasets
]


def create_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model.to(device)


def sam_step(
    model,
    optimizer,
    images,
    labels,
    criterion,
    rho
):

    optimizer.zero_grad()

    loss = criterion(
        model(images),
        labels
    )

    loss.backward()

    grad_norm = torch.norm(
        torch.stack([
            p.grad.norm()
            for p in model.parameters()
            if p.grad is not None
        ])
    )

    scale = rho / (grad_norm + 1e-12)

    perturbations = []

    with torch.no_grad():

        for parameter in model.parameters():

            if parameter.grad is not None:

                epsilon = parameter.grad * scale

                parameter.add_(epsilon)

                perturbations.append(
                    (parameter, epsilon)
                )

    optimizer.zero_grad()

    perturbed_loss = criterion(
        model(images),
        labels
    )

    perturbed_loss.backward()

    with torch.no_grad():

        for parameter, epsilon in perturbations:

            parameter.sub_(epsilon)

    optimizer.step()

    return loss.item()


def train_model(rho):

    model = create_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0

    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0
        batches = 0

        loaders = [
            iter(loader)
            for loader in source_loaders
        ]

        max_batches = max(
            len(loader)
            for loader in source_loaders
        )

        for _ in range(max_batches):

            images_list = []
            labels_list = []

            for loader_iter in loaders:

                try:
                    images, labels = next(loader_iter)

                except StopIteration:
                    loader_iter = iter(
                        source_loaders[
                            len(images_list)
                        ]
                    )

                    images, labels = next(loader_iter)

                images_list.append(images)
                labels_list.append(labels)

            images = torch.cat(
                images_list,
                dim=0
            ).to(device)

            labels = torch.cat(
                labels_list,
                dim=0
            ).to(device)

            loss = sam_step(
                model,
                optimizer,
                images,
                labels,
                criterion,
                rho
            )

            epoch_loss += loss
            batches += 1

        epoch_loss /= batches

        print(
            f"rho={rho} | "
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {epoch_loss:.4f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss
            patience_counter = 0

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(best_state)

    return model


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


for rho in RHO_VALUES:

    print("\n================================")
    print(f"TRAINING SAM WITH RHO = {rho}")
    print("================================")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = train_model(rho)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sam_rho_{rho}.pth"
    )

    torch.save(
        model.state_dict(),
        output_path
    )

    print(
        f"Saved: {output_path}"
    )