import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


SEED = 6304
NUM_CLASSES = 7
BATCH_SIZE = 96
RHO = 0.05

DATA_ROOT = "data/PACS"

MODEL_PATHS = {
    "ERM": "results/erm_resnet18.pth",
    "DAN-DG": "results/dan_dg_resnet18.pth",
    "SAM": "results/sam_resnet18.pth",
}


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


source_domains = [
    "photo",
    "art_painting",
    "cartoon"
]

datasets_dict = {}

for domain in source_domains:

    path = os.path.join(
        DATA_ROOT,
        domain
    )

    datasets_dict[domain] = datasets.ImageFolder(
        path,
        transform=transform
    )


rng = np.random.default_rng(SEED)

selected_indices = {}

for domain in source_domains:

    selected_indices[domain] = rng.choice(
        len(datasets_dict[domain]),
        size=32,
        replace=False
    )


def create_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model.to(device)


def get_validation_batch():

    images_list = []
    labels_list = []

    for domain in source_domains:

        subset = Subset(
            datasets_dict[domain],
            selected_indices[domain]
        )

        loader = DataLoader(
            subset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        images, labels = next(iter(loader))

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

    return images, labels


images, labels = get_validation_batch()

criterion = nn.CrossEntropyLoss()

sharpness_results = {}


for model_name, model_path in MODEL_PATHS.items():

    print("\n================================")
    print(model_name)
    print("================================")

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    model.zero_grad()

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

    scale = RHO / (grad_norm + 1e-12)

    perturbations = []

    with torch.no_grad():

        for parameter in model.parameters():

            if parameter.grad is not None:

                epsilon = parameter.grad * scale

                parameter.add_(epsilon)

                perturbations.append(
                    (parameter, epsilon)
                )

    with torch.no_grad():

        perturbed_loss = criterion(
            model(images),
            labels
        )

    with torch.no_grad():

        for parameter, epsilon in perturbations:

            parameter.sub_(epsilon)

    sharpness = (
        perturbed_loss.item()
        - loss.item()
    )

    sharpness_results[model_name] = sharpness

    print(
        f"Original loss: {loss.item():.6f}"
    )

    print(
        f"Perturbed loss: {perturbed_loss.item():.6f}"
    )

    print(
        f"Delta sharp: {sharpness:.6f}"
    )


print("\n================================")
print("LOCAL SHARPNESS RESULTS")
print("================================")

for model_name, value in sharpness_results.items():

    print(
        f"{model_name:8s}: "
        f"{value:.6f}"
    )