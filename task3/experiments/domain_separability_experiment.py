import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


SEED = 6304
NUM_CLASSES = 7
BATCH_SIZE = 64

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

    print(
        domain,
        "images:",
        len(datasets_dict[domain])
    )


min_size = min(
    len(datasets_dict[domain])
    for domain in source_domains
)

rng = np.random.default_rng(SEED)

selected_indices = {}

for domain in source_domains:

    selected_indices[domain] = rng.choice(
        len(datasets_dict[domain]),
        size=min_size,
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


def extract_features(model, dataset, indices):

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            features = model(images)

            all_features.append(
                features.cpu().numpy()
            )

    return np.concatenate(
        all_features,
        axis=0
    )


domain_scores = {}

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

    model.fc = nn.Identity()

    all_features = []
    all_domain_labels = []

    for domain_id, domain in enumerate(source_domains):

        features = extract_features(
            model,
            datasets_dict[domain],
            selected_indices[domain]
        )

        all_features.append(features)

        all_domain_labels.extend(
            [domain_id] * len(features)
        )

        print(
            domain,
            "features:",
            features.shape
        )

    X = np.concatenate(
        all_features,
        axis=0
    )

    y = np.array(
        all_domain_labels
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    classifier = LogisticRegression(
        C=1,
        max_iter=2000,
        multi_class="multinomial",
        random_state=SEED
    )

    classifier.fit(
        X_train,
        y_train
    )

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    domain_scores[model_name] = accuracy

    print(
        f"Source-domain separability: "
        f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
    )


print("\n================================")
print("SOURCE-DOMAIN SEPARABILITY")
print("================================")

print("Chance level: 33.33%")

for model_name, score in domain_scores.items():

    print(
        f"{model_name:8s}: "
        f"{score:.4f} ({score * 100:.2f}%)"
    )