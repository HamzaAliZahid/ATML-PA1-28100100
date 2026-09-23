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

DATA_ROOT = "common/datasets/PACS"

MODEL_PATHS = {
    "ERM": "task2/models/erm_best.pt",
    "DAN-DG": "task3/models/dan_dg.pt",
    "SAM": "task3/models/sam.pt",
}

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon",
]

BATCH_SIZE = 32
NUM_CLASSES = 7

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


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])


class FeatureClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=None
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
        features = torch.flatten(
            features,
            start_dim=1
        )

        logits = self.classifier(features)

        return features, logits


def load_checkpoint(path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    return state_dict


def convert_resnet_to_feature_classifier(state_dict):

    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("conv1."):
            new_key = "features.0." + key[len("conv1."):]

        elif key.startswith("bn1."):
            new_key = "features.1." + key[len("bn1."):]

        elif key.startswith("layer1."):
            new_key = "features.4." + key[len("layer1."):]

        elif key.startswith("layer2."):
            new_key = "features.5." + key[len("layer2."):]

        elif key.startswith("layer3."):
            new_key = "features.6." + key[len("layer3."):]

        elif key.startswith("layer4."):
            new_key = "features.7." + key[len("layer4."):]

        elif key.startswith("fc."):
            new_key = "classifier." + key[len("fc."):]

        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict


def load_model(model_name, path):

    print(
        f"\nLoading {model_name}: {path}"
    )

    state_dict = load_checkpoint(path)

    print(
        "Checkpoint keys:",
        list(state_dict.keys())[:5]
    )

    model = FeatureClassifier()

    if model_name == "ERM":

        state_dict = convert_resnet_to_feature_classifier(
            state_dict
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"{model_name} checkpoint loaded successfully"
    )

    return model


def load_domain(domain):

    path = os.path.join(
        DATA_ROOT,
        domain
    )

    return datasets.ImageFolder(
        root=path,
        transform=transform
    )


def get_balanced_indices(datasets_by_domain):

    minimum_size = min(
        len(dataset)
        for dataset in datasets_by_domain.values()
    )

    rng = np.random.default_rng(SEED)

    selected = {}

    for domain, dataset in datasets_by_domain.items():

        indices = np.arange(
            len(dataset)
        )

        rng.shuffle(indices)

        selected[domain] = indices[
            :minimum_size
        ]

    return selected


def extract_features(
    model,
    datasets_by_domain,
    selected_indices
):

    all_features = []
    all_domain_labels = []

    for domain_id, domain in enumerate(
        SOURCE_DOMAINS
    ):

        dataset = datasets_by_domain[domain]

        subset = Subset(
            dataset,
            selected_indices[domain]
        )

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(
            f"  Extracting {domain}: "
            f"{len(subset)} images"
        )

        domain_features = []

        with torch.no_grad():

            for images, _ in loader:

                images = images.to(DEVICE)

                features, _ = model(images)

                domain_features.append(
                    features.cpu().numpy()
                )

        domain_features = np.concatenate(
            domain_features,
            axis=0
        )

        all_features.append(
            domain_features
        )

        all_domain_labels.extend(
            [domain_id] * len(domain_features)
        )

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_domain_labels = np.array(
        all_domain_labels
    )

    return all_features, all_domain_labels


def evaluate_domain_separability(
    features,
    labels
):

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=SEED
    )

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=SEED
    )

    classifier.fit(
        train_x,
        train_y
    )

    predictions = classifier.predict(
        test_x
    )

    accuracy = accuracy_score(
        test_y,
        predictions
    )

    return accuracy


def main():

    print("\nLoading PACS domains...")

    datasets_by_domain = {}

    for domain in SOURCE_DOMAINS:

        dataset = load_domain(domain)

        datasets_by_domain[domain] = dataset

        print(
            f"{domain}: {len(dataset)} images"
        )

    selected_indices = get_balanced_indices(
        datasets_by_domain
    )

    print("\nBalanced sample size:")

    for domain in SOURCE_DOMAINS:

        print(
            f"{domain}: "
            f"{len(selected_indices[domain])}"
        )

    results = {}

    for model_name, model_path in MODEL_PATHS.items():

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"{model_name}"
        )

        print(
            f"{'=' * 60}"
        )

        model = load_model(
            model_name,
            model_path
        )

        features, labels = extract_features(
            model,
            datasets_by_domain,
            selected_indices
        )

        accuracy = evaluate_domain_separability(
            features,
            labels
        )

        results[model_name] = accuracy

        print(
            f"\nDomain classification accuracy: "
            f"{accuracy:.4f}"
        )

    print(
        f"\n{'=' * 60}"
    )

    print("FINAL RESULTS")

    print(
        f"{'=' * 60}"
    )

    print(
        "Chance level: 0.3333"
    )

    for model_name, accuracy in results.items():

        print(
            f"{model_name:10s}: "
            f"{accuracy:.4f}"
        )


if __name__ == "__main__":
    main()