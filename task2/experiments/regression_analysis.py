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

DATA_ROOT = "common/datasets/PACS"

ERM_CHECKPOINT = "task2/models/erm_best.pt"
DAN_CHECKPOINT = "task2/models/dan_best.pt"
DANN_CHECKPOINT = "task2/models/dann_best.pt"
CDAN_CHECKPOINT = "task2/models/cdan_best.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
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


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])


def load_domain(domain):

    path = os.path.join(
        DATA_ROOT,
        domain
    )

    return datasets.ImageFolder(
        path,
        transform=val_transform
    )


def make_split(dataset):

    indices = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    _, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    return val_idx


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


class ERM(nn.Module):

    def __init__(self):

        super().__init__()

        self.features, self.classifier = build_backbone()

    def forward(self, x):

        features = self.features(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        return self.classifier(features)


class DAN(nn.Module):

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

        return features, logits


class DANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features, self.classifier = build_backbone()

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):

        features = self.features(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        class_logits = self.classifier(features)

        return features, class_logits


class CDAN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features, self.classifier = build_backbone()

        self.domain_classifier = nn.Sequential(
            nn.Linear(
                512 * NUM_CLASSES,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):

        features = self.features(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        class_logits = self.classifier(features)

        return features, class_logits


def convert_resnet_state_dict(state_dict):

    mapping = {
        "conv1.": "features.0.",
        "bn1.": "features.1.",
        "layer1.": "features.4.",
        "layer2.": "features.5.",
        "layer3.": "features.6.",
        "layer4.": "features.7.",
        "fc.": "classifier."
    }

    converted = {}

    for key, value in state_dict.items():

        new_key = key

        for old_prefix, new_prefix in mapping.items():

            if key.startswith(old_prefix):

                new_key = (
                    new_prefix
                    + key[len(old_prefix):]
                )

                break

        converted[new_key] = value

    return converted


def load_model(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

    model_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        state_dict.keys()
    )

    if checkpoint_keys.issubset(model_keys):

        model.load_state_dict(
            state_dict,
            strict=False
        )

    else:

        converted_state_dict = convert_resnet_state_dict(
            state_dict
        )

        model.load_state_dict(
            converted_state_dict,
            strict=False
        )

    model.to(device)

    return model


def extract_features(
    model,
    loader
):

    model.eval()

    all_features = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            features = model.features(
                images
            )

            features = torch.flatten(
                features,
                start_dim=1
            )

            all_features.append(
                features.cpu().numpy()
            )

    return np.concatenate(
        all_features,
        axis=0
    )


def main():

    set_seed(SEED)

    photo = load_domain("photo")
    art = load_domain("art_painting")
    cartoon = load_domain("cartoon")
    sketch = load_domain("sketch")

    photo_val = Subset(
        photo,
        make_split(photo)
    )

    art_val = Subset(
        art,
        make_split(art)
    )

    cartoon_val = Subset(
        cartoon,
        make_split(cartoon)
    )

    source_dataset = torch.utils.data.ConcatDataset([
        photo_val,
        art_val,
        cartoon_val
    ])

    source_loader = DataLoader(
        source_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    sketch_loader = DataLoader(
        sketch,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    models_to_evaluate = {

        "ERM": (
            ERM(),
            ERM_CHECKPOINT
        ),

        "DAN": (
            DAN(),
            DAN_CHECKPOINT
        ),

        "DANN": (
            DANN(),
            DANN_CHECKPOINT
        ),

        "CDAN": (
            CDAN(),
            CDAN_CHECKPOINT
        )
    }

    results = {}

    for name, (model, checkpoint) in models_to_evaluate.items():

        print()
        print("=" * 50)
        print(name)
        print("=" * 50)

        model = load_model(
            model,
            checkpoint
        )

        source_features = extract_features(
            model,
            source_loader
        )

        target_features = extract_features(
            model,
            sketch_loader
        )

        num_samples = min(
            len(source_features),
            len(target_features)
        )

        source_features = source_features[
            :num_samples
        ]

        target_features = target_features[
            :num_samples
        ]

        X = np.concatenate(
            [
                source_features,
                target_features
            ],
            axis=0
        )

        y = np.concatenate(
            [
                np.zeros(num_samples),
                np.ones(num_samples)
            ]
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
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED
        )

        classifier.fit(
            X_train,
            y_train
        )

        predictions = classifier.predict(
            X_test
        )

        domain_accuracy = accuracy_score(
            y_test,
            predictions
        )

        results[name] = domain_accuracy

        print(
            f"Domain separability: "
            f"{domain_accuracy:.4f}"
        )

    print()
    print("=" * 50)
    print("DOMAIN SEPARABILITY")
    print("=" * 50)

    for name, score in results.items():

        print(
            f"{name}: {score:.4f} "
            f"({score * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()