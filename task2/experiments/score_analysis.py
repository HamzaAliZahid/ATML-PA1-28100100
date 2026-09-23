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

DATA_ROOT = "task2/data/PACS"

ERM_CHECKPOINT = "task2/checkpoints/pt"
DAN_CHECKPOINT = "task2/checkpoints/pt"
DANN_CHECKPOINT = "task2/checkpoints/pt"
CDAN_CHECKPOINT = "task2/checkpoints/pt"

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

            output = model(images)

            if isinstance(output, tuple):
                class_logits = output[1]
            else:
                class_logits = output

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


def load_model(model, checkpoint_path):

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.to(device)

    return model


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

    loaders = {
        "Photo": DataLoader(
            photo_val,
            batch_size=32,
            shuffle=False
        ),
        "Art": DataLoader(
            art_val,
            batch_size=32,
            shuffle=False
        ),
        "Cartoon": DataLoader(
            cartoon_val,
            batch_size=32,
            shuffle=False
        ),
        "Sketch": DataLoader(
            sketch,
            batch_size=32,
            shuffle=False
        )
    }

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

        results[name] = {}

        source_accuracies = []
        source_f1s = []

        for domain in [
            "Photo",
            "Art",
            "Cartoon"
        ]:

            accuracy, macro_f1 = evaluate(
                model,
                loaders[domain]
            )

            results[name][domain] = (
                accuracy,
                macro_f1
            )

            source_accuracies.append(
                accuracy
            )

            source_f1s.append(
                macro_f1
            )

            print(
                f"{domain}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro-F1={macro_f1:.4f}"
            )

        mean_source_accuracy = np.mean(
            source_accuracies
        )

        mean_source_f1 = np.mean(
            source_f1s
        )

        sketch_accuracy, sketch_f1 = evaluate(
            model,
            loaders["Sketch"]
        )

        results[name]["Mean Source"] = (
            mean_source_accuracy,
            mean_source_f1
        )

        results[name]["Sketch"] = (
            sketch_accuracy,
            sketch_f1
        )

        print(
            f"Mean Source: "
            f"Accuracy={mean_source_accuracy:.4f}, "
            f"Macro-F1={mean_source_f1:.4f}"
        )

        print(
            f"Sketch: "
            f"Accuracy={sketch_accuracy:.4f}, "
            f"Macro-F1={sketch_f1:.4f}"
        )

    erm_sketch_accuracy = results["ERM"]["Sketch"][0]

    print()
    print("=" * 50)
    print("TARGET ACCURACY CHANGE VS ERM")
    print("=" * 50)

    for name in results:

        sketch_accuracy = results[name]["Sketch"][0]

        change = (
            sketch_accuracy
            - erm_sketch_accuracy
        )

        print(
            f"{name}: "
            f"{change:+.4f}"
        )

