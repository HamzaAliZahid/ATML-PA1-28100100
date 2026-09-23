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

DATA_ROOT = "common/datasets/PACS"
ERM_CHECKPOINT = "task2/models/erm_best.pt"

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


def build_model():

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

    return nn.Sequential(
        features,
        nn.Flatten(),
        classifier
    )


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

            logits = model(images)

            predictions = logits.argmax(
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


def load_checkpoint(model):

    checkpoint = torch.load(
        ERM_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model_state_dict = model.state_dict()

    if set(state_dict.keys()) == set(
        model_state_dict.keys()
    ):

        model.load_state_dict(state_dict)

        return

    converted_state_dict = {}

    mapping = {
        "conv1.": "0.0.",
        "bn1.": "0.1.",
        "layer1.": "0.4.",
        "layer2.": "0.5.",
        "layer3.": "0.6.",
        "layer4.": "0.7.",
        "fc.": "2."
    }

    for key, value in state_dict.items():

        new_key = key

        for old_prefix, new_prefix in mapping.items():

            if key.startswith(old_prefix):

                new_key = (
                    new_prefix
                    + key[len(old_prefix):]
                )

                break

        converted_state_dict[new_key] = value

    model.load_state_dict(
        converted_state_dict
    )


def main():

    set_seed(SEED)

    photo = load_domain("photo")
    art = load_domain("art_painting")
    cartoon = load_domain("cartoon")

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
        )
    }

    model = build_model()

    load_checkpoint(model)

    model.to(device)

    results = {}

    accuracies = []
    f1_scores = []

    for domain in [
        "Photo",
        "Art",
        "Cartoon"
    ]:

        accuracy, macro_f1 = evaluate(
            model,
            loaders[domain]
        )

        results[domain] = (
            accuracy,
            macro_f1
        )

        accuracies.append(accuracy)
        f1_scores.append(macro_f1)

        print(
            f"{domain}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro-F1={macro_f1:.4f}"
        )

    mean_accuracy = np.mean(
        accuracies
    )

    mean_f1 = np.mean(
        f1_scores
    )

    worst_accuracy = np.min(
        accuracies
    )

    worst_f1 = np.min(
        f1_scores
    )

    print()
    print("=" * 50)
    print("ERM SOURCE VALIDATION RESULTS")
    print("=" * 50)

    print(
        f"Mean Source: "
        f"Accuracy={mean_accuracy:.4f}, "
        f"Macro-F1={mean_f1:.4f}"
    )

    print(
        f"Worst Source: "
        f"Accuracy={worst_accuracy:.4f}, "
        f"Macro-F1={worst_f1:.4f}"
    )


if __name__ == "__main__":
    main()