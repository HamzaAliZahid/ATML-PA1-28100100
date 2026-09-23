import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score


SEED = 6304
NUM_CLASSES = 7
BATCH_SIZE = 64

DATA_ROOT = "common/datasets/PACS"

MODEL_PATHS = {
    "ERM": "task2/models/erm_best.pt",
    "DAN-DG": "task3/models/dan_dg.pt",
    "SAM": "task3/models/sam.pt",
}


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


domains = [
    "photo",
    "art_painting",
    "cartoon",
    "sketch"
]


datasets_dict = {}

for domain in domains:

    domain_path = os.path.join(
        DATA_ROOT,
        domain
    )

    datasets_dict[domain] = datasets.ImageFolder(
        domain_path,
        transform=transform
    )

    print(
        domain,
        "images:",
        len(datasets_dict[domain])
    )


loaders = {}

for domain in domains:

    loaders[domain] = DataLoader(
        datasets_dict[domain],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


def build_features_classifier():

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


class FeatureClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.features, self.classifier = (
            build_features_classifier()
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return self.classifier(x)


def build_erm_model():

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


def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def load_erm_checkpoint(model, model_path):

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

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


def load_feature_classifier_checkpoint(
    model,
    model_path
):

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict
    )


def load_model(model_name, model_path):

    if model_name == "ERM":

        model = build_erm_model()

        load_erm_checkpoint(
            model,
            model_path
        )

    else:

        model = FeatureClassifier()

        load_feature_classifier_checkpoint(
            model,
            model_path
        )

    return model.to(device)


def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return accuracy, macro_f1


results = {}


for model_name, model_path in MODEL_PATHS.items():

    print("\n==============================")
    print(model_name)
    print("==============================")

    model = load_model(
        model_name,
        model_path
    )

    results[model_name] = {}

    source_accuracies = []
    source_f1s = []

    for domain in domains:

        accuracy, macro_f1 = evaluate(
            model,
            loaders[domain]
        )

        results[model_name][domain] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

        print(
            f"{domain:15s} "
            f"Accuracy: {accuracy:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

        if domain != "sketch":

            source_accuracies.append(
                accuracy
            )

            source_f1s.append(
                macro_f1
            )

    mean_accuracy = np.mean(
        source_accuracies
    )

    mean_f1 = np.mean(
        source_f1s
    )

    worst_accuracy = np.min(
        source_accuracies
    )

    worst_f1 = np.min(
        source_f1s
    )

    results[model_name]["mean_source"] = {
        "accuracy": mean_accuracy,
        "macro_f1": mean_f1
    }

    results[model_name]["worst_source"] = {
        "accuracy": worst_accuracy,
        "macro_f1": worst_f1
    }

    print(
        f"\nMean source accuracy: "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Worst source accuracy: "
        f"{worst_accuracy:.4f}"
    )

    print(
        f"Mean source Macro-F1: "
        f"{mean_f1:.4f}"
    )

    print(
        f"Worst source Macro-F1: "
        f"{worst_f1:.4f}"
    )


erm_sketch_accuracy = (
    results["ERM"]["sketch"]["accuracy"]
)


print("\n======================================")
print("SKETCH PERFORMANCE")
print("======================================")


for model_name in MODEL_PATHS:

    sketch_accuracy = (
        results[model_name]["sketch"]["accuracy"]
    )

    sketch_f1 = (
        results[model_name]["sketch"]["macro_f1"]
    )

    change = (
        sketch_accuracy
        - erm_sketch_accuracy
    )

    print(
        f"{model_name:8s} "
        f"Accuracy: {sketch_accuracy:.4f} "
        f"Macro-F1: {sketch_f1:.4f} "
        f"Δ Sketch Acc vs ERM: {change:+.4f}"
    )