import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix


SEED = 6304
NUM_CLASSES = 7
BATCH_SIZE = 64

DATA_ROOT = "common/datasets/PACS"

MODEL_PATHS = {
    "ERM": "task2/models/erm_best.pt",
    "DAN-DG": "task3/models/dan_dg.pt",
    "SAM": "task3/models/sam.pt",
}


torch.manual_seed(SEED)
np.random.seed(SEED)

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


sketch_dataset = datasets.ImageFolder(
    os.path.join(DATA_ROOT, "sketch"),
    transform=transform
)

sketch_loader = DataLoader(
    sketch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


class_names = sketch_dataset.classes

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, name)


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

        return logits


def load_checkpoint(path):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        return checkpoint

    return checkpoint


def convert_erm_checkpoint(state_dict):

    converted = {}

    for key, value in state_dict.items():

        converted[key] = value

    return converted


def create_erm_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model.to(device)


def create_feature_classifier():

    return FeatureClassifier().to(device)


def load_model(model_name, model_path):

    state_dict = load_checkpoint(model_path)

    print(
        "\nCheckpoint:",
        model_path
    )

    print(
        "First checkpoint keys:",
        list(state_dict.keys())[:5]
    )

    if model_name == "ERM":

        model = create_erm_model()

        model.load_state_dict(
            convert_erm_checkpoint(state_dict),
            strict=True
        )

    else:

        model = create_feature_classifier()

        model.load_state_dict(
            state_dict,
            strict=True
        )

    model.eval()

    print(
        f"{model_name} loaded successfully"
    )

    return model


def evaluate_model(model):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in sketch_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


results = {}


for model_name, model_path in MODEL_PATHS.items():

    print("\n================================")
    print(model_name)
    print("================================")

    model = load_model(
        model_name,
        model_path
    )

    labels, predictions = evaluate_model(
        model
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES)
    )

    row_totals = cm.sum(axis=1)

    per_class_accuracy = np.divide(
        np.diag(cm),
        row_totals,
        out=np.zeros(NUM_CLASSES, dtype=float),
        where=row_totals != 0
    )

    results[model_name] = {
        "labels": labels,
        "predictions": predictions,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy
    }

    print("\nPer-class accuracy:")

    for i, name in enumerate(class_names):

        print(
            f"{name:15s}: "
            f"{per_class_accuracy[i] * 100:.2f}%"
        )


print("\n================================")
print("PER-CLASS COMPARISON WITH ERM")
print("================================")

erm_accuracy = results["ERM"]["per_class_accuracy"]

for model_name in ["DAN-DG", "SAM"]:

    model_accuracy = (
        results[model_name]["per_class_accuracy"]
    )

    print(f"\n{model_name}")

    for i, name in enumerate(class_names):

        change = (
            model_accuracy[i] -
            erm_accuracy[i]
        ) * 100

        print(
            f"{name:15s}: "
            f"{model_accuracy[i] * 100:.2f}% "
            f"({change:+.2f} pp vs ERM)"
        )


print("\n================================")
print("CONFUSION MATRICES")
print("================================")

for model_name in MODEL_PATHS:

    print(f"\n{model_name}")

    print(
        results[model_name]["confusion_matrix"]
    )