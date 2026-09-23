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

DATA_ROOT = "data/PACS"

MODEL_PATHS = {
    "ERM": "results/erm_resnet18.pth",
    "DAN-DG": "results/dan_dg_resnet18.pth",
    "SAM": "results/sam_resnet18.pth",
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
    shuffle=False
)


class_names = sketch_dataset.classes

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)


def create_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model.to(device)


def evaluate_model(model):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in sketch_loader:

            images = images.to(device)

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

    return np.array(all_labels), np.array(all_predictions)


results = {}


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

    labels, predictions = evaluate_model(model)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES)
    )

    per_class_accuracy = (
        np.diag(cm) /
        cm.sum(axis=1)
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

    model_accuracy = results[model_name]["per_class_accuracy"]

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