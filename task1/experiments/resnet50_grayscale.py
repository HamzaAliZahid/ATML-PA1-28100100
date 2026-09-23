import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from common.utils import save_results

SEED = 6304
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

gray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

standard_dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="test",
    transform=standard_transform,
    download=True
)

gray_dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="test",
    transform=gray_transform,
    download=True
)

test_indices = np.load("task1/imports/selected_test_indices.npy")

standard_subset = Subset(
    standard_dataset,
    test_indices
)

gray_subset = Subset(
    gray_dataset,
    test_indices
)

standard_loader = DataLoader(
    standard_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

gray_loader = DataLoader(
    gray_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

weights = models.ResNet50_Weights.IMAGENET1K_V2
resnet = models.resnet50(weights=weights)

resnet.fc = nn.Identity()

for param in resnet.parameters():
    param.requires_grad = False

resnet = resnet.to(device)
resnet.eval()

classifier = nn.Linear(2048, 10).to(device)

classifier.load_state_dict(
    torch.load("task1/models/resnet50_classifier.pth")
)

classifier.eval()


def predict(loader):

    predictions = []
    confidences = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            features = resnet(images)
            logits = classifier(features)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions.append(
                torch.argmax(
                    probabilities,
                    dim=1
                ).cpu()
            )

            confidences.append(
                probabilities.max(
                    dim=1
                ).values.cpu()
            )

    return torch.cat(predictions), torch.cat(confidences)


standard_predictions, standard_confidence = predict(
    standard_loader
)

gray_predictions, gray_confidence = predict(
    gray_loader
)

labels = torch.tensor(
    standard_dataset.labels
)[test_indices]

accuracy = accuracy_score(
    labels,
    gray_predictions
)

macro_f1 = f1_score(
    labels,
    gray_predictions,
    average="macro"
)

mean_confidence = gray_confidence.mean().item()

consistency = (
    standard_predictions == gray_predictions
).float().mean().item()

standard_accuracy = accuracy_score(
    labels,
    standard_predictions
)

print("\nGrayscale Results")
print("-----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Mean Max Confidence: {mean_confidence:.4f}")
print(f"Accuracy Change: {accuracy - standard_accuracy:+.4f}")
print(f"Prediction Consistency: {consistency:.4f}")

save_results(
    model="ResNet-50",
    experiment="grayscale",
    accuracy=accuracy,
    macro_f1=macro_f1,
    mean_confidence=mean_confidence
)