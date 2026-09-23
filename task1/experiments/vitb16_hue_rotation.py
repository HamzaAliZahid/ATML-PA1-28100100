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

hue_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(
        lambda image: transforms.functional.adjust_hue(
            image,
            hue_factor=0.5
        )
    ),
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

hue_dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="test",
    transform=hue_transform,
    download=True
)

test_indices = np.load("task1/imports/selected_test_indices.npy")

standard_subset = Subset(standard_dataset, test_indices)
hue_subset = Subset(hue_dataset, test_indices)

standard_loader = DataLoader(
    standard_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

hue_loader = DataLoader(
    hue_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

weights = models.ViT_B_16_Weights.IMAGENET1K_V1
vit = models.vit_b_16(weights=weights)
vit.heads = nn.Identity()

for param in vit.parameters():
    param.requires_grad = False

vit = vit.to(device)
vit.eval()

classifier = nn.Linear(768, 10).to(device)
classifier.load_state_dict(
    torch.load("task1/models/vitb16_classifier.pth")
)
classifier.eval()

def predict(loader):

    predictions = []
    confidences = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            features = vit(images)

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

hue_predictions, hue_confidence = predict(
    hue_loader
)

labels = torch.tensor(
    standard_dataset.labels
)[test_indices]

accuracy = accuracy_score(
    labels,
    hue_predictions
)

macro_f1 = f1_score(
    labels,
    hue_predictions,
    average="macro"
)

mean_confidence = hue_confidence.mean().item()

consistency = (
    standard_predictions == hue_predictions
).float().mean().item()

standard_accuracy = accuracy_score(
    labels,
    standard_predictions
)

print("\nHue Rotation Results")
print("--------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Mean Max Confidence: {mean_confidence:.4f}")
print(f"Accuracy Change: {accuracy - standard_accuracy:+.4f}")
print(f"Prediction Consistency: {consistency:.4f}")

save_results(
    model="ViT-B/16",
    experiment="hue_rotation",
    accuracy=accuracy,
    macro_f1=macro_f1,
    mean_confidence=mean_confidence
)