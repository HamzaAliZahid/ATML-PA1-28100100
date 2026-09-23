import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import open_clip
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
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

gray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
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

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

clip_model = clip_model.to(device)
clip_model.eval()

for param in clip_model.parameters():
    param.requires_grad = False

classifier = nn.Linear(512, 10).to(device)

classifier.load_state_dict(
    torch.load("task1/models/openclip_classifier.pth")
)

classifier.eval()


def predict(loader):

    predictions = []
    confidences = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            features = clip_model.encode_image(
                images,
                normalize=True
            )

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
    model="OpenCLIP ViT-B/32",
    experiment="grayscale",
    accuracy=accuracy,
    macro_f1=macro_f1,
    mean_confidence=mean_confidence
)