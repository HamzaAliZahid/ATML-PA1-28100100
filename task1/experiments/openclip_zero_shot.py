import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import open_clip
from common.utils import save_results, select_test_indices
import csv
import os

results_file = "task1/results/openclip_zero_shot_results.csv"

SEED = 6304
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

clip_model = clip_model.to(device)
clip_model.eval()

for param in clip_model.parameters():
    param.requires_grad = False


test_dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="test",
    transform=preprocess,
    download=True
)

test_indices = select_test_indices(test_dataset.labels)
np.save("task1/imports/selected_test_indices.npy", test_indices)


test_subset = Subset(test_dataset, test_indices)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "monkey",
    "ship"
]


prompts = [
    f"a photo of a {class_name}."
    for class_name in class_names
]


text_tokens = tokenizer(prompts).to(device)


with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

image_features = []
image_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        features = clip_model.encode_image(images)

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

        image_features.append(features.cpu())
        image_labels.append(labels)


image_features = torch.cat(image_features)
image_labels = torch.cat(image_labels)

image_features = image_features.to(device)


similarities = image_features @ text_features.T


logits = similarities * clip_model.logit_scale.exp()


probabilities = torch.softmax(logits, dim=1)

predictions = torch.argmax(probabilities, dim=1)


confidence = probabilities.max(dim=1).values

accuracy = accuracy_score(
    image_labels.cpu().numpy(),
    predictions.cpu().numpy()
)

macro_f1 = f1_score(
    image_labels.cpu().numpy(),
    predictions.cpu().numpy(),
    average="macro"
)

mean_confidence = confidence.mean().item()

print("\nCLIP Zero-Shot Results")
print("----------------------")
print(f"Top-1 Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Mean Max Confidence: {mean_confidence:.4f}")

save_results(
    model="OpenCLIP ViT-B/32",
    experiment="clean_zero_shot",
    accuracy=accuracy,
    macro_f1=macro_f1,
    mean_confidence=mean_confidence
)

os.makedirs("results", exist_ok=True)

file_exists = os.path.isfile(results_file)

with open(results_file, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "model",
            "experiment",
            "accuracy",
            "macro_f1",
            "mean_confidence"
        ])

    writer.writerow([
        "OpenCLIP ViT-B/32",
        "clean_zero_shot",
        accuracy,
        macro_f1,
        mean_confidence
    ])