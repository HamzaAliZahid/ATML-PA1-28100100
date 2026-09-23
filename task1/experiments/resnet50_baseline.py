import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from common.utils import save_results, select_test_indices
import csv

csv_path = "task1/results/resnet50_final_results.csv"

SEED = 6304
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="train",
    transform=transform,
    download=True
)

test_dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="test",
    transform=transform,
    download=True
)

indices = np.arange(len(train_dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=train_dataset.labels,
    random_state=SEED
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_subset,
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

def extract_features(loader):

    features = []
    labels = []

    with torch.no_grad():

        for images, y in loader:

            images = images.to(device)

            x = resnet(images)

            features.append(x.cpu())
            labels.append(y)

    return torch.cat(features), torch.cat(labels)


train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)

classifier = nn.Linear(2048, 10).to(device)

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

loss_function = nn.CrossEntropyLoss()

best_val_accuracy = 0
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):

    classifier.train()

    x = train_features.to(device)
    y = train_labels.to(device)

    predictions = classifier(x)

    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    classifier.eval()

    with torch.no_grad():
        x_val = val_features.to(device)
        y_val = val_labels.to(device)

        val_predictions = classifier(x_val)

        predicted_classes = torch.argmax(
            val_predictions,
            dim=1
        )

        val_accuracy = (
            predicted_classes == y_val
        ).float().mean().item()


    print(
        f"Epoch {epoch + 1}: "
        f"Loss = {loss.item():.4f}, "
        f"Val Accuracy = {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_classifier = classifier.state_dict()

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

classifier.load_state_dict(best_classifier)


test_indices = select_test_indices(test_dataset.labels)
np.save("task1/imports/selected_test_indices.npy", test_indices)

test_subset = Subset(
    test_dataset,
    test_indices
)

test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_features, test_labels = extract_features(test_loader)

classifier.eval()

with torch.no_grad():

    logits = classifier(
        test_features.to(device)
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    confidence = probabilities.max(
        dim=1
    ).values


accuracy = accuracy_score(
    test_labels.numpy(),
    predictions.cpu().numpy()
)

macro_f1 = f1_score(
    test_labels.numpy(),
    predictions.cpu().numpy(),
    average="macro"
)

mean_confidence = confidence.mean().item()

print("\nFinal Results")
print("-------------")
print(f"Top-1 Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Mean Max Confidence: {mean_confidence:.4f}")

save_results(
    model="ResNet-50",
    experiment="clean",
    accuracy=accuracy,
    macro_f1=macro_f1,
    mean_confidence=mean_confidence
)

torch.save(
    classifier.state_dict(),
    "task1/models/resnet50_classifier.pth"
)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Model",
        "Experiment",
        "Top-1 Accuracy",
        "Macro-F1",
        "Mean Max Confidence"
    ])

    writer.writerow([
        "ResNet-50",
        "clean",
        accuracy,
        macro_f1,
        mean_confidence
    ])

print(f"Results saved to {csv_path}")