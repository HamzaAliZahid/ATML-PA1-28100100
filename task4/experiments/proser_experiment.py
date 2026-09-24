import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


SEED = 6304

BATCH_SIZE = 128

NUM_CLASSES = 10
NUM_DUMMY = 5

CHECKPOINT_PATH = "task4/models/proser_best.pth"
RESULTS_PATH = "task4/results/proser_results.pt"

RESULTS_DIR = "task4/results"

CIFAR10_DIR = "common/datasets/CIFAR10"
CIFAR100_DIR = "common/datasets/CIFAR100"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


class PROSER(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone.maxpool = nn.Identity()

        feature_dim = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(
            feature_dim,
            NUM_CLASSES
        )

        self.dummy_classifier = nn.Linear(
            feature_dim,
            NUM_DUMMY
        )

    def forward_to_layer2(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        return x

    def layer2_to_features(self, x):

        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):

        x = self.forward_to_layer2(x)
        features = self.layer2_to_features(x)

        known_logits = self.classifier(features)
        dummy_logits = self.dummy_classifier(features)

        return known_logits, dummy_logits


transform = transforms.Compose([
    transforms.ToTensor()
])


cifar10_train = datasets.CIFAR10(
    root=CIFAR10_DIR,
    train=True,
    download=False,
    transform=transform
)

cifar10_test = datasets.CIFAR10(
    root=CIFAR10_DIR,
    train=False,
    download=False,
    transform=transform
)

cifar100_test = datasets.CIFAR100(
    root=CIFAR100_DIR,
    train=False,
    download=False,
    transform=transform
)


train_labels = np.array(cifar10_train.targets)

all_indices = np.arange(len(cifar10_train))

_, validation_indices = train_test_split(
    all_indices,
    test_size=0.10,
    stratify=train_labels,
    random_state=SEED
)

validation_dataset = Subset(
    cifar10_train,
    validation_indices
)


known_test_dataset = cifar10_test


near_class_names = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]

far_class_names = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe"
]


cifar100_class_to_idx = {
    name: i
    for i, name in enumerate(cifar100_test.classes)
}


near_class_indices = [
    cifar100_class_to_idx[name]
    for name in near_class_names
]

far_class_indices = [
    cifar100_class_to_idx[name]
    for name in far_class_names
]


cifar100_targets = np.array(cifar100_test.targets)


near_indices = np.where(
    np.isin(cifar100_targets, near_class_indices)
)[0]

far_indices = np.where(
    np.isin(cifar100_targets, far_class_indices)
)[0]


near_dataset = Subset(
    cifar100_test,
    near_indices
)

far_dataset = Subset(
    cifar100_test,
    far_indices
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

known_test_loader = DataLoader(
    known_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

near_loader = DataLoader(
    near_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

far_loader = DataLoader(
    far_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


model = PROSER().to(device)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


def extract_outputs(loader):

    all_known_logits = []
    all_dummy_logits = []
    all_features = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            x = model.forward_to_layer2(images)
            features = model.layer2_to_features(x)

            known_logits = model.classifier(features)
            dummy_logits = model.dummy_classifier(features)

            all_known_logits.append(
                known_logits.cpu()
            )

            all_dummy_logits.append(
                dummy_logits.cpu()
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    return (
        torch.cat(all_known_logits),
        torch.cat(all_dummy_logits),
        torch.cat(all_features),
        torch.cat(all_labels)
    )


print("Extracting validation outputs...")

val_known, val_dummy, val_features, val_labels = extract_outputs(
    validation_loader
)

print("Extracting known test outputs...")

test_known, test_dummy, test_features, test_labels = extract_outputs(
    known_test_loader
)

print("Extracting near unknown outputs...")

near_known, near_dummy, near_features, near_labels = extract_outputs(
    near_loader
)

print("Extracting far unknown outputs...")

far_known, far_dummy, far_features, far_labels = extract_outputs(
    far_loader
)


os.makedirs(RESULTS_DIR, exist_ok=True)


np.save(
    os.path.join(RESULTS_DIR, "proser_validation_logits.npy"),
    val_known.numpy()
)

np.save(
    os.path.join(RESULTS_DIR, "proser_known_test_logits.npy"),
    test_known.numpy()
)

np.save(
    os.path.join(RESULTS_DIR, "proser_near_logits.npy"),
    near_known.numpy()
)

np.save(
    os.path.join(RESULTS_DIR, "proser_far_logits.npy"),
    far_known.numpy()
)


proser_validation_raw = (
    torch.max(val_dummy, dim=1).values
    - torch.max(val_known, dim=1).values
)

proser_known_raw = (
    torch.max(test_dummy, dim=1).values
    - torch.max(test_known, dim=1).values
)

proser_near_raw = (
    torch.max(near_dummy, dim=1).values
    - torch.max(near_known, dim=1).values
)

proser_far_raw = (
    torch.max(far_dummy, dim=1).values
    - torch.max(far_known, dim=1).values
)


bias = torch.quantile(
    proser_validation_raw,
    0.95
).item()


proser_validation_score = (
    proser_validation_raw - bias
)

proser_known_score = (
    proser_known_raw - bias
)

proser_near_score = (
    proser_near_raw - bias
)

proser_far_score = (
    proser_far_raw - bias
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "proser_validation_placeholder_score.npy"
    ),
    proser_validation_score.numpy()
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "proser_known_placeholder_score.npy"
    ),
    proser_known_score.numpy()
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "proser_near_placeholder_score.npy"
    ),
    proser_near_score.numpy()
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "proser_far_placeholder_score.npy"
    ),
    proser_far_score.numpy()
)


predicted_classes = torch.argmax(
    test_known,
    dim=1
)

test_accuracy = (
    predicted_classes == test_labels
).float().mean().item()


near_rejection = (
    proser_near_score > 0
).float().mean().item()

far_rejection = (
    proser_far_score > 0
).float().mean().item()


def auroc(known_scores, unknown_scores):

    y_true = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])

    y_score = np.concatenate([
        known_scores,
        unknown_scores
    ])

    return roc_auc_score(
        y_true,
        y_score
    )


near_auroc = auroc(
    proser_known_score.numpy(),
    proser_near_score.numpy()
)

far_auroc = auroc(
    proser_known_score.numpy(),
    proser_far_score.numpy()
)

all_unknown_scores = torch.cat([
    proser_near_score,
    proser_far_score
])

all_auroc = auroc(
    proser_known_score.numpy(),
    all_unknown_scores.numpy()
)


results = {
    "test_accuracy": test_accuracy,
    "bias": bias,
    "near_rejection": near_rejection,
    "far_rejection": far_rejection,
    "near_auroc": near_auroc,
    "far_auroc": far_auroc,
    "all_auroc": all_auroc
}


torch.save(
    results,
    RESULTS_PATH
)


print()
print("PROSER Results")
print("-------------------------")
print(f"Test Accuracy:    {test_accuracy:.4f}")
print(f"Bias:             {bias:.4f}")
print(f"Near Rejection:   {near_rejection:.4f}")
print(f"Far Rejection:    {far_rejection:.4f}")
print(f"Near AUROC:       {near_auroc:.4f}")
print(f"Far AUROC:        {far_auroc:.4f}")
print(f"All AUROC:        {all_auroc:.4f}")

print()
print("Saved PROSER files to:")
print(RESULTS_DIR)