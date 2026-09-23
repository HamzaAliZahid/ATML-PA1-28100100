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

CHECKPOINT_PATH = "task4/checkpoints/proser_best.pth"

DATA_DIR = "data"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

class PROSER(nn.Module):

    def __init__(
        self,
        num_classes=10,
        num_dummy=5
    ):
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
            num_classes
        )

        self.dummy_classifier = nn.Linear(
            feature_dim,
            num_dummy
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

        x = self.layer2_to_features(x)

        known_logits = self.classifier(x)

        dummy_logits = self.dummy_classifier(x)

        return known_logits, dummy_logits

eval_transform = transforms.Compose([
    transforms.ToTensor()
])


full_train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=eval_transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform
)

labels = np.array(full_train_dataset.targets)

indices = np.arange(
    len(full_train_dataset)
)

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    stratify=labels,
    random_state=SEED
)


val_dataset = Subset(
    full_train_dataset,
    val_indices
)

cifar100_dataset = datasets.CIFAR100(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform
)


cifar100_classes = cifar100_dataset.classes


near_classes = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]


far_classes = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe"
]


near_class_ids = [
    cifar100_classes.index(name)
    for name in near_classes
]

far_class_ids = [
    cifar100_classes.index(name)
    for name in far_classes
]


near_indices = [
    i
    for i, label in enumerate(cifar100_dataset.targets)
    if label in near_class_ids
]

far_indices = [
    i
    for i, label in enumerate(cifar100_dataset.targets)
    if label in far_class_ids
]


near_dataset = Subset(
    cifar100_dataset,
    near_indices
)

far_dataset = Subset(
    cifar100_dataset,
    far_indices
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

near_loader = DataLoader(
    near_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

far_loader = DataLoader(
    far_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = PROSER(
    num_classes=NUM_CLASSES,
    num_dummy=NUM_DUMMY
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

def extract_outputs(model, loader):

    all_known_logits = []
    all_dummy_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            known_logits, dummy_logits = model(
                images
            )

            all_known_logits.append(
                known_logits.cpu()
            )

            all_dummy_logits.append(
                dummy_logits.cpu()
            )

            all_labels.append(
                labels
            )


    return (
        torch.cat(all_known_logits),
        torch.cat(all_dummy_logits),
        torch.cat(all_labels)
    )


print("\nExtracting validation outputs...")

val_known, val_dummy, val_labels = extract_outputs(
    model,
    val_loader
)


print("Extracting test outputs...")

test_known, test_dummy, test_labels = extract_outputs(
    model,
    test_loader
)


print("Extracting near-unknown outputs...")

near_known, near_dummy, near_labels = extract_outputs(
    model,
    near_loader
)


print("Extracting far-unknown outputs...")

far_known, far_dummy, far_labels = extract_outputs(
    model,
    far_loader
)

def proser_raw_score(
    known_logits,
    dummy_logits
):
    max_known = known_logits.max(
        dim=1
    ).values

    max_dummy = dummy_logits.max(
        dim=1
    ).values

    return max_dummy - max_known

val_raw = proser_raw_score(
    val_known,
    val_dummy
)

test_raw = proser_raw_score(
    test_known,
    test_dummy
)

near_raw = proser_raw_score(
    near_known,
    near_dummy
)

far_raw = proser_raw_score(
    far_known,
    far_dummy
)

bias = torch.quantile(
    val_raw,
    0.95
).item()

print("PROSER calibration bias:", bias)

val_score = val_raw - bias

test_score = test_raw - bias

near_score = near_raw - bias

far_score = far_raw - bias

test_predictions = test_known.argmax(
    dim=1
)

test_accuracy = (
    test_predictions == test_labels
).float().mean().item()

print(
    f"CIFAR-10 test accuracy: "
    f"{100 * test_accuracy:.2f}%"
)

known_accepted = (
    test_score <= 0
).float().mean().item()


known_rejected = (
    test_score > 0
).float().mean().item()


print(
    f"Known accepted: "
    f"{100 * known_accepted:.2f}%"
)

print(
    f"Known rejected: "
    f"{100 * known_rejected:.2f}%"
)

near_rejected = (
    near_score > 0
).float().mean().item()


far_rejected = (
    far_score > 0
).float().mean().item()


print(
    f"Near unknown rejected: "
    f"{100 * near_rejected:.2f}%"
)

print(
    f"Far unknown rejected: "
    f"{100 * far_rejected:.2f}%"
)

known_scores = test_score.numpy()

near_scores = near_score.numpy()

far_scores = far_score.numpy()


known_labels = np.zeros(
    len(known_scores)
)

near_labels_binary = np.ones(
    len(near_scores)
)

far_labels_binary = np.ones(
    len(far_scores)
)

near_y_true = np.concatenate([
    known_labels,
    near_labels_binary
])

near_y_score = np.concatenate([
    known_scores,
    near_scores
])

near_auroc = roc_auc_score(
    near_y_true,
    near_y_score
)

far_y_true = np.concatenate([
    known_labels,
    far_labels_binary
])

far_y_score = np.concatenate([
    known_scores,
    far_scores
])


far_auroc = roc_auc_score(
    far_y_true,
    far_y_score
)

all_unknown_scores = np.concatenate([
    near_scores,
    far_scores
])

all_unknown_labels = np.ones(
    len(all_unknown_scores)
)


all_y_true = np.concatenate([
    known_labels,
    all_unknown_labels
])

all_y_score = np.concatenate([
    known_scores,
    all_unknown_scores
])


all_auroc = roc_auc_score(
    all_y_true,
    all_y_score
)

print("=" * 50)
print("PROSER RESULTS")
print("=" * 50)

print(
    f"Known vs Near AUROC: "
    f"{near_auroc:.4f}"
)

print(
    f"Known vs Far AUROC: "
    f"{far_auroc:.4f}"
)

print(
    f"Known vs All AUROC: "
    f"{all_auroc:.4f}"
)

print("=" * 50)

results = {
    "bias": bias,

    "test_accuracy": test_accuracy,

    "known_acceptance": known_accepted,

    "known_rejection": known_rejected,

    "near_rejection": near_rejected,

    "far_rejection": far_rejected,

    "near_auroc": near_auroc,

    "far_auroc": far_auroc,

    "all_auroc": all_auroc
}


os.makedirs(
    "task4/results",
    exist_ok=True
)

torch.save(
    results,
    "task4/results/proser_results.pt"
)


print(
    "Saved results to "
    "task4/results/proser_results.pt"
)