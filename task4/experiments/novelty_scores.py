import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 6304

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

CHECKPOINT_PATH = "task4/models/vanilla_best.pth"
OUTPUT_DIR = "task4/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset = datasets.CIFAR10(
    root="common/datasets/CIFAR10",
    train=True,
    download=True,
    transform=eval_transform,
)

test_dataset = datasets.CIFAR10(
    root="common/datasets/CIFAR10",
    train=False,
    download=True,
    transform=eval_transform,
)

labels = np.array(full_train_dataset.targets)
indices = np.arange(len(full_train_dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    stratify=labels,
    random_state=SEED,
)

train_dataset = Subset(
    full_train_dataset,
    train_indices,
)

val_dataset = Subset(
    full_train_dataset,
    val_indices,
)

cifar100_dataset = datasets.CIFAR100(
    root="common/datasets/CIFAR100",
    train=False,
    download=True,
    transform=eval_transform,
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
    "camel",
]

far_classes = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
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
    near_indices,
)

far_dataset = Subset(
    cifar100_dataset,
    far_indices,
)

model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
)

model.maxpool = nn.Identity()

model.fc = nn.Linear(
    model.fc.in_features,
    10,
)

def forward_features(model, x):

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)

    features = torch.flatten(x, 1)

    return features

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only = False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)


model = model.to(device)
model.eval()

print("Vanilla checkpoint loaded.")

def extract_outputs(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False
    )

    all_features = []
    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            features = forward_features(
                model,
                images,
            )

            logits = model.fc(features)

            all_features.append(
                features.cpu()
            )

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels
            )

    return (
        torch.cat(all_features),
        torch.cat(all_logits),
        torch.cat(all_labels),
    )

datasets_to_extract = {
    "cifar10_train": train_dataset,
    "cifar10_val": val_dataset,
    "cifar10_test": test_dataset,
    "cifar100_near": near_dataset,
    "cifar100_far": far_dataset,
}


for name, dataset in datasets_to_extract.items():

    print(
        f"\nExtracting {name} "
        f"({len(dataset)} images)..."
    )

    features, logits, labels = extract_outputs(
        model,
        dataset,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.pt",
    )

    torch.save(
        {
            "features": features,
            "logits": logits,
            "labels": labels,
        },
        output_path,
    )

    print(f"Saved: {output_path}")
    print("Features:", features.shape)
    print("Logits:", logits.shape)
    print("Labels:", labels.shape)


print("\nFeature/logit extraction complete.")

train_data = torch.load(
    os.path.join(OUTPUT_DIR, "cifar10_train.pt"),
    weights_only=False
)

val_data = torch.load(
    os.path.join(OUTPUT_DIR, "cifar10_val.pt"),
    weights_only=False
)

test_data = torch.load(
    os.path.join(OUTPUT_DIR, "cifar10_test.pt"),
    weights_only=False
)

near_data = torch.load(
    os.path.join(OUTPUT_DIR, "cifar100_near.pt"),
    weights_only=False
)

far_data = torch.load(
    os.path.join(OUTPUT_DIR, "cifar100_far.pt"),
    weights_only=False
)


print("\nCalculating novelty scores...")

def msp_score(logits):
    probabilities = torch.softmax(logits, dim=1)

    max_probability = probabilities.max(dim=1).values
    return 1.0 - max_probability

def mls_score(logits):
    max_logit = logits.max(dim=1).values

    return -max_logit

def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)

def compute_mahalanobis_statistics(features, labels, num_classes=10):

    feature_dim = features.shape[1]

    class_means = torch.zeros(num_classes, feature_dim)

    for c in range(num_classes):

        class_features = features[labels == c]

        class_means[c] = class_features.mean(dim=0)

    centered_features = features - class_means[labels]

    variance = centered_features.pow(2).mean(dim=0)

    variance = variance + 1e-6

    return class_means, variance

def mahalanobis_score(features, class_means, variance):

    all_distances = []

    for c in range(class_means.shape[0]):

        difference = features - class_means[c]

        distance = (
            difference.pow(2) / variance
        ).sum(dim=1)

        all_distances.append(distance)

    all_distances = torch.stack(
        all_distances,
        dim=1
    )

    min_distance = all_distances.min(dim=1).values

    return min_distance

print("Computing Mahalanobis statistics...")

class_means, variance = compute_mahalanobis_statistics(
    train_data["features"],
    train_data["labels"]
)

print("Class means:", class_means.shape)
print("Variance:", variance.shape)

datasets_to_score = {
    "cifar10_train": train_data,
    "cifar10_val": val_data,
    "cifar10_test": test_data,
    "cifar100_near": near_data,
    "cifar100_far": far_data,
}

scores = {}

for name, data in datasets_to_score.items():

    print(f"Calculating scores for {name}...")

    logits = data["logits"]
    features = data["features"]

    scores[name] = {

        "msp": msp_score(logits),

        "mls": mls_score(logits),

        "energy": energy_score(logits),

        "mahalanobis": mahalanobis_score(
            features,
            class_means,
            variance
        ),

        "labels": data["labels"]
    }

score_path = os.path.join(
    OUTPUT_DIR,
    "novelty_scores.pt"
)

torch.save(scores, score_path)

print("\nNovelty scores saved to:", score_path)

for name, data in scores.items():

    print(f"\n{name}")

    print("  MSP:          ", data["msp"].shape)
    print("  MLS:          ", data["mls"].shape)
    print("  Energy:       ", data["energy"].shape)
    print("  Mahalanobis:  ", data["mahalanobis"].shape)

print("\nEvaluating novelty scores...")


def evaluate_score(score_name):
    known_val = scores["cifar10_val"][score_name]
    known_test = scores["cifar10_test"][score_name]

    near = scores["cifar100_near"][score_name]
    far = scores["cifar100_far"][score_name]

    threshold = torch.quantile(
        known_val,
        0.95
    ).item()

    known_near_scores = torch.cat([
        known_test,
        near
    ])

    known_near_labels = torch.cat([
        torch.zeros(len(known_test)),
        torch.ones(len(near))
    ])

    near_auroc = roc_auc_score(
        known_near_labels.numpy(),
        known_near_scores.numpy()
    )

    known_far_scores = torch.cat([
        known_test,
        far
    ])

    known_far_labels = torch.cat([
        torch.zeros(len(known_test)),
        torch.ones(len(far))
    ])

    far_auroc = roc_auc_score(
        known_far_labels.numpy(),
        known_far_scores.numpy()
    )

    all_unknown = torch.cat([
        near,
        far
    ])

    known_all_scores = torch.cat([
        known_test,
        all_unknown
    ])

    known_all_labels = torch.cat([
        torch.zeros(len(known_test)),
        torch.ones(len(all_unknown))
    ])

    all_auroc = roc_auc_score(
        known_all_labels.numpy(),
        known_all_scores.numpy()
    )

    known_acceptance = (
        (known_test <= threshold).float().mean().item()
    )

    near_rejection = (
        (near > threshold).float().mean().item()
    )

    far_rejection = (
        (far > threshold).float().mean().item()
    )

    return {
        "threshold": threshold,
        "near_auroc": near_auroc,
        "far_auroc": far_auroc,
        "all_auroc": all_auroc,
        "known_acceptance": known_acceptance,
        "near_rejection": near_rejection,
        "far_rejection": far_rejection,
    }

score_names = [
    "msp",
    "mls",
    "energy",
    "mahalanobis"
]

results = {}

for score_name in score_names:

    print(f"\nEvaluating {score_name}...")

    results[score_name] = evaluate_score(score_name)

    print(
        f"Threshold: "
        f"{results[score_name]['threshold']:.4f}"
    )

    print(
        f"Near AUROC: "
        f"{results[score_name]['near_auroc']:.4f}"
    )

    print(
        f"Far AUROC: "
        f"{results[score_name]['far_auroc']:.4f}"
    )

    print(
        f"All AUROC: "
        f"{results[score_name]['all_auroc']:.4f}"
    )

    print(
        f"Known acceptance: "
        f"{results[score_name]['known_acceptance']:.4f}"
    )

    print(
        f"Near rejection: "
        f"{results[score_name]['near_rejection']:.4f}"
    )

    print(
        f"Far rejection: "
        f"{results[score_name]['far_rejection']:.4f}"
    )