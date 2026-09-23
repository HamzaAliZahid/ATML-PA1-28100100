import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


SEED = 6304
DATA_ROOT = "common/datasets/PACS"

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"

NUM_CLASSES = 7
BATCH_SIZE = 64

LAMBDA_VALUES = [0.1, 1.0, 10.0]

CHECKPOINT_DIR = "task2/models/controlled"
RESULTS_PATH = "task2/results/dan_controlled_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PACSDataset(Dataset):
    def __init__(self, root, domain, transform=None):
        self.root = root
        self.domain = domain
        self.transform = transform
        self.samples = []

        domain_path = os.path.join(root, domain)

        classes = sorted(
            [
                d
                for d in os.listdir(domain_path)
                if os.path.isdir(os.path.join(domain_path, d))
            ]
        )

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(classes)
        }

        for class_name in classes:
            class_path = os.path.join(domain_path, class_name)

            for filename in sorted(os.listdir(class_path)):
                filepath = os.path.join(class_path, filename)

                if os.path.isfile(filepath):
                    self.samples.append(
                        (filepath, self.class_to_idx[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]

        image = Image.open(filepath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def stratified_split(dataset):
    labels = np.array([label for _, label in dataset.samples])

    rng = np.random.RandomState(SEED)

    train_indices = []
    val_indices = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]

        rng.shuffle(class_indices)

        split_point = int(0.8 * len(class_indices))

        train_indices.extend(class_indices[:split_point])
        val_indices.extend(class_indices[split_point:])

    return train_indices, val_indices


class DAN(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.classifier = nn.Linear(
            512,
            NUM_CLASSES
        )

    def forward(self, x):
        features = self.features(x)
        features = features.flatten(1)

        logits = self.classifier(features)

        return features, logits


def load_model(checkpoint_path):
    model = DAN()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    if not checkpoint_keys.issubset(model_keys):

        mapping = {
            "conv1.": "features.0.",
            "bn1.": "features.1.",
            "layer1.": "features.4.",
            "layer2.": "features.5.",
            "layer3.": "features.6.",
            "layer4.": "features.7.",
            "fc.": "classifier."
        }

        converted_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            for old_prefix, new_prefix in mapping.items():
                if key.startswith(old_prefix):
                    new_key = (
                        new_prefix
                        + key[len(old_prefix):]
                    )
                    break

            converted_state_dict[new_key] = value

        state_dict = converted_state_dict

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing:
        print("Missing keys:", missing)

    if unexpected:
        print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()

    return model


def evaluate_target(model):
    dataset = PACSDataset(
        DATA_ROOT,
        TARGET_DOMAIN,
        val_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_predictions = []
    all_labels = []

    model.eval()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            _, logits = model(images)

            predictions = (
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return accuracy, macro_f1


def extract_features(model, domain, indices=None):

    dataset = PACSDataset(
        DATA_ROOT,
        domain,
        val_transform
    )

    if indices is not None:
        samples = [
            dataset.samples[i]
            for i in indices
        ]
    else:
        samples = dataset.samples

    features = []

    model.eval()

    for start in range(0, len(samples), BATCH_SIZE):

        batch_samples = samples[
            start:start + BATCH_SIZE
        ]

        images = []

        for filepath, _ in batch_samples:

            image = Image.open(filepath).convert("RGB")
            image = val_transform(image)

            images.append(image)

        images = torch.stack(images).to(device)

        with torch.no_grad():

            batch_features = model.features(images)
            batch_features = batch_features.flatten(1)

        features.append(
            batch_features.cpu().numpy()
        )

    return np.concatenate(
        features,
        axis=0
    )


def get_source_validation_indices():

    indices = {}

    for domain in SOURCE_DOMAINS:

        dataset = PACSDataset(
            DATA_ROOT,
            domain,
            val_transform
        )

        _, val_indices = stratified_split(dataset)

        indices[domain] = val_indices

    return indices


def calculate_domain_separability(
    model,
    source_val_indices
):

    source_features = []
    source_labels = []

    domain_to_id = {
        "photo": 0,
        "art_painting": 1,
        "cartoon": 2,
        "sketch": 3
    }

    for domain in SOURCE_DOMAINS:

        features = extract_features(
            model,
            domain,
            source_val_indices[domain]
        )

        source_features.append(features)

        source_labels.extend(
            [domain_to_id[domain]] * len(features)
        )

    target_features = extract_features(
        model,
        TARGET_DOMAIN
    )

    target_labels = [
        domain_to_id[TARGET_DOMAIN]
    ] * len(target_features)

    source_features = np.concatenate(
        source_features,
        axis=0
    )

    source_labels = np.array(
        source_labels
    )

    min_samples = min(
        len(source_features),
        len(target_features)
    )

    rng = np.random.RandomState(SEED)

    source_indices = rng.choice(
        len(source_features),
        min_samples,
        replace=False
    )

    target_indices = rng.choice(
        len(target_features),
        min_samples,
        replace=False
    )

    features = np.concatenate(
        [
            source_features[source_indices],
            target_features[target_indices]
        ],
        axis=0
    )

    labels = np.concatenate(
        [
            source_labels[source_indices],
            np.array(target_labels)[target_indices]
        ],
        axis=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels
    )

    domain_classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED
    )

    domain_classifier.fit(
        X_train,
        y_train
    )

    predictions = domain_classifier.predict(
        X_test
    )

    domain_accuracy = accuracy_score(
        y_test,
        predictions
    )

    return domain_accuracy


def main():

    set_seed(SEED)

    print("Device:", device)

    source_val_indices = (
        get_source_validation_indices()
    )

    results = []

    for lambda_mmd in LAMBDA_VALUES:

        print()
        print("=" * 60)
        print(
            f"Analyzing DAN with lambda = {lambda_mmd}"
        )
        print("=" * 60)

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"dan_lambda_{lambda_mmd}.pt"
        )

        if not os.path.exists(checkpoint_path):

            print(
                "Checkpoint not found:",
                checkpoint_path
            )

            continue

        model = load_model(
            checkpoint_path
        )

        target_accuracy, target_f1 = (
            evaluate_target(model)
        )

        domain_accuracy = (
            calculate_domain_separability(
                model,
                source_val_indices
            )
        )

        print(
            f"Sketch Accuracy: "
            f"{target_accuracy:.4f}"
        )

        print(
            f"Sketch Macro-F1: "
            f"{target_f1:.4f}"
        )

        print(
            f"Domain Separability Accuracy: "
            f"{domain_accuracy:.4f}"
        )

        results.append({
            "lambda_mmd": lambda_mmd,
            "sketch_accuracy": target_accuracy,
            "sketch_macro_f1": target_f1,
            "domain_separability_accuracy": domain_accuracy
        })

    results_df = pd.DataFrame(results)

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print()
    print(
        "Saved:",
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()