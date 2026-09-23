import os
import random
import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models


SEED = 6304
NUM_CLASSES = 7
RHO = 0.05

DATA_ROOT = "common/datasets/PACS"

MODEL_PATHS = {
    "ERM": "task2/models/erm_best.pt",
    "DAN-DG": "task3/models/dan_dg.pt",
    "SAM": "task3/models/sam.pt",
}

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    ),
])


def load_domain(domain):
    path = os.path.join(
        DATA_ROOT,
        domain
    )

    return datasets.ImageFolder(
        path,
        transform=transform
    )


def build_features_classifier():
    backbone = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    features = nn.Sequential(
        *list(backbone.children())[:-1]
    )

    classifier = nn.Linear(
        backbone.fc.in_features,
        NUM_CLASSES
    )

    return features, classifier


class FeatureClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features, self.classifier = (
            build_features_classifier()
        )

    def forward(self, x):
        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return self.classifier(x)


def build_erm_model():
    backbone = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    features = nn.Sequential(
        *list(backbone.children())[:-1]
    )

    classifier = nn.Linear(
        backbone.fc.in_features,
        NUM_CLASSES
    )

    return nn.Sequential(
        features,
        nn.Flatten(),
        classifier
    )


def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def load_erm_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    model_state_dict = model.state_dict()

    if set(state_dict.keys()) == set(
        model_state_dict.keys()
    ):
        model.load_state_dict(state_dict)
        return

    converted_state_dict = {}

    mapping = {
        "conv1.": "0.0.",
        "bn1.": "0.1.",
        "layer1.": "0.4.",
        "layer2.": "0.5.",
        "layer3.": "0.6.",
        "layer4.": "0.7.",
        "fc.": "2.",
    }

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

    model.load_state_dict(
        converted_state_dict
    )


def load_feature_classifier_checkpoint(
    model,
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    state_dict = extract_state_dict(checkpoint)

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict
    )


def get_fixed_batch():

    domains = [
        "photo",
        "art_painting",
        "cartoon"
    ]

    images = []
    labels = []

    rng = np.random.default_rng(SEED)

    for domain in domains:

        dataset = load_domain(domain)

        indices = rng.choice(
            len(dataset),
            size=min(32, len(dataset)),
            replace=False
        )

        for index in indices:

            image, label = dataset[index]

            images.append(image)
            labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images.to(device), labels.to(device)


def compute_sharpness(
    model,
    images,
    labels
):

    model.eval()

    model.zero_grad()

    logits = model(images)

    loss_function = nn.CrossEntropyLoss()

    loss = loss_function(
        logits,
        labels
    )

    loss.backward()

    gradients = []

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients.append(
                parameter.grad.detach().clone()
            )

    grad_norm = torch.sqrt(
        sum(
            torch.sum(gradient ** 2)
            for gradient in gradients
        )
    )

    scale = RHO / (
        grad_norm + 1e-12
    )

    perturbations = []

    with torch.no_grad():

        for parameter in model.parameters():

            if parameter.grad is None:
                perturbations.append(None)
                continue

            perturbation = (
                parameter.grad * scale
            )

            parameter.add_(perturbation)

            perturbations.append(
                perturbation
            )

    with torch.no_grad():

        perturbed_logits = model(images)

        perturbed_loss = loss_function(
            perturbed_logits,
            labels
        )

    with torch.no_grad():

        index = 0

        for parameter in model.parameters():

            if parameter.grad is None:
                continue

            parameter.sub_(
                perturbations[index]
            )

            index += 1

    model.zero_grad()

    sharpness = (
        perturbed_loss.item()
        - loss.item()
    )

    return (
        loss.item(),
        perturbed_loss.item(),
        sharpness,
        grad_norm.item()
    )


def main():

    images, labels = get_fixed_batch()

    print(
        "Fixed batch size:",
        images.size(0)
    )

    print(
        "Sharpness radius:",
        RHO
    )

    print()

    results = {}

    for model_name, checkpoint_path in MODEL_PATHS.items():

        print("=" * 60)
        print(model_name)
        print("=" * 60)

        if not os.path.exists(checkpoint_path):

            print(
                "Checkpoint not found:",
                checkpoint_path
            )

            continue

        if model_name == "ERM":

            model = build_erm_model()

            load_erm_checkpoint(
                model,
                checkpoint_path
            )

        else:

            model = FeatureClassifier()

            load_feature_classifier_checkpoint(
                model,
                checkpoint_path
            )

        model = model.to(device)

        loss, perturbed_loss, sharpness, grad_norm = (
            compute_sharpness(
                model,
                images,
                labels
            )
        )

        results[model_name] = {
            "loss": loss,
            "perturbed_loss": perturbed_loss,
            "sharpness": sharpness,
            "gradient_norm": grad_norm,
        }

        print(
            f"Loss:              {loss:.6f}"
        )

        print(
            f"Perturbed Loss:    {perturbed_loss:.6f}"
        )

        print(
            f"Sharpness:         {sharpness:.6f}"
        )

        print(
            f"Gradient Norm:     {grad_norm:.6f}"
        )

        print()

    print("=" * 60)
    print("SHARPNESS COMPARISON")
    print("=" * 60)

    print(
        f"{'Model':<12}"
        f"{'Loss':>14}"
        f"{'Perturbed':>16}"
        f"{'Sharpness':>16}"
        f"{'Grad Norm':>16}"
    )

    for model_name, result in results.items():

        print(
            f"{model_name:<12}"
            f"{result['loss']:>14.6f}"
            f"{result['perturbed_loss']:>16.6f}"
            f"{result['sharpness']:>16.6f}"
            f"{result['gradient_norm']:>16.6f}"
        )


if __name__ == "__main__":
    main()