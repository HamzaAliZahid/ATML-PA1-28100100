import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import datasets, transforms

import open_clip


SEED = 6304
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_CLASSES = 10

IMAGE_SIZE = 224
MAX_SHIFT = 32
SHIFTS = [0, 8, 16, 32]

DIRECTIONS = ["up", "down", "left", "right"]

CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck"
]

print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("\nLoading OpenCLIP ViT-B/32...")

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(DEVICE)
model.eval()

for param in model.parameters():
    param.requires_grad = False


imagenet_mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=DEVICE
).view(1, 3, 1, 1)

imagenet_std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=DEVICE
).view(1, 3, 1, 1)


class STL10Raw(Dataset):

    def __init__(self, root):

        self.dataset = datasets.STL10(
            root=root,
            split="test",
            download=True
        )

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        image = self.transform(image)

        return image, label, idx


DATA_ROOT = "common/datasets/STL10"

dataset = STL10Raw(DATA_ROOT)

print(
    "Total STL-10 test images:",
    len(dataset)
)


def select_balanced_subset(
    dataset,
    n_per_class=50
):

    rng = random.Random(SEED)

    class_indices = [
        []
        for _ in range(NUM_CLASSES)
    ]

    for idx in range(len(dataset)):

        _, label, _ = dataset[idx]

        class_indices[label].append(idx)

    selected = []

    for c in range(NUM_CLASSES):

        indices = class_indices[c].copy()

        rng.shuffle(indices)

        selected.extend(
            indices[:n_per_class]
        )

    rng.shuffle(selected)

    return selected


selected_indices = select_balanced_subset(
    dataset,
    n_per_class=50
)

print(
    "Selected images:",
    len(selected_indices)
)


images = []
labels = []
image_ids = []

for idx in selected_indices:

    image, label, image_id = dataset[idx]

    images.append(image)
    labels.append(label)
    image_ids.append(image_id)


images = torch.stack(images)

labels = torch.tensor(labels)


print(
    "Images tensor:",
    images.shape
)

print(
    "Labels tensor:",
    labels.shape
)


text_prompts = [
    f"a photo of a {class_name}"
    for class_name in CLASS_NAMES
]

text_tokens = tokenizer(
    text_prompts
).to(DEVICE)


with torch.no_grad():

    text_features = model.encode_text(
        text_tokens
    )

    text_features = F.normalize(
        text_features,
        dim=-1
    )


def translate_reflection(
    image_batch,
    shift,
    direction
):

    if shift == 0:
        return image_batch

    pad = MAX_SHIFT

    padded = F.pad(
        image_batch,
        (
            pad,
            pad,
            pad,
            pad
        ),
        mode="reflect"
    )

    if direction == "right":

        x_start = pad - shift
        y_start = pad

    elif direction == "left":

        x_start = pad + shift
        y_start = pad

    elif direction == "down":

        x_start = pad
        y_start = pad - shift

    elif direction == "up":

        x_start = pad
        y_start = pad + shift

    else:

        raise ValueError(
            "direction must be up, down, left, or right"
        )

    translated = padded[
        :,
        :,
        y_start:y_start + IMAGE_SIZE,
        x_start:x_start + IMAGE_SIZE
    ]

    return translated


@torch.no_grad()
def predict(image_batch):

    image_batch = image_batch.to(DEVICE)

    image_batch = (
        image_batch - imagenet_mean
    ) / imagenet_std

    image_features = model.encode_image(
        image_batch
    )

    image_features = F.normalize(
        image_features,
        dim=-1
    )

    logits = (
        image_features @ text_features.T
    )

    predictions = torch.argmax(
        logits,
        dim=1
    )

    return predictions


print(
    "\nCalculating clean predictions..."
)

clean_predictions = []

for start in range(
    0,
    len(images),
    BATCH_SIZE
):

    batch = images[
        start:start + BATCH_SIZE
    ]

    preds = predict(batch)

    clean_predictions.append(
        preds.cpu()
    )


clean_predictions = torch.cat(
    clean_predictions
)


clean_accuracy = (
    (clean_predictions == labels)
    .float()
    .mean()
    .item()
)


print(
    f"Clean accuracy: "
    f"{clean_accuracy * 100:.2f}%"
)


results = []

for shift in SHIFTS:

    print(
        "\n" + "=" * 60
    )

    print(
        f"SHIFT = {shift} pixels"
    )

    print(
        "=" * 60
    )

    if shift == 0:

        accuracy = clean_accuracy
        consistency = 1.0

        avg_accuracy = accuracy
        avg_consistency = consistency

    else:

        direction_results = []

        for direction in DIRECTIONS:

            print(
                f"\nTesting: "
                f"shift={shift}, "
                f"direction={direction}"
            )

            all_predictions = []

            for start in range(
                0,
                len(images),
                BATCH_SIZE
            ):

                batch = images[
                    start:start + BATCH_SIZE
                ]

                translated = translate_reflection(
                    batch,
                    shift,
                    direction
                )

                predictions = predict(
                    translated
                )

                all_predictions.append(
                    predictions.cpu()
                )


            predictions = torch.cat(
                all_predictions
            )


            accuracy = (
                predictions == labels
            ).float().mean().item()


            consistency = (
                predictions == clean_predictions
            ).float().mean().item()


            print(
                f"Accuracy: "
                f"{accuracy * 100:.2f}%"
            )

            print(
                f"Prediction consistency: "
                f"{consistency * 100:.2f}%"
            )


            direction_results.append({
                "direction": direction,
                "accuracy": accuracy,
                "consistency": consistency
            })


        avg_accuracy = np.mean([
            r["accuracy"]
            for r in direction_results
        ])

        avg_consistency = np.mean([
            r["consistency"]
            for r in direction_results
        ])


        print(
            f"\nAVERAGE @ {shift}px"
        )

        print(
            f"Accuracy: "
            f"{avg_accuracy * 100:.2f}%"
        )

        print(
            f"Consistency: "
            f"{avg_consistency * 100:.2f}%"
        )


    results.append({
        "shift": shift,
        "accuracy": avg_accuracy,
        "consistency": avg_consistency
    })


results_df = pd.DataFrame(
    results
)

results_df["accuracy_percent"] = (
    results_df["accuracy"] * 100
)

results_df["consistency_percent"] = (
    results_df["consistency"] * 100
)


print(
    "\n\n" + "=" * 70
)

print(
    "FINAL TRANSLATION RESULTS"
)

print(
    "=" * 70
)


print(
    results_df[
        [
            "shift",
            "accuracy_percent",
            "consistency_percent"
        ]
    ].to_string(index=False)
)


results_df["accuracy_drop"] = (
    clean_accuracy -
    results_df["accuracy"]
)

results_df["accuracy_drop_percent"] = (
    results_df["accuracy_drop"] * 100
)


print(
    "\n"
)

print(
    "=" * 70
)

print(
    "ACCURACY DROP FROM CLEAN"
)

print(
    "=" * 70
)


print(
    results_df[
        [
            "shift",
            "accuracy_drop_percent"
        ]
    ].to_string(index=False)
)


OUTPUT_FILE = "task1/results/clip_translation_results.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nResults saved to: {OUTPUT_FILE}"
)

print(
    "\nDone."
)