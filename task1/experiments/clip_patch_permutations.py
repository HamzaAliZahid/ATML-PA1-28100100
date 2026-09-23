import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import STL10

import open_clip


SEED = 6304
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_SIZE = 224
GRID_SIZE = 4
PATCH_SIZE = IMAGE_SIZE // GRID_SIZE

CLASSIFIER_PATH = "task1/models/openclip_classifier.pth"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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


class STL10SelectedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        image = self.transform(image)

        return image, label


def patch_shuffle(image, permutation):

    patches = image.unfold(
        1,
        PATCH_SIZE,
        PATCH_SIZE
    )

    patches = patches.unfold(
        2,
        PATCH_SIZE,
        PATCH_SIZE
    )

    patches = patches.permute(
        1, 2, 0, 3, 4
    )

    patches = patches.reshape(
        GRID_SIZE * GRID_SIZE,
        3,
        PATCH_SIZE,
        PATCH_SIZE
    )

    shuffled = patches[permutation]

    shuffled = shuffled.reshape(
        GRID_SIZE,
        GRID_SIZE,
        3,
        PATCH_SIZE,
        PATCH_SIZE
    )

    shuffled = shuffled.permute(
        2, 0, 3, 1, 4
    )

    shuffled = shuffled.reshape(
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    return shuffled


def create_permutations(num_images):

    rng = np.random.default_rng(SEED)

    permutations = []

    for _ in range(num_images):

        while True:

            permutation = rng.permutation(
                GRID_SIZE * GRID_SIZE
            )

            if not np.array_equal(
                permutation,
                np.arange(GRID_SIZE * GRID_SIZE)
            ):
                break

        permutations.append(permutation)

    return permutations

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

clip_model = clip_model.to(DEVICE)
clip_model.eval()

with torch.no_grad():

    dummy_image = torch.zeros(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    feature_dim = clip_model.encode_image(
        dummy_image
    ).shape[1]

classifier = nn.Linear(
    feature_dim,
    NUM_CLASSES
)

classifier = classifier.to(DEVICE)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE
    )
)

classifier.eval()

clip_mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=DEVICE
).view(1, 3, 1, 1)

clip_std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=DEVICE
).view(1, 3, 1, 1)

stl10_test = STL10(
    root="common/datasets/STL10",
    split="test",
    download=True
)

selected_indices = np.load(
    "task1/imports/selected_test_indices.npy"
)

dataset = STL10SelectedDataset(
    stl10_test,
    selected_indices
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

permutations = create_permutations(
    len(dataset)
)


clean_predictions = []
shuffled_predictions = []

correct_clean = 0
correct_shuffled = 0

total = 0
same_predictions = 0

with torch.no_grad():

    for batch_idx, (images, labels_batch) in enumerate(loader):

        images = images.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        batch_start = batch_idx * BATCH_SIZE
        batch_end = batch_start + len(images)

        batch_permutations = permutations[
            batch_start:batch_end
        ]

        shuffled_images = torch.stack([
            patch_shuffle(
                images[i],
                batch_permutations[i]
            )
            for i in range(len(images))
        ])

       
        images_normalized = (
            images - clip_mean
        ) / clip_std

        shuffled_normalized = (
            shuffled_images - clip_mean
        ) / clip_std

        
        clean_features = clip_model.encode_image(
            images_normalized
        )

        shuffled_features = clip_model.encode_image(
            shuffled_normalized
        )


        clean_logits = classifier(
            clean_features
        )

        shuffled_logits = classifier(
            shuffled_features
        )

        clean_preds = torch.argmax(
            clean_logits,
            dim=1
        )

        shuffled_preds = torch.argmax(
            shuffled_logits,
            dim=1
        )

       
        clean_predictions.extend(
            clean_preds.cpu().numpy()
        )

        shuffled_predictions.extend(
            shuffled_preds.cpu().numpy()
        )

       
        correct_clean += (
            clean_preds == labels_batch
        ).sum().item()

        correct_shuffled += (
            shuffled_preds == labels_batch
        ).sum().item()

     
        same_predictions += (
            clean_preds == shuffled_preds
        ).sum().item()

        total += len(labels_batch)



clean_accuracy = correct_clean / total

shuffled_accuracy = (
    correct_shuffled / total
)

consistency = (
    same_predictions / total
)

accuracy_drop = (
    clean_accuracy - shuffled_accuracy
)


results = pd.DataFrame({
    "condition": [
        "clean",
        "patch_shuffled"
    ],

    "accuracy_percent": [
        clean_accuracy * 100,
        shuffled_accuracy * 100
    ]
})

print(
    results.to_string(index=False)
)

print(
    f"Accuracy drop: "
    f"{accuracy_drop * 100:.2f} percentage points"
)

print(
    f"Prediction consistency: "
    f"{consistency * 100:.2f}%"
)