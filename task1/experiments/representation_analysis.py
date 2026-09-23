import random
import numpy as np
import pandas as pd
import csv
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader
import open_clip


SEED = 6304
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_CLASSES = 10

IMAGE_SIZE = 224
GRID_SIZE = 4
PATCH_SIZE = IMAGE_SIZE // GRID_SIZE

SELECTED_INDICES_PATH = "task1/imports/selected_test_indices.npy"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

    _, height, width = image.shape

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


stl10_test = STL10(
    root="common/datasets/STL10",
    split="test",
    download=True
)

selected_indices = np.load(
    SELECTED_INDICES_PATH
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


imagenet_mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(1, 3, 1, 1)

imagenet_std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(1, 3, 1, 1)

clip_mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=DEVICE
).view(1, 3, 1, 1)

clip_std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=DEVICE
).view(1, 3, 1, 1)


def calculate_patch_shuffle_stability(
    model,
    normalize_mean,
    normalize_std
):

    cosine_sum = 0.0
    total = 0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)

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
                images - normalize_mean
            ) / normalize_std

            shuffled_normalized = (
                shuffled_images - normalize_mean
            ) / normalize_std

            clean_features = model(
                images_normalized
            )

            shuffled_features = model(
                shuffled_normalized
            )

            cosine = F.cosine_similarity(
                clean_features,
                shuffled_features,
                dim=1
            )

            cosine_sum += cosine.sum().item()

            total += len(images)

    return cosine_sum / total


print()
print("=" * 60)
print("PATCH SHUFFLE NUMERICAL ANALYSIS")
print("=" * 60)


resnet = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)
resnet.eval()


resnet_patch_cosine = calculate_patch_shuffle_stability(
    resnet,
    imagenet_mean,
    imagenet_std
)

print(
    f"ResNet-50: {resnet_patch_cosine:.4f}"
)


vit = models.vit_b_16(
    weights=models.ViT_B_16_Weights.IMAGENET1K_V1
)

vit.heads = nn.Identity()

vit = vit.to(DEVICE)
vit.eval()


vit_patch_cosine = calculate_patch_shuffle_stability(
    vit,
    imagenet_mean,
    imagenet_std
)

print(
    f"ViT-B/16: {vit_patch_cosine:.4f}"
)


clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

clip_model = clip_model.to(DEVICE)
clip_model.eval()


def clip_patch_stability():

    cosine_sum = 0.0
    total = 0

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)

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

            cosine = F.cosine_similarity(
                clean_features,
                shuffled_features,
                dim=1
            )

            cosine_sum += cosine.sum().item()

            total += len(images)

    return cosine_sum / total


clip_patch_cosine = clip_patch_stability()

print(
    f"OpenCLIP ViT-B/32: {clip_patch_cosine:.4f}"
)


patch_shuffle_results = pd.DataFrame({
    "model": [
        "ResNet-50",
        "ViT-B/16",
        "OpenCLIP ViT-B/32"
    ],
    "cosine_stability": [
        resnet_patch_cosine,
        vit_patch_cosine,
        clip_patch_cosine
    ]
})


patch_shuffle_results.to_csv(
    "patch_shuffle_cosine_stability.csv",
    index=False
)


print()
print(patch_shuffle_results.to_string(index=False))


print()
print("=" * 60)
print("GRAYSCALE NUMERICAL ANALYSIS")
print("=" * 60)


grayscale_transform = transforms.Grayscale(
    num_output_channels=3
)


def calculate_grayscale_stability(
    model,
    normalize_mean,
    normalize_std
):

    cosine_sum = 0.0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            grayscale_images = grayscale_transform(
                images
            )

            images_normalized = (
                images - normalize_mean
            ) / normalize_std

            grayscale_normalized = (
                grayscale_images - normalize_mean
            ) / normalize_std

            clean_features = model(
                images_normalized
            )

            grayscale_features = model(
                grayscale_normalized
            )

            cosine = F.cosine_similarity(
                clean_features,
                grayscale_features,
                dim=1
            )

            cosine_sum += cosine.sum().item()

            total += len(images)

    return cosine_sum / total


resnet_gray_cosine = calculate_grayscale_stability(
    resnet,
    imagenet_mean,
    imagenet_std
)

vit_gray_cosine = calculate_grayscale_stability(
    vit,
    imagenet_mean,
    imagenet_std
)


def clip_grayscale_stability():

    cosine_sum = 0.0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            grayscale_images = grayscale_transform(
                images
            )

            images_normalized = (
                images - clip_mean
            ) / clip_std

            grayscale_normalized = (
                grayscale_images - clip_mean
            ) / clip_std

            clean_features = clip_model.encode_image(
                images_normalized
            )

            grayscale_features = clip_model.encode_image(
                grayscale_normalized
            )

            cosine = F.cosine_similarity(
                clean_features,
                grayscale_features,
                dim=1
            )

            cosine_sum += cosine.sum().item()

            total += len(images)

    return cosine_sum / total


clip_gray_cosine = clip_grayscale_stability()


grayscale_results = pd.DataFrame({
    "model": [
        "ResNet-50",
        "ViT-B/16",
        "OpenCLIP ViT-B/32"
    ],
    "cosine_stability": [
        resnet_gray_cosine,
        vit_gray_cosine,
        clip_gray_cosine
    ]
})


grayscale_results.to_csv(
    "grayscale_cosine_stability.csv",
    index=False
)


print(
    grayscale_results.to_string(index=False)
)


MAX_SHIFT = 32

SHIFTS = [
    0,
    8,
    16,
    32
]

DIRECTIONS = [
    "up",
    "down",
    "left",
    "right"
]


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


def calculate_translation_stability(
    model,
    normalize_mean,
    normalize_std
):

    clean_features_list = []

    with torch.no_grad():

        for start in range(
            0,
            len(dataset),
            BATCH_SIZE
        ):

            batch = torch.stack([
                dataset[i][0]
                for i in range(
                    start,
                    min(
                        start + BATCH_SIZE,
                        len(dataset)
                    )
                )
            ])

            batch = batch.to(DEVICE)

            batch = (
                batch - normalize_mean
            ) / normalize_std

            features = model(batch)

            clean_features_list.append(
                features.cpu()
            )

    clean_features = torch.cat(
        clean_features_list
    )

    results = []

    for shift in SHIFTS:

        if shift == 0:

            cosine_stability = 1.0

        else:

            direction_results = []

            for direction in DIRECTIONS:

                translated_features_list = []

                with torch.no_grad():

                    for start in range(
                        0,
                        len(dataset),
                        BATCH_SIZE
                    ):

                        batch = torch.stack([
                            dataset[i][0]
                            for i in range(
                                start,
                                min(
                                    start + BATCH_SIZE,
                                    len(dataset)
                                )
                            )
                        ])

                        translated = translate_reflection(
                            batch,
                            shift,
                            direction
                        )

                        translated = translated.to(
                            DEVICE
                        )

                        translated = (
                            translated - normalize_mean
                        ) / normalize_std

                        features = model(
                            translated
                        )

                        translated_features_list.append(
                            features.cpu()
                        )

                translated_features = torch.cat(
                    translated_features_list
                )

                cosine = F.cosine_similarity(
                    clean_features,
                    translated_features,
                    dim=1
                )

                direction_results.append(
                    cosine.mean().item()
                )

            cosine_stability = np.mean(
                direction_results
            )

        results.append({
            "shift": shift,
            "cosine_stability": cosine_stability
        })

    return pd.DataFrame(results)


print()
print("=" * 60)
print("TRANSLATION NUMERICAL ANALYSIS")
print("=" * 60)


vit_translation_results = calculate_translation_stability(
    vit,
    imagenet_mean,
    imagenet_std
)

vit_translation_results.to_csv(
    "vitb16_translation_cosine_stability.csv",
    index=False
)

print()
print("ViT-B/16")
print(
    vit_translation_results.to_string(
        index=False
    )
)


resnet_translation_results = calculate_translation_stability(
    resnet,
    imagenet_mean,
    imagenet_std
)

resnet_translation_results.to_csv(
    "resnet50_translation_cosine_stability.csv",
    index=False
)

print()
print("ResNet-50")
print(
    resnet_translation_results.to_string(
        index=False
    )
)


def calculate_clip_translation_stability():

    clean_features_list = []

    with torch.no_grad():

        for start in range(
            0,
            len(dataset),
            BATCH_SIZE
        ):

            batch = torch.stack([
                dataset[i][0]
                for i in range(
                    start,
                    min(
                        start + BATCH_SIZE,
                        len(dataset)
                    )
                )
            ])

            batch = batch.to(DEVICE)

            batch = (
                batch - clip_mean
            ) / clip_std

            features = clip_model.encode_image(
                batch
            )

            clean_features_list.append(
                features.cpu()
            )

    clean_features = torch.cat(
        clean_features_list
    )

    results = []

    for shift in SHIFTS:

        if shift == 0:

            cosine_stability = 1.0

        else:

            direction_results = []

            for direction in DIRECTIONS:

                translated_features_list = []

                with torch.no_grad():

                    for start in range(
                        0,
                        len(dataset),
                        BATCH_SIZE
                    ):

                        batch = torch.stack([
                            dataset[i][0]
                            for i in range(
                                start,
                                min(
                                    start + BATCH_SIZE,
                                    len(dataset)
                                )
                            )
                        ])

                        translated = translate_reflection(
                            batch,
                            shift,
                            direction
                        )

                        translated = translated.to(
                            DEVICE
                        )

                        translated = (
                            translated - clip_mean
                        ) / clip_std

                        features = clip_model.encode_image(
                            translated
                        )

                        translated_features_list.append(
                            features.cpu()
                        )

                translated_features = torch.cat(
                    translated_features_list
                )

                cosine = F.cosine_similarity(
                    clean_features,
                    translated_features,
                    dim=1
                )

                direction_results.append(
                    cosine.mean().item()
                )

            cosine_stability = np.mean(
                direction_results
            )

        results.append({
            "shift": shift,
            "cosine_stability": cosine_stability
        })

    return pd.DataFrame(results)


clip_translation_results = calculate_clip_translation_stability()

clip_translation_results.to_csv(
    "clip_translation_cosine_stability.csv",
    index=False
)

print()
print("OpenCLIP ViT-B/32")
print(
    clip_translation_results.to_string(
        index=False
    )
)


print()
print("=" * 60)
print("CUE-CONFLICT NUMERICAL ANALYSIS")
print("=" * 60)


ACCEPTED_DIR = (
    "task1/images/cue_conflict_images/accepted"
)

METADATA_PATH = (
    "task1/images/cue_conflict_images/"
    "accepted_metadata.csv"
)


metadata = []

with open(
    METADATA_PATH,
    "r",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        metadata.append(row)


print(
    "Accepted cue-conflict images:",
    len(metadata)
)


def load_cue_image(filename):

    image_path = os.path.join(
        ACCEPTED_DIR,
        filename
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )(image)

    image = transforms.ToTensor()(
        image
    )

    return image


def load_clean_image(content_index):

    original_index = int(
        content_index
    )

    image, _ = stl10_test[
        original_index
    ]

    image = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )(image)

    image = transforms.ToTensor()(
        image
    )

    return image


cue_images = []
clean_images = []

for item in metadata:

    cue_images.append(
        load_cue_image(
            item["filename"]
        )
    )

    clean_images.append(
        load_clean_image(
            item["content_index"]
        )
    )


cue_images = torch.stack(
    cue_images
)

clean_images = torch.stack(
    clean_images
)


def calculate_cue_stability(
    model,
    normalize_mean,
    normalize_std,
    is_clip=False
):

    clean_features = []
    cue_features = []

    with torch.no_grad():

        for start in range(
            0,
            len(clean_images),
            BATCH_SIZE
        ):

            clean_batch = clean_images[
                start:start + BATCH_SIZE
            ]

            cue_batch = cue_images[
                start:start + BATCH_SIZE
            ]

            clean_batch = clean_batch.to(
                DEVICE
            )

            cue_batch = cue_batch.to(
                DEVICE
            )

            clean_batch = (
                clean_batch - normalize_mean
            ) / normalize_std

            cue_batch = (
                cue_batch - normalize_mean
            ) / normalize_std

            if is_clip:

                clean_feature = (
                    model.encode_image(
                        clean_batch,
                        normalize=True
                    )
                )

                cue_feature = (
                    model.encode_image(
                        cue_batch,
                        normalize=True
                    )
                )

            else:

                clean_feature = model(
                    clean_batch
                )

                cue_feature = model(
                    cue_batch
                )

            clean_features.append(
                clean_feature.cpu()
            )

            cue_features.append(
                cue_feature.cpu()
            )

    clean_features = torch.cat(
        clean_features
    )

    cue_features = torch.cat(
        cue_features
    )

    cosine = F.cosine_similarity(
        clean_features,
        cue_features,
        dim=1
    )

    return cosine.mean().item()


resnet_cue_cosine = calculate_cue_stability(
    resnet,
    imagenet_mean,
    imagenet_std
)

vit_cue_cosine = calculate_cue_stability(
    vit,
    imagenet_mean,
    imagenet_std
)

clip_cue_cosine = calculate_cue_stability(
    clip_model,
    clip_mean,
    clip_std,
    is_clip=True
)


cue_conflict_results = pd.DataFrame({
    "model": [
        "ResNet-50",
        "ViT-B/16",
        "OpenCLIP ViT-B/32"
    ],
    "cosine_stability": [
        resnet_cue_cosine,
        vit_cue_cosine,
        clip_cue_cosine
    ]
})


cue_conflict_results.to_csv(
    "cue_conflict_cosine_stability.csv",
    index=False
)


print()
print(
    cue_conflict_results.to_string(
        index=False
    )
)


print()
print("=" * 60)
print("NUMERICAL ANALYSIS COMPLETE")
print("=" * 60)