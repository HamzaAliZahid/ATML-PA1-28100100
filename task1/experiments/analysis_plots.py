import random
import numpy as np
import csv
import os

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from torchvision.datasets import STL10
from torch.utils.data import Dataset

import open_clip

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


SEED = 6304

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64

NUM_CLASSES = 10

IMAGE_SIZE = 224
GRID_SIZE = 4
PATCH_SIZE = IMAGE_SIZE // GRID_SIZE

SELECTED_INDICES_PATH = (
    "task1/imports/selected_test_indices.npy"
)

ACCEPTED_DIR = (
    "task1/images/cue_conflict_images/accepted"
)

METADATA_PATH = (
    "task1/images/cue_conflict_images/"
    "accepted_metadata.csv"
)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class STL10SelectedDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices
    ):

        self.dataset = dataset
        self.indices = indices

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        image, label = self.dataset[
            self.indices[idx]
        ]

        image = self.transform(image)

        return image, label


def patch_shuffle(
    image,
    permutation
):

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
        1,
        2,
        0,
        3,
        4
    )

    patches = patches.reshape(
        GRID_SIZE * GRID_SIZE,
        3,
        PATCH_SIZE,
        PATCH_SIZE
    )

    shuffled = patches[
        permutation
    ]

    shuffled = shuffled.reshape(
        GRID_SIZE,
        GRID_SIZE,
        3,
        PATCH_SIZE,
        PATCH_SIZE
    )

    shuffled = shuffled.permute(
        2,
        0,
        3,
        1,
        4
    )

    shuffled = shuffled.reshape(
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    return shuffled


def create_permutations(
    num_images
):

    rng = np.random.default_rng(
        SEED
    )

    permutations = []

    for _ in range(num_images):

        while True:

            permutation = rng.permutation(
                GRID_SIZE * GRID_SIZE
            )

            if not np.array_equal(
                permutation,
                np.arange(
                    GRID_SIZE * GRID_SIZE
                )
            ):
                break

        permutations.append(
            permutation
        )

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


print(
    "Number of selected images:",
    len(dataset)
)


all_images = []
all_labels = []


for i in range(
    len(dataset)
):

    image, label = dataset[i]

    all_images.append(
        image
    )

    all_labels.append(
        label
    )


all_images = torch.stack(
    all_images
)

all_labels = np.array(
    all_labels
)


print(
    "Image tensor:",
    all_images.shape
)


permutations = create_permutations(
    len(all_images)
)


grayscale_transform = transforms.Grayscale(
    num_output_channels=3
)


grayscale_images = torch.stack([
    grayscale_transform(
        image
    )
    for image in all_images
])


shuffled_images = torch.stack([
    patch_shuffle(
        all_images[i],
        permutations[i]
    )
    for i in range(
        len(all_images)
    )
])


MAX_SHIFT = 32


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
            "Invalid direction"
        )

    translated = padded[
        :,
        :,
        y_start:y_start + IMAGE_SIZE,
        x_start:x_start + IMAGE_SIZE
    ]

    return translated


translation_images = {}

for direction in DIRECTIONS:

    translation_images[
        direction
    ] = translate_reflection(
        all_images,
        32,
        direction
    )


imagenet_mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(
    1,
    3,
    1,
    1
)


imagenet_std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(
    1,
    3,
    1,
    1
)


clip_mean = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073],
    device=DEVICE
).view(
    1,
    3,
    1,
    1
)


clip_std = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711],
    device=DEVICE
).view(
    1,
    3,
    1,
    1
)


print()
print("=" * 60)
print("LOADING MODELS")
print("=" * 60)


resnet = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

resnet.fc = nn.Identity()

resnet = resnet.to(
    DEVICE
)

resnet.eval()


vit = models.vit_b_16(
    weights=models.ViT_B_16_Weights.IMAGENET1K_V1
)

vit.heads = nn.Identity()

vit = vit.to(
    DEVICE
)

vit.eval()


clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

clip_model = clip_model.to(
    DEVICE
)

clip_model.eval()


for model in [
    resnet,
    vit,
    clip_model
]:

    for param in model.parameters():

        param.requires_grad = False


@torch.no_grad()
def get_resnet_features(
    image_batch
):

    image_batch = image_batch.to(
        DEVICE
    )

    image_batch = (
        image_batch - imagenet_mean
    ) / imagenet_std

    features = resnet(
        image_batch
    )

    return features.cpu()


@torch.no_grad()
def get_vit_features(
    image_batch
):

    image_batch = image_batch.to(
        DEVICE
    )

    image_batch = (
        image_batch - imagenet_mean
    ) / imagenet_std

    features = vit(
        image_batch
    )

    return features.cpu()


@torch.no_grad()
def get_clip_features(
    image_batch
):

    image_batch = image_batch.to(
        DEVICE
    )

    image_batch = (
        image_batch - clip_mean
    ) / clip_std

    features = clip_model.encode_image(
        image_batch,
        normalize=True
    )

    return features.cpu()


def extract_features(
    image_tensor,
    model_name
):

    feature_list = []

    for start in range(
        0,
        len(image_tensor),
        BATCH_SIZE
    ):

        batch = image_tensor[
            start:start + BATCH_SIZE
        ]

        if model_name == "ResNet-50":

            features = get_resnet_features(
                batch
            )

        elif model_name == "ViT-B/16":

            features = get_vit_features(
                batch
            )

        elif model_name == "CLIP ViT-B/32":

            features = get_clip_features(
                batch
            )

        else:

            raise ValueError(
                "Unknown model"
            )

        feature_list.append(
            features
        )

    return torch.cat(
        feature_list
    )


def get_average_translation_features(
    model_name
):

    direction_features = []

    for direction in DIRECTIONS:

        print(
            f"Extracting {model_name} "
            f"translation: {direction}"
        )

        features = extract_features(
            translation_images[
                direction
            ],
            model_name
        )

        direction_features.append(
            features
        )

    stacked = torch.stack(
        direction_features
    )

    return stacked.mean(
        dim=0
    )


print()
print("=" * 60)
print("LOADING CUE-CONFLICT DATA")
print("=" * 60)


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


cue_images = []
clean_cue_images = []


for item in metadata:

    image_path = os.path.join(
        ACCEPTED_DIR,
        item["filename"]
    )

    cue_image = Image.open(
        image_path
    ).convert("RGB")

    cue_image = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )(cue_image)

    cue_image = transforms.ToTensor()(
        cue_image
    )

    cue_images.append(
        cue_image
    )

    content_index = int(
        item["content_index"]
    )

    clean_image, _ = stl10_test[
        content_index
    ]

    clean_image = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )(clean_image)

    clean_image = transforms.ToTensor()(
        clean_image
    )

    clean_cue_images.append(
        clean_image
    )


cue_images = torch.stack(
    cue_images
)

clean_cue_images = torch.stack(
    clean_cue_images
)


class_to_id = {
    "airplane": 0,
    "bird": 1,
    "car": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "horse": 6,
    "monkey": 7,
    "ship": 8,
    "truck": 9
}


cue_labels = np.array([
    class_to_id[
        item["content_class"]
    ]
    for item in metadata
])


print(
    "Cue tensor:",
    cue_images.shape
)


print(
    "Cue labels:",
    cue_labels.shape
)


def extract_cue_features(
    image_tensor,
    model_name
):

    return extract_features(
        image_tensor,
        model_name
    )


def run_tsne_plot(
    clean_features,
    transformed_features,
    labels,
    model_name,
    transformation_name
):

    print()
    print(
        f"Running t-SNE: "
        f"{model_name} + "
        f"{transformation_name}"
    )

    combined_features = torch.cat(
        [
            clean_features,
            transformed_features
        ],
        dim=0
    )

    combined_labels = np.concatenate(
        [
            labels,
            labels
        ]
    )

    condition = np.array(
        ["Clean"] * len(labels)
        +
        ["Transformed"] * len(labels)
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=SEED,
        init="pca",
        learning_rate="auto"
    )

    features_2d = tsne.fit_transform(
        combined_features.numpy()
    )

    plt.figure(
        figsize=(10, 8)
    )

    class_names = [
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

    markers = {
        "Clean": "o",
        "Transformed": "x"
    }

    for class_id in range(
        NUM_CLASSES
    ):

        for condition_name in [
            "Clean",
            "Transformed"
        ]:

            mask = (
                (combined_labels == class_id)
                &
                (condition == condition_name)
            )

            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                marker=markers[
                    condition_name
                ],
                label=(
                    f"{class_names[class_id]} "
                    f"({condition_name})"
                ),
                alpha=0.65,
                s=35
            )

    plt.title(
        f"{model_name} - "
        f"{transformation_name} t-SNE"
    )

    plt.xlabel(
        "t-SNE 1"
    )

    plt.ylabel(
        "t-SNE 2"
    )

    plt.legend(
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()

    filename = (
        "tsne_"
        +
        model_name.lower()
        .replace("/", "")
        .replace("-", "")
        .replace(" ", "_")
        +
        "_"
        +
        transformation_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        +
        ".png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "Saved:",
        filename
    )


print()
print("=" * 60)
print("RESNET-50 t-SNE")
print("=" * 60)


resnet_clean = extract_features(
    all_images,
    "ResNet-50"
)

resnet_grayscale = extract_features(
    grayscale_images,
    "ResNet-50"
)

resnet_patch = extract_features(
    shuffled_images,
    "ResNet-50"
)

resnet_translation = (
    get_average_translation_features(
        "ResNet-50"
    )
)

resnet_cue = extract_cue_features(
    cue_images,
    "ResNet-50"
)

resnet_clean_cue = extract_features(
    clean_cue_images,
    "ResNet-50"
)


run_tsne_plot(
    resnet_clean,
    resnet_grayscale,
    all_labels,
    "ResNet-50",
    "Grayscale"
)


run_tsne_plot(
    resnet_clean_cue,
    resnet_cue,
    cue_labels,
    "ResNet-50",
    "Cue Conflict"
)


run_tsne_plot(
    resnet_clean,
    resnet_translation,
    all_labels,
    "ResNet-50",
    "Translation 32px"
)


run_tsne_plot(
    resnet_clean,
    resnet_patch,
    all_labels,
    "ResNet-50",
    "Patch Shuffle"
)


print()
print("=" * 60)
print("ViT-B/16 t-SNE")
print("=" * 60)


vit_clean = extract_features(
    all_images,
    "ViT-B/16"
)

vit_grayscale = extract_features(
    grayscale_images,
    "ViT-B/16"
)

vit_patch = extract_features(
    shuffled_images,
    "ViT-B/16"
)

vit_translation = (
    get_average_translation_features(
        "ViT-B/16"
    )
)

vit_cue = extract_cue_features(
    cue_images,
    "ViT-B/16"
)

vit_clean_cue = extract_features(
    clean_cue_images,
    "ViT-B/16"
)


run_tsne_plot(
    vit_clean,
    vit_grayscale,
    all_labels,
    "ViT-B/16",
    "Grayscale"
)


run_tsne_plot(
    vit_clean_cue,
    vit_cue,
    cue_labels,
    "ViT-B/16",
    "Cue Conflict"
)


run_tsne_plot(
    vit_clean,
    vit_translation,
    all_labels,
    "ViT-B/16",
    "Translation 32px"
)


run_tsne_plot(
    vit_clean,
    vit_patch,
    all_labels,
    "ViT-B/16",
    "Patch Shuffle"
)


print()
print("=" * 60)
print("CLIP ViT-B/32 t-SNE")
print("=" * 60)


clip_clean = extract_features(
    all_images,
    "CLIP ViT-B/32"
)

clip_grayscale = extract_features(
    grayscale_images,
    "CLIP ViT-B/32"
)

clip_patch = extract_features(
    shuffled_images,
    "CLIP ViT-B/32"
)

clip_translation = (
    get_average_translation_features(
        "CLIP ViT-B/32"
    )
)

clip_cue = extract_cue_features(
    cue_images,
    "CLIP ViT-B/32"
)

clip_clean_cue = extract_features(
    clean_cue_images,
    "CLIP ViT-B/32"
)


run_tsne_plot(
    clip_clean,
    clip_grayscale,
    all_labels,
    "CLIP ViT-B/32",
    "Grayscale"
)


run_tsne_plot(
    clip_clean_cue,
    clip_cue,
    cue_labels,
    "CLIP ViT-B/32",
    "Cue Conflict"
)


run_tsne_plot(
    clip_clean,
    clip_translation,
    all_labels,
    "CLIP ViT-B/32",
    "Translation 32px"
)


run_tsne_plot(
    clip_clean,
    clip_patch,
    all_labels,
    "CLIP ViT-B/32",
    "Patch Shuffle"
)


print()
print("=" * 60)
print("ALL t-SNE PLOTS COMPLETE")
print("=" * 60)