import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms

from imports import net, function

SEED = 6304

IMAGE_SIZE = 256

CLASS_PAIRS = [
    ("cat", "dog"),
    ("horse", "deer"),
    ("car", "truck"),
    ("airplane", "bird"),
    ("ship", "monkey")
]

IMAGES_PER_DIRECTION = 30

OUTPUT_DIR = "task1/images/cue_conflict_images"
GENERATED_DIR = os.path.join(OUTPUT_DIR, "generated")
ACCEPTED_DIR = os.path.join(OUTPUT_DIR, "accepted")

SELECTED_INDICES_PATH = "task1/imports/selected_test_indices.npy"

VGG_PATH = "task1/imports/vgg_normalised.pth"
DECODER_PATH = "task1/imports/decoder.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(ACCEPTED_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.STL10(
    root="common/datasets/STL10",
    split="test",
    download=True,
    transform=transform
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

class_to_idx = {
    name: i
    for i, name in enumerate(class_names)
}

selected_indices = np.load(SELECTED_INDICES_PATH)

selected_by_class = {
    class_name: []
    for class_name in class_names
}

for index in selected_indices:

    label = dataset.labels[index]

    class_name = class_names[label]

    selected_by_class[class_name].append(int(index))

metadata_path = os.path.join(
    OUTPUT_DIR,
    "generated_metadata.csv"
)

expected_images = (
    len(CLASS_PAIRS)
    * 2
    * IMAGES_PER_DIRECTION
)

existing_images = [
    f for f in os.listdir(GENERATED_DIR)
    if f.endswith(".png")
]

metadata = []

if len(existing_images) < expected_images:

    print("\nGenerated images are incomplete.")
    print("Generating cue-conflict images...")

    vgg = net.vgg
    decoder = net.decoder

    vgg.load_state_dict(
        torch.load(
            VGG_PATH,
            map_location=device
        )
    )

    decoder.load_state_dict(
        torch.load(
            DECODER_PATH,
            map_location=device
        )
    )

    vgg = nn.Sequential(
        *list(vgg.children())[:31]
    )

    vgg = vgg.to(device)
    decoder = decoder.to(device)

    vgg.eval()
    decoder.eval()

    def stylize(content_image, style_image):

        content_image = content_image.unsqueeze(0).to(device)
        style_image = style_image.unsqueeze(0).to(device)

        with torch.no_grad():

            content_features = vgg(content_image)

            style_features = vgg(style_image)

            target_features = function.adaptive_instance_normalization(
                content_features,
                style_features
            )

            output = decoder(target_features)

        output = output.squeeze(0).cpu()

        output = torch.clamp(output, 0, 1)

        return output

    def save_tensor_image(image_tensor, path):

        image = transforms.ToPILImage()(image_tensor)

        image.save(path)

    image_id = 0

    for content_class, style_class in CLASS_PAIRS:

        content_indices = selected_by_class[content_class]
        style_indices = selected_by_class[style_class]

        print(
            "\nPair:",
            content_class,
            "<->",
            style_class
        )

        print(
            "Generating:",
            content_class,
            "shape +",
            style_class,
            "style"
        )

        content_indices_selected = random.sample(
            content_indices,
            IMAGES_PER_DIRECTION
        )

        style_indices_selected = random.sample(
            style_indices,
            IMAGES_PER_DIRECTION
        )

        for i in range(IMAGES_PER_DIRECTION):

            content_index = content_indices_selected[i]
            style_index = style_indices_selected[i]

            content_image, _ = dataset[content_index]

            style_image, _ = dataset[style_index]

            output = stylize(
                content_image,
                style_image
            )

            filename = (
                f"{image_id:04d}_"
                f"{content_class}_"
                f"{style_class}.png"
            )

            output_path = os.path.join(
                GENERATED_DIR,
                filename
            )

            save_tensor_image(
                output,
                output_path
            )

            metadata.append({
                "image_id": image_id,
                "filename": filename,
                "content_class": content_class,
                "style_class": style_class,
                "content_index": content_index,
                "style_index": style_index
            })

            image_id += 1

        print(
            "Generating:",
            style_class,
            "shape +",
            content_class,
            "style"
        )

        content_indices_selected = random.sample(
            style_indices,
            IMAGES_PER_DIRECTION
        )

        style_indices_selected = random.sample(
            content_indices,
            IMAGES_PER_DIRECTION
        )

        for i in range(IMAGES_PER_DIRECTION):

            content_index = content_indices_selected[i]
            style_index = style_indices_selected[i]

            content_image, _ = dataset[content_index]

            style_image, _ = dataset[style_index]

            output = stylize(
                content_image,
                style_image
            )

            filename = (
                f"{image_id:04d}_"
                f"{style_class}_"
                f"{content_class}.png"
            )

            output_path = os.path.join(
                GENERATED_DIR,
                filename
            )

            save_tensor_image(
                output,
                output_path
            )

            metadata.append({
                "image_id": image_id,
                "filename": filename,
                "content_class": style_class,
                "style_class": content_class,
                "content_index": content_index,
                "style_index": style_index
            })

            image_id += 1

    with open(
        metadata_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "filename",
                "content_class",
                "style_class",
                "content_index",
                "style_index"
            ]
        )

        writer.writeheader()

        writer.writerows(metadata)

    print("\nGeneration complete.")

else:

    print("\nAll generated images already exist.")
    print("Skipping image generation.")

    if not os.path.exists(metadata_path):

        raise FileNotFoundError(
            f"Generated images exist, but metadata file is missing: "
            f"{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        metadata = list(reader)

    for item in metadata:

        item["image_id"] = int(item["image_id"])

        item["content_index"] = int(
            item["content_index"]
        )

        item["style_index"] = int(
            item["style_index"]
        )

    print("Loaded metadata from:")
    print(metadata_path)


print(
    "\nTotal candidates generated:",
    len(metadata)
)


ACCEPTED_INDICES = [
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20,
    21, 23, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 39, 40, 41, 42,
    43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 56, 57, 60, 61, 62,
    63, 65, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 80, 82, 85,
    86, 87, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103,
    104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147,
    148, 149, 150, 151, 152, 153, 155, 158, 160, 161, 162, 165, 166,
    167, 170, 171, 172, 173, 174, 175, 179, 180, 183, 184, 186, 187,
    188, 189, 190, 191, 192, 195, 196, 197, 199, 200, 201, 202, 205,
    206, 208, 209, 211, 212, 216, 217, 219, 220, 221, 222, 225, 227, 230, 233, 234,
    237, 238, 242, 244, 246, 247, 248, 250, 251, 253, 255, 257, 258,
    259, 261, 263, 265, 269, 271, 273, 280, 281, 282, 287, 289, 290,
    298, 299
]


all_ids = set(
    item["image_id"]
    for item in metadata
)

accepted_set = set(
    ACCEPTED_INDICES
)

invalid_ids = accepted_set - all_ids

if len(invalid_ids) > 0:

    raise ValueError(
        f"Invalid image IDs: {sorted(invalid_ids)}"
    )


rejected_indices = sorted(
    all_ids - accepted_set
)


print("\nHuman filtering results:")

print(
    "Generated:",
    len(metadata)
)

print(
    "Accepted:",
    len(accepted_set)
)

print(
    "Rejected:",
    len(rejected_indices)
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "accepted_indices.npy"
    ),
    np.array(
        sorted(accepted_set)
    )
)


for item in metadata:

    if item["image_id"] not in accepted_set:
        continue

    source_path = os.path.join(
        GENERATED_DIR,
        item["filename"]
    )

    destination_path = os.path.join(
        ACCEPTED_DIR,
        item["filename"]
    )

    image = Image.open(source_path)

    image.save(destination_path)


accepted_metadata_path = os.path.join(
    OUTPUT_DIR,
    "accepted_metadata.csv"
)

accepted_metadata = [
    item
    for item in metadata
    if item["image_id"] in accepted_set
]


with open(
    accepted_metadata_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id",
            "filename",
            "content_class",
            "style_class",
            "content_index",
            "style_index"
        ]
    )

    writer.writeheader()

    writer.writerows(accepted_metadata)


print("\nSaved accepted images to:")

print(ACCEPTED_DIR)

print("\nDone.")