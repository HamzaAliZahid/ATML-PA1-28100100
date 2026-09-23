import os
import csv

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


ACCEPTED_DIR = (
    "task1/images/cue_conflict_images/accepted"
)

METADATA_PATH = (
    "task1/images/cue_conflict_images/"
    "accepted_metadata.csv"
)

CLASSIFIER_PATH = (
    "task1/models/vitb16_classifier.pth"
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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


weights = models.ViT_B_16_Weights.IMAGENET1K_V1

vit = models.vit_b_16(
    weights=weights
)

vit.heads = nn.Identity()


for param in vit.parameters():
    param.requires_grad = False


vit = vit.to(device)
vit.eval()


classifier = nn.Linear(
    768,
    10
).to(device)


classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=device
    )
)

classifier.eval()


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
    "Accepted images:",
    len(metadata)
)


results = []

n_shape = 0
n_texture = 0
n_other = 0


for i, item in enumerate(metadata):

    image_path = os.path.join(
        ACCEPTED_DIR,
        item["filename"]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(
        image
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        features = vit(image)

        logits = classifier(features)

        prediction = torch.argmax(
            logits,
            dim=1
        ).item()


    predicted_class = class_names[prediction]

    content_class = item["content_class"]

    style_class = item["style_class"]


    if predicted_class == content_class:

        prediction_type = "shape"

        n_shape += 1

    elif predicted_class == style_class:

        prediction_type = "texture"

        n_texture += 1

    else:

        prediction_type = "other"

        n_other += 1


    results.append({
        "image_id": item["image_id"],
        "filename": item["filename"],
        "content_class": content_class,
        "style_class": style_class,
        "prediction": predicted_class,
        "prediction_type": prediction_type
    })


    if (i + 1) % 20 == 0:

        print(
            f"Processed {i + 1}/"
            f"{len(metadata)}"
        )


total = (
    n_shape +
    n_texture +
    n_other
)


shape_bias = (
    n_shape /
    (n_shape + n_texture)
) * 100


coverage = (
    (n_shape + n_texture) /
    total
) * 100


print("\nViT-B/16 Cue-Conflict Results")
print("--------------------------------")

print(
    "Total images:",
    total
)

print(
    "Shape predictions:",
    n_shape
)

print(
    "Texture predictions:",
    n_texture
)

print(
    "Other predictions:",
    n_other
)

print(
    f"Shape Bias: {shape_bias:.2f}%"
)

print(
    f"Coverage: {coverage:.2f}%"
)


output_path = (
    "task1/results/"
    "vitb16_cue_conflict_predictions.csv"
)


with open(
    output_path,
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
            "prediction",
            "prediction_type"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print(
    "\nPredictions saved to:",
    output_path
)