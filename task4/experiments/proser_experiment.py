import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split

SEED = 6304

BATCH_SIZE = 128
EPOCHS = 50

LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 10
NUM_DUMMY = 5

BETA = 1.0
GAMMA = 0.1

CHECKPOINT_PATH = "task4/checkpoints/vanilla_best.pth"
OUTPUT_PATH = "task4/checkpoints/proser_best.pth"

os.makedirs("task4/checkpoints", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])


full_train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

eval_train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=eval_transform
)


labels = np.array(full_train_dataset.targets)
indices = np.arange(len(full_train_dataset))


train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    stratify=labels,
    random_state=SEED
)


train_dataset = Subset(
    full_train_dataset,
    train_indices
)

val_dataset = Subset(
    eval_train_dataset,
    val_indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

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

model = PROSER(
    num_classes=NUM_CLASSES,
    num_dummy=NUM_DUMMY
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

vanilla_state = checkpoint["model_state_dict"]

backbone_state = {}

for key, value in vanilla_state.items():

    if key.startswith("fc."):
        continue

    backbone_state[key] = value


model.backbone.load_state_dict(
    backbone_state,
    strict=True
)

model.classifier.load_state_dict({
    "weight": vanilla_state["fc.weight"],
    "bias": vanilla_state["fc.bias"]
})

nn.init.normal_(
    model.dummy_classifier.weight,
    mean=0.0,
    std=0.01
)

nn.init.constant_(
    model.dummy_classifier.bias,
    0.0
)


model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            known_logits, _ = model(images)

            predictions = known_logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        half = images.size(0) // 2

        images_mix = images[:half]
        labels_mix = labels[:half]

        images_normal = images[half:]
        labels_normal = labels[half:]

        normal_known_logits, normal_dummy_logits = model(
            images_normal
        )

        classification_loss = criterion(
            normal_known_logits,
            labels_normal
        )

        max_dummy = normal_dummy_logits.max(
            dim=1
        ).values

        known_without_true = normal_known_logits.clone()

        known_without_true[
            torch.arange(
                labels_normal.size(0),
                device=device
            ),
            labels_normal
        ] = -1e9

        max_wrong_known = known_without_true.max(
            dim=1
        ).values

        placeholder_logits = torch.cat(
            [
                normal_known_logits,
                max_dummy.unsqueeze(1)
            ],
            dim=1
        )

        placeholder_targets = torch.full(
            (labels_normal.size(0),),
            NUM_CLASSES,
            dtype=torch.long,
            device=device
        )

        placeholder_logits = placeholder_logits.clone()

        placeholder_logits[
            torch.arange(
                labels_normal.size(0),
                device=device
            ),
            labels_normal
        ] = -1e9


        classifier_placeholder_loss = criterion(
            placeholder_logits,
            placeholder_targets
        )

        features = model.forward_to_layer2(
            images_mix
        )

        permutation = torch.randperm(
            features.size(0),
            device=device
        )


        shuffled_features = features[
            permutation
        ]

        shuffled_labels = labels_mix[
            permutation
        ]

        different_class = (
            labels_mix != shuffled_labels
        )

        valid_features = features[different_class]
        valid_shuffled_features = shuffled_features[
            different_class
        ]


        if valid_features.size(0) > 0:
            lam = torch.distributions.Beta(
                BETA,
                BETA
            ).sample().item()

            mixed_features = (
                lam * valid_features
                +
                (1.0 - lam) * valid_shuffled_features
            )
            mixed_final_features = model.layer2_to_features(
                mixed_features
            )


            mixed_known_logits = model.classifier(
                mixed_final_features
            )

            mixed_dummy_logits = model.dummy_classifier(
                mixed_final_features
            )

            mixed_max_dummy = mixed_dummy_logits.max(
                dim=1
            ).values


            mixed_placeholder_logits = torch.cat(
                [
                    mixed_known_logits,
                    mixed_max_dummy.unsqueeze(1)
                ],
                dim=1
            )


            mixed_targets = torch.full(
                (mixed_features.size(0),),
                NUM_CLASSES,
                dtype=torch.long,
                device=device
            )


            data_placeholder_loss = criterion(
                mixed_placeholder_logits,
                mixed_targets
            )

        else:

            data_placeholder_loss = torch.tensor(
                0.0,
                device=device
            )

        loss = (
            classification_loss
            +
            BETA * classifier_placeholder_loss
            +
            GAMMA * data_placeholder_loss
        )


        loss.backward()

        optimizer.step()

        predictions = normal_known_logits.argmax(dim=1)

        running_correct += (
            predictions == labels_normal
        ).sum().item()

        running_total += labels_normal.size(0)

        running_loss += loss.item()


    scheduler.step()


    train_accuracy = (
        100.0
        * running_correct
        / running_total
    )


    val_accuracy = evaluate(
        model,
        val_loader
    )


    current_lr = scheduler.get_last_lr()[0]


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}% "
        f"LR: {current_lr:.6f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_accuracy,
            },
            OUTPUT_PATH
        )

        print(
            f"  -> Saved best PROSER checkpoint "
            f"(Val Acc: {val_accuracy:.2f}%)"
        )


print()
print("Training complete.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)
print(
    f"Checkpoint saved to: {OUTPUT_PATH}"
)