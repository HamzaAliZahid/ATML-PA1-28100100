import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score


SEED = 6304

DATA_ROOT = "task2/PACS"

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]

TARGET_DOMAIN = "sketch"

NUM_CLASSES = 7
BATCH_SIZE_SOURCE = 8
BATCH_SIZE_TARGET = 24

EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

LAMBDA_VALUES = [0.1, 1.0, 10.0]

CHECKPOINT_DIR = "task2/checkpoints/controlled"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

class PACSDataset(Dataset):
    def __init__(self, root, domain, transform=None):
        self.root = os.path.join(root, domain)
        self.transform = transform

        self.samples = []

        classes = sorted(os.listdir(self.root))

        for label, class_name in enumerate(classes):

            class_dir = os.path.join(self.root, class_name)

            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):

                path = os.path.join(class_dir, filename)

                if os.path.isfile(path):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def stratified_split(dataset, train_ratio=0.8):

    labels = np.array([
        label for _, label in dataset.samples
    ])

    train_indices = []
    val_indices = []

    rng = np.random.RandomState(SEED)

    for label in np.unique(labels):

        indices = np.where(labels == label)[0]

        rng.shuffle(indices)

        split = int(len(indices) * train_ratio)

        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

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

        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):

        features = self.features(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        logits = self.classifier(features)

        return features, logits


def freeze_bn_running_stats(model):

    for module in model.modules():

        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def gaussian_kernel(x, y, bandwidth):

    x_squared = torch.sum(x ** 2, dim=1, keepdim=True)
    y_squared = torch.sum(y ** 2, dim=1, keepdim=True)

    distances = (
        x_squared
        + y_squared.t()
        - 2 * torch.mm(x, y.t())
    )

    return torch.exp(
        -distances / (2 * bandwidth)
    )


def mmd_loss(source_features, target_features):

    combined = torch.cat(
        [source_features, target_features],
        dim=0
    )

    with torch.no_grad():

        distances = torch.cdist(
            combined,
            combined
        ).pow(2)

        median_distance = torch.median(distances)

    kernels = []

    for multiplier in [0.5, 1.0, 2.0]:

        bandwidth = median_distance * multiplier

        kernels.append(
            gaussian_kernel(
                combined,
                combined,
                bandwidth
            )
        )

    kernel = sum(kernels)

    ns = source_features.size(0)
    nt = target_features.size(0)

    ss = kernel[:ns, :ns]
    tt = kernel[ns:, ns:]
    st = kernel[:ns, ns:]

    loss = (
        ss.mean()
        + tt.mean()
        - 2 * st.mean()
    )

    return loss


def evaluate(model, loader):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for images, y in loader:

            images = images.to(device)

            _, logits = model(images)

            preds = torch.argmax(
                logits,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                y.numpy()
            )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return accuracy, macro_f1


def train_one_lambda(lambda_mmd):

    print()
    print("=" * 60)
    print("Training DAN with lambda =", lambda_mmd)
    print("=" * 60)

    train_datasets = []
    val_loaders = []

    for domain in SOURCE_DOMAINS:

        full_train = PACSDataset(
            DATA_ROOT,
            domain,
            train_transform
        )

        train_indices, val_indices = stratified_split(
            full_train
        )

        train_subset = torch.utils.data.Subset(
            full_train,
            train_indices
        )

        val_dataset = PACSDataset(
            DATA_ROOT,
            domain,
            val_transform
        )

        val_subset = torch.utils.data.Subset(
            val_dataset,
            val_indices
        )

        train_datasets.append(train_subset)

        val_loaders.append(
            DataLoader(
                val_subset,
                batch_size=32,
                shuffle=False
            )
        )

    sketch_dataset = PACSDataset(
        DATA_ROOT,
        TARGET_DOMAIN,
        train_transform
    )

    source_loaders = [
        DataLoader(
            dataset,
            batch_size=BATCH_SIZE_SOURCE,
            shuffle=True,
            drop_last=True
        )
        for dataset in train_datasets
    ]

    target_loader = DataLoader(
        sketch_dataset,
        batch_size=BATCH_SIZE_TARGET,
        shuffle=True,
        drop_last=True
    )

    model = DAN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_mean_f1 = -1
    patience_counter = 0

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"dan_lambda_{lambda_mmd}.pt"
    )

    for epoch in range(EPOCHS):

        model.train()
        freeze_bn_running_stats(model)

        source_iters = [
            iter(loader)
            for loader in source_loaders
        ]

        target_iter = iter(target_loader)

        num_steps = min(
            len(loader)
            for loader in source_loaders
        )

        num_steps = min(
            num_steps,
            len(target_loader)
        )

        epoch_loss = 0.0

        for _ in range(num_steps):

            source_images = []
            source_labels = []

            for source_iter in source_iters:

                images, labels = next(source_iter)

                source_images.append(images)
                source_labels.append(labels)

            target_images, _ = next(target_iter)

            source_images = torch.cat(
                source_images,
                dim=0
            ).to(device)

            source_labels = torch.cat(
                source_labels,
                dim=0
            ).to(device)

            target_images = target_images.to(device)

            source_features, source_logits = model(
                source_images
            )

            target_features, _ = model(
                target_images
            )

            classification_loss = F.cross_entropy(
                source_logits,
                source_labels
            )

            alignment_loss = mmd_loss(
                source_features,
                target_features
            )

            loss = (
                classification_loss
                + lambda_mmd * alignment_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        source_metrics = []

        for loader in val_loaders:

            acc, f1 = evaluate(
                model,
                loader
            )

            source_metrics.append(
                (acc, f1)
            )

        mean_f1 = np.mean([
            f1 for _, f1 in source_metrics
        ])

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {epoch_loss / num_steps:.4f} | "
            f"Mean source F1: {mean_f1:.4f}"
        )

        if mean_f1 > best_mean_f1:

            best_mean_f1 = mean_f1

            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print("Early stopping.")

                break

    print(
        "Saved:",
        checkpoint_path
    )


def main():

    for lambda_mmd in LAMBDA_VALUES:

        set_seed(SEED)

        train_one_lambda(lambda_mmd)