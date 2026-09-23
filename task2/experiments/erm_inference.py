import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score


SEED = 6304
NUM_CLASSES = 7
BATCH_SIZE = 32

DATA_ROOT = "task2/data/PACS"
CHECKPOINT = "task2/checkpoints/erm_best.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


sketch_dataset = datasets.ImageFolder(
    root=f"{DATA_ROOT}/sketch",
    transform=transform
)

sketch_loader = DataLoader(
    sketch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("Sketch images:", len(sketch_dataset))


model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in sketch_loader:

        images = images.to(DEVICE)

        logits = model(images)

        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)


print("\n===== ERM ON SKETCH =====")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Macro-F1: {macro_f1:.4f}"
)