import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── 1. Define paths ──────────────────────────────────────────
TRAIN_DIR = "chest_xray/train"
VAL_DIR   = "chest_xray/val"
TEST_DIR  = "chest_xray/test"

# ── 2. Define transforms ─────────────────────────────────────
# Every image needs to be the same size and normalized
# These are the standard ImageNet mean & std values
# because ResNet18 was trained on ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # ResNet expects 224x224
    transforms.ToTensor(),               # converts image to tensor (0-1)
    transforms.Normalize(                # normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── 3. Load datasets ─────────────────────────────────────────
# ImageFolder automatically assigns labels based on folder names
# NORMAL = 0, PNEUMONIA = 1
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=transform)

# ── 4. Create DataLoaders ─────────────────────────────────────
# DataLoader batches and shuffles data during training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ── 5. Explore the data ───────────────────────────────────────
print("=" * 40)
print("       DATASET SUMMARY")
print("=" * 40)
print(f"Classes        : {train_dataset.classes}")
print(f"Class mapping  : {train_dataset.class_to_idx}")
print(f"Training images: {len(train_dataset)}")
print(f"Val images     : {len(val_dataset)}")
print(f"Test images    : {len(test_dataset)}")
print("=" * 40)

# ── 6. Visualize sample images ────────────────────────────────
# Un-normalize for display (reverse the normalization)
def imshow(tensor, title):
    img = tensor.clone()
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)              # reverse normalization
    img = img.clamp(0, 1)
    plt.imshow(img.permute(1, 2, 0))  # (C,H,W) → (H,W,C) for matplotlib
    plt.title(title)
    plt.axis("off")

# Grab one batch
images, labels = next(iter(train_loader))
class_names = train_dataset.classes

print(f"\nOne batch shape : {images.shape}")   # (32, 3, 224, 224)
print(f"Labels in batch : {labels[:8].tolist()}")

# Plot 8 sample images
plt.figure(figsize=(16, 4))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    imshow(images[i], class_names[labels[i]])
plt.suptitle("Sample Training Images", fontsize=14)
plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()
print("\nSample images saved as sample_images.png")