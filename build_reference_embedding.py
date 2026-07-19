"""
build_reference_embedding.py
-----------------------------
One-time script: run this once after training.
Computes the "average feature fingerprint" of real chest X-rays by running
all training images through ResNet18's backbone (everything except the
final classification layer) and averaging their embeddings.

This average embedding acts as a reference point. At inference time, if a
new image's embedding is too "far" from this reference, it's probably not
a chest X-ray at all (a cat photo, a random object, a screenshot, etc.)

Run:
    python build_reference_embedding.py
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("chest_xray/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Load your trained model, but strip off the final classification layer
# so we get raw 512-dim feature embeddings instead of class scores.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.fc = nn.Identity()  # removes classification head, outputs raw features
model.to(device)
model.eval()

print("Computing embeddings for all training images...")

all_embeddings = []
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        embeddings = model(images)          # shape: [batch, 512]
        all_embeddings.append(embeddings.cpu())

all_embeddings = torch.cat(all_embeddings)  # shape: [N, 512]

# The reference point = average embedding across all real X-rays
reference_embedding = all_embeddings.mean(dim=0)

# Also compute a sensible similarity threshold: the average distance
# real X-rays have from the reference point. Anything much further
# than this is likely not an X-ray.
similarities = torch.nn.functional.cosine_similarity(
    all_embeddings, reference_embedding.unsqueeze(0)
)
threshold = similarities.mean().item() - 2 * similarities.std().item()

np.save("reference_embedding.npy", reference_embedding.numpy())
with open("ood_threshold.txt", "w") as f:
    f.write(str(threshold))

print(f"Reference embedding saved.")
print(f"Mean similarity of real X-rays to reference: {similarities.mean().item():.4f}")
print(f"Suggested OOD threshold: {threshold:.4f}")
print("(Any new image with similarity BELOW this is flagged as 'not an X-ray')")