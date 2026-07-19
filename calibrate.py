"""
calibrate.py
------------
One-time script: run this AFTER train.py has produced best_model.pth.
It learns a single "temperature" value that fixes overconfident predictions,
and saves it to temperature.txt for app.py to load.

Run:
    python calibrate.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from temperature_scaling import ModelWithTemperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Same transform + val set as train.py ──────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder("chest_xray/val", transform=val_transform)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Classes: {val_dataset.classes}")  # should print ['NORMAL', 'PNEUMONIA']

# ── Load the trained model exactly like app.py does ───────────
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# ── Calibrate ───────────────────────────────────────────────
scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(val_loader, device=device)

learned_T = scaled_model.temperature.item()

# ── Save just the number so app.py can load it ─────────────
with open("temperature.txt", "w") as f:
    f.write(str(learned_T))

print(f"\nSaved temperature = {learned_T:.4f} to temperature.txt")
print("Now update app.py to load and apply it (see integration notes).")