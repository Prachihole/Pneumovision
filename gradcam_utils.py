import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def get_gradcam_heatmap(model, image_tensor, device):
    """
    Generates Grad-CAM heatmap for the predicted class.
    Returns a numpy array (H, W, 3) as RGB heatmap overlay.
    """
    model.eval()
    features = {}
    gradients = {}

    # Hook to capture last conv layer output and gradients
    def forward_hook(module, input, output):
        features['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Register hooks on last conv layer of ResNet18
    target_layer = model.layer4[1].conv2
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass for predicted class
    model.zero_grad()
    output[0, pred_class].backward()

    # Remove hooks
    fh.remove()
    bh.remove()

    # Compute Grad-CAM
    grads   = gradients['value'].squeeze()       # (C, H, W)
    fmaps   = features['value'].squeeze()        # (C, H, W)
    weights = grads.mean(dim=(1, 2))             # global average pooling

    cam = torch.zeros(fmaps.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]

    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Convert to color heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap, pred_class


def overlay_heatmap(original_pil, heatmap_np, alpha=0.45):
    """
    Overlays Grad-CAM heatmap on original image.
    Returns PIL Image.
    """
    original = np.array(original_pil.resize((224, 224)).convert("RGB"))
    overlay  = (alpha * heatmap_np + (1 - alpha) * original).astype(np.uint8)
    return Image.fromarray(overlay)