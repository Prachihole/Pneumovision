# PneumoVision — Clinical Chest X-Ray Pneumonia Detector

PneumoVision is a deep learning web app that analyzes chest X-ray images and predicts whether they show signs of pneumonia. It uses a fine-tuned ResNet18 model, Grad-CAM for visual explainability, an out-of-distribution (OOD) check to reject non-X-ray images, and auto-generates a downloadable clinical PDF report.

Built as a learning project to explore transfer learning, model interpretability, and deploying ML models behind a real interface — not a certified medical device.

---

## Features

- **Single-image analysis** — upload one chest X-ray and get an instant prediction (Normal / Pneumonia) with confidence scores.
- **Batch analysis** — upload multiple X-rays at once and get results in a table.
- **Grad-CAM visualization** — a heatmap overlay showing which regions of the X-ray influenced the model's decision, for interpretability.
- **Out-of-distribution (OOD) detection** — uses cosine similarity against a reference embedding to flag uploads that aren't valid chest X-rays, instead of confidently misclassifying random images.
- **Severity scoring** — for positive pneumonia cases, estimates severity based on the percentage of lung area highlighted by Grad-CAM.
- **PDF report generation** — auto-generates a downloadable clinical-style report per analysis using ReportLab.
- **Dark-mode clinical UI** — custom Gradio interface styled to resemble a real diagnostic tool.

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | ResNet18 (transfer learning, ImageNet pretrained) |
| Framework | PyTorch |
| Interpretability | Grad-CAM |
| Interface | Gradio |
| Reports | ReportLab |
| OOD Detection | Cosine similarity on feature embeddings |

---

## How It Works

1. An uploaded image is resized to 224×224 and normalized using ImageNet statistics.
2. The image's feature embedding is compared (cosine similarity) against a reference embedding of known chest X-rays. If similarity falls below a set threshold, the image is rejected as "not a valid chest X-ray" rather than forced through the classifier.
3. Valid images are passed through the fine-tuned ResNet18 classifier, producing Normal/Pneumonia probabilities.
4. Grad-CAM generates a heatmap over the final convolutional layer, highlighting regions that most influenced the prediction.
5. For pneumonia-positive cases, a severity score is estimated from the percentage of lung area covered by high-attention regions.
6. Results, the Grad-CAM overlay, and the verdict are compiled into a downloadable PDF report.

---

## Model Performance

- **Test Accuracy:** 89.74%
- **Dataset:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Architecture:** ResNet18, fine-tuned via transfer learning on top of ImageNet-pretrained weights

> **Note on limitations:** This dataset consists of pediatric chest X-rays from a single medical center (Guangzhou Women and Children's Medical Center), so performance may not generalize to adult patients or X-rays from other imaging equipment/hospitals. Accuracy alone doesn't capture the full picture for a medical screening task — sensitivity and specificity matter more in practice, since false negatives (missed pneumonia cases) and false positives (unnecessary alarm) carry very different costs.

---

## Installation

```bash
git clone https://github.com/<your-username>/pneumovision.git
cd pneumovision
pip install -r requirements.txt
```

**Requirements** (add to `requirements.txt`):
```
torch
torchvision
gradio
pillow
numpy
reportlab
```

You'll also need:
- `best_model.pth` — the trained model weights
- `reference_embedding.npy` — reference feature embedding for OOD detection
- `ood_threshold.txt` — similarity threshold for OOD detection

---

## Usage

```bash
python app.py
```

This launches a local Gradio server (default: `http://127.0.0.1:7860`) with two tabs:

- **Single Analysis** — upload one X-ray, get prediction, Grad-CAM heatmap, verdict, and PDF report.
- **Batch Analysis** — upload multiple X-rays, get a results table with predictions and confidence scores for each.

---

## Project Structure

```
pneumovision/
├── app.py                    # Main Gradio app
├── gradcam_utils.py          # Grad-CAM heatmap generation
├── report_utils.py           # PDF report generation
├── severity_utils.py         # Severity scoring from Grad-CAM coverage
├── best_model.pth            # Trained model weights
├── reference_embedding.npy   # OOD reference embedding
├── ood_threshold.txt         # OOD similarity threshold
└── requirements.txt
```

---

## Future Improvements

- Multi-class classification (e.g. bacterial vs. viral pneumonia, other findings)
- Report sensitivity/specificity/AUC-ROC alongside accuracy
- Validate on an external, out-of-distribution dataset to test generalization
- Uncertainty estimation for low-confidence predictions
- Batch PDF export (currently only single-image PDF export is supported)

---

## Disclaimer

**For educational use only. Not a certified medical device.** This tool is a personal/academic project exploring computer vision and medical imaging, and should not be used for real clinical diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## Acknowledgments

Built with PyTorch, Gradio, and ReportLab. Dataset from Kaggle (Chest X-Ray Images - Pneumonia).
