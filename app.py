import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import tempfile
import os
from gradcam_utils import get_gradcam_heatmap, overlay_heatmap
from report_utils import generate_report
from severity_utils import compute_severity

# ── Model setup ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

LABELS = ['NORMAL', 'PNEUMONIA']

# ── Single image analysis ─────────────────────────────────────
def analyze_single(image):
    if image is None:
        return None, None, None, None

    tensor = transform(image).unsqueeze(0)

    # ── OOD check FIRST, before diagnosing ──────────────
    with torch.no_grad():
        embedding = feature_model(tensor.to(device))
        similarity = torch.nn.functional.cosine_similarity(
            embedding.cpu(), reference_embedding.unsqueeze(0)
        ).item()

        print(f"[OOD DEBUG] similarity = {similarity:.4f} | threshold = {ood_threshold:.4f}") 

    if similarity < ood_threshold:
        verdict = "⚠ NOT A VALID CHEST X-RAY — please upload a real X-ray image"
        return None, None, None, verdict

    # ── Normal diagnosis flow (unchanged) ───────────────
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(device)), dim=1)[0]

    pneumonia_prob = round(probs[1].item(), 4)
    normal_prob    = round(probs[0].item(), 4)
    pred_label     = LABELS[probs.argmax().item()]

    heatmap_np, _ = get_gradcam_heatmap(model, tensor.clone(), device)
    overlay_pil   = overlay_heatmap(image, heatmap_np)

    # ── Severity score (only meaningful for Pneumonia cases) ──
    if pred_label == "PNEUMONIA":
        severity_label, coverage = compute_severity(heatmap_np)
        verdict = f"{pred_label} — Severity: {severity_label} ({coverage}% lung coverage)"
    else:
        verdict = pred_label

    pdf_bytes = generate_report(
        original_pil=image.resize((224, 224)),
        heatmap_pil=overlay_pil,
        filename="uploaded_xray.jpg",
        pneumonia_prob=pneumonia_prob,
        normal_prob=normal_prob,
        predicted_label=pred_label
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.close()

    label_out = {"PNEUMONIA": pneumonia_prob, "NORMAL": normal_prob}
    return label_out, overlay_pil, tmp.name, verdict

# ── Batch analysis ────────────────────────────────────────────
def analyze_batch(files):
    if not files:
        return "No files uploaded."

    rows = [["Filename", "Prediction", "Pneumonia %", "Normal %", "Confidence"]]

    for f in files:
        img = Image.open(f.name).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor.to(device)), dim=1)[0]
        p = round(probs[1].item() * 100, 2)
        n = round(probs[0].item() * 100, 2)
        pred = "PNEUMONIA" if p > 50 else "NORMAL"
        conf = max(p, n)
        rows.append([os.path.basename(f.name), pred, f"{p}%", f"{n}%", f"{conf}%"])

    # Format as markdown table
    header = "| " + " | ".join(rows[0]) + " |"
    divider = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return f"{header}\n{divider}\n{body}"

reference_embedding = torch.tensor(np.load("reference_embedding.npy"))
with open("ood_threshold.txt") as f:
    ood_threshold = float(f.read().strip())

# Build a feature-extractor version of the model (same weights, no classifier head)
feature_model = models.resnet18(weights=None)
feature_model.fc = nn.Linear(feature_model.fc.in_features, 2)
feature_model.load_state_dict(torch.load("best_model.pth", map_location=device))
feature_model.fc = nn.Identity()
feature_model.to(device)
feature_model.eval()


# ── CSS ───────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
body, .gradio-container { background:#060a07 !important; font-family:'IBM Plex Sans',sans-serif !important; }
.gradio-container { max-width:960px !important; margin:auto !important; padding:2rem !important; }
footer { display:none !important; }
.gr-button { font-family:'IBM Plex Mono',monospace !important; letter-spacing:0.06em !important; background:#0d1f12 !important; border:0.5px solid #2ecc71 !important; color:#2ecc71 !important; border-radius:6px !important; }
.gr-button:hover { background:#102816 !important; }
.gr-box, .gr-form, .gr-panel { background:#0e1510 !important; border:0.5px solid #1e2b22 !important; border-radius:10px !important; }
label, .gr-label { color:#4a6b52 !important; font-family:'IBM Plex Mono',monospace !important; font-size:10px !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
.gr-tab-nav button { font-family:'IBM Plex Mono',monospace !important; font-size:11px !important; letter-spacing:0.06em !important; color:#4a6b52 !important; background:transparent !important; border:none !important; }
.gr-tab-nav button.selected { color:#2ecc71 !important; border-bottom:1.5px solid #2ecc71 !important; }
"""

# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="PneumoVision Clinical") as demo:

    gr.HTML("""
    <link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&display=swap' rel='stylesheet'>
    <div style='margin-bottom:24px;padding-bottom:18px;border-bottom:0.5px solid #1e2b22;
                display:flex;align-items:center;justify-content:space-between;'>
      <div style='display:flex;align-items:center;gap:12px;'>
        <div style='width:36px;height:36px;border:1.5px solid #2ecc71;border-radius:6px;
                    display:flex;align-items:center;justify-content:center;'>
          <svg width='18' height='18' viewBox='0 0 16 16' fill='none' stroke='#2ecc71'
               stroke-width='1.5' stroke-linecap='round'>
            <path d='M4 8 C4 5 6 3 8 3 C10 3 12 5 12 8'/>
            <path d='M3 10 C3 12 5 14 8 14 C11 14 13 12 13 10'/>
            <circle cx='8' cy='8' r='1.5' fill='#2ecc71' stroke='none'/>
          </svg>
        </div>
        <div>
          <div style='font-family:IBM Plex Mono,monospace;font-size:16px;
                      font-weight:500;color:#e8ede9;'>
            Pneumo<span style="color:#2ecc71;">Vision</span>
            <span style='font-size:11px;color:#2a3d2e;margin-left:8px;'>Clinical</span>
          </div>
          <div style='font-family:IBM Plex Mono,monospace;font-size:9px;
                      color:#2a3d2e;letter-spacing:0.06em;'>
            ResNet18 · Transfer Learning · Grad-CAM · PDF Reports · Batch Analysis
          </div>
        </div>
      </div>
      <div style='display:flex;gap:8px;'>
        <span style='font-family:IBM Plex Mono,monospace;font-size:10px;background:#0d1f12;
                     border:0.5px solid #1e3b1e;border-radius:4px;padding:4px 10px;color:#4a6b52;'>
          89.74% accuracy
        </span>
        <span style='font-family:IBM Plex Mono,monospace;font-size:10px;background:#0d1f12;
                     border:0.5px solid #1e3b1e;border-radius:4px;padding:4px 10px;color:#2ecc71;'>
          ● SYSTEM READY
        </span>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Single Analysis ─────────────────────────────
        with gr.Tab("[ SINGLE ANALYSIS ]"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="INPUT · CHEST X-RAY")
                    run_btn   = gr.Button("[ RUN ANALYSIS ]")

                with gr.Column(scale=1):
                    label_out  = gr.Label(num_top_classes=2, label="OUTPUT · DIAGNOSIS")
                    heatmap_out= gr.Image(label="GRAD-CAM · ATTENTION MAP", type="pil")

            with gr.Row():
                report_out = gr.File(label="DOWNLOAD · CLINICAL PDF REPORT")
                verdict_out = gr.Textbox(label="VERDICT", interactive=False)

            run_btn.click(
                fn=analyze_single,
                inputs=img_input,
                outputs=[label_out, heatmap_out, report_out, verdict_out]
            )

        # ── Tab 2: Batch Analysis ──────────────────────────────
        with gr.Tab("[ BATCH ANALYSIS ]"):
            gr.HTML("""<p style='font-family:IBM Plex Mono,monospace;font-size:11px;
                        color:#4a6b52;margin-bottom:16px;'>
                        Upload multiple X-rays for simultaneous analysis</p>""")
            batch_input  = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="INPUT · MULTIPLE X-RAYS"
            )
            batch_btn    = gr.Button("[ RUN BATCH ANALYSIS ]")
            batch_output = gr.Markdown(label="BATCH RESULTS")

            batch_btn.click(
                fn=analyze_batch,
                inputs=batch_input,
                outputs=batch_output
            )

    gr.HTML("""
    <div style='margin-top:20px;padding-top:14px;border-top:0.5px solid #1e2b22;
                font-family:IBM Plex Mono,monospace;font-size:9px;color:#2a3d2e;
                text-align:center;letter-spacing:0.04em;'>
      FOR EDUCATIONAL USE ONLY · NOT A CERTIFIED MEDICAL DEVICE ·
      BUILT WITH PyTorch · Gradio · ReportLab
    </div>
    """)

demo.launch()