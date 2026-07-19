"""
severity_utils.py
------------------
Turns a Grad-CAM heatmap into a rough "severity" estimate by measuring
how much of the image is covered by high-attention (hot) regions.

NOTE: This is a heuristic, not a clinically validated severity score.
Good talking point in interviews: mention this limitation explicitly —
it shows scientific honesty, which interviewers respect.
"""

import numpy as np


def compute_severity(heatmap_np, hot_threshold=0.5):
    """
    heatmap_np: the raw Grad-CAM heatmap array (values 0.0 - 1.0),
                same one your gradcam_utils.py already produces
                before it gets colorized/overlaid.
    hot_threshold: pixel intensity above which we count it as "hot" / concerning

    Returns: (severity_label, coverage_percent)
    """
    heatmap_np = np.asarray(heatmap_np)

    # Normalize to 0-1 range just in case it isn't already
    if heatmap_np.max() > 1.0:
        heatmap_np = heatmap_np / 255.0

    hot_pixels = (heatmap_np > hot_threshold).sum()
    total_pixels = heatmap_np.size
    coverage_percent = round((hot_pixels / total_pixels) * 100, 1)

    if coverage_percent < 10:
        severity_label = "MILD"
    elif coverage_percent < 25:
        severity_label = "MODERATE"
    else:
        severity_label = "SEVERE"

    return severity_label, coverage_percent