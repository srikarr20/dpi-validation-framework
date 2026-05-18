from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE = Path(
    "/Users/rallabandisailesh/Desktop/Consulting Firm/validation/cases/003-nereids-bragg-edge"
)

ARRAYS = BASE / "outputs" / "arrays"
OVERLAYS = BASE / "overlays"

OVERLAYS.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Load arrays
# -------------------------------------------------

chi2 = np.load(ARRAYS / "chi2.npy")
bias = np.load(ARRAYS / "bias.npy")

# -------------------------------------------------
# Normalize helper
# -------------------------------------------------

def normalize(arr):

    arr = np.nan_to_num(arr)

    amin = arr.min()
    amax = arr.max()

    if amax - amin == 0:
        return np.zeros_like(arr)

    return (arr - amin) / (amax - amin)

# -------------------------------------------------
# DPI observability transform
# -------------------------------------------------

def dpi_observability_transform(arr):

    norm = normalize(arr)

    img = Image.fromarray(
        (norm * 255).astype(np.uint8)
    )

    persistence = img.filter(
        ImageFilter.FIND_EDGES
    )

    persistence = ImageEnhance.Contrast(
        persistence
    ).enhance(3.0)

    topology = persistence.filter(
        ImageFilter.SMOOTH_MORE
    )

    return topology

# -------------------------------------------------
# Generate overlays
# -------------------------------------------------

chi2_overlay = dpi_observability_transform(chi2)
bias_overlay = dpi_observability_transform(bias)

# -------------------------------------------------
# Save overlays
# -------------------------------------------------

chi2_overlay_path = OVERLAYS / "chi2_overlay_v1.png"
bias_overlay_path = OVERLAYS / "bias_overlay_v1.png"

chi2_overlay.save(chi2_overlay_path)
bias_overlay.save(bias_overlay_path)

print(f"Saved -> {chi2_overlay_path}")
print(f"Saved -> {bias_overlay_path}")

# -------------------------------------------------
# Visualization
# -------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].imshow(normalize(chi2), cmap="viridis")
axes[0,0].set_title("Raw chi²")

axes[0,1].imshow(chi2_overlay, cmap="inferno")
axes[0,1].set_title("DPI chi² Overlay")

axes[1,0].imshow(normalize(bias), cmap="viridis")
axes[1,0].set_title("Raw Bias")

axes[1,1].imshow(bias_overlay, cmap="inferno")
axes[1,1].set_title("DPI Bias Overlay")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()
