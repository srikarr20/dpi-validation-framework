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
# Detector-space fields
# -------------------------------------------------

FIELDS = {
    "chi2": "chi2.npy",
    "bias": "bias.npy",
    "transmission_noisy": "transmission_noisy.npy",
    "unc_maps": "unc_maps.npy",
}

# -------------------------------------------------
# Normalize helper
# -------------------------------------------------

def normalize(arr):

    arr = np.nan_to_num(arr)

    # flatten extra dimensions if needed
    while arr.ndim > 2:
        arr = arr[0]

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
# Generate comparison
# -------------------------------------------------

fig, axes = plt.subplots(
    len(FIELDS),
    2,
    figsize=(12, 4 * len(FIELDS))
)

for row, (field_name, filename) in enumerate(FIELDS.items()):

    path = ARRAYS / filename

    arr = np.load(path)

    norm = normalize(arr)

    overlay = dpi_observability_transform(arr)

    # save overlay
    overlay_path = OVERLAYS / f"{field_name}_overlay_v1.png"

    overlay.save(overlay_path)

    print(f"Saved -> {overlay_path}")

    # raw field
    axes[row, 0].imshow(norm, cmap="viridis")
    axes[row, 0].set_title(f"Raw {field_name}")

    # overlay
    axes[row, 1].imshow(overlay, cmap="inferno")
    axes[row, 1].set_title(f"DPI {field_name} Overlay")

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")

plt.tight_layout()

comparison_path = OVERLAYS / "field_comparison_observability_v1.png"

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved comparison -> {comparison_path}")

plt.show()
