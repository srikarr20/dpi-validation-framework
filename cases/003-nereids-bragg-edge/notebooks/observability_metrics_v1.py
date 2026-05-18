from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

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

    return np.array(topology)

# -------------------------------------------------
# Metrics
# -------------------------------------------------

def edge_density(binary):

    return binary.mean()

def largest_connected_fraction(binary):

    labeled, n = ndimage.label(binary)

    if n == 0:
        return 0

    sizes = ndimage.sum(
        binary,
        labeled,
        range(1, n + 1)
    )

    largest = sizes.max()

    return largest / binary.sum()

def fragmentation_index(binary):

    labeled, n = ndimage.label(binary)

    return n

def persistence_entropy(binary):

    p = binary.mean()

    if p <= 0 or p >= 1:
        return 0

    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

# -------------------------------------------------
# Analysis
# -------------------------------------------------

results = []

fig, axes = plt.subplots(
    len(FIELDS),
    2,
    figsize=(12, 4 * len(FIELDS))
)

for row, (field_name, filename) in enumerate(FIELDS.items()):

    arr = np.load(ARRAYS / filename)

    overlay = dpi_observability_transform(arr)

    binary = overlay > np.percentile(
        overlay,
        75
    )

    density = edge_density(binary)

    connected = largest_connected_fraction(binary)

    fragmentation = fragmentation_index(binary)

    entropy = persistence_entropy(binary)

    results.append([
        field_name,
        density,
        connected,
        fragmentation,
        entropy
    ])

    # visualization
    axes[row, 0].imshow(normalize(arr), cmap="viridis")
    axes[row, 0].set_title(f"Raw {field_name}")

    axes[row, 1].imshow(overlay, cmap="inferno")
    axes[row, 1].set_title(
        f"{field_name}\n"
        f"density={density:.3f} | "
        f"connected={connected:.3f} | "
        f"frag={fragmentation}"
    )

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")

plt.tight_layout()

# -------------------------------------------------
# Save figure
# -------------------------------------------------

comparison_path = OVERLAYS / "observability_metrics_v1.png"

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved -> {comparison_path}")

# -------------------------------------------------
# Print metrics
# -------------------------------------------------

print("\nOBSERVABILITY METRICS\n")

for r in results:

    print(
        f"{r[0]:20s} | "
        f"density={r[1]:.4f} | "
        f"connected={r[2]:.4f} | "
        f"fragmentation={r[3]:3d} | "
        f"entropy={r[4]:.4f}"
    )

plt.show()
