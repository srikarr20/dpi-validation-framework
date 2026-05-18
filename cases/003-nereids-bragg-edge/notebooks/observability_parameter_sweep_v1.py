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
# Fields
# -------------------------------------------------

FIELDS = {
    "chi2": "chi2.npy",
    "bias": "bias.npy",
    "transmission_noisy": "transmission_noisy.npy",
    "unc_maps": "unc_maps.npy",
}

# -------------------------------------------------
# Sweep parameters
# -------------------------------------------------

CONTRASTS = [1.5, 3.0, 5.0]
SMOOTHING = [0, 1, 2]
THRESHOLDS = [70, 75, 80]

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
# DPI transform
# -------------------------------------------------

def dpi_transform(arr, contrast, smooth_passes):

    norm = normalize(arr)

    img = Image.fromarray(
        (norm * 255).astype(np.uint8)
    )

    persistence = img.filter(
        ImageFilter.FIND_EDGES
    )

    persistence = ImageEnhance.Contrast(
        persistence
    ).enhance(contrast)

    topology = persistence

    for _ in range(smooth_passes):

        topology = topology.filter(
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

for field_name, filename in FIELDS.items():

    arr = np.load(ARRAYS / filename)

    for contrast in CONTRASTS:

        for smooth in SMOOTHING:

            for threshold in THRESHOLDS:

                overlay = dpi_transform(
                    arr,
                    contrast,
                    smooth
                )

                binary = overlay > np.percentile(
                    overlay,
                    threshold
                )

                density = edge_density(binary)

                connected = largest_connected_fraction(binary)

                fragmentation = fragmentation_index(binary)

                entropy = persistence_entropy(binary)

                results.append([
                    field_name,
                    contrast,
                    smooth,
                    threshold,
                    density,
                    connected,
                    fragmentation,
                    entropy
                ])

# -------------------------------------------------
# Print summary
# -------------------------------------------------

print("\nOBSERVABILITY PARAMETER SWEEP\n")

for r in results:

    print(
        f"{r[0]:20s} | "
        f"contrast={r[1]:3.1f} | "
        f"smooth={r[2]} | "
        f"threshold={r[3]} | "
        f"density={r[4]:.3f} | "
        f"connected={r[5]:.3f} | "
        f"frag={r[6]:3d} | "
        f"entropy={r[7]:.3f}"
    )

# -------------------------------------------------
# Stability visualization
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

field_colors = {
    "chi2": "tab:red",
    "bias": "tab:orange",
    "transmission_noisy": "tab:green",
    "unc_maps": "tab:blue",
}

for field_name in FIELDS.keys():

    connected_vals = [
        r[5]
        for r in results
        if r[0] == field_name
    ]

    ax.plot(
        connected_vals,
        label=field_name
    )

ax.set_title(
    "Observability Connectedness Stability"
)

ax.set_ylabel("Connected Fraction")
ax.set_xlabel("Parameter Sweep Index")

ax.legend()

plt.tight_layout()

output_path = OVERLAYS / "observability_parameter_sweep_v1.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved -> {output_path}")

plt.show()

# -------------------------------------------------
# Save metrics
# -------------------------------------------------

import csv
import json

csv_path = OVERLAYS / "observability_parameter_sweep_v1.csv"
json_path = OVERLAYS / "observability_parameter_sweep_v1.json"

headers = [
    "field",
    "contrast",
    "smooth",
    "threshold",
    "density",
    "connected",
    "fragmentation",
    "entropy"
]

# CSV
with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(headers)

    writer.writerows(results)

# JSON
json_results = []

for r in results:

    json_results.append({
        "field": r[0],
        "contrast": r[1],
        "smooth": r[2],
        "threshold": r[3],
        "density": r[4],
        "connected": r[5],
        "fragmentation": r[6],
        "entropy": r[7]
    })

with open(json_path, "w") as f:

    json.dump(
        json_results,
        f,
        indent=2
    )

print(f"\nSaved CSV -> {csv_path}")
print(f"Saved JSON -> {json_path}")

