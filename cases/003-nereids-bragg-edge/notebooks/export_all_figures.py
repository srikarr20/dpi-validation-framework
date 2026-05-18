from pathlib import Path
import matplotlib.pyplot as plt

output_dir = Path(
    "/Users/rallabandisailesh/Desktop/Consulting Firm/validation/cases/003-nereids-bragg-edge/outputs"
)

output_dir.mkdir(parents=True, exist_ok=True)

figures = [plt.figure(n) for n in plt.get_fignums()]

print(f"Found {len(figures)} figures")

for i, fig in enumerate(figures):

    output_path = output_dir / f"nereids_figure_{i+1:02d}.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")
