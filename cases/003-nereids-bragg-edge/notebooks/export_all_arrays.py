from pathlib import Path
import numpy as np

output_dir = Path(
    "/Users/rallabandisailesh/Desktop/Consulting Firm/validation/cases/003-nereids-bragg-edge/outputs/arrays"
)

output_dir.mkdir(parents=True, exist_ok=True)

count = 0

for name, value in list(globals().items()):

    if isinstance(value, np.ndarray):

        try:

            output_path = output_dir / f"{name}.npy"

            np.save(output_path, value)

            print(f"Saved array -> {output_path}")

            count += 1

        except Exception as e:

            print(f"Skipped {name}: {e}")

print(f"\nExported {count} arrays")
