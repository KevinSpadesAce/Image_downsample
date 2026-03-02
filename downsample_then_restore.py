import os
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

# ---------------------------
# Resize core function
# ---------------------------

def resize_image(img_path, output_path, down_size=128, interpolation="bilinear"):
    img = Image.open(img_path).convert("RGB")

    if img.size != (512, 512):
        raise ValueError(f"{img_path} is not 512x512, got {img.size}")

    # Select interpolation
    interp_map = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS
    }

    if interpolation not in interp_map:
        raise ValueError("Interpolation must be bilinear | bicubic | lanczos")

    interp = interp_map[interpolation]

    # Downsample
    img_down = img.resize((down_size, down_size), interp)

    # Upsample back to 512x512
    img_up = img_down.resize((512, 512), interp)

    img_up.save(output_path)


# ---------------------------
# Batch process directory
# ---------------------------

def process_directory(input_dir, output_dir, down_size, interpolation):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for img_path in tqdm(list(input_dir.rglob("*.*"))):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        relative_path = img_path.relative_to(input_dir)
        save_path = output_dir / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        resize_image(
            img_path,
            save_path,
            down_size=down_size,
            interpolation=interpolation
        )


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--down_size", type=int, default=128)
    parser.add_argument("--interpolation", type=str, default="bilinear")

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
        args.down_size,
        args.interpolation
    )