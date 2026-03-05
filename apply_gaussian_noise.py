import os
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------
# Gaussian noise core function
# ---------------------------

def add_gaussian_noise(img: Image.Image, sigma=10):
    """
    给图像加入 Gaussian noise
    sigma 控制噪声强度
    """
    img_np = np.array(img).astype(np.float32)

    noise = np.random.normal(0, sigma, img_np.shape)

    noisy = img_np + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)

# ---------------------------
# Batch process directory
# ---------------------------

def process_tree(input_dir: Path, output_dir: Path, sigma: float):

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    all_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

    for src in tqdm(all_files, desc="Adding Gaussian Noise"):

        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")

        noisy_img = add_gaussian_noise(img, sigma)

        noisy_img.save(dst)


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sigma", type=float, default=10)

    args = parser.parse_args()

    process_tree(
        Path(args.input_dir),
        Path(args.output_dir),
        sigma=args.sigma
    )