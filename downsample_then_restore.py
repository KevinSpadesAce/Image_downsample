import os
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------
# Resize core function
# ---------------------------

def resize_image(img: Image.Image, down_size=128, interpolation="bilinear") -> Image.Image:
    
    orig_w, orig_h = img.size

    interp_map = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS
    }

    if interpolation not in interp_map:
        raise ValueError("Interpolation must be bilinear | bicubic | lanczos")

    interp = interp_map[interpolation]

    # 下采样
    img_down = img.resize((down_size, down_size), interp)

    # 再拉回原始尺寸
    img_up = img_down.resize((orig_w, orig_h), interp)

    return img_up

# ---------------------------
# Batch process directory
# ---------------------------

def process_tree(input_dir: Path, output_dir: Path, down_size: int, interpolation: str, keep_format: bool):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    all_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

    for src in tqdm(all_files, desc="Processing"):
        rel = src.relative_to(input_dir)                    # e.g. CelebA/Real/0/12.jpg
        dst = output_dir / rel                              # e.g. downSampled_Image_Selection/CelebA/Real/0/12.jpg
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")
        out = resize_image(img, down_size=down_size, interpolation=interpolation)

        # 保留原后缀写回（jpg/png等）
        if keep_format:
            out.save(dst)
        else:
            # 不保留就统一输出成PNG（路径后缀会变成 .png）
            dst = dst.with_suffix(".png")
            out.save(dst)


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--down_size", type=int, default=128,
                        help="Downsample size (e.g., 128, 128)")
    parser.add_argument("--interpolation", type=str, default="bilinear")

    args = parser.parse_args()

    process_tree(
        Path(args.input_dir),
        Path(args.output_dir),
        down_size=args.down_size,
        interpolation=args.interpolation,
        keep_format=True
    )