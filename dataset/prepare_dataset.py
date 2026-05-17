"""
dataset/prepare_dataset.py

Downloads and prepares publicly available handwriting / OCR datasets:
  1. IAM Handwriting Database (sample subset via HuggingFace)
  2. Synthetic whiteboard images (generated locally)
  3. IIIT-HWS (offline, instructions provided)

Run: python dataset/prepare_dataset.py
"""

import os
import json
import random
import string
import textwrap
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DATASET_DIR = Path(__file__).parent
SAMPLES_DIR = DATASET_DIR / "samples"
SYNTH_DIR   = DATASET_DIR / "synthetic"
META_FILE   = DATASET_DIR / "dataset_meta.json"

SAMPLES_DIR.mkdir(exist_ok=True)
SYNTH_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────── #
#  1. Synthetic whiteboard generator                                    #
# ──────────────────────────────────────────────────────────────────── #

SAMPLE_TEXTS = [
    "Machine Learning\n- Supervised\n- Unsupervised\n- Reinforcement",
    "TODO:\n1. Fix bug #234\n2. Write tests\n3. Deploy to staging",
    "y = mx + b\nwhere m = slope\nb = y-intercept",
    "API Gateway\n  → Lambda\n    → DynamoDB",
    "Meeting Notes:\nProject Alpha\nDeadline: Q3",
    "def fibonacci(n):\n  if n <= 1:\n    return n\n  return f(n-1)+f(n-2)",
]


def generate_synthetic_whiteboard(text: str, idx: int) -> Path:
    """Create a synthetic whiteboard image with given text."""
    if not HAS_PIL:
        print("  PIL not installed, skipping synthetic generation")
        return None

    W, H = 800, 600
    # White/cream background
    bg_color = (random.randint(240, 255), random.randint(238, 252), random.randint(235, 250))
    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)

    # Add subtle texture
    noise = np.random.randint(0, 12, (H, W, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise)
    img = Image.blend(img, noise_img.resize((W, H)), alpha=0.05)
    draw = ImageDraw.Draw(img)

    # Try system fonts; fall back to default
    font_size = random.randint(20, 30)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Random slight tilt
    from PIL import Image as PILImage
    import math
    tilt = random.uniform(-3, 3)

    # Draw lines
    lines = text.split("\n")
    x0, y0 = random.randint(40, 80), random.randint(40, 80)
    line_height = int(font_size * 1.5)
    ink_r = random.randint(0, 30)
    ink_color = (ink_r, ink_r + random.randint(0, 15), random.randint(0, 30))

    for i, line in enumerate(lines):
        y = y0 + i * line_height
        if y > H - 40:
            break
        # Slight x jitter for handwriting feel
        jitter_x = random.randint(-2, 2)
        draw.text((x0 + jitter_x, y), line, fill=ink_color, font=font)

    # Random shadow / vignette
    if random.random() > 0.5:
        shadow = Image.new("RGB", (W, H), (200, 200, 200))
        img = Image.blend(img, shadow, alpha=0.08)

    # Slight blur to simulate camera
    if random.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))

    out_path = SYNTH_DIR / f"whiteboard_{idx:04d}.png"
    img.save(out_path)
    return out_path


def generate_synthetic_handwritten(text: str, idx: int) -> Path:
    """Simulate lined notebook paper with handwritten-style text."""
    if not HAS_PIL:
        return None

    W, H = 700, 500
    # Slightly yellow notebook bg
    img = Image.new("RGB", (W, H), (255, 253, 240))
    draw = ImageDraw.Draw(img)

    # Draw ruled lines
    line_spacing = 30
    for y in range(60, H, line_spacing):
        draw.line([(40, y), (W - 40, y)], fill=(185, 210, 235), width=1)

    # Red margin line
    draw.line([(80, 0), (80, H)], fill=(220, 80, 80), width=1)

    font_size = random.randint(18, 24)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    lines = text.split("\n")
    x0, y_start = 90, 48
    ink = (20 + random.randint(0, 20), 10, 140 + random.randint(0, 40))  # blue ink

    for i, line in enumerate(lines):
        y = y_start + i * line_spacing
        if y > H - 40:
            break
        jitter = random.randint(-1, 1)
        draw.text((x0, y + jitter), line, fill=ink, font=font)

    out_path = SYNTH_DIR / f"handwritten_{idx:04d}.png"
    img.save(out_path)
    return out_path


# ──────────────────────────────────────────────────────────────────── #
#  2. Try to pull IAM sample subset via HuggingFace datasets           #
# ──────────────────────────────────────────────────────────────────── #

def try_download_iam_sample():
    try:
        from datasets import load_dataset
        print("Downloading IAM handwriting sample (HuggingFace)…")
        ds = load_dataset("Teklia/IAM-line", split="test", streaming=True, trust_remote_code=True)
        count = 0
        for item in ds:
            if count >= 20:
                break
            img = item.get("image")
            label = item.get("text", "unknown")
            if img:
                out = SAMPLES_DIR / f"iam_{count:04d}.png"
                img.save(out)
                print(f"  Saved {out.name}: {label[:40]}")
            count += 1
        print(f"  Downloaded {count} IAM samples.")
        return True
    except Exception as e:
        print(f"  IAM download skipped: {e}")
        return False


# ──────────────────────────────────────────────────────────────────── #
#  Main                                                                 #
# ──────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 60)
    print("  HandwriteAI Dataset Preparation")
    print("=" * 60)

    meta = {"samples": [], "synthetic": []}

    # 1. Synthetic
    print("\n[1/3] Generating synthetic images…")
    for i, text in enumerate(SAMPLE_TEXTS):
        p = generate_synthetic_whiteboard(text, i)
        if p:
            meta["synthetic"].append({"file": str(p), "type": "whiteboard", "text": text})
            print(f"  ✓ whiteboard_{i:04d}.png")
        p = generate_synthetic_handwritten(text, i)
        if p:
            meta["synthetic"].append({"file": str(p), "type": "handwritten", "text": text})
            print(f"  ✓ handwritten_{i:04d}.png")

    # 2. IAM sample
    print("\n[2/3] Attempting IAM dataset download…")
    try_download_iam_sample()

    # 3. Metadata
    print("\n[3/3] Writing dataset_meta.json…")
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ {META_FILE}")

    print("\n" + "=" * 60)
    print(f"  Done. Synthetic: {len(meta['synthetic'])} images")
    print(f"  Samples dir: {SAMPLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
