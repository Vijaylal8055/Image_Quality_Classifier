import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageFilter

# Paths
BASE_DIR = Path("custom_dataset_hd")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Classes
CLASSES = ["normal", "blurred", "edge_cut", "half_visible"]

# Create dirs
for split in [TRAIN_DIR, VAL_DIR]:
    for cls in CLASSES:
        os.makedirs(split / cls, exist_ok=True)

# Download base dataset (Caltech101 from TensorFlow/Keras)
from tensorflow.keras.datasets import cifar10

print("📥 Loading CIFAR-10 dataset as base...")
(x_train, _), (x_test, _) = cifar10.load_data()
images = list(x_train)[:4000] + list(x_test)[:1000]  # 5000 total

def save_img(img_array, path):
    img = Image.fromarray(img_array)
    img = img.resize((256, 256))
    img.save(path)

print("⚙️ Generating augmented dataset...")
for idx, img_array in enumerate(images):
    img = Image.fromarray(img_array).resize((256, 256))

    # Pick split
    split_dir = TRAIN_DIR if random.random() > 0.2 else VAL_DIR

    # Save normal
    img.save(split_dir / "normal" / f"normal_{idx}.jpg")

    # Blurred
    img.filter(ImageFilter.GaussianBlur(radius=5)).save(split_dir / "blurred" / f"blurred_{idx}.jpg")

    # Edge-cut (crop left or right)
    w, h = img.size
    crop = img.crop((0, 0, int(w * 0.7), h))  # crop 30% right side
    new = Image.new("RGB", (w, h), (0, 0, 0))
    new.paste(crop, (0, 0))
    new.save(split_dir / "edge_cut" / f"edge_{idx}.jpg")

    # Half-visible (mask half black)
    half = img.copy()
    black = Image.new("RGB", (w // 2, h), (0, 0, 0))
    half.paste(black, (0, 0))  # mask left half
    half.save(split_dir / "half_visible" / f"half_{idx}.jpg")

print("✅ Dataset ready at:", BASE_DIR)
