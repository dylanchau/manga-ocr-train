# Manga OCR — Complete Beginner Guide

## Project Structure

```
manga_ocr/
├── README.md
├── requirements.txt
├── config.py                  # All settings in one place
├── data/
│   ├── raw/                   # Your manga images go here
│   ├── annotated/             # Labeled data (bounding boxes + text)
│   └── processed/             # Cleaned, ready-to-train data
├── src/
│   ├── preprocess.py          # Image cleaning & preparation
│   ├── dataset.py             # PyTorch Dataset class
│   ├── detect/
│   │   ├── model.py           # Text detection model (YOLOv8)
│   │   └── train.py           # Train the detector
│   ├── recognize/
│   │   ├── model.py           # Text recognition model (CRNN)
│   │   └── train.py           # Train the recognizer
│   └── pipeline.py            # End-to-end OCR pipeline
├── tools/
│   ├── annotate.py            # Helper to label your images
│   └── evaluate.py            # Measure accuracy
└── inference.py               # Run OCR on a new image
```

---

## Step 0 — Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics          # YOLOv8 (detection)
pip install opencv-python        # Image processing
pip install Pillow               # Image I/O
pip install numpy pandas         # Data helpers
pip install matplotlib           # Visualize results
pip install editdistance         # Measure OCR accuracy
pip install labelme              # Annotation tool (GUI)
```

---

## config.py — Central settings

```python
# config.py
import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_ANNOTATED= os.path.join(BASE_DIR, "data", "annotated")
DATA_PROCESSED= os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

# ── Image settings ─────────────────────────────────────────────────────
IMG_HEIGHT    = 640          # Detection model input height
IMG_WIDTH     = 640          # Detection model input width

RECOG_HEIGHT  = 32           # Recognition model input height (fixed)
RECOG_WIDTH   = 128          # Recognition model input width  (fixed)

# ── Characters the model can read ─────────────────────────────────────
# Add Japanese/Chinese characters if your manga uses them
CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
    # Japanese example (add more as needed):
    # "あいうえおかきくけこ..."
)
BLANK_IDX = 0                # CTC blank token index (must be 0)
NUM_CLASSES = len(CHARSET) + 1   # +1 for CTC blank

# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE    = 16
EPOCHS_DETECT = 50
EPOCHS_RECOG  = 100
LR            = 1e-3
DEVICE        = "cuda"       # Change to "cpu" if no GPU

# ── Detection confidence ───────────────────────────────────────────────
CONF_THRESHOLD  = 0.4
IOU_THRESHOLD   = 0.5
```

---

## src/preprocess.py — Clean your images

```python
# src/preprocess.py
"""
Prepares manga images for training.

What this does:
  1. Converts to grayscale (manga is usually black & white)
  2. Applies adaptive thresholding to separate ink from paper
  3. Resizes to a standard size
  4. Normalizes pixel values to [0, 1]
"""

import cv2
import numpy as np
from PIL import Image
import os


def load_image(path: str) -> np.ndarray:
    """Load image from disk as a NumPy array (BGR format)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding: makes text pure black, background pure white.
    Better than a fixed threshold because manga scans vary in brightness.
    """
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,    # Neighbourhood size (must be odd)
        C=8              # Constant subtracted from mean
    )


def denoise(img: np.ndarray) -> np.ndarray:
    """Remove small noise dots from scanned manga pages."""
    # fastNlMeansDenoising only works on grayscale
    return cv2.fastNlMeansDenoising(img, h=10)


def resize_for_detection(img: np.ndarray, size=(640, 640)) -> np.ndarray:
    """Resize image to the detection model's expected input size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0,255] to [0,1] as float32."""
    return img.astype(np.float32) / 255.0


def preprocess_for_recognition(crop: np.ndarray, target_h=32, target_w=128) -> np.ndarray:
    """
    Prepare a cropped text region for the recognition model.
    
    - Convert to grayscale
    - Resize to fixed height while keeping aspect ratio
    - Pad width to target_w if needed
    """
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = crop.shape
    new_w = int(w * (target_h / h))          # Scale width proportionally
    new_w = min(new_w, target_w)             # Cap at max width
    crop = cv2.resize(crop, (new_w, target_h))

    # Pad on the right to reach target_w
    if new_w < target_w:
        pad = np.full((target_h, target_w - new_w), 255, dtype=np.uint8)
        crop = np.hstack([crop, pad])

    return normalize(crop)


def full_pipeline(image_path: str) -> dict:
    """
    Run all preprocessing steps and return a dict with each result.
    Useful for debugging — you can visualize each stage.
    """
    original = load_image(image_path)
    gray     = to_grayscale(original)
    denoised = denoise(gray)
    binary   = binarize(denoised)
    resized  = resize_for_detection(binary)
    normalized = normalize(resized)

    return {
        "original":   original,
        "gray":       gray,
        "denoised":   denoised,
        "binary":     binary,
        "resized":    resized,
        "normalized": normalized,
    }


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, matplotlib.pyplot as plt

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.jpg"
    stages = full_pipeline(path)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["original", "gray", "binary", "resized"]
    for ax, key in zip(axes, titles):
        img = stages[key]
        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(key)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("preprocess_stages.png")
    print("Saved: preprocess_stages.png")
```

---

## src/dataset.py — Feed data to the model

```python
# src/dataset.py
"""
PyTorch Dataset classes.

Why do we need this?
  PyTorch models learn from batches of images.
  A Dataset object tells PyTorch: "here's how to load one example".
  A DataLoader then batches them automatically.
"""

import os, json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from config import CHARSET, RECOG_HEIGHT, RECOG_WIDTH
from src.preprocess import preprocess_for_recognition


# ── Helpers for encoding / decoding text ───────────────────────────────

CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARSET)}   # blank=0
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARSET)}

def encode_text(text: str) -> list[int]:
    """Convert a string to a list of integer indices."""
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]

def decode_indices(indices: list[int]) -> str:
    """Convert a list of indices back to a string (removes blanks & repeats)."""
    result = []
    prev = None
    for idx in indices:
        if idx != 0 and idx != prev:       # 0 = CTC blank
            result.append(IDX_TO_CHAR.get(idx, "?"))
        prev = idx
    return "".join(result)


# ── Detection Dataset ───────────────────────────────────────────────────

class MangaDetectionDataset(Dataset):
    """
    Dataset for the TEXT DETECTION model.
    
    Expects a folder with:
      - images/   (*.jpg or *.png)
      - labels/   (*.txt  — YOLO format: class cx cy w h, all normalized)
    
    YOLO label format (one line per text box):
      0 0.512 0.340 0.180 0.095
      ^class  ^center_x  ^center_y  ^width  ^height
      (all values are 0–1, relative to image size)
    """

    def __init__(self, root: str, img_size=640, augment=False):
        self.img_dir   = os.path.join(root, "images")
        self.label_dir = os.path.join(root, "labels")
        self.img_size  = img_size
        self.augment   = augment

        # Collect all image paths that have a matching label file
        self.samples = [
            f for f in os.listdir(self.img_dir)
            if f.endswith((".jpg", ".png")) and
               os.path.exists(os.path.join(self.label_dir, f.replace(".jpg", ".txt").replace(".png", ".txt")))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        img_path   = os.path.join(self.img_dir, fname)
        label_path = os.path.join(self.label_dir, fname.rsplit(".", 1)[0] + ".txt")

        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # HWC → CHW

        # Load labels (each row: class cx cy w h)
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(p) for p in parts])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

        return img, boxes


# ── Recognition Dataset ─────────────────────────────────────────────────

class MangaRecognitionDataset(Dataset):
    """
    Dataset for the TEXT RECOGNITION model.
    
    Expects a JSON file with entries like:
      [{"image": "crops/001.png", "text": "Hello world"}, ...]
    
    Images should be pre-cropped text regions.
    """

    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        img = preprocess_for_recognition(img, RECOG_HEIGHT, RECOG_WIDTH)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)   # (1, H, W)

        label = encode_text(entry["text"])
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor, len(label)
```

---

## src/detect/model.py — Text detection with YOLOv8

```python
# src/detect/model.py
"""
Text Detection Model using YOLOv8.

What is YOLOv8?
  YOLO = "You Only Look Once". It's a fast, accurate model that
  draws bounding boxes around objects in images.
  Here, our "object" is a speech bubble or text region.

We use Ultralytics YOLOv8 which is pre-trained on general objects
(cars, people, etc.) and fine-tune it to detect manga text boxes.
This is called TRANSFER LEARNING — we borrow a brain that already
knows shapes and edges, and teach it a new specific task.
"""

from ultralytics import YOLO
import os
from config import MODELS_DIR


def get_detection_model(pretrained=True):
    """
    Load a YOLOv8 model.
    
    pretrained=True  → Start from Ultralytics pre-trained weights (recommended)
    pretrained=False → Start from scratch (needs much more data)
    """
    if pretrained:
        # 'yolov8n' = nano (fastest, smallest). Options: n, s, m, l, x
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO("yolov8n.yaml")    # Architecture only, random weights
    return model


def train_detection(data_yaml: str, epochs=50, img_size=640):
    """
    Fine-tune YOLOv8 on your manga dataset.
    
    data_yaml must be a file like:
    
      train: data/processed/train/images
      val:   data/processed/val/images
      nc: 1                    # number of classes (just "text")
      names: ["text_region"]
    """
    model = get_detection_model(pretrained=True)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        patience=10,          # Stop early if no improvement for 10 epochs
        save=True,
        project=MODELS_DIR,
        name="detect",
        device="0",           # GPU 0; use "cpu" if no GPU
        augment=True,         # Random flips, brightness shifts, etc.
    )
    return results


def detect_text_regions(model_path: str, image_path: str, conf=0.4):
    """
    Run detection on one image.
    Returns a list of bounding boxes: [(x1,y1,x2,y2, confidence), ...]
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf)[0]

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf_score = float(box.conf[0])
        boxes.append((int(x1), int(y1), int(x2), int(y2), conf_score))
    return boxes
```

---

## src/recognize/model.py — CRNN: reading the text

```python
# src/recognize/model.py
"""
Text Recognition Model — CRNN (CNN + RNN + CTC loss).

Architecture in plain English:
  1. CNN  — Reads visual patterns from the image (strokes, curves)
  2. RNN  — Reads the sequence left-to-right (characters in order)
  3. CTC  — A special loss function that aligns predicted characters
             to the actual text without needing exact position labels

Input:  Grayscale image of shape (1, 32, 128)
Output: Sequence of character probabilities at each time step
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """A single conv → batchnorm → relu block. Stacked to build the CNN."""
    def __init__(self, in_ch, out_ch, kernel=3, pool=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """
    Full CRNN model.
    
    CNN output shape:  (batch, channels, 1, time_steps)
    After squeeze:     (batch, channels, time_steps)
    RNN input:         (time_steps, batch, channels)  — sequence first
    RNN output:        (time_steps, batch, hidden*2)
    FC output:         (time_steps, batch, num_classes)
    """

    def __init__(self, num_classes: int, img_h=32, rnn_hidden=256):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────
        # Each MaxPool2d((2,2)) halves the height.
        # After 4 pooling ops: height 32 → 2 → 1 (we need height=1 for RNN)
        self.cnn = nn.Sequential(
            ConvBlock(1,  64,  3, pool=(2, 2)),    # 32×128 → 16×64
            ConvBlock(64, 128, 3, pool=(2, 2)),    # 16×64  → 8×32
            ConvBlock(128, 256, 3),                # 8×32   → 8×32
            ConvBlock(256, 256, 3, pool=(2, 1)),   # 8×32   → 4×32
            ConvBlock(256, 512, 3),                # 4×32   → 4×32
            ConvBlock(512, 512, 3, pool=(2, 1)),   # 4×32   → 2×32
            ConvBlock(512, 512, 3, pool=(2, 1)),   # 2×32   → 1×32
        )

        # ── Bidirectional RNN ─────────────────────────────────────────
        # Bidirectional = reads left-to-right AND right-to-left.
        # Gives richer context for each character.
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,     # (seq, batch, features)
            dropout=0.3,
        )

        # ── Classifier head ───────────────────────────────────────────
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)   # *2 for bidirectional

    def forward(self, x):
        # x: (batch, 1, H, W)
        features = self.cnn(x)                    # (batch, 512, 1, W')
        features = features.squeeze(2)            # (batch, 512, W')
        features = features.permute(2, 0, 1)      # (W', batch, 512) = (seq, batch, feat)

        rnn_out, _ = self.rnn(features)           # (seq, batch, hidden*2)
        logits = self.fc(rnn_out)                 # (seq, batch, num_classes)
        return logits   # CTC loss expects this shape
```

---

## src/recognize/train.py — Train the recognizer

```python
# src/recognize/train.py
"""
Training loop for the CRNN recognition model.

Key concept — CTC Loss:
  We don't know exactly WHERE each character appears in the image.
  CTC (Connectionist Temporal Classification) loss handles this by
  exploring all valid alignments and picking the most likely one.
  This is why we don't need to annotate exact character positions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
import os

from config import (DEVICE, BATCH_SIZE, EPOCHS_RECOG, LR, 
                    NUM_CLASSES, MODELS_DIR)
from src.dataset import MangaRecognitionDataset, decode_indices
from src.recognize.model import CRNN


def collate_fn(batch):
    """
    Custom collate: pad variable-length label sequences so they can
    be batched together.
    """
    images, labels, label_lens = zip(*batch)
    images = torch.stack(images)
    label_lens = torch.tensor(label_lens, dtype=torch.long)
    labels_padded = torch.cat(labels)                  # CTC wants flat labels
    return images, labels_padded, label_lens


def train_recognition(json_path: str):
    # ── Dataset ────────────────────────────────────────────────────────
    dataset = MangaRecognitionDataset(json_path)
    val_size = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ── Model ──────────────────────────────────────────────────────────
    model = CRNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS_RECOG + 1):
        # ── Train phase ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for imgs, labels, label_lens in train_loader:
            imgs       = imgs.to(DEVICE)
            labels     = labels.to(DEVICE)
            label_lens = label_lens.to(DEVICE)

            logits = model(imgs)                        # (T, N, C)
            log_probs = logits.log_softmax(2)

            # CTC requires input lengths = number of time steps per sample
            input_lens = torch.full(
                (imgs.size(0),), logits.size(0), dtype=torch.long
            )

            loss = ctc_loss(log_probs, labels, input_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)   # Prevent exploding gradients
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation phase ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels, label_lens in val_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                label_lens = label_lens.to(DEVICE)
                logits = model(imgs)
                log_probs = logits.log_softmax(2)
                input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long)
                val_loss += ctc_loss(log_probs, labels, input_lens, label_lens).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS_RECOG}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODELS_DIR, "crnn_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
```

---

## src/pipeline.py — End-to-end OCR

```python
# src/pipeline.py
"""
Full OCR pipeline: detect text regions, then recognize text in each.

This is the main class you'll use after training both models.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO

from config import DEVICE, CONF_THRESHOLD, NUM_CLASSES, RECOG_HEIGHT, RECOG_WIDTH
from src.recognize.model import CRNN
from src.dataset import decode_indices
from src.preprocess import preprocess_for_recognition


class MangaOCR:
    def __init__(self, detect_model_path: str, recog_model_path: str):
        print("Loading detection model...")
        self.detector = YOLO(detect_model_path)

        print("Loading recognition model...")
        self.recognizer = CRNN(NUM_CLASSES).to(DEVICE)
        state = torch.load(recog_model_path, map_location=DEVICE)
        self.recognizer.load_state_dict(state)
        self.recognizer.eval()
        print("Models loaded ✓")

    def detect(self, image: np.ndarray) -> list[tuple]:
        """Return list of (x1, y1, x2, y2, confidence) bounding boxes."""
        results = self.detector(image, conf=CONF_THRESHOLD)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2, float(box.conf[0])))
        return boxes

    def recognize(self, crop: np.ndarray) -> str:
        """Run recognition on a cropped text region."""
        img = preprocess_for_recognition(crop, RECOG_HEIGHT, RECOG_WIDTH)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            logits = self.recognizer(tensor)     # (T, 1, C)
            preds  = logits.softmax(2).argmax(2) # (T, 1)
            preds  = preds.squeeze(1).tolist()   # [T]

        return decode_indices(preds)

    def sort_reading_order(self, boxes: list[tuple]) -> list[tuple]:
        """
        Sort speech bubbles in manga reading order.
        Manga is typically read right-to-left, top-to-bottom.
        We divide the page into rows, then sort each row right→left.
        """
        if not boxes:
            return boxes

        # Cluster boxes into rows by their vertical center
        boxes_sorted = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        row_height_threshold = 60   # boxes within 60px vertically = same row

        rows, current_row = [], [boxes_sorted[0]]
        for box in boxes_sorted[1:]:
            cy = (box[1] + box[3]) / 2
            last_cy = (current_row[-1][1] + current_row[-1][3]) / 2
            if abs(cy - last_cy) < row_height_threshold:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
        rows.append(current_row)

        # Within each row, sort right → left (manga direction)
        ordered = []
        for row in rows:
            row.sort(key=lambda b: b[0], reverse=True)
            ordered.extend(row)
        return ordered

    def run(self, image_path: str) -> list[dict]:
        """
        Full pipeline on one image.
        Returns a list of dicts: [{"box": (x1,y1,x2,y2), "text": "..."}, ...]
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        boxes = self.detect(image)
        boxes = self.sort_reading_order(boxes)

        results = []
        for (x1, y1, x2, y2, conf) in boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text = self.recognize(crop)
            results.append({
                "box":  (x1, y1, x2, y2),
                "conf": conf,
                "text": text,
            })

        return results

    def visualize(self, image_path: str, results: list[dict], save_path: str = None):
        """Draw bounding boxes and recognized text on the image."""
        img = cv2.imread(image_path)

        for item in results:
            x1, y1, x2, y2 = item["box"]
            text = item["text"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, text, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Saved annotated image → {save_path}")
        return img
```

---

## inference.py — Run on a new image

```python
# inference.py
"""
Usage:
  python inference.py path/to/manga_page.jpg

This loads both trained models and prints all detected text.
"""

import sys
from src.pipeline import MangaOCR


DETECT_MODEL = "models/detect/weights/best.pt"
RECOG_MODEL  = "models/crnn_best.pt"


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.jpg"

    ocr = MangaOCR(DETECT_MODEL, RECOG_MODEL)
    results = ocr.run(image_path)

    print(f"\nFound {len(results)} text regions:\n")
    for i, item in enumerate(results, 1):
        print(f"  [{i}] \"{item['text']}\"  (confidence: {item['conf']:.2f})")

    ocr.visualize(image_path, results, save_path="output_annotated.jpg")


if __name__ == "__main__":
    main()
```

---

## tools/evaluate.py — Measure accuracy

```python
# tools/evaluate.py
"""
Measures how accurate your model is using two metrics:

  CER (Character Error Rate):
    What fraction of individual characters are wrong.
    0.0 = perfect. Lower is better.

  WER (Word Error Rate):
    What fraction of words are wrong.
    0.0 = perfect. Lower is better.
"""

import editdistance
import json
from src.pipeline import MangaOCR


def cer(predicted: str, ground_truth: str) -> float:
    """Character Error Rate."""
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return editdistance.eval(predicted, ground_truth) / len(ground_truth)


def wer(predicted: str, ground_truth: str) -> float:
    """Word Error Rate."""
    pred_words = predicted.split()
    true_words = ground_truth.split()
    if len(true_words) == 0:
        return 0.0
    return editdistance.eval(pred_words, true_words) / len(true_words)


def evaluate(ocr: MangaOCR, test_json: str):
    """
    test_json format:
    [{"image": "path/to/page.jpg", "texts": ["Hello", "World", ...]}, ...]
    """
    with open(test_json) as f:
        test_data = json.load(f)

    all_cer, all_wer = [], []

    for item in test_data:
        results = ocr.run(item["image"])
        predicted_texts = [r["text"] for r in results]

        for pred, truth in zip(predicted_texts, item["texts"]):
            all_cer.append(cer(pred, truth))
            all_wer.append(wer(pred, truth))

    avg_cer = sum(all_cer) / len(all_cer) if all_cer else 0
    avg_wer = sum(all_wer) / len(all_wer) if all_wer else 0
    print(f"CER: {avg_cer:.4f}  ({avg_cer*100:.1f}% character errors)")
    print(f"WER: {avg_wer:.4f}  ({avg_wer*100:.1f}% word errors)")
```

---

## Recommended learning path (for beginners)

```
Week 1  Collect 50-100 manga images. Install all tools.
        Run labelme to draw boxes around text regions.
        Understand preprocess.py by visualizing each stage.

Week 2  Train the detection model on your labeled data.
        It only needs ~100 labeled images to start working.
        Visualize detections to see where it fails.

Week 3  Crop all detected text regions. Build the recognition dataset.
        Train the CRNN recognizer.

Week 4  Connect both models with pipeline.py.
        Run inference.py on fresh manga pages.
        Measure CER/WER. Collect more data for weak spots.
```

## Tips

- Start with English manga (simpler character set) before Japanese
- More annotated data = better accuracy, always
- A CER below 10% is a solid first target
- EasyOCR (pip install easyocr) is a great quick baseline to compare against
