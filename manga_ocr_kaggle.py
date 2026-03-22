# =============================================================================
# Manga OCR — Kaggle Notebook
# Copy each "# %% [cell N]" block into a separate Kaggle cell.
# Run cells top-to-bottom.  GPU accelerator must be ON (Settings → Accelerator).
# =============================================================================


# %% [cell 1]  ── Setup: verify GPU, install missing packages ──────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# These are NOT pre-installed on Kaggle — install them first
pip("ultralytics")     # YOLOv8
pip("editdistance")    # For CER/WER evaluation
pip("manga-ocr")       # Optional: pre-trained baseline to compare against

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("GPU name       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)")

# Kaggle gives you a T4 (16 GB) or P100 (16 GB).
# If you see "None" here, go to:  Settings (right panel) → Accelerator → GPU T4 x1


# %% [cell 2]  ── Config: all Kaggle paths and hyperparameters ────────────────
import os

# ── Kaggle-specific paths ──────────────────────────────────────────────────────
# /kaggle/input/   = read-only; your datasets live here after you "Add Data"
# /kaggle/working/ = writable; save all outputs here
#
# HOW TO ADD A DATASET:
#   Right panel → Add Data → search "Roboflow Manga Text Detection" (or upload your own)
#   It appears at /kaggle/input/<dataset-slug>/

INPUT_DIR   = "/kaggle/input"
WORKING_DIR = "/kaggle/working"

# Update this slug to match your added dataset's folder name
DATASET_SLUG     = "manga-text-detection"          # change me
DATASET_ROOT     = os.path.join(INPUT_DIR, DATASET_SLUG)

# Where we save trained models (persists until session ends — download before closing!)
MODELS_DIR       = os.path.join(WORKING_DIR, "models")
DETECT_MODEL_DIR = os.path.join(MODELS_DIR,  "detect")
RECOG_MODEL_DIR  = os.path.join(MODELS_DIR,  "recog")
LOG_DIR          = os.path.join(WORKING_DIR, "logs")

for d in [MODELS_DIR, DETECT_MODEL_DIR, RECOG_MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Image sizes ────────────────────────────────────────────────────────────────
IMG_SIZE     = 640    # Detection model input (square)
RECOG_H      = 32     # Recognition crop height (fixed)
RECOG_W      = 128    # Recognition crop width  (fixed)

# ── Character set ──────────────────────────────────────────────────────────────
# English only to start. Add Japanese later:
#   pip install jaconv   →   from jaconv import h2z
#   Then append hiragana/katakana strings here.
CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
NUM_CLASSES = len(CHARSET) + 1    # +1 for CTC blank (index 0)

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE        = 16
EPOCHS_DETECT     = 30      # Reduced from 50 — Kaggle session is 9 hrs max
EPOCHS_RECOG      = 60      # Reduced from 100
LR                = 1e-3
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ── Detection thresholds ───────────────────────────────────────────────────────
CONF_THRESHOLD    = 0.4
IOU_THRESHOLD     = 0.5

# ── Checkpoint: resume from last saved epoch if session was interrupted ────────
RESUME_DETECT     = False   # Set True to resume detection training
RESUME_RECOG      = False   # Set True to resume recognition training

print(f"Device: {DEVICE}")
print(f"Dataset path: {DATASET_ROOT}")
print(f"Models will be saved to: {MODELS_DIR}")


# %% [cell 3]  ── Download dataset from Roboflow (optional, needs internet ON) ──
# Skip this cell if you added the dataset via the Kaggle "Add Data" button.
#
# To enable internet: Settings (right panel) → Internet → On
#
# Get your Roboflow API key: roboflow.com → account settings → API keys

ROBOFLOW_API_KEY  = "PASTE_YOUR_KEY_HERE"   # ← replace this
ROBOFLOW_WORKSPACE = "ocr-9ocgg"
ROBOFLOW_PROJECT   = "manga-text-detection-xyvbw-iipaw"
ROBOFLOW_VERSION   = 1

if ROBOFLOW_API_KEY != "PASTE_YOUR_KEY_HERE":
    pip("roboflow")
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download(
        "yolov8",
        location=os.path.join(WORKING_DIR, "dataset")
    )
    DATASET_ROOT = dataset.location
    print(f"Downloaded to: {DATASET_ROOT}")
else:
    print("Skipped — using dataset from /kaggle/input/")


# %% [cell 4]  ── Preprocessing helpers ────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold — handles uneven brightness in manga scans."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=8
    )


def preprocess_for_recognition(crop: np.ndarray,
                                 target_h=RECOG_H,
                                 target_w=RECOG_W) -> np.ndarray:
    """
    Resize a cropped text region to (target_h, target_w).
    Keeps aspect ratio, pads the right side with white.
    """
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = crop.shape
    new_w = min(int(w * (target_h / h)), target_w)
    crop  = cv2.resize(crop, (new_w, target_h))

    if new_w < target_w:
        pad  = np.full((target_h, target_w - new_w), 255, dtype=np.uint8)
        crop = np.hstack([crop, pad])

    return crop.astype(np.float32) / 255.0


def preview_preprocessing(image_path: str):
    """Show before/after in the notebook output."""
    import matplotlib.pyplot as plt

    orig  = load_image(image_path)
    gray  = to_grayscale(orig)
    binz  = binarize(gray)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes,
                               [orig[:, :, ::-1], gray, binz],
                               ["Original", "Grayscale", "Binarized"]):
        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.suptitle("Preprocessing stages", fontsize=14)
    plt.tight_layout()
    plt.show()

# Test on the first image in the dataset (edit the path as needed)
sample_images = []
for root, _, files in os.walk(DATASET_ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            sample_images.append(os.path.join(root, f))
            break
    if sample_images:
        break

if sample_images:
    preview_preprocessing(sample_images[0])
    print(f"Sample image: {sample_images[0]}")
else:
    print("No images found in dataset path. Check DATASET_ROOT.")


# %% [cell 5]  ── Dataset class (detection) ────────────────────────────────────
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARSET)}   # 0 = CTC blank
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARSET)}


def encode_text(text: str) -> list:
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def decode_indices(indices: list) -> str:
    """CTC greedy decode: remove blanks (0) and consecutive duplicates."""
    result, prev = [], None
    for idx in indices:
        if idx != 0 and idx != prev:
            result.append(IDX_TO_CHAR.get(idx, "?"))
        prev = idx
    return "".join(result)


class MangaRecognitionDataset(Dataset):
    """
    Expects a JSON file:
      [{"image": "path/to/crop.png", "text": "Hello"}, ...]
    """
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry  = self.samples[idx]
        img    = cv2.imread(entry["image"], cv2.IMREAD_GRAYSCALE)
        img    = preprocess_for_recognition(img)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        label  = encode_text(entry["text"])
        return tensor, torch.tensor(label, dtype=torch.long), len(label)


def collate_recognition(batch):
    images, labels, lengths = zip(*batch)
    return (
        torch.stack(images),
        torch.cat(labels),
        torch.tensor(lengths, dtype=torch.long)
    )


print("Dataset classes defined.")


# %% [cell 6]  ── CRNN recognition model ──────────────────────────────────────
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2),
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
    CNN + Bidirectional LSTM + CTC.
    Input : (batch, 1, 32, 128)
    Output: (T, batch, num_classes)  — for CTCLoss
    """
    def __init__(self, num_classes: int, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1,   64,  pool=(2, 2)),
            ConvBlock(64,  128, pool=(2, 2)),
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=(2, 1)),
            ConvBlock(256, 512),
            ConvBlock(512, 512, pool=(2, 1)),
            ConvBlock(512, 512, pool=(2, 1)),
        )
        self.rnn = nn.LSTM(
            512, rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        f = self.cnn(x)           # (B, 512, 1, W')
        f = f.squeeze(2)          # (B, 512, W')
        f = f.permute(2, 0, 1)    # (W', B, 512)
        out, _ = self.rnn(f)      # (W', B, hidden*2)
        return self.fc(out)       # (W', B, num_classes)


model_test = CRNN(NUM_CLASSES).to(DEVICE)
dummy      = torch.randn(2, 1, RECOG_H, RECOG_W).to(DEVICE)
out        = model_test(dummy)
print(f"CRNN output shape: {out.shape}")   # Should be (T, 2, NUM_CLASSES)
del model_test, dummy, out


# %% [cell 7]  ── Train detection model (YOLOv8) ───────────────────────────────
# YOLOv8 needs a data.yaml file.  If your Roboflow download created one, use it.
# Otherwise create it below.

from ultralytics import YOLO
import yaml

# Auto-detect data.yaml from the dataset folder
data_yaml_path = None
for root, _, files in os.walk(DATASET_ROOT):
    for f in files:
        if f == "data.yaml":
            data_yaml_path = os.path.join(root, f)
            break

if data_yaml_path is None:
    # Create a minimal data.yaml (update paths if your folder differs)
    data_yaml_path = os.path.join(WORKING_DIR, "data.yaml")
    yaml_content = {
        "train": os.path.join(DATASET_ROOT, "train", "images"),
        "val":   os.path.join(DATASET_ROOT, "valid", "images"),
        "nc":    1,
        "names": ["text_region"],
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    print(f"Created data.yaml: {data_yaml_path}")
else:
    print(f"Found data.yaml: {data_yaml_path}")

# ── Resume support ─────────────────────────────────────────────────────────────
detect_weights = "yolov8n.pt"   # Start from pre-trained
best_detect    = os.path.join(DETECT_MODEL_DIR, "weights", "best.pt")

if RESUME_DETECT and os.path.exists(best_detect):
    detect_weights = best_detect
    print(f"Resuming detection training from: {best_detect}")
else:
    print("Starting detection training from pre-trained yolov8n.pt")

# ── Train ──────────────────────────────────────────────────────────────────────
detect_model = YOLO(detect_weights)

results = detect_model.train(
    data    = data_yaml_path,
    epochs  = EPOCHS_DETECT,
    imgsz   = IMG_SIZE,
    batch   = BATCH_SIZE,
    patience= 10,
    save    = True,
    project = MODELS_DIR,
    name    = "detect",
    device  = "0" if DEVICE == "cuda" else "cpu",
    augment = True,
    # ── Kaggle-specific: save every 5 epochs so you can recover if session ends ──
    save_period = 5,
)

print(f"\nDetection training complete.")
print(f"Best model saved at: {os.path.join(MODELS_DIR, 'detect', 'weights', 'best.pt')}")


# %% [cell 8]  ── Train recognition model (CRNN) with AMP ────────────────────
#
# KEY KAGGLE CHANGE: Mixed-precision training (AMP)
#   - torch.cuda.amp.autocast()   — runs forward pass in float16 (faster, less VRAM)
#   - GradScaler                  — scales gradients to avoid float16 underflow
#   - Effect: ~2x faster training, ~40% less VRAM on T4
#
# KEY KAGGLE CHANGE: Checkpoint saving every N epochs
#   If your session ends at hour 8, you can set RESUME_RECOG=True and continue.

from torch.utils.data import random_split
from torch.nn.utils import clip_grad_norm_
from tqdm.notebook import tqdm    # tqdm.notebook renders a progress bar in Kaggle

# ── Load dataset ───────────────────────────────────────────────────────────────
# Change this path to your recognition JSON file.
# If you don't have one yet, skip this cell and use the pre-trained manga-ocr baseline.
RECOG_JSON = os.path.join(WORKING_DIR, "recognition_data.json")

if not os.path.exists(RECOG_JSON):
    print(f"No recognition dataset found at: {RECOG_JSON}")
    print("Skipping recognition training.")
    print("Tip: Use the pre-trained manga-ocr model in cell 10 as a baseline.")
else:
    dataset  = MangaRecognitionDataset(RECOG_JSON)
    val_size = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              collate_fn=collate_recognition, num_workers=2,
                              pin_memory=True)    # pin_memory=True speeds up GPU transfer
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              collate_fn=collate_recognition, num_workers=2,
                              pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    crnn      = CRNN(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True)

    # ── AMP (mixed precision) — Kaggle-specific ────────────────────────────────
    use_amp = (DEVICE == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_epoch    = 1
    best_val_loss  = float("inf")
    checkpoint_path = os.path.join(RECOG_MODEL_DIR, "checkpoint_latest.pt")

    if RESUME_RECOG and os.path.exists(checkpoint_path):
        ckpt        = torch.load(checkpoint_path, map_location=DEVICE)
        crnn.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}  (best val loss: {best_val_loss:.4f})")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS_RECOG + 1):
        # Train
        crnn.train()
        train_loss = 0.0

        for imgs, labels, label_lens in tqdm(train_loader,
                                              desc=f"Epoch {epoch}/{EPOCHS_RECOG} [train]",
                                              leave=False):
            imgs       = imgs.to(DEVICE, non_blocking=True)
            labels     = labels.to(DEVICE, non_blocking=True)
            label_lens = label_lens.to(DEVICE, non_blocking=True)

            # ── AMP forward pass ───────────────────────────────────────────────
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits    = crnn(imgs)                          # (T, N, C)
                log_probs = logits.log_softmax(2)
                input_lens = torch.full(
                    (imgs.size(0),), logits.size(0), dtype=torch.long, device=DEVICE
                )
                loss = ctc_loss(log_probs, labels, input_lens, label_lens)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(crnn.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        crnn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, label_lens in val_loader:
                imgs       = imgs.to(DEVICE, non_blocking=True)
                labels     = labels.to(DEVICE, non_blocking=True)
                label_lens = label_lens.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits    = crnn(imgs)
                    log_probs = logits.log_softmax(2)
                    input_lens = torch.full(
                        (imgs.size(0),), logits.size(0), dtype=torch.long, device=DEVICE
                    )
                    val_loss += ctc_loss(log_probs, labels, input_lens, label_lens).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS_RECOG}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(RECOG_MODEL_DIR, "crnn_best.pt")
            torch.save(crnn.state_dict(), best_path)
            print(f"  ✓ New best saved → {best_path}")

        # ── Checkpoint every 10 epochs (Kaggle session safety) ────────────────
        if epoch % 10 == 0:
            torch.save({
                "epoch":         epoch,
                "model":         crnn.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scaler":        scaler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"  [checkpoint saved] epoch {epoch}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {os.path.join(RECOG_MODEL_DIR, 'crnn_best.pt')}")


# %% [cell 9]  ── End-to-end OCR pipeline ─────────────────────────────────────
class MangaOCR:
    def __init__(self, detect_model_path: str, recog_model_path: str):
        self.detector = YOLO(detect_model_path)
        self.recognizer = CRNN(NUM_CLASSES).to(DEVICE)
        self.recognizer.load_state_dict(
            torch.load(recog_model_path, map_location=DEVICE)
        )
        self.recognizer.eval()
        print("Models loaded.")

    def detect(self, image: np.ndarray) -> list:
        results = self.detector(image, conf=CONF_THRESHOLD)[0]
        return [
            (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
             int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0]))
            for b in results.boxes
        ]

    def recognize(self, crop: np.ndarray) -> str:
        img = preprocess_for_recognition(crop)
        t   = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
        with torch.no_grad():
            logits = self.recognizer(t)
            preds  = logits.softmax(2).argmax(2).squeeze(1).tolist()
        return decode_indices(preds)

    def sort_reading_order(self, boxes: list) -> list:
        """Right-to-left, top-to-bottom (standard manga reading order)."""
        if not boxes:
            return boxes
        sorted_boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        rows, cur = [], [sorted_boxes[0]]
        for box in sorted_boxes[1:]:
            cy      = (box[1] + box[3]) / 2
            last_cy = (cur[-1][1] + cur[-1][3]) / 2
            if abs(cy - last_cy) < 60:
                cur.append(box)
            else:
                rows.append(cur)
                cur = [box]
        rows.append(cur)
        ordered = []
        for row in rows:
            row.sort(key=lambda b: b[0], reverse=True)   # right → left
            ordered.extend(row)
        return ordered

    def run(self, image_path: str) -> list:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        boxes   = self.detect(image)
        boxes   = self.sort_reading_order(boxes)
        results = []
        for (x1, y1, x2, y2, conf) in boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results.append({
                "box":  (x1, y1, x2, y2),
                "conf": conf,
                "text": self.recognize(crop),
            })
        return results

    def visualize(self, image_path: str, results: list,
                  save_path: str = None):
        import matplotlib.pyplot as plt
        img = cv2.imread(image_path)[:, :, ::-1].copy()   # BGR → RGB
        for item in results:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(img, item["text"], (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("OCR results")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()

print("MangaOCR class defined.")


# %% [cell 10]  ── Quick baseline with pre-trained manga-ocr ──────────────────
# Use this BEFORE your own models are trained to see what good output looks like.
# manga-ocr is optimised for Japanese text but works on English too.

from manga_ocr import MangaOcr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

mocr = MangaOcr()    # Downloads ~400 MB model on first run

def run_baseline(image_path: str):
    """Detect with your YOLO model, recognize with manga-ocr baseline."""
    detect_path = os.path.join(MODELS_DIR, "detect", "weights", "best.pt")

    if not os.path.exists(detect_path):
        print("Detection model not trained yet. Run cell 7 first.")
        return

    yolo  = YOLO(detect_path)
    image = cv2.imread(image_path)
    results_raw = yolo(image, conf=CONF_THRESHOLD)[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image[:, :, ::-1])

    for box in results_raw.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop  = image[y1:y2, x1:x2]
        text  = mocr(Image.fromarray(crop[:, :, ::-1]))

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, text, fontsize=8, color="lime",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))

    ax.set_axis_off()
    out_path = os.path.join(WORKING_DIR, "baseline_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {out_path}")

# Run on a sample — change the path to your image
if sample_images:
    run_baseline(sample_images[0])


# %% [cell 11]  ── Evaluate: CER and WER ─────────────────────────────────────
import editdistance


def cer(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    return editdistance.eval(pred, truth) / len(truth)


def wer(pred: str, truth: str) -> float:
    p, t = pred.split(), truth.split()
    if not t:
        return 0.0
    return editdistance.eval(p, t) / len(t)


def evaluate(ocr_instance: MangaOCR, test_json: str):
    """
    test_json format:
      [{"image": "path.jpg", "texts": ["bubble1 text", "bubble2 text"]}, ...]
    """
    with open(test_json) as f:
        data = json.load(f)

    all_cer, all_wer = [], []

    for item in tqdm(data, desc="Evaluating"):
        results = ocr_instance.run(item["image"])
        preds   = [r["text"] for r in results]
        for pred, truth in zip(preds, item["texts"]):
            all_cer.append(cer(pred, truth))
            all_wer.append(wer(pred, truth))

    avg_cer = sum(all_cer) / len(all_cer) if all_cer else 0
    avg_wer = sum(all_wer) / len(all_wer) if all_wer else 0

    print(f"Character Error Rate (CER): {avg_cer:.4f}  ({avg_cer*100:.1f}% errors)")
    print(f"Word Error Rate      (WER): {avg_wer:.4f}  ({avg_wer*100:.1f}% errors)")
    print("Target for a first model: CER < 0.15 (15%)")
    return avg_cer, avg_wer

print("Evaluation functions defined.")


# %% [cell 12]  ── IMPORTANT: save your models before session ends ─────────────
#
# Kaggle wipes /kaggle/working/ when the session closes.
# Run this cell to zip everything up — the output appears in the
# "Output" tab on the right, and you can download it from there.
#
# Alternative: go to  File → Save & Run All  which commits all outputs
# to a Kaggle dataset version automatically.

import shutil

zip_path = os.path.join(WORKING_DIR, "manga_ocr_models")
shutil.make_archive(zip_path, "zip", MODELS_DIR)
print(f"Models zipped: {zip_path}.zip")
print()
print("Download instructions:")
print("  Option A: Right panel → Output tab → find the .zip → Download")
print("  Option B: File menu → Save & Run All  (commits outputs to Kaggle)")
print()
print("Files saved:")
for root, _, files in os.walk(MODELS_DIR):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1e6
        print(f"  {path.replace(WORKING_DIR, '')}  ({size:.1f} MB)")
