# tools/convert_annotations.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts labelme .json annotation files into two training formats:
#
#   1. YOLO detection labels (.txt per image)
#      → used to train the text DETECTION model (YOLOv8)
#
#   2. Recognition dataset (labels.json)
#      → used to train the text RECOGNITION model (CRNN)
#      → also crops each text region and saves it as a small image
#
# Usage:
#   python tools/convert_annotations.py \
#       --input  data/raw/ \
#       --output data/processed/
#
# After running, your data/processed/ folder will look like:
#   data/processed/
#     detection/
#       train/
#         images/   ← full manga pages (symlinked or copied)
#         labels/   ← YOLO .txt files
#       val/
#         images/
#         labels/
#     recognition/
#       crops/      ← small cropped images of each text region
#       labels.json ← [{"image": "crops/001.png", "text": "こんにちは"}, ...]
#       data.yaml   ← YOLO config file (ready for YOLOv8 training)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import shutil
import argparse
import random
import cv2
import yaml


def load_labelme_json(json_path: str) -> dict:
    """Load one labelme annotation file."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def labelme_to_yolo(shape: dict, img_w: int, img_h: int) -> str:
    """
    Convert one labelme rectangle shape to a YOLO label line.

    YOLO format:  class_id  cx  cy  w  h
    All values are normalised to [0, 1] relative to image size.
    We use class_id = 0 (there is only one class: "text_region").
    """
    points = shape["points"]  # [[x1,y1], [x2,y2]] for rectangles

    x1 = min(p[0] for p in points)
    y1 = min(p[1] for p in points)
    x2 = max(p[0] for p in points)
    y2 = max(p[1] for p in points)

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    # Clamp to [0, 1] just in case a box was drawn slightly outside the image
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def crop_region(image: "np.ndarray", shape: dict,
                padding: int = 4) -> "np.ndarray":
    """
    Crop the text region from the full image with a small padding border.
    Padding helps the recogniser see character edges that touch the box boundary.
    """
    points = shape["points"]
    x1 = int(min(p[0] for p in points)) - padding
    y1 = int(min(p[1] for p in points)) - padding
    x2 = int(max(p[0] for p in points)) + padding
    y2 = int(max(p[1] for p in points)) + padding

    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return image[y1:y2, x1:x2]


def convert(input_dir: str, output_dir: str,
            val_split: float = 0.1, seed: int = 42):
    """
    Main conversion function.

    input_dir : folder containing manga .jpg/.png and their labelme .json files
    output_dir: where to write detection/ and recognition/ folders
    val_split : fraction of images to use as validation set
    """
    import numpy as np  # imported here so the script header stays clean

    random.seed(seed)

    # ── Collect all annotated image/json pairs ─────────────────────────────
    pairs = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(input_dir, json_name)
        if os.path.exists(json_path):
            pairs.append((
                os.path.join(input_dir, fname),
                json_path
            ))

    if not pairs:
        print(f"No annotated images found in: {input_dir}")
        print("Make sure each image has a matching .json file from labelme.")
        return

    print(f"Found {len(pairs)} annotated image(s).")

    # ── Train / val split ──────────────────────────────────────────────────
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_split))
    val_set = set(range(len(pairs) - n_val, len(pairs)))

    # ── Output directory structure ─────────────────────────────────────────
    det_train_img = os.path.join(output_dir, "detection", "train", "images")
    det_train_lbl = os.path.join(output_dir, "detection", "train", "labels")
    det_val_img   = os.path.join(output_dir, "detection", "val",   "images")
    det_val_lbl   = os.path.join(output_dir, "detection", "val",   "labels")
    rec_crops_dir = os.path.join(output_dir, "recognition", "crops")

    for d in [det_train_img, det_train_lbl, det_val_img, det_val_lbl, rec_crops_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Process each image ─────────────────────────────────────────────────
    recognition_entries = []
    crop_index = 0
    skipped_boxes = 0

    for idx, (img_path, json_path) in enumerate(pairs):
        is_val  = idx in val_set
        split   = "val" if is_val else "train"
        img_dir = det_val_img if is_val else det_train_img
        lbl_dir = det_val_lbl if is_val else det_train_lbl

        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [skip] Cannot read image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]
        fname_base   = os.path.splitext(os.path.basename(img_path))[0]

        # Copy image to detection split folder
        dest_img = os.path.join(img_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dest_img)

        # Load labelme annotation
        annotation = load_labelme_json(json_path)

        # Filter to rectangle shapes only (ignore polygons, etc.)
        shapes = [s for s in annotation.get("shapes", [])
                  if s["shape_type"] == "rectangle"]

        if not shapes:
            print(f"  [warn] No rectangles in: {json_path}")
            continue

        # ── Write YOLO label file ──────────────────────────────────────────
        yolo_lines = []
        for shape in shapes:
            yolo_line = labelme_to_yolo(shape, img_w, img_h)
            yolo_lines.append(yolo_line)

        label_file = os.path.join(lbl_dir, fname_base + ".txt")
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # ── Crop and save each text region for recognition ─────────────────
        for shape in shapes:
            text = shape.get("label", "").strip()

            if not text:
                # Box was drawn but no text was typed — skip
                skipped_boxes += 1
                continue

            crop = crop_region(image, shape)

            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                skipped_boxes += 1
                continue

            crop_filename = f"{crop_index:06d}.png"
            crop_path     = os.path.join(rec_crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            recognition_entries.append({
                "image": crop_path,
                "text":  text,
            })
            crop_index += 1

        print(f"  [{split}] {fname_base}: {len(shapes)} box(es)")

    # ── Write recognition labels.json ─────────────────────────────────────
    labels_json_path = os.path.join(output_dir, "recognition", "labels.json")
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(recognition_entries, f, ensure_ascii=False, indent=2)

    # ── Write YOLO data.yaml ───────────────────────────────────────────────
    data_yaml_path = os.path.join(output_dir, "detection", "data.yaml")
    yaml_content = {
        "train": os.path.abspath(det_train_img),
        "val":   os.path.abspath(det_val_img),
        "nc":    1,
        "names": ["text_region"],
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # ── Summary ────────────────────────────────────────────────────────────
    n_train = len(pairs) - n_val
    print()
    print("=" * 50)
    print("Conversion complete!")
    print(f"  Detection  — train: {n_train} images, val: {n_val} images")
    print(f"  Recognition— {len(recognition_entries)} crops saved")
    if skipped_boxes:
        print(f"  Skipped    — {skipped_boxes} boxes (empty label or bad crop)")
    print()
    print("Output files:")
    print(f"  YOLO labels  → {output_dir}/detection/")
    print(f"  data.yaml    → {data_yaml_path}")
    print(f"  labels.json  → {labels_json_path}")
    print()
    print("Next steps:")
    print("  1. Train detection : pass data.yaml to YOLOv8")
    print("  2. Train recognition: pass labels.json to CRNN trainer")


# ── Kaggle version: paste into a cell ─────────────────────────────────────────

KAGGLE_CELL = '''
# ── Run this in Kaggle after uploading your labelme .json files ───────────────
# Upload your annotated folder as a Kaggle dataset first.

import os, json, shutil, random, cv2, yaml

INPUT_DIR  = "/kaggle/input/my-manga-annotations/"   # ← your dataset slug
OUTPUT_DIR = "/kaggle/working/processed/"

# Then just call convert(INPUT_DIR, OUTPUT_DIR)
# The rest of the code is identical to the local version above.
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert labelme annotations to YOLO + recognition formats"
    )
    parser.add_argument("--input",  required=True,
                        help="Folder with manga images + labelme .json files")
    parser.add_argument("--output", required=True,
                        help="Output folder for processed data")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction for validation set (default: 0.1 = 10%%)")
    args = parser.parse_args()

    convert(args.input, args.output, args.val_split)