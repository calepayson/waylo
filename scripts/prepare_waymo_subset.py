# scripts/prepare_waymo_subset.py
# ! IMP: Preferably run in a seperate conda env
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
# ---- CONFIG ----
# Folder where you put the extracted TFRecords from training_0000.tar
WAYMO_TFRECORD_DIR = Path("waymo_tf_records") 

# Output in YOLO-style layout (similar to voc_data)
OUT_ROOT = Path("waymo_data")
IMG_DIR = OUT_ROOT / "images"
LBL_DIR = OUT_ROOT / "labels"

OUT_ROOT.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# Use only the FRONT camera to keep life simple
CAMERA_NAME = open_dataset.CameraName.FRONT

# Map Waymo label types -> YOLO class IDs
CLASS_MAP = {
    label_pb2.Label.Type.TYPE_VEHICLE: 0,
    label_pb2.Label.Type.TYPE_PEDESTRIAN: 1,
    label_pb2.Label.Type.TYPE_CYCLIST: 2,
    label_pb2.Label.Type.TYPE_SIGN: 3,
}

MAX_FRAMES_PER_FILE = 50   # keep this small so you don't explode disk usage
TRAIN_SPLIT = 0.8


def process_tfrecord(tfrecord_path: Path, rows: list[tuple[str, str]]):
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

    for frame_idx, data in enumerate(dataset):
        if MAX_FRAMES_PER_FILE and frame_idx >= MAX_FRAMES_PER_FILE:
            break

        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        # Get FRONT camera image
        cam_image = None
        for img in frame.images:
            if img.name == CAMERA_NAME:
                cam_image = img
                break
        if cam_image is None:
            continue

        # Decode JPEG bytes to numpy
        img_tensor = tf.image.decode_jpeg(cam_image.image)
        img_np = img_tensor.numpy()
        h, w, _ = img_np.shape

        # Collect YOLO boxes for this image
        yolo_lines = []

        for cam_labels in frame.camera_labels:
            if cam_labels.name != CAMERA_NAME:
                continue

            for lab in cam_labels.labels:
                cls_id = CLASS_MAP.get(lab.type, None)
                if cls_id is None:
                    continue  # ignore unknown classes

                # Waymo 2D box: center_x, center_y, length (width in x), width (height in y) in pixels 
                cx = lab.box.center_x / w
                cy = lab.box.center_y / h
                bw = lab.box.length / w
                bh = lab.box.width / h

                # Skip super tiny / weird boxes
                if bw <= 0 or bh <= 0:
                    continue

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if not yolo_lines:
            # No usable labels for this frame → skip
            continue

        # Save image + label
        base = f"{tfrecord_path.stem}_{frame_idx:05d}"
        img_path = IMG_DIR / f"{base}.jpg"
        lbl_path = LBL_DIR / f"{base}.txt"

        Image.fromarray(img_np).save(img_path)
        with lbl_path.open("w") as f:
            f.writelines(yolo_lines)

        rows.append((img_path.name, lbl_path.name))


def main():
    tfrecords = sorted(WAYMO_TFRECORD_DIR.glob("*.tfrecord"))
    if not tfrecords:
        raise SystemExit(f"No TFRecords found in {WAYMO_TFRECORD_DIR}")

    rows: list[tuple[str, str]] = []

    for tfrecord in tfrecords:
        print(f"Processing {tfrecord} ...")
        process_tfrecord(tfrecord, rows)

    print(f"Total labeled images: {len(rows)}")

    # Shuffle + split into train/val
    random.shuffle(rows)
    split_idx = int(len(rows) * TRAIN_SPLIT)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    # Write CSVs compatible with VOCDataset
    with (OUT_ROOT / "train.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(train_rows)

    with (OUT_ROOT / "val.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(val_rows)

    print("Wrote:")
    print(f"  {OUT_ROOT / 'train.csv'}  ({len(train_rows)} rows)")
    print(f"  {OUT_ROOT / 'val.csv'}    ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
