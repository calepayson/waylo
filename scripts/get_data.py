# scripts/get_data.py

import shutil
import zipfile
from pathlib import Path

import kagglehub  # pip install kagglehub
import gdown      # pip install gdown


PROJECT_ROOT = Path(__file__).parent.parent

# ---------- PASCAL VOC (KaggleHub) ----------
VOC_TARGET = PROJECT_ROOT / "voc_data"

# ---------- WAYMO (preprocessed YOLO subset) ----------
WAYMO_TARGET = PROJECT_ROOT / "waymo_data"
WAYMO_ZIP = PROJECT_ROOT / "waymo_data.zip"

WAYMO_FILE_ID = "1IXqatW_53k6p5GFosTQNggXIvwjaobgI"


def download_voc():
    """Download Pascal VOC YOLO dataset via KaggleHub (only on first run)."""
    if VOC_TARGET.exists():
        print(f"[VOC] Found existing folder: {VOC_TARGET}")
        return

    print("[VOC] Downloading Pascal VOC YOLO via KaggleHub...")
    path = kagglehub.dataset_download("aladdinpersson/pascalvoc-yolo")
    shutil.move(path, VOC_TARGET)
    print("[VOC] Moved dataset to:", VOC_TARGET)


def download_waymo():
    """Download preprocessed Waymo YOLO subset from Google Drive."""
    if WAYMO_TARGET.exists():
        print(f"[Waymo] Found existing folder: {WAYMO_TARGET}")
        return

    if WAYMO_FILE_ID == "PUT_YOUR_FILE_ID_HERE":
        raise RuntimeError(
            "You must set WAYMO_FILE_ID in scripts/get_data.py "
            "to your Google Drive file ID for waymo_data.zip."
        )

    url = f"https://drive.google.com/uc?id={WAYMO_FILE_ID}"

    print("[Waymo] Downloading waymo_data.zip from:", url)
    gdown.download(url, str(WAYMO_ZIP), quiet=False)

    print("[Waymo] Extracting waymo_data.zip ...")
    with zipfile.ZipFile(WAYMO_ZIP, "r") as zf:
        zf.extractall(PROJECT_ROOT)

    if not WAYMO_TARGET.exists():
        raise RuntimeError(
            f"[Waymo] Expected {WAYMO_TARGET} after extraction. "
            f"Check the contents of {WAYMO_ZIP} – it should contain a "
            f"top-level 'waymo_data/' directory."
        )

    print("[Waymo] Extracted to:", WAYMO_TARGET)


def main():
    download_voc()
    download_waymo()


if __name__ == "__main__":
    main()
