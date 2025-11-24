"""Loads VOCDataset for object detection."""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path

from config import CurrentConfig


class VOCDataset(Dataset):
    """
    PyTorch Dataset for PASCAL VOC in YOLO format.

    Loads images and labels from paths in a csv file. Converts bounding boxes
    into YOLO's grid-based label matrix form.

    Args:
        csv_path: Path to CSV with columns [image_filename, label_filename].
        config: CurrentConfig instance for img_dir/label_dir paths.
        S: Grid size (the image is divided into SxS cells).
        B: Number of bounding boxes predicted per cell.
        C: Number of object classes.
        transform: Optional torchvision transform for images.
    """

    def __init__(
        self,
        csv_path: Path,
        config: CurrentConfig | None = None,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        transform=None,
    ):
        self.config = config or CurrentConfig()
        self.annotations = pd.read_csv(csv_path)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Returns:
            image: Transformed image tensor.
            label_matrix: Tensor of shape (S, S, C + 5*B) encoding class probs
                and bounding boxes for each grid cell.
        """
        label_path = self.config.label_dir / self.annotations.iloc[index, 1]
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([int(class_label), x, y, w, h])

        img_path = self.config.img_dir / self.annotations.iloc[index, 0]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, w, h = box
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                label_matrix[i, j, 21:25] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
