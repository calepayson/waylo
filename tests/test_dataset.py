"""Tests for VOCDataset."""

import pytest
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import VOCDataset
from config import CurrentConfig


@pytest.fixture
def tmp_voc_data(tmp_path):
    """Create minimal VOC-style dataset for testing."""
    img_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    img_dir.mkdir()
    label_dir.mkdir()

    # Create dummy image (100x100 RGB)
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_dir / "test.jpg")

    # Create label: class 5 at center (0.5, 0.5) with w=0.2, h=0.3
    (label_dir / "test.txt").write_text("5 0.5 0.5 0.2 0.3\n")

    # Create CSV
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("image,label\ntest.jpg,test.txt\n")

    return tmp_path, csv_path


@pytest.fixture
def config(tmp_voc_data):
    """CurrentConfig pointing to temp data."""
    tmp_path, _ = tmp_voc_data
    cfg = CurrentConfig()
    cfg.data_root = tmp_path
    return cfg


class TestVOCDataset:
    def test_len(self, tmp_voc_data, config):
        _, csv_path = tmp_voc_data
        ds = VOCDataset(csv_path, config=config)
        assert len(ds) == 1

    def test_getitem_shapes(self, tmp_voc_data, config):
        _, csv_path = tmp_voc_data
        transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
            ]
        )
        ds = VOCDataset(csv_path, config=config, transform=transform)
        image, label_matrix = ds[0]

        assert image.shape == (3, 448, 448)
        assert label_matrix.shape == (7, 7, 30)  # S=7, C + 5*B = 20 + 10

    def test_label_encoding(self, tmp_voc_data, config):
        _, csv_path = tmp_voc_data
        ds = VOCDataset(csv_path, config=config)
        _, label_matrix = ds[0]

        # Box at (0.5, 0.5) -> cell (3, 3) for S=7
        i, j = 3, 3

        # Objectness should be 1
        assert label_matrix[i, j, 20] == 1

        # Class 5 should be one-hot
        assert label_matrix[i, j, 5] == 1
        assert label_matrix[i, j, :20].sum() == 1  # Only one class active

        # Box coords: x_cell, y_cell should be 0.5 (center of cell)
        # w_cell, h_cell should be 0.2*7=1.4, 0.3*7=2.1
        assert torch.isclose(label_matrix[i, j, 21], torch.tensor(0.5), atol=0.01)
        assert torch.isclose(label_matrix[i, j, 22], torch.tensor(0.5), atol=0.01)
        assert torch.isclose(label_matrix[i, j, 23], torch.tensor(1.4), atol=0.01)
        assert torch.isclose(label_matrix[i, j, 24], torch.tensor(2.1), atol=0.01)

    def test_empty_cells_are_zero(self, tmp_voc_data, config):
        _, csv_path = tmp_voc_data
        ds = VOCDataset(csv_path, config=config)
        _, label_matrix = ds[0]

        # Cell (0, 0) should have no object
        assert label_matrix[0, 0, 20] == 0
        assert label_matrix[0, 0, :20].sum() == 0

    def test_no_transform(self, tmp_voc_data, config):
        _, csv_path = tmp_voc_data
        ds = VOCDataset(csv_path, config=config, transform=None)
        image, _ = ds[0]

        # Should return PIL Image when no transform
        assert isinstance(image, Image.Image)
