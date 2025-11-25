"""Tests for YoloLoss."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import YOLOConfig
from loss import YoloLoss


@pytest.fixture
def config():
    return YOLOConfig()


@pytest.fixture
def loss_fn(config):
    return YoloLoss(config)


class TestYoloLossShape:
    """Test that loss returns correct shape for various inputs."""

    def test_returns_scalar(self, loss_fn, config):
        batch_size = 4
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        predictions = torch.randn(batch_size, S * S * (C + B * 5))
        target = torch.zeros(batch_size, S, S, C + 5)

        loss = loss_fn(predictions, target)

        assert loss.shape == torch.Size([])

    def test_batch_size_one(self, loss_fn, config):
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        predictions = torch.randn(1, S * S * (C + B * 5))
        target = torch.zeros(1, S, S, C + 5)

        loss = loss_fn(predictions, target)

        assert loss.shape == torch.Size([])


class TestYoloLossValues:
    """Test loss values under specific conditions."""

    def test_zero_when_no_objects_and_zero_preds(self, loss_fn, config):
        """When no objects exist and predictions are zero, loss should be zero."""
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        predictions = torch.zeros(1, S * S * (C + B * 5))
        target = torch.zeros(1, S, S, C + 5)

        loss = loss_fn(predictions, target)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_loss_with_mismatch(self, loss_fn, config):
        """Loss should be positive when predictions don't match."""
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        predictions = torch.randn(2, S * S * (C + B * 5))
        target = torch.randn(2, S, S, C + 5).abs()
        target[..., C] = 1  # object exists

        loss = loss_fn(predictions, target)

        assert loss.item() > 0

    def test_object_loss_increases_with_confidence_error(self, config):
        """Larger confidence errors should produce larger object loss."""
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        loss_fn = YoloLoss(config)

        # Target with object at cell (0,0)
        target = torch.zeros(1, S, S, C + 5)
        target[0, 0, 0, C] = 1  # objectness
        target[0, 0, 0, C + 1 : C + 5] = torch.tensor([0.5, 0.5, 0.2, 0.2])  # box

        # Prediction with small confidence error
        pred_small = torch.zeros(1, S * S * (C + B * 5))
        pred_small = pred_small.reshape(1, S, S, C + B * 5)
        pred_small[0, 0, 0, C] = 0.9  # close to 1
        pred_small[0, 0, 0, C + 1 : C + 5] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        pred_small = pred_small.reshape(1, -1)

        # Prediction with large confidence error
        pred_large = torch.zeros(1, S * S * (C + B * 5))
        pred_large = pred_large.reshape(1, S, S, C + B * 5)
        pred_large[0, 0, 0, C] = 0.1  # far from 1
        pred_large[0, 0, 0, C + 1 : C + 5] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        pred_large = pred_large.reshape(1, -1)

        loss_small = loss_fn(pred_small, target)
        loss_large = loss_fn(pred_large, target)

        assert loss_large > loss_small


class TestYoloLossLambdaWeights:
    """Test that lambda weights affect loss correctly."""

    def test_lambda_noobj_scales_no_object_loss(self):
        """Higher lambda_noobj should increase loss when no objects."""
        config_high = YOLOConfig(lambda_noobj=1.0)
        config_low = YOLOConfig(lambda_noobj=0.1)

        loss_high = YoloLoss(config_high)
        loss_low = YoloLoss(config_low)

        S, B, C = config_high.split_size, config_high.n_boxes, config_high.n_classes
        predictions = torch.randn(2, S * S * (C + B * 5))
        target = torch.zeros(2, S, S, C + 5)  # no objects

        loss_high_val = loss_high(predictions, target)
        loss_low_val = loss_low(predictions, target)

        assert loss_high_val.item() > loss_low_val.item()

    def test_lambda_coord_scales_box_loss(self):
        """Higher lambda_coord should increase loss from box errors."""
        config_high = YOLOConfig(lambda_coord=10.0)
        config_low = YOLOConfig(lambda_coord=1.0)

        loss_high = YoloLoss(config_high)
        loss_low = YoloLoss(config_low)

        S, B, C = config_high.split_size, config_high.n_boxes, config_high.n_classes

        # Target with object
        target = torch.zeros(1, S, S, C + 5)
        target[0, 3, 3, C] = 1
        target[0, 3, 3, C + 1 : C + 5] = torch.tensor([0.5, 0.5, 0.5, 0.5])

        # Predictions with box error
        torch.manual_seed(42)
        predictions = torch.randn(1, S * S * (C + B * 5))

        loss_high_val = loss_high(predictions, target)
        loss_low_val = loss_low(predictions, target)

        assert loss_high_val.item() > loss_low_val.item()


class TestYoloLossGradients:
    """Test gradient flow through the loss."""

    def test_gradients_flow_to_predictions(self, loss_fn, config):
        """Ensure gradients flow back through predictions."""
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        predictions = torch.randn(2, S * S * (C + B * 5), requires_grad=True)
        target = torch.randn(2, S, S, C + 5).abs()
        target[..., C] = 1

        loss = loss_fn(predictions, target)
        loss.backward()

        assert predictions.grad is not None
        assert not torch.all(predictions.grad == 0)

    def test_no_nan_gradients(self, loss_fn, config):
        """Gradients should not contain NaN values."""
        S, B, C = config.split_size, config.n_boxes, config.n_classes
        predictions = torch.randn(2, S * S * (C + B * 5), requires_grad=True)
        target = torch.randn(2, S, S, C + 5).abs()
        target[..., C] = 1

        loss = loss_fn(predictions, target)
        loss.backward()

        assert not torch.isnan(predictions.grad).any()


class TestYoloLossConfig:
    """Test that config changes affect loss computation."""

    def test_different_class_counts(self):
        """Loss should work with different numbers of classes."""
        for n_classes in [5, 10, 20, 80]:
            config = YOLOConfig(n_classes=n_classes)
            loss_fn = YoloLoss(config)

            S, B, C = config.split_size, config.n_boxes, n_classes
            predictions = torch.randn(2, S * S * (C + B * 5))
            target = torch.zeros(2, S, S, C + 5)

            loss = loss_fn(predictions, target)

            assert loss.shape == torch.Size([])
            assert not torch.isnan(loss)

    def test_different_split_sizes(self):
        """Loss should work with different grid sizes."""
        for split_size in [7, 13, 19]:
            config = YOLOConfig(split_size=split_size)
            loss_fn = YoloLoss(config)

            S, B, C = split_size, config.n_boxes, config.n_classes
            predictions = torch.randn(2, S * S * (C + B * 5))
            target = torch.zeros(2, S, S, C + 5)

            loss = loss_fn(predictions, target)

            assert loss.shape == torch.Size([])
            assert not torch.isnan(loss)


class TestYoloLossBestBox:
    """Test best box selection logic."""

    def test_selects_box_with_higher_iou(self, config):
        """Should use the box with higher IoU for loss computation."""
        loss_fn = YoloLoss(config)
        S, B, C = config.split_size, config.n_boxes, config.n_classes

        # Target with object at (3,3)
        target = torch.zeros(1, S, S, C + 5)
        target[0, 3, 3, C] = 1
        target[0, 3, 3, C + 1 : C + 5] = torch.tensor([0.5, 0.5, 0.3, 0.3])

        # Pred: box1 matches well, box2 matches poorly
        pred = torch.zeros(1, S, S, C + B * 5)
        pred[0, 3, 3, C] = 0.9  # conf1
        pred[0, 3, 3, C + 1 : C + 5] = torch.tensor([0.5, 0.5, 0.3, 0.3])  # box1 exact
        pred[0, 3, 3, C + 5] = 0.9  # conf2
        pred[0, 3, 3, C + 6 : C + 10] = torch.tensor([0.0, 0.0, 0.1, 0.1])  # box2 wrong
        pred = pred.reshape(1, -1)

        loss = loss_fn(pred, target)

        # Loss should be small since box1 is selected and matches
        assert loss.item() < 1.0
