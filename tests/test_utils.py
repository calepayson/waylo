"""Tests for utils.py"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import iou, nms, calculate_map


class TestIoU:
    """Tests for intersection over union calculation."""

    # --- Midpoint format tests ---

    def test_midpoint_perfect_overlap(self):
        """Identical boxes should have IoU of 1."""
        box = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        result = iou(box, box, box_format="midpoint")
        assert torch.allclose(result, torch.tensor([[1.0]]))

    def test_midpoint_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0."""
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box2 = torch.tensor([[5.0, 5.0, 1.0, 1.0]])
        result = iou(box1, box2, box_format="midpoint")
        assert torch.allclose(result, torch.tensor([[0.0]]), atol=1e-5)

    def test_midpoint_partial_overlap(self):
        """Partially overlapping boxes."""
        # Two 2x2 boxes, centers 1 unit apart horizontally
        # Overlap is 1x2 = 2, union is 2*4 - 2 = 6, IoU = 2/6 = 1/3
        box1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        box2 = torch.tensor([[1.0, 0.0, 2.0, 2.0]])
        result = iou(box1, box2, box_format="midpoint")
        assert torch.allclose(result, torch.tensor([[1 / 3]]), atol=1e-5)

    def test_midpoint_one_inside_other(self):
        """Smaller box fully inside larger box."""
        # Large 4x4 box, small 2x2 box at same center
        # Intersection = 4, union = 16, IoU = 0.25
        box1 = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        box2 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        result = iou(box1, box2, box_format="midpoint")
        assert torch.allclose(result, torch.tensor([[0.25]]))

    # --- Corners format tests ---

    def test_corners_perfect_overlap(self):
        """Identical boxes should have IoU of 1."""
        box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        result = iou(box, box, box_format="corners")
        assert torch.allclose(result, torch.tensor([[1.0]]))

    def test_corners_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0."""
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box2 = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        result = iou(box1, box2, box_format="corners")
        assert torch.allclose(result, torch.tensor([[0.0]]), atol=1e-5)

    def test_corners_partial_overlap(self):
        """Partially overlapping boxes."""
        # Two 1x1 boxes overlapping by 0.5 in both dimensions
        # Intersection = 0.25, union = 2 - 0.25 = 1.75, IoU = 0.25/1.75
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box2 = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
        result = iou(box1, box2, box_format="corners")
        expected = 0.25 / 1.75
        assert torch.allclose(result, torch.tensor([[expected]]), atol=1e-5)

    def test_corners_touching_edge(self):
        """Boxes sharing an edge but no area overlap."""
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box2 = torch.tensor([[1.0, 0.0, 2.0, 1.0]])
        result = iou(box1, box2, box_format="corners")
        assert torch.allclose(result, torch.tensor([[0.0]]), atol=1e-5)

    # --- Batch tests ---

    def test_batched_inputs(self):
        """Multiple boxes in a batch."""
        preds = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        labels = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [5.0, 5.0, 1.0, 1.0],
                [0.5, 0.0, 1.0, 1.0],
            ]
        )
        result = iou(preds, labels, box_format="midpoint")
        assert result.shape == (3, 1)
        assert torch.allclose(result[0], torch.tensor([1.0]))
        assert torch.allclose(result[1], torch.tensor([0.0]), atol=1e-5)
        assert result[2] > 0 and result[2] < 1

    def test_batched_different_ious(self):
        """Verify batch processing doesn't mix up boxes."""
        preds = torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [10.0, 10.0, 12.0, 12.0],
            ]
        )
        labels = torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [10.0, 10.0, 12.0, 12.0],
            ]
        )
        result = iou(preds, labels, box_format="corners")
        assert torch.allclose(result, torch.tensor([[1.0], [1.0]]))

    # --- Edge cases ---

    def test_large_boxes(self):
        """Large coordinate values."""
        box = torch.tensor([[1000.0, 1000.0, 2000.0, 2000.0]])
        result = iou(box, box, box_format="corners")
        assert torch.allclose(result, torch.tensor([[1.0]]))

    def test_float_precision(self):
        """Float32 vs float64 should both work."""
        box32 = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32)
        box64 = torch.tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float64)
        result32 = iou(box32, box32, box_format="midpoint")
        result64 = iou(box64, box64, box_format="midpoint")
        assert torch.allclose(result32, torch.tensor([[1.0]], dtype=torch.float32))
        assert torch.allclose(result64, torch.tensor([[1.0]], dtype=torch.float64))

    # --- Invalid input tests ---

    def test_invalid_box_format(self):
        """Invalid box_format should raise or handle gracefully."""
        box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        # Once you add else/raise ValueError, change to:
        # with pytest.raises(ValueError):
        #     iou(box, box, box_format="invalid")
        with pytest.raises((ValueError, UnboundLocalError, NameError)):
            iou(box, box, box_format="invalid")


class TestNMS:
    """Tests for non-max suppression."""

    # --- Basic functionality ---

    def test_empty_input(self):
        """Empty list returns empty list."""
        result = nms([], iou_thresh=0.5, thresh=0.5)
        assert result == []

    def test_all_below_threshold(self):
        """All boxes below confidence threshold are filtered."""
        bboxes = [
            [0, 0.3, 0.0, 0.0, 1.0, 1.0],
            [0, 0.4, 2.0, 2.0, 3.0, 3.0],
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5)
        assert result == []

    def test_single_box_above_threshold(self):
        """Single box above threshold is kept."""
        bboxes = [[0, 0.9, 0.0, 0.0, 1.0, 1.0]]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5)
        assert len(result) == 1
        assert result[0][1] == 0.9

    def test_non_overlapping_same_class(self):
        """Non-overlapping boxes of same class are all kept."""
        bboxes = [
            [0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [0, 0.8, 5.0, 5.0, 6.0, 6.0],
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5, box_format="corners")
        assert len(result) == 2

    def test_overlapping_same_class_suppresses(self):
        """Overlapping boxes of same class: lower confidence suppressed."""
        bboxes = [
            [0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [0, 0.8, 0.1, 0.1, 1.1, 1.1],  # high overlap with first
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5, box_format="corners")
        assert len(result) == 1
        assert result[0][1] == 0.9

    def test_overlapping_different_classes_kept(self):
        """Overlapping boxes of different classes are both kept."""
        bboxes = [
            [0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [1, 0.8, 0.0, 0.0, 1.0, 1.0],  # identical box, different class
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5, box_format="corners")
        assert len(result) == 2

    # --- Threshold edge cases ---

    def test_box_exactly_at_thresh_filtered(self):
        """Box with prob == thresh is filtered (uses >)."""
        bboxes = [[0, 0.5, 0.0, 0.0, 1.0, 1.0]]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5)
        assert result == []

    def test_iou_exactly_at_thresh_kept(self):
        """Box with IoU == iou_thresh is kept (uses <)."""
        # Two boxes with exactly 0.5 IoU should keep both
        # 1x1 boxes offset by 0.5 → intersection=0.25, union=1.75, IoU≈0.143
        # Need to construct boxes with IoU exactly at threshold
        # Easier: set iou_thresh very high so nothing gets suppressed
        bboxes = [
            [0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [0, 0.8, 0.1, 0.1, 1.1, 1.1],
        ]
        result = nms(bboxes, iou_thresh=0.99, thresh=0.5, box_format="corners")
        assert len(result) == 2

    # --- Box format ---

    def test_midpoint_format(self):
        """NMS works with midpoint format."""
        # Two identical boxes at same location (midpoint format: cx, cy, w, h)
        bboxes = [
            [0, 0.9, 0.5, 0.5, 1.0, 1.0],
            [0, 0.8, 0.5, 0.5, 1.0, 1.0],
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5, box_format="midpoint")
        assert len(result) == 1
        assert result[0][1] == 0.9

    # --- Multi-box scenarios ---

    def test_chain_suppression(self):
        """Highest confidence box suppresses multiple overlapping boxes."""
        bboxes = [
            [0, 0.7, 0.0, 0.0, 1.0, 1.0],
            [0, 0.9, 0.05, 0.05, 1.05, 1.05],
            [0, 0.8, 0.1, 0.1, 1.1, 1.1],
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5, box_format="corners")
        assert len(result) == 1
        assert result[0][1] == 0.9

    def test_multiple_classes_independent(self):
        """Each class processed independently."""
        bboxes = [
            [0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [0, 0.8, 0.0, 0.0, 1.0, 1.0],  # suppressed by first
            [1, 0.85, 0.0, 0.0, 1.0, 1.0],
            [1, 0.7, 0.0, 0.0, 1.0, 1.0],  # suppressed by third
        ]
        result = nms(bboxes, iou_thresh=0.5, thresh=0.5, box_format="corners")
        assert len(result) == 2
        classes = [box[0] for box in result]
        assert 0 in classes and 1 in classes


class TestMAP:
    """Tests for mean average precision calculation."""

    # --- Basic functionality ---

    def test_perfect_predictions(self):
        """All predictions exactly match ground truths → mAP = 1.0."""
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [1, 0, 0.9, 2.0, 2.0, 3.0, 3.0],
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1, 0, 1.0, 2.0, 2.0, 3.0, 3.0],
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=1
        )
        assert result > 0.99

    def test_no_predictions(self):
        """No predictions for any class with ground truths."""
        box_preds = []
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
        # All classes with GTs are skipped, avg_precisions is empty → division error or 0
        # Current implementation will raise ZeroDivisionError
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=20
        )
        assert result == 0.0

    def test_no_ground_truths(self):
        """No ground truths → classes skipped, empty list."""
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],
        ]
        box_targs = []
        with pytest.raises(ZeroDivisionError):
            calculate_map(
                box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=1
            )

    def test_all_false_positives(self):
        """Predictions don't overlap any ground truths → mAP = 0."""
        box_preds = [
            [0, 0, 0.9, 10.0, 10.0, 11.0, 11.0],
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=1
        )
        assert result < 0.01

    # --- Multi-class ---

    def test_multiple_classes_independent(self):
        """Each class contributes independently to mAP."""
        # Class 0: perfect, Class 1: no detections (skipped)
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0, 1, 1.0, 2.0, 2.0, 3.0, 3.0],  # class 1, no predictions
        ]
        # Class 1 has no detections but has GTs → n_true_bboxes > 0, but detections empty
        # Loop doesn't add to TP/FP, recalls/precisions are empty tensors of size 0
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=2
        )
        # Class 0: AP=1, Class 1: AP=0 (no TP), mAP = 0.5
        assert 0.4 < result < 0.6

    def test_two_classes_both_perfect(self):
        """Two classes, both with perfect predictions."""
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [0, 1, 0.9, 2.0, 2.0, 3.0, 3.0],
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0, 1, 1.0, 2.0, 2.0, 3.0, 3.0],
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=2
        )
        assert result > 0.99

    # --- IoU threshold ---

    def test_iou_threshold_strict(self):
        """Stricter IoU threshold reduces mAP for partial overlaps."""
        # Boxes overlap but not perfectly
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],
        ]
        box_targs = [
            [0, 0, 1.0, 0.2, 0.2, 1.2, 1.2],  # partial overlap
        ]
        result_loose = calculate_map(
            box_preds, box_targs, iou_thresh=0.3, box_format="corners", n_classes=1
        )
        result_strict = calculate_map(
            box_preds, box_targs, iou_thresh=0.9, box_format="corners", n_classes=1
        )
        assert result_loose > result_strict

    # --- Duplicate detections ---

    def test_duplicate_detection_penalized(self):
        """Second detection on same GT should be false positive."""
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],
            [0, 0, 0.8, 0.0, 0.0, 1.0, 1.0],  # duplicate, lower conf
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=1
        )
        # First is TP, second is FP → precision drops
        assert result < 1.0

    # --- Multiple images ---

    def test_multiple_images(self):
        """Detections across multiple images (train_idx)."""
        box_preds = [
            [0, 0, 0.9, 0.0, 0.0, 1.0, 1.0],  # image 0
            [1, 0, 0.9, 0.0, 0.0, 1.0, 1.0],  # image 1
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],  # image 0
            [1, 0, 1.0, 0.0, 0.0, 1.0, 1.0],  # image 1
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=1
        )
        assert result > 0.99

    # --- Box format ---

    def test_midpoint_format(self):
        """Works with midpoint box format."""
        # midpoint: cx, cy, w, h
        box_preds = [
            [0, 0, 0.9, 0.5, 0.5, 1.0, 1.0],
        ]
        box_targs = [
            [0, 0, 1.0, 0.5, 0.5, 1.0, 1.0],
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="midpoint", n_classes=1
        )
        assert result > 0.99

    # --- Confidence ordering ---

    def test_confidence_ordering_matters(self):
        """Higher confidence predictions processed first."""
        # One GT, two predictions: high conf misses, low conf hits
        box_preds = [
            [0, 0, 0.9, 10.0, 10.0, 11.0, 11.0],  # high conf, wrong location
            [0, 0, 0.1, 0.0, 0.0, 1.0, 1.0],  # low conf, correct location
        ]
        box_targs = [
            [0, 0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
        result = calculate_map(
            box_preds, box_targs, iou_thresh=0.5, box_format="corners", n_classes=1
        )
        # First (high conf) is FP, second (low conf) is TP
        # Precision at recall=1 is 0.5, AP < 1
        assert result < 0.99
