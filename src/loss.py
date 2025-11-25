import torch
import torch.nn as nn

from config import CurrentConfig
from utils import iou


class YoloLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, target):
        S = self.config.split_size
        B = self.config.n_boxes
        C = self.config.n_classes

        predictions = predictions.reshape(-1, S, S, C + B * 5)

        box1_start = C + 1
        box2_start = C + 6
        conf1_idx = C
        conf2_idx = C + 5
        target_box_start = C + 1

        iou_b1 = iou(
            predictions[..., box1_start : box1_start + 4],
            target[..., target_box_start : target_box_start + 4],
        )
        iou_b2 = iou(
            predictions[..., box2_start : box2_start + 4],
            target[..., target_box_start : target_box_start + 4],
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        exists_box = target[..., C].unsqueeze(3)

        # Box coordinates
        box_predictions = exists_box * (
            bestbox * predictions[..., box2_start : box2_start + 4]
            + (1 - bestbox) * predictions[..., box1_start : box1_start + 4]
        )
        box_targets = exists_box * target[..., target_box_start : target_box_start + 4]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # Object loss
        pred_box = (
            bestbox * predictions[..., conf2_idx : conf2_idx + 1]
            + (1 - bestbox) * predictions[..., conf1_idx : conf1_idx + 1]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., C : C + 1]),
        )

        # No object loss
        no_object_loss = self.mse(
            torch.flatten(
                (1 - exists_box) * predictions[..., conf1_idx : conf1_idx + 1],
                start_dim=1,
            ),
            torch.flatten((1 - exists_box) * target[..., C : C + 1], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten(
                (1 - exists_box) * predictions[..., conf2_idx : conf2_idx + 1],
                start_dim=1,
            ),
            torch.flatten((1 - exists_box) * target[..., C : C + 1], start_dim=1),
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :C], end_dim=-2),
            torch.flatten(exists_box * target[..., :C], end_dim=-2),
        )

        loss = (
            self.config.lambda_coord * box_loss
            + object_loss
            + self.config.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
