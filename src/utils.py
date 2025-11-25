import torch

from collections import Counter


def iou(box_preds, box_labels, box_format="midpoint"):
    """
    Calculates the intersection over union of two boxes.

    Args:
        box_preds (tensor): Bounding box predictions (N, 4)
        box_labels (tensor): Bounding box targets (N, 4)
        box_format (str): midpoint|corners, (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all pred/target boxes
    """
    if box_format == "midpoint":
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3] / 2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4] / 2
        box1_x2 = box_preds[..., 0:1] + box_preds[..., 2:3] / 2
        box1_y2 = box_preds[..., 1:2] + box_preds[..., 3:4] / 2
        box2_x1 = box_labels[..., 0:1] - box_labels[..., 2:3] / 2
        box2_y1 = box_labels[..., 1:2] - box_labels[..., 3:4] / 2
        box2_x2 = box_labels[..., 0:1] + box_labels[..., 2:3] / 2
        box2_y2 = box_labels[..., 1:2] + box_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = box_preds[..., 0:1]
        box1_y1 = box_preds[..., 1:2]
        box1_x2 = box_preds[..., 2:3]
        box1_y2 = box_preds[..., 3:4]
        box2_x1 = box_labels[..., 0:1]
        box2_y1 = box_labels[..., 1:2]
        box2_x2 = box_labels[..., 2:3]
        box2_y2 = box_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def nms(bboxes, iou_thresh, thresh, box_format="corners"):
    """
    Implements non-max suppression on given bounding boxes.

    Args:
        bboxes (list): A list of bounding boxes where each bounding box is in
            the format [class_pred, prob_score, x1, y1, x2, y2].
        iou_thresh (float): A threshold gating whether a predicted bounding box
            is correct.
        thresh (float): A threshold gating whether to keep or remove a predicted
            bounding box.
        box_format (str): midpoint|corners, (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        list: Bounding boxes after non-max suppression
    """

    bboxes = [box for box in bboxes if box[1] > thresh]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        current_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != current_box[0]
            or iou(
                torch.tensor(current_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_thresh
        ]

        bboxes_after_nms.append(current_box)

    return bboxes_after_nms


def map(box_preds, box_targs, iou_thresh=0.5, box_format="midpoint", n_classes=20):
    """
    Calculates mean average precision for a set of labeled and predicted
    bounding boxes.

    Args:
        box_preds (list): A list of bounding boxes where each bounding box is in
            the format [train_idx, class_pred, prob_score, x1, y1, x2, y2].
        box_targs (list): Same as box_preds but with target boxes
        iou_thresh (float): A threshold above which a predicted bounding box is
            correct.
        box_format (str): midpoint|corners, (x,y,w,h) or (x1,y1,x2,y2).
        n_classes (int): The number of different classes to predict.

    Returns:
        float: The mean average precision across all classes.
    """
    avg_precisions = []
    epsilon = 1e-6

    for c in range(n_classes):
        detections = []
        ground_truths = []

        for detection in box_preds:
            if detection[1] == c:
                detections.append(detection)

        for gt in box_targs:
            if gt[1] == c:
                ground_truths.append(gt)

        n_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in n_bboxes.items():
            n_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        true_pos = torch.zeros((len(detections)))
        false_pos = torch.zeros((len(detections)))
        n_true_bboxes = len(ground_truths)

        if n_true_bboxes == 0:
            continue

        for i, detection in enumerate(detections):
            gt_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0
            for j, gt in enumerate(gt_img):
                iou_score = iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou_score > best_iou:
                    best_iou = iou_score
                    best_gt_idx = j

            if best_iou > iou_thresh:
                if n_bboxes[detection[0]][best_gt_idx] == 0:
                    true_pos[i] = 1
                    n_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    false_pos[i] = 1
            else:
                false_pos[i] = 1

        true_pos_cumsum = torch.cumsum(true_pos, dim=0)
        false_pos_cumsum = torch.cumsum(false_pos, dim=0)
        recalls = true_pos_cumsum / (n_true_bboxes + epsilon)
        precisions = torch.divide(
            true_pos_cumsum, (true_pos_cumsum + false_pos_cumsum + epsilon)
        )
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        avg_precisions.append(torch.trapz(precisions, recalls))

    return sum(avg_precisions) / len(avg_precisions)
