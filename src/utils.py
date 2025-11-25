import torch


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
