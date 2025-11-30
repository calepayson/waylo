import os
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from config import WaymoConfig, CurrentConfig
from dataset import VOCDataset
from model import Yolo
from loss import YoloLoss
from utils import get_bboxes, calculate_map, convert_cellboxes_to_list

# Try to use real SummaryWriter, but fall back to a no-op stub if tensorboard is broken
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    print("⚠️ TensorBoard import failed. Using dummy SummaryWriter (no logging).")

    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def add_image(self, *args, **kwargs): pass
        def add_histogram(self, *args, **kwargs): pass
        def close(self): pass


seed = 42
torch.manual_seed(seed)

#! IMP Change to *CurrentConfig* to rerun experiment on Pascal VOC
conf = WaymoConfig()

# Use results directory provided by config
RESULTS_DIR = conf.results_dir
RESULTS_DIR.mkdir(exist_ok=True)
print("Saving results to:", RESULTS_DIR)


class Compose(object):
    def __init__(self, transforms_):
        self.transforms = transforms_

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose(
    [
        transforms.Resize((conf.img_size, conf.img_size)),
        transforms.ToTensor(),
    ]
)


def train_one_epoch(train_loader, model, optimizer, loss_fn, writer, epoch: int):
    from tqdm import tqdm

    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    model.train()
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(conf.device), y.to(conf.device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/batch", loss.item(), global_step)

        loop.set_postfix(loss=loss.item())

    epoch_loss = sum(mean_loss) / max(len(mean_loss), 1)
    writer.add_scalar("Loss/epoch", epoch_loss, epoch)
    return epoch_loss


def save_metrics(loss_history, map_history, out_dir: Path):
    # Save loss per epoch
    loss_file = RESULTS_DIR / "loss_history.csv"
    with loss_file.open("w") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(loss_history, start=1):
            f.write(f"{i},{loss}\n")

    # Save mAP (epoch, mAP)
    map_file = RESULTS_DIR / "map_history.csv"
    with map_file.open("w") as f:
        f.write("epoch,map\n")
        for epoch, m in map_history:
            f.write(f"{epoch},{m}\n")

    print(f"Saved metrics to {loss_file} and {map_file}")


def visualize_predictions(model, dataset, num_images: int, out_dir: Path, conf_thresh=0.4):
    """
    Save a few images with predicted bounding boxes drawn on them.
    """
    model.eval()
    to_pil = transforms.ToPILImage()

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    saved = 0
    for batch_idx, (img_tensor, label_matrix) in enumerate(loader):
        if saved >= num_images:
            break

        img_tensor = img_tensor.to(conf.device)
        with torch.no_grad():
            preds = model(img_tensor)

        # Convert predictions to [class, conf, x, y, w, h] list
        # (uses current utils.convert_cellboxes_to_list signature)
        pred_boxes_batch = convert_cellboxes_to_list(preds)
        boxes = pred_boxes_batch[0]  # first (and only) image in batch

        # Filter by confidence
        boxes = [b for b in boxes if b[1] >= conf_thresh]

        if not boxes:
            continue

        # Convert tensor image back to PIL
        img_np = img_tensor[0].detach().cpu()
        pil_img = to_pil(img_np)
        draw = ImageDraw.Draw(pil_img)
        W, H = pil_img.size

        for cls, score, xc, yc, w, h in boxes:
            # convert normalized xywh to pixel corners
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1 + 2, y1 + 2), f"{int(cls)}:{score:.2f}", fill="yellow")

        out_path = RESULTS_DIR / f"prediction_{saved+1}.png"
        pil_img.save(out_path)
        saved += 1

    print(f"Saved {saved} annotated prediction image(s) to {out_dir}")


def main():
    print("Using device:", conf.device)

    writer = SummaryWriter("runs/waylo")

    model = Yolo(config=conf).to(conf.device)
    optimizer = optim.Adam(
        model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay
    )
    loss_fn = YoloLoss(config=conf)

    # Use data_root from config (this handles VOC or Waymo depending on the config)
    data_root = conf.data_root.resolve()

    # Use paths from config (these differ for VOC vs Waymo)
    train_csv = conf.train_csv
    test_csv  = conf.val_csv   # we use val as "test" loader 

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    print("Train CSV:", train_csv)
    print("Test  CSV:", test_csv)

    train_dataset = VOCDataset(train_csv, transform=transform, config=conf)
    test_dataset = VOCDataset(test_csv, transform=transform, config=conf)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        shuffle=True,
        drop_last=True,
    )

    loss_history = []
    map_history = []  # list of (epoch_idx, mAP)

    for epoch in range(conf.epochs):
        print(f"\nEpoch {epoch + 1}/{conf.epochs}")

        epoch_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, writer, epoch)
        print(f"Mean loss: {epoch_loss:.4f}")
        loss_history.append(epoch_loss)

        # Compute mAP every 5 epochs (or last)
        if (epoch + 1) % 5 == 0 or epoch == conf.epochs - 1:
            box_preds, box_targs = get_bboxes(
                train_loader,
                model,
                iou_thresh=0.5,
                thresh=0.4,
                device=conf.device,
            )
            mean_avg_prec = calculate_map(
                box_preds,
                box_targs,
                iou_thresh=0.5,
                box_format="midpoint",
                n_classes=conf.n_classes,
            )
            writer.add_scalar("mAP/train", mean_avg_prec, epoch)
            map_history.append((epoch + 1, float(mean_avg_prec)))
            print(f"Train mAP: {mean_avg_prec:.4f}")

    writer.close()

    # Save loss & mAP as CSV for plotting later
    save_metrics(loss_history, map_history, RESULTS_DIR)

    # Save a few annotated prediction images from the test set
    visualize_predictions(model, test_dataset, num_images=5, out_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()