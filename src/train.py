import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import CurrentConfig
from dataset import VOCDataset
from model import Yolo
from loss import YoloLoss
from utils import get_bboxes

seed = 42
torch.manual_seed(seed)

conf = CurrentConfig()


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

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


def train(train_loader, model, optimizer, loss_fn, writer, epoch):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

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

    epoch_loss = sum(mean_loss) / len(mean_loss)
    writer.add_scalar("Loss/epoch", epoch_loss, epoch)
    return epoch_loss


def main():
    print(conf.device)

    writer = SummaryWriter("runs/waylo")

    model = Yolo(config=conf).to(conf.device)
    optimizer = optim.Adam(
        model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay
    )
    loss_fn = YoloLoss(config=conf)

    train_dataset = VOCDataset(
        "voc_data/100examples.csv", transform=transform, config=conf
    )
    test_dataset = VOCDataset("voc_data/test.csv", transform=transform, config=conf)

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

    for epoch in range(conf.epochs):
        print(f"\nEpoch {epoch + 1}/{conf.epochs}")

        epoch_loss = train(train_loader, model, optimizer, loss_fn, writer, epoch)
        print(f"Mean loss: {epoch_loss:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == conf.epochs - 1:
            box_preds, box_targs = get_bboxes(
                train_loader, model, iou_thresh=0.5, thresh=0.4
            )
            mean_avg_prec = map(
                box_preds, box_targs, iou_thresh=0.5, box_format="midpoint"
            )
            writer.add_scalar("mAP/train", mean_avg_prec, epoch)
            print(f"Train mAP: {mean_avg_prec:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
