import argparse
from tqdm import tqdm
from glob import glob

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import VOCDataset
from loss import YOLOLoss
from model import YOLOv1
from utils import non_max_suppression, mean_average_precision, cellboxes_to_boxes, get_bboxes, \
    plot_image, save_checkpoint, load_checkpoint


def get_last_model_file():
    model_files = glob('drive/MyDrive/YOLOv1/*.pth.tar')
    if len(model_files) == 0:
        return None, 1
    else:
        last_model_file = model_files[-1]
        epoch_number = int(last_model_file.split("_")[2].split(".")[0])
        return model_files[-1], epoch_number + 1


# PARSE ARGS
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--plot')
args = arg_parse.parse_args()

SEED = 123
torch.manual_seed(SEED)

# GLOBAL VARIABLES
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
PLOT_RESULT = args.plot == "true"
LOAD_MODEL_FILE, CURRENT_EPOCH = get_last_model_file()
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

transform = T.Compose([T.Resize((448, 448)),
                       T.ToTensor()
                       ])


def train_fn(train_loader, model, optimizer, loss_function, epoch):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        loss = loss_function(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_description(f"Epoch [{epoch}]")
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss: {sum(mean_loss) / len(mean_loss)}")


def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = YOLOLoss()

    if LOAD_MODEL_FILE is not None:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("data/train.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_dataset = VOCDataset("data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=True,
                             drop_last=True)

    if PLOT_RESULT:
        for img_idx, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            for idx in range(8):
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx],
                                             iou_threshold=0.5,
                                             probability_threshold=0.4,
                                             box_format="midpoint")
                plot_image(x[idx].permute(1, 2, 0).to('cpu'), bboxes)
            if img_idx == 8:
                import sys
                sys.exit()

    for epoch in range(CURRENT_EPOCH, EPOCHS + 1):
        train_fn(train_loader, model, optimizer, loss_function, epoch)

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)

        mean_average_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        print(f"Train mAP: {mean_average_prec}")

        checkpoint = {
            'state_dict': model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"drive/MyDrive/YOLOv1/model_epoch_{epoch}.pth.tar")

    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4)
    mean_average_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
    print(f"Test mAP: {mean_average_prec}")


if __name__ == '__main__':
    main()
