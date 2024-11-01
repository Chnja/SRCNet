# -*- encoding: utf-8 -*-
# -----------------------------------
# train.py
# Written by Chnja from WHU
# chj1997@whu.edu.cn
# -----------------------------------

import torch
import os
from tensorboardX import SummaryWriter
from TrainUse import seed_torch, CDMetrics, CTime
from Loss import CombineLoss

from nloaders import nloaders
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 2e-3
BATCH_SIZE = 16
EPOCH = 300
PRI_EPOCH = 0

from SRCNet import SRCNet as trainNet

net = trainNet().to(device)
netName = "SRCNet_LEVIR-cd"

netName = "proj/" + netName

writer = SummaryWriter(netName + "/")

seed_torch(seed=3407)

criterion = CombineLoss()

train_loader, val_loader = nloaders(BATCH_SIZE, dataset_dir="/mnt/ramdisk/LEVIR-cd/")
optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

if __name__ == "__main__":
    Ctime = CTime()
    if not os.path.exists(netName):
        os.mkdir(netName)
    if PRI_EPOCH > 0:
        net.load_state_dict(torch.load(netName + "/epoch_" + str(PRI_EPOCH) + ".pth"))
    if PRI_EPOCH != 0:
        for epoch in range(0, PRI_EPOCH):
            scheduler.step()
    maxf1 = 0
    maxepoch = 0
    Ctime.born()
    for epoch in range(PRI_EPOCH, EPOCH):
        trainMetrics = CDMetrics()
        valMetrics = CDMetrics()
        net.train()
        length = len(train_loader)
        Ctime.start()
        with tqdm(total=length, desc=f"[epoch: {epoch+1}/{EPOCH}]", ncols=125) as pbar:
            for data in train_loader:
                batch_img1, batch_img2, labels = data
                # Set variables for training
                batch_img1 = batch_img1.float().to(device)
                batch_img2 = batch_img2.float().to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                cd_preds, cosDis, diff, sigma = net(batch_img1, batch_img2)
                cd_loss = criterion(cd_preds, labels, cosDis, diff, sigma)

                loss = cd_loss
                loss.backward()
                optimizer.step()

                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                trainMetrics.set(cd_loss, cd_preds, labels, scheduler.get_last_lr())

                # log the batch mean metrics
                mean_train_metrics = trainMetrics.get()

                # clear batch variables from memory
                del batch_img1, batch_img2, labels

                pbar.set_postfix(
                    {
                        "Loss": "{:.4f}/{:.4f}".format(
                            cd_loss.item(), mean_train_metrics["loss"]
                        ),
                        "F1": "{:.4f}".format(mean_train_metrics["f1"]),
                        "LR": "{:3.2e}".format(mean_train_metrics["lr"]),
                    }
                )
                pbar.update(1)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {"train": v}, epoch + 1)

        scheduler.step()
        net.eval()
        with torch.no_grad():
            length = len(val_loader)
            for i, data in enumerate(val_loader, 0):
                batch_img1, batch_img2, labels = data
                # Set variables for training
                batch_img1 = batch_img1.float().to(device)
                batch_img2 = batch_img2.float().to(device)
                labels = labels.long().to(device)

                # Get predictions and calculate loss
                cd_preds, cosDis, diff, sigma = net(batch_img1, batch_img2)
                cd_loss = criterion(cd_preds, labels, cosDis, diff, sigma)

                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                valMetrics.set(cd_loss, cd_preds, labels, scheduler.get_last_lr())

                # log the batch mean metrics
                mean_val_metrics = valMetrics.get()

                # clear batch variables from memory
                del batch_img1, batch_img2, labels
                print("  Test: %d / %d\r" % (i, length), end="")

            if mean_val_metrics["f1"] > maxf1:
                maxf1 = mean_val_metrics["f1"]
                if epoch + 1 - maxepoch <= 5 and maxepoch != 0:
                    os.remove(netName + "/epoch_" + str(maxepoch) + ".pth")
                maxepoch = epoch + 1
                torch.save(
                    net.state_dict(), netName + "/epoch_" + str(maxepoch) + ".pth"
                )

            print(
                "  Loss: %.05f | F1: %.04f | MAX-F1: %.04f (%d)"
                % (
                    mean_val_metrics["loss"],
                    mean_val_metrics["f1"],
                    maxf1,
                    maxepoch,
                )
            )
            for k, v in mean_val_metrics.items():
                writer.add_scalars(str(k), {"test": v}, epoch + 1)

        Ctime.end()

        Ctime.show(epoch, PRI_EPOCH, EPOCH)

    writer.close()
