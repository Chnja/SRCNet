import torch
import random
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
import time


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CTime:
    def __init__(self):
        self.time_born = ""
        self.time_start = ""
        self.time_end = ""

    def born(self):
        self.time_born = time.time()

    def start(self):
        self.time_start = time.time()

    def end(self):
        self.time_end = time.time()

    def showtime(self, delta):
        # delta = end-start
        delta = int(delta + 0.5)
        hour = delta // 60 // 60
        delta = delta - hour * 60 * 60
        if hour >= 100:
            hour = str(hour)
        else:
            hour = "0" + str(hour)
            hour = hour[-2:]
        minu = delta // 60
        minu = "0" + str(minu)
        minu = minu[-2:]
        sec = delta % 60
        sec = "0" + str(sec)
        sec = sec[-2:]
        return hour + ":" + minu + ":" + sec

    def show(self, epoch, PRI_EPOCH, EPOCH):
        print(
            "  Time: "
            + self.showtime(self.time_end - self.time_start)
            + " | "
            + self.showtime(self.time_end - self.time_born)
            + " + "
            + self.showtime((self.time_end - self.time_start) * (EPOCH - epoch - 1))
        )


def calF1(preds, labels):
    preds = preds.data.cpu().numpy().flatten()
    labels = labels.data.cpu().numpy().flatten()
    sumNum = labels.shape[0]
    TP = np.sum(preds * labels) / sumNum
    TN = np.sum((1 - preds) * (1 - labels)) / sumNum
    FP = np.sum(preds * (1 - labels)) / sumNum
    FN = np.sum((1 - preds) * labels) / sumNum
    return TP, TN, FP, FN


class CDMetrics:
    def __init__(self):
        self.metrics = {
            "loss": [],
            "matrix": [],
            "lr": [],
        }

    def set(self, cd_loss, cd_preds, labels, lr=[0.0]):
        TP, TN, FP, FN = calF1(cd_preds, labels)
        self.metrics["loss"].append(cd_loss.item())
        self.metrics["matrix"].append([TP, TN, FP, FN])
        self.metrics["lr"].append(lr[0])

    def get(self):
        [TP, TN, FP, FN] = np.mean(self.metrics["matrix"], axis=0)
        oa = TP + TN
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = TP / (1 - TN + 1e-10)
        return {
            "loss": np.mean(self.metrics["loss"]),
            "lr": self.metrics["lr"][-1],
            "oa": oa,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
