# -*- encoding: utf-8 -*-
# -----------------------------------
# cLoss.py
# Written by Chnja from WHU
# chj1997@whu.edu.cn
# -----------------------------------


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        # logpt = F.log_softmax(input)
        logpt = torch.log(input + 1e-10)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def DiceLoss(prob, target):
    # prob = F.softmax(prob, dim=1)
    target = target.type(prob.type())
    prob = prob[:, 1, :, :]
    dims = (1, 2)
    I = torch.sum(prob * target, dims)
    U = torch.sum(prob + target, dims)
    _loss = (2.0 * I / (U + 1e-10)).mean()
    return 1 - _loss


class EdgeLoss:
    def __init__(self, KSIZE=7):
        self.KSIZE = KSIZE
        self.MASK = torch.zeros([KSIZE, KSIZE])
        self.cal_mask(KSIZE)

    def cal_mask(self, ksize):
        num = 0
        MASK = self.MASK
        for x in range(0, ksize):
            for y in range(0, ksize):
                if (x + 0.5 - ksize / 2) ** 2 + (y + 0.5 - ksize / 2) ** 2 <= (
                    (ksize - 1) / 2
                ) ** 2:
                    MASK[x][y] = 1
                    num += 1
        MASK = MASK.reshape(1, 1, 1, 1, -1).float() / num
        MASK = MASK.to(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.MASK = MASK

    def tensor_average(self, bin_img, ksize):
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode="constant", value=0)

        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)

        eroded = torch.sum(patches.reshape(B, C, H, W, -1).float() * self.MASK, dim=-1)
        return eroded

    def edgeLoss(self, input, target):
        targets = target.unsqueeze(dim=1)
        targetAve = self.tensor_average(targets, ksize=self.KSIZE)
        at = torch.abs(targets.float() - targetAve)
        # at[at == 0] = 0.2
        at = at.view(-1)

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        # logpt = F.log_softmax(input)
        logpt = torch.log(input + 1e-10)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt * at
        return loss.mean()


class CombineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.EL = EdgeLoss(KSIZE=7)

    def calloss(self, prediction, target, sigmas):
        focal0 = FocalLoss(gamma=0, alpha=None)
        bce = focal0(prediction, target)
        dice = DiceLoss(prediction, target)
        edge = self.EL.edgeLoss(prediction, target)
        return bce / sigmas[0] + dice / sigmas[1] + edge / sigmas[2]

    def forward(self, predictions, target, Diss, diff, sigma):
        loss = 0
        sigmas = sigma
        sigmas = sigmas * sigmas

        for prediction in predictions:
            prediction = F.softmax(prediction, dim=1)
            loss += self.calloss(prediction, target, sigmas)

        for Dis in Diss:
            loss += self.calloss(Dis, target, sigmas)

        if len(diff) != 0:
            (dif,) = diff
            loss += dif

        loss += torch.sum(torch.log(sigmas)) / 2

        return loss
