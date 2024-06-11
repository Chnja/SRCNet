import torch
import random
import numpy as np

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

import copy


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        # mask = np.array(mask).astype(np.float32)
        mask = np.array(mask).astype(np.float32) / 255.0

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return {"image": (img1, img2), "label": mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": (img1, img2), "label": mask}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {"image": (img1, img2), "label": mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img1 = img1.rotate(rotate_degree, Image.BILINEAR)
            img2 = img2.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {"image": (img1, img2), "label": mask}


class Shift(object):
    # def __init__(self, size):
    #     self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]

        Rx = random.randint(-32, 32)
        Rx = Rx + 256 if Rx < 0 else Rx
        Ry = random.randint(-32, 32)
        Ry = Ry + 256 if Ry < 0 else Ry

        img1_p = copy.deepcopy(img1)
        img2_p = copy.deepcopy(img2)
        mask_p = copy.deepcopy(mask)

        img1.paste(img1_p, (Rx, Ry))
        img1.paste(img1_p, (Rx - 256, Ry - 256))
        img1.paste(img1_p, (Rx - 256, Ry))
        img1.paste(img1_p, (Rx, Ry - 256))

        img2.paste(img2_p, (Rx, Ry))
        img2.paste(img2_p, (Rx - 256, Ry - 256))
        img2.paste(img2_p, (Rx - 256, Ry))
        img2.paste(img2_p, (Rx, Ry - 256))

        mask.paste(mask_p, (Rx, Ry))
        mask.paste(mask_p, (Rx - 256, Ry - 256))
        mask.paste(mask_p, (Rx - 256, Ry))
        mask.paste(mask_p, (Rx, Ry - 256))

        return {"image": (img1, img2), "label": mask}


def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """

    returnImage = image

    random_factor = np.random.rand() * 0.7 + 0.8  # 随机因子 0.8~1.5
    returnImage = ImageEnhance.Color(returnImage).enhance(
        random_factor
    )  # 调整图像的饱和度

    random_factor = np.random.rand() * 0.7 + 0.8  # 随机因子 0.8~1.5
    returnImage = ImageEnhance.Brightness(returnImage).enhance(
        random_factor
    )  # 调整图像的亮度

    random_factor = np.random.rand() * 0.7 + 0.8  # 随机因子 0.8~1.5
    returnImage = ImageEnhance.Contrast(returnImage).enhance(
        random_factor
    )  # 调整图像对比度

    random_factor = np.random.rand() * 2.2 + 0.8  # 随机因子 0.8~3.0
    returnImage = ImageEnhance.Sharpness(returnImage).enhance(
        random_factor
    )  # 调整图像锐度

    return returnImage


class RandomColor(object):
    # def __init__(self, size):
    #     self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample["image"][0]
        img2 = sample["image"][1]
        mask = sample["label"]

        img1 = randomColor(img1)
        img2 = randomColor(img2)

        return {"image": (img1, img2), "label": mask}


"""
We don't use Normalize here, because it will bring negative effects.
the mask of ground truth is converted to [0,1] in ToTensor() function.
"""
train_transforms = transforms.Compose(
    [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomColor(),
        Shift(),
        RandomRotate(180),
        ToTensor(),
    ]
)

test_transforms = transforms.Compose([ToTensor()])
