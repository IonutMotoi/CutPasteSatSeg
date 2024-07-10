import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A

from cut_and_paste import CutAndPaste


def get_train_augmentations(cfg):
    transforms = []
    if cfg["hor_flip"]:
        transforms.append(A.HorizontalFlip(p=0.5))
    if cfg["ver_flip"]:
        transforms.append(A.VerticalFlip(p=0.5))
    if cfg["random_rotate"]:
        transforms.append(A.RandomRotate90(p=1.0))
    return A.Compose(transforms)


class DynamicEarthNetPlanet(Dataset):
    def __init__(self, cfg, split):
        """
        Args:
            root: the path of the folder which contains the sentinel 2 dataset
            split: train/val
        """
        self.root = cfg["root"]
        self.split = split
        self.num_classes = cfg["num_classes"]
        self.normalize = transforms.Normalize(mean=cfg["mean"], std=cfg["std"])

        if split == "train":
            self.augmentations = get_train_augmentations(cfg["augmentations"])

        with open(os.path.join(self.root, self.split + ".txt"), "r") as f:
            file_list = [line.rstrip().split(' ') for line in f]
        self.files, self.labels = list(zip(*file_list))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Load image
        img = rasterio.open(os.path.join(self.root, self.files[index])).read()
        img = img.astype(np.float32)

        # Load label
        label = rasterio.open(os.path.join(self.root, self.labels[index])).read()
        mask = self.label2mask(label)

        # Apply augmentations
        if self.split == "train":
            aug = self.augmentations(image=img.transpose(1, 2, 0), mask=mask)
            img = aug["image"].transpose(2, 0, 1)
            mask = aug["mask"]

        # Normalize
        img = self.normalize(torch.from_numpy(img))
        mask = torch.from_numpy(mask).long()

        return img, mask

    def label2mask(self, label):
        mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int64)
        for i in range(self.num_classes + 1):
            if i == 6:  # ignore the snow and ice class
                mask[label[i, :, :] == 255] = -1
            else:
                mask[label[i, :, :] == 255] = i
        return mask


class DynamicEarthNetPlanet_CAP(DynamicEarthNetPlanet):
    """
    DynamicEarthNet Planet dataset with Cut And Paste augmentation
    """

    def __init__(self, cfg, split):
        """
        Args:
            root: the path of the folder which contains the sentinel 2 dataset
            split: train/val
        """
        super().__init__(cfg, split)
        self.cut_and_paste = CutAndPaste(cfg["cut_and_paste"])

    def __getitem__(self, index):
        # Load image
        img = rasterio.open(os.path.join(self.root, self.files[index])).read()
        img = img.astype(np.float32)

        # Load label
        label = rasterio.open(os.path.join(self.root, self.labels[index])).read()
        mask = self.label2mask(label)

        # Cut and Paste augmentation
        self.cut_and_paste.paste_instances(img, mask)

        # Apply augmentations and normalize
        aug = self.augmentations(image=img.transpose(1, 2, 0), mask=mask)
        img = torch.from_numpy((aug["image"]).transpose(2, 0, 1))
        img = self.normalize(img)
        mask = torch.from_numpy(aug["mask"]).long()

        return img, mask
