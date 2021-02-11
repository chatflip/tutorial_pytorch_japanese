# -*- coding: utf-8 -*-
import json
import os

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset


class VOCSegmentation2012(Dataset):
    # 初期化
    def __init__(self, root, phase, num_classes, transform=None):
        self.transform = transform  # 画像変形用
        self.image_paths = []  # 画像のパス格納用
        self.mask_paths = []  # 画像のラベル格納用
        self.num_classes = num_classes
        image_list_path = os.path.join(root, "ImageSets", "Segmentation", f"{phase}.txt")
        image_lists = pd.read_table(image_list_path, header=None)

        for _, image_list in image_lists.iterrows():
            image_path = f"{root}/JPEGImages/{image_list.values[0]}.jpg"
            mask_path = f"{root}/SegmentationRaw/{image_list.values[0]}.png"
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)

    # num_worker数で並列処理される関数
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[index], 0)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)  # 画像変形適用
        image, mask = augmented['image'], augmented['mask']
        height, width = mask.shape
        binary_mask = torch.zeros(self.num_classes, height, width)
        for i in range(self.num_classes):
            one_hot_mask = torch.where(mask == i, 1, 0)
            binary_mask[i, :, :] = one_hot_mask
        return image, binary_mask  # 画像とラベルを返す

    # データセットの画像数宣言(これが無いとエラー)
    def __len__(self):
        return len(self.image_paths)
