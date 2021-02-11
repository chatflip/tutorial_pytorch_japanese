# -*- coding: utf-8 -*-
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    # 初期化
    def __init__(self, root, phase, transform=None):
        self.transform = transform  # 画像変形用
        self.image_paths = []  # 画像のパス格納用
        self.image_labels = []  # 画像のラベル格納用
        class_names = np.genfromtxt(os.path.join(root, 'meta', 'classes.txt'), dtype=str)  # クラス名を取得
        with open(os.path.join(root, 'meta', '{}.json'.format(phase))) as f:
            filenames = json.load(f)

        for class_index, class_name in enumerate(class_names):
            image_paths = filenames[class_name]
            num_image = len(image_paths)
            fullpaths = [os.path.join('{}/images/{}.jpg'.format(root, image_path)) for image_path in image_paths]
            self.image_paths.extend(fullpaths)
            self.image_labels.extend([class_index] * num_image)

    # num_worker数で並列処理される関数
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')  # 画像をPILで開く
        if self.transform is not None:
            image = self.transform(image)  # 画像変形適用
        return image, self.image_labels[index]  # 画像とラベルを返す

    # データセットの画像数宣言(これが無いとエラー)
    def __len__(self):
        return len(self.image_paths)
