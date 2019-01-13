# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
require pytorch 0.4.0
        torchvision 0.2.1
"""

import os
import glob

from PIL import Image
from torch.utils.data import Dataset

class AnimeFaceDB(Dataset):
    #初期化
    def __init__(self, root, transform=None):
        self.transform = transform#画像変形用
        self.image_paths = []#画像のパス格納用
        self.image_labels = []#画像のラベル格納用
        class_names = os.listdir(root)
        class_names.sort()#クラスをアルファベット順にソート
        for (i,x) in enumerate(class_names):
            temp = glob.glob(root+x+"/*")
            temp.sort()
            self.image_labels.extend([i]*len(temp))
            self.image_paths.extend(temp)

    #num_worker数で並列処理される関数
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')#画像をPILで開く
        if self.transform is not None:
            image = self.transform(image)#画像変形適用
        return image,self.image_labels[index]#画像とラベルを返す

    #データセットの画像数宣言(これが無いとエラー)
    def __len__(self):
        return len(self.image_paths)