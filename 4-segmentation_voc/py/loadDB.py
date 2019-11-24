# -*- coding: utf-8 -*-
import os

from PIL import Image
from torch.utils.data import Dataset


class VOCdetectionDB(Dataset):
    # 初期化
    def __init__(self, root, phase, transform=None):
        self.phase = phase
        self.transform = transform  # 画像変形用
        self.img_paths, self.ann_paths = split_trainval(root, phase)
        print(len(self.img_paths))
    # データセットの画像数宣言(これが無いとエラー)
    def __len__(self):
        return len(self.img_paths)

# txtファイルからアノテーション情報を取り出す
def split_trainval(root, phase):
    id_txt = os.path.join(root, 'ImageSets', 'Main', phase+'.txt')
    imgs = []
    anns = []
    with open(id_txt, 'r') as f:
        file_ids = [i.replace('\n', '') for i in f.readlines()]
        for file_id in file_ids:
            imgs.append(os.path.join(root, 'JPEGImages', file_id+'.jpg'))
            anns.append(os.path.join(root, 'Annotations', file_id+'.xml'))
    return imgs, anns

def parse_xml(anns):
    return 

"""
    # 初期化
    def __init__(self, root, transform=None):
        self.transform = transform  # 画像変形用
        self.image_paths = []  # 画像のパス格納用
        self.image_labels = []  # 画像のラベル格納用
        class_names = os.listdir(root)
        class_names.sort()  # クラスをアルファベット順にソート
        for (i, x) in enumerate(class_names):
            temp = glob.glob(root+x+"/*")
            temp.sort()
            self.image_labels.extend([i]*len(temp))
            self.image_paths.extend(temp)

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
"""