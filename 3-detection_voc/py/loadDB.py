# -*- coding: utf-8 -*-
from enum import IntEnum
import os
import xml.etree.ElementTree as ET


import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class detAnn(IntEnum):
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3
    label = 4
    width = 5
    height = 6


voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']


class VOCdetectionDB(Dataset):
    # 初期化
    def __init__(self, root, phase, transform=None):
        self.phase = phase
        self.transform = transform  # 画像変形用
        self.img_paths, self.ann_paths = split_trainval(root, phase)

    # num_worker数で並列処理される関数
    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = Image.open(image_path).convert('RGB')  # 画像をPILで開く
        ann_list = parse_xml(self.ann_paths[index])
        samples = {'img': image, 'ann': np.array(ann_list)}
        if self.transform is not None:
            samples = self.transform(samples)  # 画像とアノテーションを変換
        return samples['img'], samples['ann']  # 画像とラベルを返す

    # データセットの画像数宣言(これが無いとエラー)
    def __len__(self):
        return len(self.img_paths)


# __getitem__で集めた画像とアノテーションを
# 画像(batch_size, channel, height, width)
# アノテーション(batch_size, batch内の最大アノテーション数, アノテーション数5)に変更する
def detect_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, targets = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(target) for target in targets]
    batch_size = images.shape[0]
    num_annotations = 5
    annotations = torch.zeros((batch_size, max(lengths), num_annotations), dtype=torch.double)
    for i, target in enumerate(targets):
        end = lengths[i]
        annotations[i, :end, :] = torch.from_numpy(target[:, :num_annotations])
    return images, annotations

# txtファイルから画像とアノテーションのファイル名を取り出す
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


# アノテーションファイルからアノテーションを取り出す
def parse_xml(file_ann):
    xml = ET.parse(file_ann).getroot()
    size = xml.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    bbox_list = []
    for obj in xml.iter('object'):
        difficult = int(obj.find('difficult').text)
        if difficult == 1:  # 難しいやつ使わない
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        # 原点(1, 1)から(0, 0)に変更する
        xmin = float(bbox.find('xmin').text) - 1.0
        ymin = float(bbox.find('ymin').text) - 1.0
        xmax = float(bbox.find('xmax').text) - 1.0
        ymax = float(bbox.find('ymax').text) - 1.0
        label_idx = voc_classes.index(name)
        ann_dict = [xmin, ymin, xmax, ymax,
                    label_idx, width, height]
        bbox_list.append(ann_dict)
    return bbox_list
