# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
"""

from __future__ import print_function
import fnmatch
import glob
import os
from PIL import Image
import random


def make_dir(save_root, folder_name):
    if folder_name == "pass":
        if not os.path.exists(save_root):
            os.mkdir(save_root)
    elif not os.path.exists(os.path.join(save_root, folder_name)):
        os.mkdir(os.path.join(save_root, folder_name))

if __name__ == "__main__":
    dataset_root = "./animeface-character-dataset/thumb/"
    save_root = "./dataset"
    make_dir(save_root, "pass")
    make_dir(save_root, "train")
    make_dir(save_root, "test")
    train_rate = 0.75
    class_names = os.listdir(dataset_root)
    class_names.sort()
    random.seed(a=1)
    for class_name in class_names:
        file_names = glob.glob(os.path.join(dataset_root, class_name, "*.png"))
        img_names = fnmatch.filter(file_names, "*.png")
        class_length = len(img_names)
        if class_length <= 4:
            print("passed ", class_name)
            continue
        else:
            print(class_name)
            random.shuffle(img_names)
            make_dir(save_root, os.path.join("train", class_name))
            make_dir(save_root, os.path.join("test", class_name))
            for i, img_name in enumerate(img_names):
                img = Image.open(img_name)
                name = os.path.basename(img_name)
                if i < class_length * train_rate:
                    img.save(os.path.join(save_root, "train", class_name, name))
                else:
                    img.save(os.path.join(save_root, "test", class_name, name))
                img.close()
