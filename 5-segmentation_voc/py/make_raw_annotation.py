import argparse
import os

import cv2
import numpy as np
import pandas as pd


def opt():
    parser = argparse.ArgumentParser(description="PyTorch Penn-Fudan")
    parser.add_argument(
        "--voc_root", type=str, default="./../datasets/VOCdevkit/VOC2012"
    )
    args = parser.parse_args()
    return args


def get_pascal_labels():
    VOC_COLOR_MAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    color_map = np.array(VOC_COLOR_MAP, dtype=np.uint8)
    color_map = color_map[:, ::-1]  # RGB2BGR
    return color_map


def main(args):
    color_map = get_pascal_labels()
    num_class = color_map.shape[0]
    dst_root = os.path.join(args.voc_root, "SegmentationRaw")
    os.makedirs(dst_root, exist_ok=True)
    image_list_path = os.path.join(
        args.voc_root, "ImageSets", "Segmentation", "trainval.txt"
    )
    image_lists = pd.read_table(image_list_path, header=None)
    for idx, image_list in image_lists.iterrows():
        src_path = f"{args.voc_root}/SegmentationClass/{image_list.values[0]}.png"
        dst_path = f"{args.voc_root}/SegmentationRaw/{image_list.values[0]}.png"
        annotation_color = cv2.imread(src_path)
        raw_annotation = np.zeros(annotation_color.shape[:2])
        for idx in range(num_class):
            valid = np.all(annotation_color == color_map[idx], axis=-1)
            rs, cs = valid.nonzero()
            raw_annotation[rs, cs] = idx
        raw_annotation = np.array(raw_annotation, dtype=np.uint8)
        cv2.imwrite(dst_path, raw_annotation)


if __name__ == "__main__":
    args = opt()
    main(args)
