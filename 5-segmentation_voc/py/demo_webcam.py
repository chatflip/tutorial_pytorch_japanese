import argparse
import sys
import time

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def opt():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Model")
    # parser.add_argument('--weight_path', type=str, default='weight/voc2012_DeepLabV3_mobilenet_v2_512_512.pth')
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weight/voc2012_Unet_efficientnet-b3_512_512.pth",
    )
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--inference_height", type=int, default=256)
    parser.add_argument("--inference_width", type=int, default=256)
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


def get_transform(args):
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
    )
    transform = A.Compose(
        [
            A.Resize(args.inference_height, args.inference_width),
            normalize,
            ToTensorV2(),
        ]
    )
    return transform


def preprocess_image(image, transform):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image=image)["image"]
    return tensor.unsqueeze(0)


def decode_result(result):
    result = result.squeeze(0)
    index = result.argmax(0)
    return index.cpu().numpy()


def make_overlay(frame, seg_image):
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, seg_image.shape[:2], cv2.INTER_AREA)
    overlay_image = cv2.addWeighted(frame, 0.4, seg_image, 0.6, 0.8)
    overlay_image = cv2.resize(overlay_image, (width, height), cv2.INTER_CUBIC)
    return overlay_image


def main(args):
    print(args)
    color_map = get_pascal_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cpuとgpu自動選択
    model = torch.load(args.weight_path).eval().to(device)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 60.0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"video FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    transform = get_transform(args)

    while cap.isOpened():
        start_time = time.perf_counter()
        ret, frame = cap.read()

        tensor = preprocess_image(frame, transform)
        tensor = tensor.to(device)
        preprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        with torch.no_grad():
            result = model(tensor)
        inference_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        index = decode_result(result)
        seg_image = color_map[index]
        overlay = make_overlay(frame, seg_image)
        postprocess_time = time.perf_counter() - start_time

        cv2.imshow("aaa", overlay)
        interval = preprocess_time + inference_time + postprocess_time
        sys.stdout.write("\rFPS: {:.1f}".format(1.0 / interval))
        sys.stdout.flush()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    args = opt()
    main(args)
