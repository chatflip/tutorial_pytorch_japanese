# -*- coding: utf-8 -*-
import copy
import os
import time
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from loadDB import AnimeFaceDB
from model_quantize import mobilenet_v2
from train_val import train, validate
from utils import seed_everything


def load_weight(model, weight_name):
    assert os.path.isfile(weight_name), "don't exists weight: {}".format(weight_name)
    print("use pretrained model : %s" % weight_name)
    param = torch.load(weight_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(param)
    return model


if __name__ == '__main__':
    args = opt()
    print(args)
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定

    device = torch.device('cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir='log/AnimeFace/static_quantize')  # tensorboard用のwriter作成
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=2),  # リサイズ
        transforms.RandomCrop(args.crop_size),  # クロップ
        transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
        transforms.ToTensor(),  # テンソル化
        normalize  # 標準化
    ])

    train_AnimeFace = AnimeFaceDB(
        os.path.join(args.path2db, 'train'),
        transform=train_transform)
    indices = list(range(len(train_AnimeFace)))
    random.shuffle(indices)
    indices = indices[:args.batch_size * args.num_calibration_batches]
    subset = torch.utils.data.Subset(
            train_AnimeFace,
            indices=indices)
    train_loader = torch.utils.data.DataLoader(
        dataset=subset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=False, drop_last=True,
        worker_init_fn=worker_init)

    criterion = nn.CrossEntropyLoss()

    model = mobilenet_v2(pretrained=False, num_classes=args.num_classes)
    weight_name = 'weight/AnimeFace_mobilenetv2_float_best.pth'
    model = load_weight(model, weight_name)

    quantized_model = copy.deepcopy(model)
    quantized_model.eval()
    quantized_model.fuse_model()
    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    quantized_model.qconfig = torch.quantization.get_default_qconfig(args.backend)
    torch.quantization.prepare(quantized_model, inplace=True)

    iteration = 0
    starttime = time.time()  # 実行時間計測(実時間)
    validate(args, quantized_model, device, train_loader, criterion, writer, iteration)

    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)
    saved_weight = 'weight/AnimeFace_mobilenetv2_static_quantization_best.pth'
    torch.save(quantized_model.state_dict(), saved_weight)
    saved_script = 'weight/AnimeFace_mobilenetv2_static_quantization_script_best.pth'
    torch.jit.save(torch.jit.script(quantized_model), saved_script)

    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))