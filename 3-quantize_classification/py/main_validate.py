# -*- coding: utf-8 -*-
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from loadDB import AnimeFaceDB
from train_val import validate
from utils import seed_everything


def print_size_of_model(model):
    torch.jit.save(model, "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/(1e+6))
    os.remove('temp.p')


if __name__ == '__main__':
    args = opt()
    print(args)
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定
    device = torch.device('cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir='log/AnimeFace/test')  # tensorboard用のwriter作成
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    val_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=2),  # リサイズ
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),  # テンソル化
        normalize  # 標準化
    ])

    # AnimeFaceの評価用データ設定
    val_AnimeFace = AnimeFaceDB(
        args.path2db+'/val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_AnimeFace, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
        worker_init_fn=worker_init)

    criterion = nn.CrossEntropyLoss()
    iteration = 0  # 反復回数保存用

    torch.set_num_threads(1)
    model = torch.jit.load('weight/AnimeFace_mobilenetv2_script_float_epoch100.pth')
    print('float model')
    print_size_of_model(model)
    validate(args, model, device, val_loader, criterion, writer, iteration)

    model = torch.jit.load('weight/AnimeFace_mobilenetv2_script_dynamic_quantization_epoch100.pth')
    print('dynamic quantization model')
    print_size_of_model(model)
    validate(args, model, device, val_loader, criterion, writer, iteration)

    model = torch.jit.load('weight/AnimeFace_mobilenetv2_script_static_quantization_epoch100.pth')
    print('dynamic quantization model')
    print_size_of_model(model)
    validate(args, model, device, val_loader, criterion, writer, iteration)

    model = torch.jit.load('weight/AnimeFace_mobilenetv2_script_qat_epoch20.pth')
    print('quantization aware training model')
    print_size_of_model(model)
    validate(args, model, device, val_loader, criterion, writer, iteration)

    writer.close()  # tensorboard用のwriter閉じる
