# -*- coding: utf-8 -*-
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
from model_float import mobilenet_v2
from train_val import validate
from utils import seed_everything


def print_size_of_model(model):
    torch.jit.save(model, "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/(1e+6))
    os.remove('temp.p')


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
        os.path.join(args.path2db, 'val'),
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_AnimeFace, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
        worker_init_fn=worker_init)

    criterion = nn.CrossEntropyLoss()
    iteration = 0  # 反復回数保存用

    saved_weight = 'weight/AnimeFace_mobilenetv2_float_best.pth'
    saved_script = 'weight/AnimeFace_mobilenetv2_float_script_best.pth'
    model = mobilenet_v2(num_classes=args.num_classes).eval()
    model = load_weight(model, saved_weight)
    torch.jit.save(torch.jit.script(model), saved_script)

    torch.set_num_threads(1)
    model = torch.jit.load(saved_script)
    print('float model')
    print_size_of_model(model)
    #validate(args, model, device, val_loader, criterion, writer, iteration)

    model = torch.jit.load('weight/AnimeFace_mobilenetv2_dynamic_quantization_script_best.pth')
    print('dynamic quantization model')
    print_size_of_model(model)
    #validate(args, model, device, val_loader, criterion, writer, iteration)

    model = torch.jit.load('weight/AnimeFace_mobilenetv2_static_quantization_script_best.pth')
    print('static quantization model')
    print_size_of_model(model)
    validate(args, model, device, val_loader, criterion, writer, iteration)

    model = torch.jit.load('weight/AnimeFace_mobilenetv2_qat_script_best.pth')
    print('quantization aware training model')
    print_size_of_model(model)
    #validate(args, model, device, val_loader, criterion, writer, iteration)

    writer.close()  # tensorboard用のwriter閉じる
