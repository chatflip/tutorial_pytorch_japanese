# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from datasets import Food101Dataset
from model import mobilenet_v2
from train_val import calculate_validate
from utils import get_worker_init, seed_everything


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


def load_data(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),  # リサイズ&クロップ
        transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
        transforms.ToTensor(),  # テンソル化
        normalize  # 標準化
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=2),  # リサイズ
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),  # テンソル化
        normalize  # 標準化
    ])

    # Food101の学習用データ設定
    train_dataset = Food101Dataset(
        args.path2db, 'train',
        transform=train_transform,
    )

    # Food101の評価用データ設定
    val_dataset = Food101Dataset(
        args.path2db, 'test',
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=get_worker_init(args.seed))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        worker_init_fn=get_worker_init(args.seed))

    return train_loader, val_loader


if __name__ == '__main__':
    args = opt()
    print(args)
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定
    device = torch.device('cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir='log/food101/test')  # tensorboard用のwriter作成
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader, val_loader = load_data(args)

    criterion = nn.CrossEntropyLoss()
    iteration = 0  # 反復回数保存用

    saved_weight = 'weight/food101_mobilenetv2_best.pth'
    saved_script = 'weight/food101_mobilenetv2_script_float.pth'
    model = mobilenet_v2(num_classes=args.num_classes).eval()
    model = load_weight(model, saved_weight)
    torch.jit.save(torch.jit.script(model), saved_script)

    # torch.set_num_threads(1)
    model = torch.jit.load(saved_script)
    print('float model')
    print_size_of_model(model)
    acc, inf_time = calculate_validate(args, model, device, val_loader, criterion, writer, iteration)
    print('Accracy: {} infTime: {}'.format(acc, inf_time))

    model = torch.jit.load('weight/food101_mobilenetv2_dynamic_quantize_script.pt')
    print('dynamic quantization model')
    print_size_of_model(model)
    acc, inf_time = calculate_validate(args, model, device, val_loader, criterion, writer, iteration)
    print('Accracy: {} infTime: {}'.format(acc, inf_time))


    model = torch.jit.load('weight/food101_mobilenetv2_static_quantize_script.pt')
    print('static quantization model')
    print_size_of_model(model)
    acc, inf_time = calculate_validate(args, model, device, val_loader, criterion, writer, iteration)
    print('Accracy: {} infTime: {}'.format(acc, inf_time))


    model = torch.jit.load('weight/food101_mobilenetv2_qat_script.pt')
    print('quantization aware training model')
    print_size_of_model(model)
    acc, inf_time = calculate_validate(args, model, device, val_loader, criterion, writer, iteration)
    print('Accracy: {} infTime: {}'.format(acc, inf_time))

    writer.close()  # tensorboard用のwriter閉じる
