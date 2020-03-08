# -*- coding: utf-8 -*-
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from loadDB import AnimeFaceDB
from model import mobilenet_v2
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

    device = torch.device('cpu')  # cpu only
    writer = SummaryWriter(log_dir='log/AnimeFace')  # tensorboard用のwriter作成
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
        dataset=val_AnimeFace, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
        worker_init_fn=worker_init)

    model = mobilenet_v2(pretrained=False, num_classes=args.num_classes)
    weight_name = "weight/AnimeFace_mobilenetv2_float_epoch100.pth"
    model = load_weight(model, weight_name)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    saved_weight = 'weight/AnimeFace_dynamic_quantization_mobilenetv2_100.pth'
    torch.save(quantized_model.state_dict(), saved_weight)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)  # 最適化方法定義
    criterion = nn.CrossEntropyLoss()
    iteration = 0  # 反復回数保存用

    torch.set_num_threads(1)

    starttime = time.time()  # 実行時間計測(実時間)
    validate(args, model, device, val_loader, criterion, writer, iteration)
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('float model elapsed time = {0:d}m {1:d}s'.format(
        int((interval % 3600) / 60), int((interval % 3600) % 60)))

    starttime = time.time()  # 実行時間計測(実時間)
    validate(args, quantized_model, device, val_loader, criterion, writer, iteration)
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('dynamic quantized model elapsed time = {0:d}m {1:d}s'.format(
        int((interval % 3600) / 60), int((interval % 3600) % 60)))

    writer.close()  # tensorboard用のwriter閉じる
