# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
require pytorch 0.4.0
        torchvision 0.2.1
"""

from __future__ import print_function
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from alex import alex
from args import opt
from loadDB import AnimeFaceDB
from train_test import train, test


if __name__ == "__main__":
    args = opt()
    # フォルダが存在してなければ作る
    if not os.path.exists(args.path2weight):
        os.mkdir(args.path2weight)
    torch.manual_seed(args.seed)  # torchとtorchvisionで使う乱数固定
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # gpu使えるか and 使うか
    device = torch.device("cuda" if use_cuda else "cpu")  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir="log/AnimeFace")  # tensorboard用のwriter作成

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([
                      transforms.Resize((args.img_size, args.img_size), interpolation=5),  # リサイズ
                      transforms.RandomCrop((args.crop_size, args.crop_size)),  # クロップ
                      transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                      transforms.ToTensor(),  # テンソル化
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
                      ])
    test_transform = transforms.Compose([
                     transforms.Resize((args.crop_size, args.crop_size), interpolation=5),  # リサイズ
                     transforms.ToTensor(),  # テンソル化
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
                     ])

    # AnimeFaceの学習用データ設定
    train_AnimeFace = AnimeFaceDB(
        args.path2db+"train/", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_AnimeFace, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True)
    # AnimeFaceの評価用データ設定
    test_AnimeFace = AnimeFaceDB(
        args.path2db+"test/", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_AnimeFace, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False)

    model = alex(pretrained=True, num_classes=args.numof_classes).to(device)  # ネットワーク定義 + gpu使うならcuda化
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)  # 最適化方法定義
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 37], gamma=0.1)  # 学習率の軽減スケジュール
    criterion = nn.CrossEntropyLoss().to(device)
    starttime = time.time()  # 実行時間計測(実時間)
    iteration = 0  # 反復回数保存用
    # 学習と評価
    for epoch in range(1, args.epochs + 1):
        scheduler.step()  # epoch 0 スタートだから+1して数値を合わせてスケジューラ開始
        train(args, model, device, train_loader, writer, criterion, optimizer, epoch, iteration)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        test(args, model, device, test_loader, writer, criterion, iteration)
        # 重み保存
        if epoch % args.save_interval == 0:
            saved_weight = "{}AnimeFace_alex_{}.pth".format(
                args.path2weight, epoch)
            torch.save(model.cpu().state_dict(), saved_weight)
            model.to(device)
    writer.close()  # tensorboard用のwriter閉じる
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))
