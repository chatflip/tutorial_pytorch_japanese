# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from loadDB import VOCdetectionDB, detect_collate_fn
from loss import MultiBoxLoss
from model import SSD
from train_test import train, test
from transform_detector import ColorJitter, Expand, RandomResizedCrop, RandomHorizontalFlip, Resize, ToTensor, Normalize
from utils import seed_everything

ssd_cfg = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}


if __name__ == "__main__":
    args = opt()
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定
    # フォルダが存在してなければ作る
    if not os.path.exists(args.path2weight):
        os.mkdir(args.path2weight)
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # gpu使えるか and 使うか
    device = torch.device("cuda" if use_cuda else "cpu")  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir="log/VOCdetect")  # tensorboard用のwriter作成

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([
                      ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4, hue=0.25),
                      Expand([0.485, 0.456, 0.406]),
                      RandomResizedCrop((args.crop_size, args.crop_size), scale=(0.5, 1.5)),  # クロップ
                      RandomHorizontalFlip(p=0.5),  # 左右反転
                      Resize(size=(args.img_size, args.img_size), interpolation=5),  # リサイズ
                      ToTensor(),  # テンソル化
                      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 標準化
                      ])
    val_transform = transforms.Compose([
                    Resize(size=(args.img_size, args.img_size), interpolation=5),  # リサイズ
                    ToTensor(),  # テンソル化
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 標準化
                    ])

    # PascalVOCの学習用データ設定
    train_VOCdetect = VOCdetectionDB(
        root='data/VOCdevkit/VOC2007', phase='train',
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_VOCdetect, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=detect_collate_fn, pin_memory=True,
        drop_last=True, worker_init_fn=worker_init)

    # AnimeFaceの評価用データ設定
    val_VOCdetect = VOCdetectionDB(
        root='data/VOCdevkit/VOC2007', phase='val',
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_VOCdetect, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=detect_collate_fn, pin_memory=True,
        drop_last=False, worker_init_fn=worker_init)
    
    model = SSD(phase='train', cfg=ssd_cfg)
    vgg_weights = torch.load('weight/vgg16_reducedfc.pth')
    model.vgg.load_state_dict(vgg_weights)
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:  # バイアス項がある場合
                nn.init.constant_(m.bias, 0.0)
    # Heの初期値を適用
    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)


    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)  # 最適化方法定義
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 37], gamma=0.1)  # 学習率の軽減スケジュール
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
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
            saved_weight = "{}VOC07_SSD300_{}.pth".format(
                args.path2weight, epoch)
            torch.save(model.cpu().state_dict(), saved_weight)
            model.to(device)
    writer.close()  # tensorboard用のwriter閉じる
    endtime = time.time()
    interval = endtime - starttime
    # 実行時間表示
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))
