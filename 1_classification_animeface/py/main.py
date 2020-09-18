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
from datasets import AnimeFaceDataset
from model import mobilenet_v2
from train_val import train, validate
from utils import get_worker_init, seed_everything


def load_data(args):
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

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=2),  # リサイズ
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),  # テンソル化
        normalize  # 標準化
    ])

    # AnimeFaceの学習用データ設定
    train_dataset = AnimeFaceDataset(
        os.path.join(args.path2db, 'train'),
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=get_worker_init(args.seed))

    # AnimeFaceの評価用データ設定
    val_dataset = AnimeFaceDataset(
        os.path.join(args.path2db, 'val'),
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        worker_init_fn=get_worker_init(args.seed))

    return train_loader, val_loader


def main():
    args = opt()
    print(args)
    seed_everything(args.seed)  # 乱数テーブル固定
    os.makedirs(args.path2weight, exist_ok=True)
    writer = SummaryWriter(log_dir='log/animeface')  # tensorboard用のwriter作成
    # torch.backends.cudnn.benchmark = True  # 再現性を無くして高速化

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    multigpu = torch.cuda.device_count() > 1  # グラボ2つ以上ならmultigpuにする

    train_loader, val_loader = load_data(args)

    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 最適化方法定義

    iteration = 0  # 反復回数保存用
    # 評価だけやる
    if args.evaluate:
        print("use pretrained model : %s" % args.resume)
        weight_name = '{}/{}_mobilenetv2_best.pth'.format(args.path2weight, args.exp_name)
        param = torch.load(weight_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(param)
        if multigpu:
            model = nn.DataParallel(model)
        model.to(device)  # gpu使うならcuda化
        validate(args, model, device, val_loader, criterion, writer, iteration)
        sys.exit()

    model_without_dp = model
    if multigpu:
        model = nn.DataParallel(model)
        model_without_dp = model.module

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)  # 学習率の軽減スケジュール

    best_acc = 0.0
    # 学習再開時の設定
    if args.restart:
        checkpoint = torch.load(
            'weight/{}_checkpoint.pth'.format(args.exp_name), map_location='cpu'
        )
        model_without_dp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        iteration = args.epochs * len(train_loader)

    starttime = time.time()  # 実行時間計測(実時間)
    # 学習と評価
    for epoch in range(args.start_epoch, args.epochs + 1):
        train(args, model, device, train_loader, writer,
              criterion, optimizer, epoch, iteration, args.apex)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        acc = validate(args, model, device, val_loader, criterion, writer, iteration)
        scheduler.step()  # 学習率のスケジューリング更新
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            print('Acc@1 best: {}'.format(best_acc))
            weight_name = '{}/{}_mobilenetv2_best.pth'.format(args.path2weight, args.exp_name)
            torch.save(model_without_dp.cpu().state_dict(), weight_name)
            checkpoint = {
                'model': model_without_dp.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'args': args
            }
            torch.save(
                checkpoint, 'weight/{}_checkpoint.pth'.format(args.exp_name)
            )
            model.to(device)

    writer.close()  # tensorboard用のwriter閉じる
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))


if __name__ == '__main__':
    main()
