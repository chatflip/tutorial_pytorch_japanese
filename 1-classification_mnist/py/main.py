# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms

from model import LeNet
from utils import AverageMeter, ProgressMeter, accuracy, seed_everything


def train(args, model, device, train_loader, criterion,
          optimizer, writer, epoch, iteration):

    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # ネットワークを学習用に設定
    # ex.)dropout,batchnormを有効
    model.train()

    end = time.time()  # 1回目の読み込み時間計測用
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)  # 画像のロード時間記録

        images, target = images.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化
        output = model(images)  # sofmaxm前まで出力(forward)
        loss = criterion(output, target)  # ネットワークの出力をsoftmax + ラベルとのloss計算

        # losss, accuracyを計算して更新
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()  # 勾配初期化
        loss.backward()  # 勾配計算(backprop)
        optimizer.step()  # パラメータ更新

        batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
        end = time.time()  # 基準の時間更新

        # print_freqごとに進行具合とloss表示
        if i % args.print_freq == 0:
            progress.display(i)
            writer.add_scalars('loss', {'train': losses.val}, iteration)
            writer.add_scalars('accuracy', {'train': top1.val}, iteration)
        iteration += 1


def validate(args, model, device, val_loader,
             criterion, writer, iteration):

    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix='Validate: ')

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        end = time.time()  # 基準の時間更新
        for i, (images, target) in enumerate(val_loader):
            data_time.update(time.time() - end)  # 画像のロード時間記録

            images, target = images.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化
            output = model(images)  # sofmaxm前まで出力(forward)
            loss = criterion(output, target)  # 評価データセットでのloss計算

            # losss, accuracyを計算して更新
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
            end = time.time()  # 基準の時間更新
            if i % args.print_freq == 0:
                progress.display(i)

    # 精度等格納
    progress.display(i + 1)
    writer.add_scalars('loss', {'val': losses.avg}, iteration)
    writer.add_scalars('accuracy', {'val': top1.avg}, iteration)


def opt():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # Train Validate settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='mini-batch size in train')
    parser.add_argument('--val-batch-size', type=int, default=1000,
                        help='mini-batch size in validate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of total epochs to run')

    # network parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-04,
                        help='weight decay')
    # etc
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--no_cuda',  action='store_true', default=False,
                        help='no cuda')
    parser.add_argument('--resume', type=str, default='weight/MNIST_lenet_10.pth',
                        help='load weight')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for initializing training. ')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='print frequency (default: 10)')
    parser.add_argument('--save-freq', type=int, default=2,
                        help='save every N epoch')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = opt()
    print(args)
    if not os.path.exists('weight'):
        os.mkdir('weight')
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')  # cpuとgpu自動選択
    writer = SummaryWriter(log_dir='log/MNIST')  # tensorboard用のwriter作成

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=2),  # リサイズ
        transforms.ToTensor(),  # テンソル化
        transforms.Normalize((0.1307,), (0.3081,))  # 標準化
    ])

    # MNISTの学習用データ設定
    train_MNIST = datasets.MNIST(
        'data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_MNIST, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_init)

    # MNISTの評価用データ設定
    val_MNIST = datasets.MNIST(
        'data', train=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_MNIST, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_init)

    model = LeNet()  # ネットワーク定義 + gpu使うならcuda化
    if args.evaluate:
        print("use pretrained model : %s" % args.resume)
        param = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(param)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )  # 最適化方法定義
    iteration = 0  # 反復回数保存用

    model.to(device)  # gpu使うならcuda化

    if args.evaluate:
        validate(args, model, device, val_loader, criterion, writer, iteration)
        sys.exit()

    starttime = time.time()  # 実行時間計測(実時間)
    # 学習と評価
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion,
              optimizer, writer, epoch, iteration)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        validate(args, model, device, val_loader, criterion, writer, iteration)
        if epoch % args.save_freq == 0:
            saved_weight = 'weight/MNIST_lenet_{:02d}.pth'.format(epoch)
            torch.save(model.cpu().state_dict(), saved_weight)  # cpuにして保存しないとgpuメモリに若干残る
            model.to(device)

    writer.close()  # tensorboard用のwriter閉じる
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {:d}h {:d}m {:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))
