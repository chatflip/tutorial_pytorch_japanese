# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
require pytorch 0.4.0
        torchvision 0.2.1
"""

from __future__ import print_function
import argparse
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from alex import alex
from loadDB import AnimeFaceDB
from utils import AverageMeter


# 精度出す関数
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(args, model, device, train_loader, writer, criterion, optimizer, epoch, iteration):
    # ネットワークを学習用に設定
    # ex.)dropout,batchnormを有効
    model.train()
    # AverageMeterの値の初期化
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()  # 1回目の読み込み時間計測用
    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)  # 画像のロード時間記録
        data, target = data.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化
        optimizer.zero_grad()  # 勾配初期化
        output = model(data)  # sofmaxm前まで出力(forward)
        loss = criterion(output, target)  # ネットワークの出力をsoftmax + ラベルとのloss計算
        acc1, acc3 = accuracy(output, target, topk=(1, 3))  # 予測した中で1番目と3番目までに正解がある率
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top3.update(acc3[0], data.size(0))
        loss.backward()  # 勾配計算(backprop)
        optimizer.step()  # パラメータ更新
        batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
        end = time.time()  # 基準の時間更新
        # log_intervalごとに進行具合とloss表示
        if i % args.log_interval == 0:
            print('Epoch: [{0}][{1:5d}/{2:5d}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))
            writer.add_scalars("loss", {"train": losses.val}, iteration)
            writer.add_scalars("accuracy", {"train": top1.val}, iteration)
        iteration += 1


def test(args, model, device, test_loader, writer, criterion, iteration):
    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()
    # AverageMeterの値の初期化
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        end = time.time()  # 基準の時間更新
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化
            output = model(data)  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
            loss = criterion(output, target)  # sum up batch loss
            losses.update(loss.item(), data.size(0))
            acc1, _ = accuracy(output, target, topk=(1, 3))  # ラベルと合ってる率を算出
            top1.update(acc1[0], data.size(0))
            batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
            end = time.time()  # 基準の時間更新
    # test_loss格納
    writer.add_scalars("loss", {"test": losses.avg}, iteration)
    writer.add_scalars("accuracy", {"test": 100. * top1.avg}, iteration)
    print('Test Acc@1 {top1.avg:.4f}\t loss {loss.avg:.4f}\t Time {batch_time.avg:.3f}'.format(top1=top1, loss=losses, batch_time=batch_time))


def opt():
    parser = argparse.ArgumentParser(description="PyTorch AnimeFace")
    parser.add_argument("--path2db", type=str, default="./dataset/", help="path to database")
    parser.add_argument("--path2weight", type=str, default="./weight/", help="path to weight")
    # Train Test settings
    parser.add_argument("--batch-size", type=int, default=50, metavar="N", help="input batch size for training (default: 50)")
    parser.add_argument("--test-batch-size", type=int, default=100, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--save-interval", type=int, default=10, help="save every N epoch")
    parser.add_argument("--numof_classes", type=int, default=176, help="num of classes")
    # network parameters
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    # etc
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--img_size", type=int, default=256, help="image size")
    parser.add_argument("--crop_size", type=int, default=224, help="crop size")
    parser.add_argument("--num_workers", type=int, default=4, help="num of pallarel threads(dataloader)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = opt()
    # フォルダが存在してなければ作る
    if not os.path.exists(args.path2weight):
        os.mkdir(args.path2weight)
    torch.manual_seed(args.seed)  # torchとtorchvisionで使う乱数固定
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # gpu使えるか and 使うか
    device = torch.device("cuda" if use_cuda else "cpu")  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir="./log/AnimeFace")  # tensorboard用のwriter作成

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), interpolation=5),  # リサイズ
                                          transforms.RandomCrop((args.crop_size, args.crop_size)),  # クロップ
                                          transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                                          transforms.ToTensor(),  # テンソル化
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
                                          ])
    test_transform = transforms.Compose([transforms.Resize((args.crop_size, args.crop_size), interpolation=5),  # リサイズ
                                         transforms.ToTensor(),  # テンソル化
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
                                         ])

    # AnimeFaceの学習用データ設定
    train_AnimeFace = AnimeFaceDB(args.path2db+"train/", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_AnimeFace, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)
    # AnimeFaceの評価用データ設定
    test_AnimeFace = AnimeFaceDB(args.path2db+"test/", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_AnimeFace, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers,
                                              pin_memory=True, drop_last=False)

    model = alex(pretrained=True, num_classes=args.numof_classes).to(device)  # ネットワーク定義 + gpu使うならcuda化
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 最適化方法定義
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 37], gamma=0.1)  # 学習率の軽減スケジュール
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
            saved_weight = "{}AnimeFace_alex_{}.pth".format(args.path2weight, epoch)
            torch.save(model.cpu().state_dict(), saved_weight)
            model.to(device)
    writer.close()  # tensorboard用のwriter閉じる
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print("elapsed time = {0:d}h {1:d}m {2:d}s".format(int(interval / 3600), int((interval % 3600) / 60), int((interval % 3600) % 60)))
