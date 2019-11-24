# -*- coding: utf-8 -*-
from __future__ import print_function
import time

import torch

from utils import AverageMeter


# 精度出す関数
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
       for the specified values of k"""
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


def train(args, model, device, train_loader,
          writer, criterion, optimizer, epoch, iteration):
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
    for i, (data, targets) in enumerate(train_loader):
        print(data.shape)
        print(targets.shape)
        data_time.update(time.time() - end)  # 画像のロード時間記録
        data, targets = data.to(device), targets.to(device)  # gpu使うなら画像とラベルcuda化
        optimizer.zero_grad()  # 勾配初期化
        output = model(data)  # sofmaxm前まで出力(forward)
        print(output)

        # 損失の計算
        loss_l, loss_c = criterion(outputs, targets)
        loss = loss_l + loss_c
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
        #acc1, acc3 = accuracy(output, target, topk=(1, 3))  # 予測した中で1番目と3番目までに正解がある率
        losses.update(loss.item(), data.size(0))
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
            writer.add_scalars("loss",
                               {"train": losses.val}, iteration)
            writer.add_scalars("accuracy",
                               {"train": top1.val}, iteration)
        iteration += 1


def test(args, model, device, test_loader,
         writer, criterion, iteration):
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
    writer.add_scalars("loss",
                       {"test": losses.avg}, iteration)
    writer.add_scalars("accuracy", {"test": 100. * top1.avg}, iteration)
    print('Test Acc@1 {top1.avg:.4f}\t'
          'loss {loss.avg:.4f}\t'
          'Time {batch_time.avg:.3f}'.format(
            top1=top1, loss=losses, batch_time=batch_time))
