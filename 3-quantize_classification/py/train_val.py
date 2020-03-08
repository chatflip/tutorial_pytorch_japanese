# -*- coding: utf-8 -*-
import time

import torch

from utils import AverageMeter, ProgressMeter, accuracy


def train(args, model, device, train_loader, writer, criterion,
          optimizer, epoch, iteration):

    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
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
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 予測した中で1番目と3番目までに正解がある率
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
    losses = AverageMeter('Loss', ':6.5f')
    inf_time = AverageMeter('infTime', ':6.3f')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, inf_time, losses, top1, top5],
        prefix='Validate: ')

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        end = time.time()  # 基準の時間更新
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化

            inf_start = time.time()
            output = model(images)  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
            inf_time.update(time.time() - inf_start)

            loss = criterion(output, target)  # sum up batch loss

            # losss, accuracyを計算して更新
            losses.update(loss.item(), images.size(0))
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # ラベルと合ってる率を算出
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
