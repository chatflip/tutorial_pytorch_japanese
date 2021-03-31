# -*- coding: utf-8 -*-
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from utils import AverageMeter, ProgressMeter, accuracy


def train(
    args,
    model,
    device,
    train_loader,
    writer,
    criterion,
    optimizer,
    scheduler,
    epoch,
    iteration,
    apex=False,
):

    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.5f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # ネットワークを学習用に設定
    # ex.)dropout,batchnormを有効
    model.train()

    if apex:
        scaler = GradScaler()

    end = time.time()  # 1回目の読み込み時間計測用
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)  # 画像のロード時間記録

        images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
        target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送

        if apex:
            with autocast():
                output = model(images)  # sofmaxm前まで出力(forward)
                loss = criterion(output, target)  # ネットワークの出力をsoftmax + ラベルとのloss計算
        else:
            output = model(images)  # sofmaxm前まで出力(forward)
            loss = criterion(output, target)  # ネットワークの出力をsoftmax + ラベルとのloss計算

        # losss, accuracyを計算して更新
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 予測した中で1番目と3番目までに正解がある率
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()  # 勾配初期化

        if apex:
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # パラメータ更新
            scaler.update()
        else:
            loss.backward()
            optimizer.step()  # パラメータ更新
        scheduler.step();
        batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
        end = time.time()  # 基準の時間更新

        # print_freqごとに進行具合とloss表示
        if i % args.print_freq == 0:
            progress.display(i)
            writer.log_metric("train/loss", losses.val, step=iteration)
            writer.log_metric("train/Acc1", top1.val.item(), step=iteration)
            writer.log_metric("train/Acc5", top5.val.item(), step=iteration)
        iteration += 1


def validate(args, model, device, val_loader, criterion, writer, iteration):

    # ProgressMeter, AverageMeterの値初期化
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.5f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Validate: ",
    )

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        end = time.time()  # 基準の時間更新
        for i, (images, target) in enumerate(val_loader):
            data_time.update(time.time() - end)  # 画像のロード時間記録

            images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
            target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送
            output = model(images)  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
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
    writer.log_metric("val/loss", losses.avg, step=iteration)
    writer.log_metric("val/Acc1", top1.avg.item(), step=iteration)
    writer.log_metric("val/Acc5", top5.avg.item(), step=iteration)
    return top1.avg.item()
