# -*- coding: utf-8 -*-
import time

import torch

from utils import MetricLogger, SmoothedValue, accuracy, is_main_process

try:
    from apex import amp
except ImportError:
    amp = None


def train(args, model, device, train_loader, writer, criterion,
          optimizer, epoch, iteration, apex=False):

    # ネットワークを学習用に設定
    # ex.)dropout,batchnormを有効
    model.train()

    # ProgressMeter, AverageMeterの値初期化
    metric_logger = MetricLogger(delimiter=' ')
    header = 'Epoch: [{}]'.format(epoch)

    for images, target in metric_logger.log_every(train_loader, args.print_freq, header):
        images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
        target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送

        output = model(images)  # sofmaxm前まで出力(forward)
        loss = criterion(output, target)  # ネットワークの出力をsoftmax + ラベルとのloss計算

        # losss, accuracyを計算して更新
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  # 予測した中で1番目と3番目までに正解がある率

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
        metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))

        optimizer.zero_grad()  # 勾配初期化

        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()  # パラメータ更新

        if iteration % args.print_freq == 0 and is_main_process():
            # print_freqごとに進行具合とloss表示
            writer.add_scalars('loss', {'train': loss.item()}, iteration)
            writer.add_scalars('acc1', {'train': acc1.item()}, iteration)
            writer.add_scalars('acc5', {'train': acc5.item()}, iteration)
        iteration += 1


def validate(args, model, device, val_loader,
             criterion, writer, iteration):

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Validate:'

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        for images, target in metric_logger.log_every(val_loader, args.print_freq, header):
            images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
            target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送
            output = model(images)  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
            loss = criterion(output, target)  # sum up batch loss

            # losss, accuracyを計算して更新
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # ラベルと合ってる率を算出
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
            metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main_process():
        writer.add_scalars('loss', {'val': metric_logger.loss.global_avg}, iteration)
        writer.add_scalars('acc1', {'val': metric_logger.acc1.global_avg}, iteration)
        writer.add_scalars('acc5', {'val': metric_logger.acc5.global_avg}, iteration)
    return metric_logger.acc1.global_avg


def calibrate_validate(args, model, device, val_loader, criterion, writer, iteration, neval_batches):

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Calibrate:'
    cnt = 0

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        for images, target in metric_logger.log_every(val_loader, args.print_freq, header):
            images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
            target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送
            output = model(images)  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
            loss = criterion(output, target)  # sum up batch loss
            cnt += 1
            # losss, accuracyを計算して更新
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # ラベルと合ってる率を算出
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
            metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))
            if cnt >= neval_batches:
                 break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main_process():
        writer.add_scalars('loss', {'val': metric_logger.loss.global_avg}, iteration)
        writer.add_scalars('acc1', {'val': metric_logger.acc1.global_avg}, iteration)
        writer.add_scalars('acc5', {'val': metric_logger.acc5.global_avg}, iteration)
    return metric_logger.acc1.global_avg


def calculate_validate(args, model, device, val_loader, criterion, writer, iteration):

    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Calibrate:'
    cnt = 0

    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        for images, target in metric_logger.log_every(val_loader, args.print_freq, header):
            images = images.to(device, non_blocking=True)  # gpu使うなら画像をgpuに転送
            target = target.to(device, non_blocking=True)  # gpu使うならラベルをgpuに転送

            inf_start = time.perf_counter()
            output = model(images)  # sofmaxm前まで出力(forward)#評価データセットでのloss計算
            inf_time = time.perf_counter() - inf_start
            loss = criterion(output, target)  # sum up batch loss
            cnt += 1
            # losss, accuracyを計算して更新
            acc1, acc5 = accuracy(output, target, topk=(1, 5))  # ラベルと合ってる率を算出
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=images.size(0))
            metric_logger.meters['acc5'].update(acc5.item(), n=images.size(0))
            metric_logger.meters['inf_time'].update(inf_time, n=images.size(0))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if is_main_process():
        writer.add_scalars('loss', {'val': metric_logger.loss.global_avg}, iteration)
        writer.add_scalars('acc1', {'val': metric_logger.acc1.global_avg}, iteration)
        writer.add_scalars('acc5', {'val': metric_logger.acc5.global_avg}, iteration)
    return metric_logger.acc1.global_avg, metric_logger.inf_time.global_avg
