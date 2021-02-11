# -*- coding: utf-8 -*-
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from datasets import Food101Dataset
from model import mobilenet_v2
from train_val import train, validate
from utils import get_worker_init, seed_everything, init_distributed_mode, save_on_master

try:
    from apex import amp
except ImportError:
    amp = None


def load_data(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.crop_size),  # リサイズ&クロップ
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

    # Food101の学習用データ設定
    train_dataset = Food101Dataset(
        args.path2db, 'train',
        transform=train_transform,
    )

    # Food101の評価用データ設定
    val_dataset = Food101Dataset(
        args.path2db, 'test',
        transform=val_transform
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        worker_init_fn=get_worker_init(args.seed))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        worker_init_fn=get_worker_init(args.seed))

    return train_loader, val_loader, train_sampler, val_sampler


def main():
    args = opt()
    print(args)
    if args.apex and amp is None:
        raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                           "to enable mixed-precision training.")
    seed_everything(args.seed)  # 乱数テーブル固定
    os.makedirs(args.path2weight, exist_ok=True)
    writer = SummaryWriter(log_dir='log/{}'.format(args.exp_name))  # tensorboard用のwriter作成
    torch.backends.cudnn.benchmark = True  # 再現性を無くして高速化

    init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    train_loader, val_loader, train_sampler, val_sampler = load_data(args)

    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes).to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 最適化方法定義

    iteration = 0  # 反復回数保存用
    # 評価だけやる
    ############
    if args.evaluate:
        weight_name = '{}/{}_mobilenetv2_best.pth'.format(args.path2weight, args.exp_name)
        print("use pretrained model : {}".format(weight_name))
        param = torch.load(weight_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(param)
        model = nn.DataParallel(model)
        model.to(device)  # gpu使うならcuda化
        validate(args, model, device, val_loader, criterion, writer, iteration)
        return

    if args.apex:
        model, optimizer = amp.initialize(
            model, optimizer,
            opt_level=args.apex_opt_level
        )

    model_without_ddp = model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)  # 学習率の軽減スケジュール

    best_acc = 0.0
    # 学習再開時の設定
    if args.restart:
        checkpoint = torch.load(
            'weight/{}_checkpoint.pth'.format(args.exp_name), map_location='cpu'
        )
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        iteration = args.epochs * len(train_loader)

    starttime = time.time()  # 実行時間計測(実時間)
    # 学習と評価
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, writer,
              criterion, optimizer, epoch, iteration, args.apex)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        acc = validate(args, model, device, val_loader, criterion, writer, iteration)
        scheduler.step()  # 学習率のスケジューリング更新
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            print('Acc@1 best: {:6.2f}%'.format(best_acc))
            weight_name = '{}/{}_mobilenetv2_best.pth'.format(args.path2weight, args.exp_name)
            save_on_master(model_without_ddp.cpu().state_dict(), weight_name)
            checkpoint = {
                'model': model_without_ddp.cpu().state_dict(),
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
        else:
            print('Acc@1 : {:6.2f}%'.format(acc))

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
