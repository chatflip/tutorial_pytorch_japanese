# -*- coding: utf-8 -*-
import copy
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
from utils import seed_everything, get_worker_init, init_distributed_mode, is_main_process, save_on_master


def load_weight(model, weight_name):
    assert os.path.isfile(weight_name), "don't exists weight: {}".format(weight_name)
    print("use pretrained model : %s" % weight_name)
    param = torch.load(weight_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(param)
    return model


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


if __name__ == '__main__':
    args = opt()
    print(args)
    seed_everything(args.seed)  # 乱数テーブル固定
    os.makedirs(args.path2weight, exist_ok=True)
    writer = SummaryWriter(log_dir='log/{}_qat'.format(args.exp_name))  # tensorboard用のwriter作成
    torch.backends.cudnn.benchmark = True  # 再現性を無くして高速化

    init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    ngpus_per_node = torch.cuda.device_count()

    if args.backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported: " + str(args.backend))

    if args.distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    train_loader, val_loader, train_sampler, val_sampler = load_data(args)

    model = mobilenet_v2(
        pretrained=False, quantize=True, backend=args.backend, num_classes=args.num_classes)
    weight_name = 'weight/food101_mobilenetv2_best.pth'
    model = load_weight(model, weight_name)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 最適化方法定義

    iteration = 0  # 反復回数保存用

    model.to(device)

    torch.backends.quantized.engine = args.backend
    model.qconfig = torch.quantization.get_default_qat_qconfig(args.backend)
    torch.quantization.prepare_qat(model, inplace=True)
    model.apply(torch.quantization.enable_fake_quant)
    model.apply(torch.quantization.enable_observer)  # observer有効にする

    model_without_ddp = model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    starttime = time.time()  # 実行時間計測(実時間)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    best_acc = 0

    # 学習と評価
    for epoch in range(1, args.qat_epochs + 1):
        model.to(device)  # validateのときにcpuに強制変換したので
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, writer,
              criterion, optimizer, epoch, iteration)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す

        if epoch >= args.num_observer_update_epochs:
            print('Disabling observer for subseq epochs, epoch = ', epoch)
            model.apply(torch.quantization.disable_observer)
        if epoch >= args.num_batch_norm_update_epochs:
            print('Freezing BN for subseq epochs, epoch = ', epoch)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        quantized_model = copy.deepcopy(model_without_ddp.to('cpu'))
        quantized_model.to(device)
        acc = validate(args, quantized_model, device, val_loader, criterion, writer, iteration)
        scheduler.step()  # 学習率のスケジューリング更新

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        # 重み保存
        if is_best:
            weight_name = '{}/{}_mobilenetv2_qat_best.pth'.format(args.path2weight, args.exp_name)
            save_on_master(model_without_ddp.cpu().state_dict(), weight_name)
            model.to(device)

    writer.close()  # tensorboard用のwriter閉じる

    if is_main_process():
        # 量子化モデル
        model.to('cpu')
        weight_name = '{}/{}_mobilenetv2_qat_best.pth'.format(args.path2weight, args.exp_name)
        quantized_model = load_weight(model_without_ddp, weight_name)
        quantized_model.eval()
        torch.quantization.convert(quantized_model, inplace=True)  # 量子化
        saved_script = 'weight/food101_mobilenetv2_qat_script.pt'
        torch.jit.save(torch.jit.script(quantized_model), saved_script)

    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))
