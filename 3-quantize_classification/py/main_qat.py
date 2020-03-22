# -*- coding: utf-8 -*-
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from loadDB import AnimeFaceDB
from model_quantize import mobilenet_v2
from train_val import train, validate
from utils import seed_everything


def load_weight(model, weight_name):
    assert os.path.isfile(weight_name), "don't exists weight: {}".format(weight_name)
    print("use pretrained model : %s" % weight_name)
    param = torch.load(weight_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(param)
    return model


if __name__ == '__main__':
    args = opt()
    print(args)
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定


    if args.backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported: " + str(args.backend))

    # フォルダが存在してなければ作る
    if not os.path.exists('weight'):
        os.mkdir('weight')

    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir='log/quantize_AnimeFace')  # tensorboard用のwriter作成
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
    train_AnimeFace = AnimeFaceDB(
        args.path2db+'/train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_AnimeFace, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=False, drop_last=True,
        worker_init_fn=worker_init)

    # AnimeFaceの評価用データ設定
    val_AnimeFace = AnimeFaceDB(
        args.path2db+'/val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_AnimeFace, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
        worker_init_fn=worker_init)

    model = mobilenet_v2(pretrained=False, num_classes=args.num_classes, quantize=False)
    weight_name = "weight/AnimeFace_mobilenetv2_float_epoch100.pth"
    model = load_weight(model, weight_name)
    model.fuse_model()

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 最適化方法定義
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    criterion = nn.CrossEntropyLoss().to(device)
    iteration = 0  # 反復回数保存用

    if args.evaluate:
        validate(args, model, device, val_loader, criterion, writer, iteration)
        sys.exit()

    torch.backends.quantized.engine = args.backend
    model.qconfig = torch.quantization.get_default_qat_qconfig(args.backend)
    torch.quantization.prepare_qat(model, inplace=True)
    model.apply(torch.quantization.enable_fake_quant)
    model.apply(torch.quantization.enable_observer)  # observer有効にする

    starttime = time.time()  # 実行時間計測(実時間)
    # 学習と評価
    for epoch in range(1, args.epochs + 1):
        model.to(device)  # validateのときにcpuに強制変換したので
        train(args, model, device, train_loader, writer,
              criterion, optimizer, epoch, iteration)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す

        if epoch >= args.num_observer_update_epochs:
            print('Disabling observer for subseq epochs, epoch = ', epoch)
            model.apply(torch.quantization.disable_observer)
        if epoch >= args.num_batch_norm_update_epochs:
            print('Freezing BN for subseq epochs, epoch = ', epoch)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        quantized_model = copy.deepcopy(model.to('cpu'))
        quantized_model.to(device)
        validate(args, quantized_model, device, val_loader, criterion, writer, iteration)
        scheduler.step()  # 学習率のスケジューリング更新
        # 重み保存
        if epoch % args.save_freq == 0:
            saved_weight = 'weight/AnimeFace_mobilenetv2_qat_epoch{}.pth'.format(epoch)
            torch.save(model.cpu().state_dict(), saved_weight)
            model.to(device)

    writer.close()  # tensorboard用のwriter閉じる

    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))

    # 量子化モデル
    quantized_model = copy.deepcopy(model.to('cpu'))
    quantized_model.eval()
    torch.quantization.convert(quantized_model, inplace=True)  # 量子化
    saved_weight = 'weight/AnimeFace_mobilenetv2_qat_epoch{}.pth'.format(args.epochs)
    torch.save(quantized_model.state_dict(), saved_weight)
    saved_script = 'weight/AnimeFace_mobilenetv2_script_qat_epoch{}.pth'.format(args.epochs)
    torch.jit.save(torch.jit.script(quantized_model), saved_script)
