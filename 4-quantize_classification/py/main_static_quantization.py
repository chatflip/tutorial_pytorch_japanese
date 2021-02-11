# -*- coding: utf-8 -*-
import copy
import os
import time

import torch
import torch.nn as nn
import torch.utils.mobile_optimizer as mobile_optimizer
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from datasets import Food101Dataset
from model import mobilenet_v2
from train_val import calibrate_validate
from utils import seed_everything, get_worker_init, init_distributed_mode, is_main_process


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
    writer = SummaryWriter(log_dir='log/{}_static_quantize'.format(args.exp_name))  # tensorboard用のwriter作成
    torch.backends.cudnn.benchmark = True  # 再現性を無くして高速化
    
    if args.backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported: " + str(args.backend))

    init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    train_loader, _, _, _ = load_data(args)

    model = mobilenet_v2(
        pretrained=False, quantize=True, backend=args.backend, num_classes=args.num_classes)

    criterion = nn.CrossEntropyLoss().to(device)

    weight_name = 'weight/food101_mobilenetv2_best.pth'
    model = load_weight(model, weight_name)
    model.to(device)
    quantized_model = copy.deepcopy(model)
    quantized_model.eval()
    quantized_model.fuse_model()
    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    quantized_model.qconfig = torch.quantization.get_default_qconfig(args.backend)
    print(quantized_model.qconfig) ##channel
    torch.quantization.prepare(quantized_model, inplace=True)

    iteration = 0
    starttime = time.time()  # 実行時間計測(実時間)
    calibrate_validate(args, quantized_model, device, train_loader, criterion, writer, iteration, args.num_calibration_batches)
    if is_main_process():
        quantized_model.to('cpu')
        # Convert to quantized model
        torch.quantization.convert(quantized_model, inplace=True)
        saved_weight = 'weight/food101_mobilenetv2_static_quantize.pth'
        torch.save(quantized_model.state_dict(), saved_weight)
        saved_script = 'weight/food101_mobilenetv2_static_quantize_script.pt'
        torch.jit.save(torch.jit.script(quantized_model), saved_script)

    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))