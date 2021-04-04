# -*- coding: utf-8 -*-
import os
import time

import albumentations as A
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from datasets import Food101Dataset
from efficientnet_pytorch import EfficientNet
from MlflowWriter import MlflowWriter
from mobilenet_v2 import mobilenet_v2
from resnet import resnet50, resnet101
from train_val import train, validate
from utils import get_worker_init, seed_everything


def load_data(args):
    cwd = hydra.utils.get_original_cwd()
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
    )
    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(args.arch.crop_size, args.arch.crop_size),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(args.arch.image_size, args.arch.image_size),
            A.CenterCrop(args.arch.crop_size, args.arch.crop_size),
            normalize,
            ToTensorV2(),
        ]
    )

    # AnimeFaceの学習用データ設定
    train_dataset = Food101Dataset(
        os.path.join(cwd, args.path2db),
        "train",
        transform=train_transform,
    )

    # Food101の評価用データ設定
    val_dataset = Food101Dataset(
        os.path.join(cwd, args.path2db), "test", transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.arch.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=get_worker_init(args.seed),
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.arch.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=get_worker_init(args.seed),
    )
    return train_loader, val_loader


@hydra.main(config_path="./../config", config_name="config")
def main(args):
    print(args)
    cwd = hydra.utils.get_original_cwd()
    seed_everything(args.seed)  # 乱数テーブル固定
    os.makedirs(os.path.join(cwd, args.path2weight), exist_ok=True)
    writer = MlflowWriter("{}-{}".format(args.exp_name, args.arch.name))
    for key in args:
        writer.log_param(key, args[key])
    writer.log_params_from_omegaconf_dict(args)
    # torch.backends.cudnn.benchmark = True  # 再現性を無くして高速化
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    multigpu = torch.cuda.device_count() > 1  # グラボ2つ以上ならmultigpuにする

    train_loader, val_loader = load_data(args)

    if args.arch.name == "mobilenet_v2":
        model = mobilenet_v2(pretrained=True, num_classes=args.num_classes)
    elif args.arch.name == "resnet50":
        model = resnet50(pretrained=True)
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, args.num_classes)
    elif args.arch.name == "resnet101":
        model = resnet101(pretrained=True)
        in_channels = model.fc.in_features
        model.fc = nn.Linear(in_channels, args.num_classes)
    elif "efficientnet" in args.arch.name:
        model = EfficientNet.from_pretrained(args.arch.name)
        in_channels = model._fc.in_features
        model._fc = nn.Linear(in_channels, args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.arch.max_lr,
    )  # 最適化方法定義

    iteration = 0  # 反復回数保存用
    # 評価だけやる
    if args.evaluate:
        weight_name = "{}/{}/{}_mobilenetv2_best.pth".format(
            cwd, args.path2weight, args.exp_name
        )
        print("use pretrained model : {}".format(weight_name))
        param = torch.load(weight_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(param)
        if multigpu:
            model = nn.DataParallel(model)
        model.to(device)  # gpu使うならcuda化
        validate(args, model, device, val_loader, criterion, writer, iteration)
        return

    model_without_dp = model
    if multigpu:
        model = nn.DataParallel(model)
        model_without_dp = model.module

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader),
        T_mult=1,
        eta_min=args.arch.min_lr,
        last_epoch=-1,
    )  # 学習率の軽減スケジュール

    best_acc = 0.0
    # 学習再開時の設定
    if args.restart:
        checkpoint = torch.load(
            "{}/{}/{}_checkpoint.pth".format(cwd, args.path2weight, args.exp_name),
            map_location="cpu",
        )
        model_without_dp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        iteration = args.epochs * len(train_loader)

    starttime = time.time()  # 実行時間計測(実時間)
    # 学習と評価
    for epoch in range(args.start_epoch, args.epochs + 1):
        train(
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
            args.apex,
        )
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        acc = validate(args, model, device, val_loader, criterion, writer, iteration)
        scheduler.step()  # 学習率のスケジューリング更新
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if is_best:
            print("Acc@1 best: {:6.2f}%".format(best_acc))
            weight_name = "{}/{}/{}_mobilenetv2_best.pth".format(
                cwd, args.path2weight, args.exp_name
            )
            torch.save(model_without_dp.cpu().state_dict(), weight_name)
            checkpoint = {
                "model": model_without_dp.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "args": args,
            }
            torch.save(
                checkpoint,
                "{}/{}/{}_checkpoint.pth".format(cwd, args.path2weight, args.exp_name),
            )
            writer.log_artifact(
                "{}/{}/{}_checkpoint.pth".format(cwd, args.path2weight, args.exp_name)
            )
            model.to(device)

    # Hydraの成果物をArtifactに保存
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), "main.log"))
    writer.set_terminated()  # mlflow用のwriter閉じる
    writer.move_mlruns()
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print(
        "elapsed time = {0:d}h {1:d}m {2:d}s".format(
            int(interval / 3600),
            int((interval % 3600) / 60),
            int((interval % 3600) % 60),
        )
    )


if __name__ == "__main__":
    main()
