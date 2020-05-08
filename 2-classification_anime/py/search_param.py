# -*- coding: utf-8 -*-
import os
import time

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from args import opt
from loadDB import AnimeFaceDB
from model import resnet18
from train_val import train, validate
from utils import seed_everything

def objective(trial):
    args = opt()
    worker_init = seed_everything(args.seed)  # 乱数テーブル固定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    multigpu = torch.cuda.device_count() > 1  # グラボ2つ以上ならmultigpuにする
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
        os.path.join(args.path2db, 'train'),
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_AnimeFace, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=worker_init)

    # AnimeFaceの評価用データ設定
    val_AnimeFace = AnimeFaceDB(
        os.path.join(args.path2db, 'val'),
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_AnimeFace, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        worker_init_fn=worker_init)

    criterion = nn.CrossEntropyLoss().to(device)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

    writer = SummaryWriter(log_dir='log/temp')  # tensorboard用のwriter作成
    model = resnet18(pretrained=True, num_classes=args.num_classes)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 最適化方法定義
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5*args.epochs), int(0.75*args.epochs)], gamma=0.1)  # 学習率の軽減スケジュール

    iteration = 0  # 反復回数保存用
    if multigpu:
        model = nn.DataParallel(model)
    model.to(device)

    best_acc = 0.0
    # 学習と評価
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, writer,
              criterion, optimizer, epoch, iteration)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        acc = validate(args, model, device, val_loader, criterion, writer, iteration)
        scheduler.step()  # 学習率のスケジューリング更新
        best_acc1 = max(acc, best_acc)
        #if trial.should_prune():
        #    print("early stop")
        #    raise optuna.structs.TrialPruned()
    writer.close()  # tensorboard用のwriter閉じる
    return best_acc1

if __name__ == '__main__':
    starttime = time.time()  # 実行時間計測(実時間)

    # optuna設定
    study_name = "AnimeFace_lr"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///log/AnimeFace.db",
        direction="maximize",
        load_if_exists=True)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {0:d}h {1:d}m {2:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))
