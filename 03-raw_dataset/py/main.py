# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
require pytorch 0.4.0
        torchvision 0.2.1
"""

from __future__ import print_function
import os
import time
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from alex import alex
from loadDB import AnimeFaceDB

def make_dir(save_root,folder_name):
    if folder_name == "pass":
        if not os.path.exists(save_root):
            os.mkdir(save_root)
    elif not os.path.exists(os.path.join(save_root,folder_name)):
        os.mkdir(os.path.join(save_root,folder_name))

def train(args, model, device, train_loader, writer, optimizer, epoch, iteration):
    #ネットワークを学習用に設定
    #ex.)dropout,batchnormを有効
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)#gpu使うなら画像とラベルcuda化
        optimizer.zero_grad()#勾配初期化
        output = model(data)#sofmaxm前まで出力(forward)
        loss = nn.CrossEntropyLoss(size_average=True)(output, target)#ネットワークの出力をsoftmax + ラベルとのloss計算
        loss.backward()#勾配計算(backprop)
        optimizer.step()#パラメータ更新
        #log_intervalごとに進行具合とloss表示
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalars("loss", {"train": loss}, iteration)
        iteration += 1

def test(args, model, device, test_loader, writer, iteration):
    #ネットワークを評価用に設定
    #ex.)dropout,batchnormを恒等関数に
    model.eval()
    test_loss = 0
    correct = 0
    #勾配計算しない(計算量低減)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)#gpu使うなら画像とラベルcuda化
            output = model(data)#sofmaxm前まで出力(forward)#評価データセットでのloss計算
            test_loss += nn.CrossEntropyLoss(size_average=False)(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] #softmaxでargmax計算
            correct += pred.eq(target.view_as(pred)).sum().item()#バッチ内の正解数計算
    test_loss /= len(test_loader.dataset)#dataset数で割って正解計算
    #test_loss格納
    writer.add_scalars("loss", {"test": test_loss}, iteration)
    writer.add_scalars("accuracy", {"test": 100. * correct / len(test_loader.dataset)}, iteration)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def opt():
    parser = argparse.ArgumentParser(description="PyTorch AnimeFace")
    parser.add_argument("--path2db", type = str, default="./dataset/", help="path to database")
    parser.add_argument("--path2weight", type = str, default="./weight/", help="path to weight")
    # Train Test settings
    parser.add_argument("--batch-size", type=int, default=50, metavar="N", help="input batch size for training (default: 50)")
    parser.add_argument("--test-batch-size", type=int, default=100, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--save-interval", type = int, default=10, help="save every N epoch")
    parser.add_argument("--numof_classes", type = int, default=176, help="num of classes")
    #network parameters
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    #etc
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--img_size", type = int, default=256, help="image size")
    parser.add_argument("--crop_size", type = int, default=224, help="crop size")
    parser.add_argument("--num_workers", type=int, default=10, help="num of pallarel threads(dataloader)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = opt()
    make_dir("./weight","pass")
    torch.manual_seed(args.seed)#どこに効いてるか分からないけど乱数テーブル固定
    use_cuda = not args.no_cuda and torch.cuda.is_available()#gpu使えるか and 使うか
    device = torch.device("cuda" if use_cuda else "cpu")#cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir="./log/AnimeFace")#tensorboard用のwriter作成

    #画像開いたところからtensorでNNに使えるようにするまでの変形
    train_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), interpolation=5),#リサイズ
                                          transforms.RandomCrop((args.crop_size, args.crop_size)),#クロップ
                                          transforms.RandomHorizontalFlip(p=0.5),#左右反転
                                          transforms.ToTensor(),#テンソル化
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5 ,0.5))#標準化
                                          ])
    test_transform = transforms.Compose([transforms.Resize((args.crop_size, args.crop_size), interpolation=5),#リサイズ
                                          transforms.ToTensor(),#テンソル化
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5 ,0.5))#標準化
                                          ])

    #AnimeFaceの学習用データ設定
    train_AnimeFace = AnimeFaceDB(args.path2db+"train/", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_AnimeFace, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True)
    #AnimeFaceの評価用データ設定
    test_AnimeFace = AnimeFaceDB(args.path2db+"test/", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_AnimeFace, batch_size=args.batch_size,
                                               shuffle=False, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=False)

    model = alex(pretrained=True, num_classes=args.numof_classes).to(device)#ネットワーク定義 + gpu使うならcuda化
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)#最適化方法定義
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,37], gamma=0.1)#学習率の軽減スケジュール
    starttime = time.time()#実行時間計測(実時間)
    iteration = 0#反復回数保存用
    #学習と評価
    for epoch in range(1, args.epochs + 1):
        scheduler.step()#epoch 0 スタートだから+1して数値を合わせてスケジューラ開始
        train(args, model, device, train_loader, writer, optimizer, epoch, iteration)
        iteration += len(train_loader)#1epoch終わった時のiterationを足す
        test(args, model, device, test_loader, writer, iteration)
        #重み保存
        if epoch % args.save_interval == 0:
            saved_weight = args.path2weight+"AnimeFace_alex_" + str(epoch) +".pth"
            torch.save(model.cpu().state_dict(), saved_weight)
            model.to(device)
    #tensorboard用のwriter閉じる
    writer.close()
    #実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print ("elapsed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))