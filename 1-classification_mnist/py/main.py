# -*- coding: utf-8 -*-
'''
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
require pytorch 0.4.0
        torchvision 0.2.1
'''

from __future__ import print_function
import argparse
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

from model import LeNet
from utils import AverageMeter


def train(args, model, device, train_loader,
          writer, optimizer, epoch, iteration):
    # ネットワークを学習用に設定
    # ex.)dropout,batchnormを有効
    model.train()
    # AverageMeterの値初期化
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()  # 1回目の読み込み時間計測用
    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)  # 画像のロード時間記録
        data, target = data.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化
        optimizer.zero_grad()  # 勾配初期化
        output = model(data)  # sofmaxm前まで出力(forward)
        loss = nn.CrossEntropyLoss(size_average=True)(output, target)  # ネットワークの出力をsoftmax + ラベルとのloss計算
        losses.update(loss.item(), data.size(0))  # ossの値更新
        loss.backward()  # 勾配計算(backprop)
        optimizer.step()  # パラメータ更新
        batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
        end = time.time()  # 基準の時間更新
        # log_intervalごとに進行具合とloss表示
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {0} [{1:5d}/{2:5d} ({3:.0f}%)]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f}'.format(
                   epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses))
            writer.add_scalars('loss', {'train': losses.val}, iteration)
        iteration += 1


def test(args, model, device, test_loader, writer, iteration):
    # ネットワークを評価用に設定
    # ex.)dropout,batchnormを恒等関数に
    model.eval()
    test_loss = 0
    correct = 0
    # AverageMeterの値初期化
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()  # 基準の時間更新
    # 勾配計算しない(計算量低減)
    with torch.no_grad():
        for data, target in test_loader:
            data_time.update(time.time() - end)  # 画像のロード時間記録
            data, target = data.to(device), target.to(device)  # gpu使うなら画像とラベルcuda化
            output = model(data)  # sofmaxm前まで出力(forward)
            test_loss += nn.CrossEntropyLoss(size_average=False)(output, target).item()  # 評価データセットでのloss計算
            pred = output.max(1, keepdim=True)[1]  # softmaxでargmax計算
            correct += pred.eq(target.view_as(pred)).sum().item()  # バッチ内の正解数計算
            batch_time.update(time.time() - end)  # 画像ロードからパラメータ更新にかかった時間記録
            end = time.time()  # 基準の時間更新
    test_loss /= len(test_loader.dataset)  # dataset数で割って正解計算
    # test_loss格納
    writer.add_scalars('loss',
                       {'test': test_loss}, iteration)
    writer.add_scalars('accuracy',
                       {'test': 100. * correct / len(test_loader.dataset)}, iteration)
    print('\nTest Accuracy: {}/{} ({:.0f}%)\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Average Loss: {test_loss:.4f}\t\n'.format(
           correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset),
           batch_time=batch_time, data_time=data_time, test_loss=test_loss))


def opt():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # Train Test settings
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    # network parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # etc
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='num of pallarel threads(dataloader)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = opt()
    torch.manual_seed(args.seed)  # どこに効いてるか分からないけど乱数テーブル固定
    use_cuda = not args.no_cuda and torch.cuda.is_available()  # gpu使えるか and 使うか
    device = torch.device('cuda' if use_cuda else 'cpu')  # cpuとgpu自動選択 (pytorch0.4.0以降の書き方)
    writer = SummaryWriter(log_dir='log/MNIST')  # tensorboard用のwriter作成

    # 画像開いたところからtensorでNNに使えるようにするまでの変形
    transform = transforms.Compose([
                transforms.Resize((32, 32), interpolation=5),  # リサイズ
                transforms.ToTensor(),  # テンソル化
                transforms.Normalize((0.1307,), (0.3081,))  # 標準化
                ])

    # MNISTの学習用データ設定
    train_MNIST = datasets.MNIST(
        'data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_MNIST, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    # MNISTの評価用データ設定
    test_MNIST = datasets.MNIST(
        'data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_MNIST, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    model = LeNet().to(device)  # ネットワーク定義 + gpu使うならcuda化
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)  # 最適化方法定義
    starttime = time.time()  # 実行時間計測(実時間)
    iteration = 0  # 反復回数保存用
    # 学習と評価
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, writer, optimizer, epoch, iteration)
        iteration += len(train_loader)  # 1epoch終わった時のiterationを足す
        test(args, model, device, test_loader, writer, iteration)
    writer.close()  # tensorboard用のwriter閉じる
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print('elapsed time = {:d}h {:d}m {:d}s'.format(
        int(interval / 3600),
        int((interval % 3600) / 60),
        int((interval % 3600) % 60)))
