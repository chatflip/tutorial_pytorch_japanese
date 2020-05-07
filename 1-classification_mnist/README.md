classification MNIST
====
手書き文字(0-9)認識とtensorboardを用いたLoss, Accuracyの管理

## Description
### 使用データセット
[MNIST](http://yann.lecun.com/exdb/mnist/)

### 使用ネットワーク
[Lenet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)[1]

## Usage
### 実行
```
# 学習, 識別
$ python main.py
# ログ確認
$ tensorboard --logdir=log/MNIST
#macの場合表示されるURLではなく
http://localhost:(port番号)/#scalars
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti  
cuda 10.2  
cudnn 7.6.5  
Accuracy 98.98%  
Elapsed time = 0h 0m 33s  

## 参考文献
[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, november, 1998.
