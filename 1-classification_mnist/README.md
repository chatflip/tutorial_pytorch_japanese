classification MNIST
====
手書き文字(0-9)認識とtensorboardを用いたLoss, Accuracyの管理

## Description
### 使用データセット
[MNIST](http://yann.lecun.com/exdb/mnist/)

### 使用ネットワーク
[Lenet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)[1]

## Usage
### 学習, 識別
```
$ python main.py
```

### 実行
```
$ tensorboard --logdir=./log/MNIST
#macの場合表示されるURLではなく
http://localhost:(port番号)/#scalars
```

## 動作環境(確認済み)
accuracy 99.0% 
### on cpu
OS: macOS Mojave
プロセッサ 2.5 GHz Intel Core i7  
elapsed time 0h 1m 58s  
### on gpu
OS: Ubuntu 16.04  
プロセッサ Intel® Core™ i9-9900K CPU @ 3.60GHz × 16  
グラフィック GeForce RTX 2080 Ti  
cuda 10.0  
cudnn 7.5  
accuracy 99.0%  
elapsed time 0h 1m 16s  

## 参考文献
[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," Proceedings of the IEEE, november, 1998.

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  
