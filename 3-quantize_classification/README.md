quantize classification
====
量子化したモデルの学習

## Description
### 使用データセット
[MNIST](http://yann.lecun.com/exdb/mnist/)

### 使用ネットワーク
[Mobilenet v2](https://arxiv.org/abs/1801.04381)[1]

### 参考文献
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, 
"MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.  

## Usage
### 実行
```
# 学習，識別
$ python py/main.py
# ログ確認
$ tensorboard --logdir=log/quantize
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti  
cuda 10.0  
cudnn 7.5  
Top1 Accuracy 98.0%  
Elapsed time 0h 11m 3s  

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  