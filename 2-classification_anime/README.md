classification　anime
====
アニメのキャラクターの顔を集めたデータセットを用いた識別

## Description
### 使用データセット
[AnimeFace Character Dataset](http://www.nurs.or.jp/%7Enagadomi/animeface-character-dataset/README.html)

### 使用ネットワーク
[pytorchのalexnet実装](https://arxiv.org/abs/1404.5997)[1]  
[元のalexnet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)[2]  

### 参考文献
[1] A. Krizhevsky, "One weird trick for parallelizing convolutional neural networks," arXiv, 2014.  
[2] A. Krizhevsky, et al.: “imageNet Classification with Deep Convolutional Neural Networks,” Advances in Neural Information Processing Systems, 2012.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
$ python py/preprocess.py
# 学習，識別
$ python py/main.py
# ログ確認
$ tensorboard --logdir=log/AnimeFace
```

## 動作環境(確認済み)
### on gpu
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti  
cuda 10.0  
cudnn 7.5  
Accuracy .%  
Elapsed time 0h m s  



## Author
chatflip
[[github](https://github.com/chatflip)]  
[[Qiita](https://qiita.com/chat-flip)]  