classification　anime
====
アニメのキャラクターの顔を集めたデータセットを用いた識別

## Description
### 使用データセット
[AnimeFace Character Dataset](http://www.nurs.or.jp/%7Enagadomi/animeface-character-dataset/README.html)

### 使用ネットワーク
[Resnet](https://arxiv.org/abs/1512.03385)[1]  

### 参考文献
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition," CVPR, 2015.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
$ python py/preprocess.py
# 学習，識別
$ python py/main.py
# ログ確認
$ tensorboard --logdir=log/AnimeFace
# パラメータ探索
$ python py/search_param.py --epoch 2 --print-freq 100
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti x2  
cuda 10.2  
cudnn 7.6.5  
Top1 Accuracy 93.81%  
Elapsed time 0h 3m 19s  
