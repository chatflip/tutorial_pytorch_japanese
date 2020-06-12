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
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env py/main.py 
# ログ確認
$ tensorboard --logdir=log/AnimeFace
# apex使ったMixed-Precision Training
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env py/main.py --apex
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti x2  
cuda 10.2  
cudnn 7.6.5  

float training  
Top1 Accuracy 96.23%  
Elapsed time 0h 3m 16s

mixed-precision training  
Top1 Accuracy 96.26%  
Elapsed time 0h 2m 27s  