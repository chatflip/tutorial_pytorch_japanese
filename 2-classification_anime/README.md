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