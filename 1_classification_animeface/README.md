1_classification_animeface
====
アニメのキャラクターの顔を集めたデータセットを用いた識別

## Description
### 使用データセット
[AnimeFace Character Dataset](http://www.nurs.or.jp/%7Enagadomi/animeface-character-dataset/README.html)

### 使用ネットワーク
[Mobilenet v2](https://arxiv.org/abs/1801.04381) [1]

### 参考文献
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
python py/setup.py
# 学習，識別
bash start.sh
python py/main.py --apex
# ログ確認
tensorboard --logdir=log/animeface
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti x2  
cuda 10.2  
cudnn 7.6.5  

Top1 Accuracy 91.92%  
elapsed time = 0h 2m 12s
