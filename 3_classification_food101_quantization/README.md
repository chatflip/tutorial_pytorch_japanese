2_classification_food101
===
101種類の料理の画像識別  
分散学習対応  
MLFlowの実験管理導入  

## Description
### 使用データセット
[The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) [1][2]  

### 使用ネットワーク
[Mobilenet v2](https://arxiv.org/abs/1801.04381) [3]  

### 参考文献
[1] http://www.foodspotting.com/  
[2] http://www.foodspotting.com/terms/  
[3] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
python py/setup.py
# 学習，識別
bash start.sh
# 量子化
bash quantize.sh
# ログ確認
tensorboard --logdir=log/food101
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti x2  
cuda 10.2  
cudnn 7.6.5  

DP  
Acc@1 best:  82.64%  
elapsed time = 0h 29m 11s  

DDP  
Acc@1 best:  81.10%  
elapsed time = 0h 26m 21s  
 