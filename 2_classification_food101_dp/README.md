2_classification_food101
===
101種類の料理の画像識別  
albumentationsを用いた画像拡張
様々なarchitectureで実験

## Description
### 使用データセット
[The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) [1][2]  

### 使用ネットワーク
[Resnet](https://arxiv.org/abs/1512.03385) [3]  
[MobilenetV2](https://arxiv.org/abs/1801.04381) [4]  
[Efficientnet](https://arxiv.org/abs/1905.11946) [5]  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
python py/setup.py
# 学習，識別
bash train.sh
# ログ確認
cd outputs/{date}/{time}
mlflow ui
```

## Results
Acc@1 best:  81.11%
elapsed time = 0h 29m 2s(albumentations)  
elapsed time = 0h 14m 30s(torchvision)  



### 参考文献
[1] http://www.foodspotting.com/  
[2] http://www.foodspotting.com/terms/  
[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition," CVPR, 2015.
[4] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.  
[5] Mingxing Tan, Quoc V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML 2019.