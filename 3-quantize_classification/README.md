quantize classification
====
量子化したモデルの学習

## Description
### 使用データセット
[AnimeFace Character Dataset](http://www.nurs.or.jp/%7Enagadomi/animeface-character-dataset/README.html)

### 使用ネットワーク
[Mobilenet v2](https://arxiv.org/abs/1801.04381)[1]

### 参考文献
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen, 
"MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
$ python py/preprocess.py

# float model training
$ python py/main_float.py --apex

# dynamic quantization
$ python py/main_dynamic_quantization.py

# static quantization
$ python py/main_static_quantization.py --batch-size=32 --backend=fbgemm

# Quantization Aware Training
$ python py/main_qat.py --batch-size=64 --val-batch-size=128 --epoch=20 --lr=0.00001 --lr-step-size=30 --lr-gamma=0.1

# evaliate
$ python py/main_validate.py --val-batch-size=1 --print-freq=1000
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti x2  
cuda 10.2  
cudnn 7.6.5  
elapsed time = 0h 8m 56s

## Results
```
# float model  
Size (MB): 10.117735
infTime 0.040  Loss 0.03600  Acc@1 98.92  Acc@5 99.92  

# Dynamic quantization model  
Size (MB): 9.446549
infTime 0.023  Loss 0.04866  Acc@1 98.79  Acc@5 99.87  

# Static quantization model  
Size (MB): 2.535883  
infTime 0.010  Loss 0.06769  Acc@1 98.45  Acc@5 99.79

# Quantization Aware Training model  
Size (MB): 2.535883
infTime 0.008  Loss 0.04767  Acc@1 98.78  Acc@5 99.86
```
