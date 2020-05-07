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
$ python py/main_float.py
# dynamic quantization
$ python py/dynamic_quantization.py
# Quantization Aware Training
$ python py/main_qat.py --epochs=10
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti  
cuda 10.0  
cudnn 7.5  


## Results
```
# float model  
Size (MB): 10.485055  
Validate: [6209/6209]   Time  0.040 ( 0.041)    infTime  0.039 ( 0.040) Loss 0.00039 (0.03600)  Acc@1 100.00 ( 98.92)   Acc@5 100.00 ( 99.92)  

# Dynamic quantization model  
Size (MB): 9.811573  
Validate: [6209/6209]   Time  0.040 ( 0.042)    infTime  0.039 ( 0.041) Loss 0.00038 (0.03604)  Acc@1 100.00 ( 98.92)   Acc@5 100.00 ( 99.92)  

# Static quantization model  
Size (MB): 2.724299  
Validate: [6209/6209]   Time  0.013 ( 0.013)    infTime  0.012 ( 0.012) Loss 0.00168 (0.05084)  Acc@1 100.00 ( 98.73)   Acc@5 100.00 ( 99.89)  

# Quantization Aware Training model  
Size (MB): 2.534283  
Validate: [6209/6209]   Time  0.011 ( 0.012)    infTime  0.010 ( 0.011) Loss 0.00003 (0.04603)  Acc@1 100.00 ( 98.76)   Acc@5 100.00 ( 99.89)  
```
