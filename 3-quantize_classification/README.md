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
Validate: [ 0/98]       Time  3.709 ( 3.709)    infTime  3.410 ( 3.410) Loss 0.10274 (0.10274)  Acc@1 100.00 (100.00)   Acc@5 100.00 (100.00)
Validate: [98/98]       Time  0.048 ( 3.156)    infTime  0.044 ( 3.180) Loss 0.10520 (0.15849)  Acc@1 100.00 ( 97.58)   Acc@5 100.00 ( 99.76)

# Dynamic quantization model  
Validate: [ 0/98]       Time  3.860 ( 3.860)    infTime  3.580 ( 3.580) Loss 0.10271 (0.10271)  Acc@1 100.00 (100.00)   Acc@5 100.00 (100.00)
Validate: [98/98]       Time  0.051 ( 3.175)    infTime  0.047 ( 3.199) Loss 0.10442 (0.15847)  Acc@1 100.00 ( 97.57)   Acc@5 100.00 ( 99.76)


# Quantization Aware Training model  
python py/main_qat.py --batch-size=32 --epoch=20 --lr=0.00001 --lr-step-size=30 --lr-gamma=0.1  

```

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  