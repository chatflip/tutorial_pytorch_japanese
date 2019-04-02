03-raw_dataset
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
### 画像の準備
```
$ bash scripts/download.sh
```

### 実行
```
$ python py/main.py
```

### 動作環境(確認済み)
accuracy 86.7%
<!-- 
#### on cpu
OS macOS High Sierra  
プロセッサ 2.5 GHz Intel Core i7  
メモリ 8 GB 2133 MHz LPDDR3  
elapsed time 0h 0m 0s  
-->

#### on gpu
OS: Ubuntu 16.04  
プロセッサ Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz  
グラボ GeForce GTX 1080 × 2  
cuda 8.0  
cudnn 5.1  
elapsed time 0h 10m 3s  

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  