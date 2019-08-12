detection voc
====
SSDで物体検出

## Description
### 使用データセット
[Visual Object Classes Challenge 2012](http://host.robots.ox.ac.uk/pascal/VOC/)[1]  

### 使用ネットワーク
[SSD-300](https://arxiv.org/abs/1512.02325)[2]  

### 参考文献
[1] Mark Everingham, "The PASCAL Visual Object Classes Challenge: A Retrospective," International Journal of Computer Vision, Vol. 111, pp. 98-136, 2015.  
[2] Wei Liu, et al.: "SSD: Single Shot MultiBox Detector," European Conference on Computer Vision, 2016.  

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
Accuracy .%  
Elapsed time 0h m s  

## Author
chatflip
[[github](https://github.com/chatflip)]  
[[Qiita](https://qiita.com/chat-flip)]  