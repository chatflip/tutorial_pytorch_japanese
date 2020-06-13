detection voc
====
Mask R-CNNを用いたセクメンテーション
Pytorchの[tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) 参考  

## Description
### 使用データセット
[Penn-Fudan Database](https://www.cis.upenn.edu/~jshi/ped_html/) [1]  

### 使用ネットワーク
[Mask R-CNN](https://arxiv.org/abs/1703.06870) [2] 

### 参考文献
[1] Liming Wang, Jianbo Shi, Gang Song, I-fan Shen, "Object Detection Combining Recognition and Segmentation", ACCV, 2007.    
[2] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick: "Mask R-CNN," ICCV, 2017.  

## Usage
### 実行
```
# ダウンロード，フォルダ構成
$ python py/preprocess.py
# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0
cp references/detection/utils.py ../py/
cp references/detection/transforms.py ../py/
cp references/detection/coco_eval.py ../py/
cp references/detection/engine.py ../py/
cp references/detection/coco_utils.py ../py/

# 学習，識別
$ python py/main.py
```

## 動作環境(確認済み)
OS: Ubuntu 16.04  
プロセッサ Intel Core i9 3.6GHz  
グラフィック GeForce RTX 2080 Ti x2  
cuda 10.2  
cudnn 7.6.5  
