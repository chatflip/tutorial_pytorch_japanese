pytorch tutorial for japanese
====

昔の自分を確認する用

## Description
[1-classification_mnist](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/1-classification_mnist)  
手書き文字(0-9)認識とtensorboardxを用いたloss, accuracyの管理  

[2-classification_anime](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/2-classification_anime)  
pytorchのデータセットを使わない画像識別  

[3-detection_voc](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/3-detection_voc)  
SSDを用いた物体検出  

[4-segmentation_voc](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/4-segmentation_voc)  
PSPNetを用いた領域分割  


## Requirement
Mac or Ubuntu

## Installation
### anaconda導入
最新バージョンを使う場合[ここ](https://www.anaconda.com/distribution/)からダウンロード, インストール  
実行時の環境は[ここ](https://repo.continuum.io/archive/)から```Anaconda3-5.2.0-MacOSX-x86_64.sh```をダウンロード, インストール

### pytorch導入(仮想環境)
```
$ conda create -n tutorial python=3.6 -y
$ source activate tutorial
$ conda install -c pytorch pytorch=1.1.0 torchvision=0.3.0 -y
$ pip install tb-nightly
```

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  
