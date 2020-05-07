pytorch tutorial for japanese
====

昔の自分を確認する用  
[ハードウェア構成](hardware.md)  

## Description
[1-classification_mnist](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/1-classification_mnist)  
手書き文字(0-9)認識とtensorboardを用いたloss, accuracyの管理  

[2-classification_anime](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/2-classification_anime)  
pytorchのデータセットを使わない画像識別  

[3-quantize_classification](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/3-quantize_classification)  
量子化したmodelの学習  

[pruning.ipynb](https://github.com/chatflip/tutorial_pytorch_japanese/blob/master/notebook/pruning.ipynb)  
1-classification_mnistのmodelの枝刈り  
[torchscript.ipynb](https://github.com/chatflip/tutorial_pytorch_japanese/blob/master/notebook/torchscript.ipynb)  
mobilenetv2のmodelのtorchscript変換  

## Requirement
Mac or Ubuntu

## Installation
### anaconda導入
最新バージョンを使う場合[ここ](https://www.anaconda.com/distribution/)からダウンロード, インストール  
実行時の環境は[ここ](https://repo.continuum.io/archive/)から```Anaconda3-5.2.0-MacOSX-x86_64.sh```をダウンロード, インストール

### pytorch導入(仮想環境)
```
$ conda create -n pt15 python=3.7 -y
$ source activate pt15
$ conda install -c pytorch pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.2 -y
$ pip install tb-nightly optuna
```
