pytorch tutorial for japanese
====

昔の自分を確認する用

## Description
[1-classification_mnist](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/1-classification_mnist)  
手書き文字(0-9)認識とtensorboardを用いたloss, accuracyの管理  

[2-classification_anime](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/2-classification_anime)  
pytorchのデータセットを使わない画像識別  

[3-quantize_classification](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/3-quantize_classification)  
量子化したmodelの学習  

[pruning.ipynb](https://github.com/chatflip/tutorial_pytorch_japanese/blob/master/notebook/pruning.ipynb)  
1-classification_mnistのmodelの枝刈り  

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
$ conda install -c pytorch pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -y
$ pip install tb-nightly
```

### notebook導入
```
pip install ipython
ipython kernel install --user --name=pt14 --display-name=pt14
```

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  
