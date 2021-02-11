pytorch tutorial for japanese
====

昔の自分を確認する用  
[ハードウェア構成](hardware.md)  

## Description
[1_animeface_classification](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/1_animeface_classification)  
- フォルダ分けしたデータセットの画像識別 + 実験管理  

[2_classification_food101](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/2_classification_food101)  
- 101種類の料理の画像識別
- hydra+mlflowを用いた実験管理
- albumentationsを用いた画像拡張
- 様々なarchitectureを使用した画像識別

[3_classification_food101_ddp] 編集中  
- 分散学習

[4-quantize_classification] 編集中  
- 学習モデルの量子化

[5-segmentation_voc](https://github.com/chatflip/tutorial_pytorch_japanese/tree/master/5-segmentation_voc)  
- Semantic Segmentationを用いた領域分割

## Requirement
Mac or Ubuntu

## Installation
### anaconda導入
最新バージョンを使う場合[ここ](https://www.anaconda.com/distribution/)からダウンロード, インストール  
実行時の環境は[ここ](https://repo.continuum.io/archive/) から```Anaconda3-5.2.0-MacOSX-x86_64.sh``` をダウンロード, インストール

### pytorch導入(仮想環境)
``` 
conda create -n pt17 python=3.7 -y  
source activate pt17  
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch -y
pip install hydra-core==1.0.5 mlflow==1.13.1 efficientnet-pytorch==0.7.0 albumentations==0.5.2 segmentation-models-pytorch==0.1.3

```

### 環境確認
```
torch.cuda.is_available()
-> True
torch.backends.cudnn.version()
-> 8005
torch.distributed.is_nccl_available()
-> True
```