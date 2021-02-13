5-segmentation_voc
===
Semantic Segmentationを用いた領域分割

## Description
### 使用データセット

### 使用ネットワーク

## Usage
### 実行
```
# ダウンロード，フォルダ構成
python py/download.py
python py/make_raw_annotation.py

# 学習，識別
bash train.sh
# ログ確認
cd outputs/{date}/{time}
mlflow ui

# 速度
v4l2-ctl --list-formats-ext 
# 学習モデルを使ってwebcam推論
python py/demo_webcam.py
```

## Results

### 参考文献
