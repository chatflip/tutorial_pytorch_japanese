
02-log_plot
====
tensorboardxを用いたloss, accuracyの管理

## Description
tensorflowのtensorboardを用いてlossを確認

### tensorboardx
[tensorboardx](https://github.com/lanpa/tensorboard-pytorch)

## Usage
### tensorboard確認用の仮想環境作成
```
$ conda create -n tb python=3.5.0 -y
$ source activate tb
$ pip install -r txt/requirements.txt
$ source deactivate
```

### pytorchでlog取るライブラリインストール
```
#pytorchの環境で
$ pip install tensorboardX
```

### 実行
```
$ python py/main.py
$ bash scripts/tensorboard.sh
#macの場合表示されるURLではなく
#http://localhost:(port番号)/#scalars
```

## Author
chatflip
[[github](https://github.com/chatflip)]
[[Qiita](https://qiita.com/chat-flip)]  
