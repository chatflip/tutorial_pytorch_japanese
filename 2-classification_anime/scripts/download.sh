#!/bin/sh
wget http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip
unzip animeface-character-dataset.zip
python py/train_test_split.py
rm animeface-character-dataset.zip
rm -r animeface-character-dataset