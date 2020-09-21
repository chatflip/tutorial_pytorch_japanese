#!/bin/bash
source activate pt16
# mixed precision training実行
python py/main.py --apex
