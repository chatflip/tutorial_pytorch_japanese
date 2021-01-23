#!/bin/bash
source activate pt17
# mixed precision training実行
python py/main.py common.apex=true