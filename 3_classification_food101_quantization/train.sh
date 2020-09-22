#!/bin/bash
source activate pt16
# mixed precision training実行
python -m torch.distributed.launch --nproc_per_node=2 --use_env py/main.py --apex
