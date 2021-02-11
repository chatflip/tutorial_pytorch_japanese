#!/bin/bash
source activate pt16
# mixed precision training実行
python py/main_dynamic_quantization.py
python -m torch.distributed.launch --nproc_per_node=2 --use_env py/main_static_quantization.py
python -m torch.distributed.launch --nproc_per_node=2 --use_env py/main_qat.py --batch-size 96 --qat-epoch 1
python py/main_evaluate.py
