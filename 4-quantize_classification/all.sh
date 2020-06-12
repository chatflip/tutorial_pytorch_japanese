#!/bin/bash
source activate pt15
python py/main_float.py --apex
python py/main_dynamic_quantization.py
python py/main_static_quantization.py --batch-size=32 --backend=fbgemm
python py/main_qat.py --batch-size=64 --val-batch-size=128 --epoch=20 --lr=0.00001 --lr-step-size=30 --lr-gamma=0.1
python py/main_validate.py --val-batch-size=1 --print-freq=1000
