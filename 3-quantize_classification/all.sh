#!/bin/bash
source activate pt14
python py/main_float.py
python py/main_dynamic_quantization.py
python py/main_static_quantization.py
python py/main_qat.py --batch-size=32 --epoch=20 --lr=0.00001 --lr-step-size=30 --lr-gamma=0.1
python py/main_test.py --val-batch-size=1
