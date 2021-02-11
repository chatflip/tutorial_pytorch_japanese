#!/bin/bash
source activate pt17
python py/main.py backbone=mobilenet_v2 arch=DeepLabV3
python py/main.py backbone=efficientnet_b3 arch=Unet
