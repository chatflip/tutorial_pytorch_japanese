#!/bin/bash
source activate pt17
#python py/main.py backbone=timm-efficientnet-b0 arch=DeepLabV3Plus
python py/main.py backbone=timm-efficientnet-b4 arch=UnetPlusPlus image_height=512 image_width=512 epochs=50
