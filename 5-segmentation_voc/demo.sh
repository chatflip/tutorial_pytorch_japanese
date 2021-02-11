#!/bin/bash
source activate pt17
python py/demo_webcam.py backbone=mobilenet_v2 arch=DeepLabV3
python py/demo_webcam.py backbone=efficientnet_b3 arch=Unet
