#!/bin/bash
# poetry shell
python py/main.py -m apex=true arch=mobilenet_v2,resnet50
#python py/main.py -m arch=resnet50 arch=efficientnet_b0,efficientnet_b1
