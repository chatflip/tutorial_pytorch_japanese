#!/bin/bash
source activate pt17
# mixed precision training実行
python py/main.py common.apex=true
# python resnet50 --apex
# python mobilenetv2 --apex
# python efficienetnetb~