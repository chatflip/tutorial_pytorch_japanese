#!/bin/bash
source activate pt17
# mixed precision training実行
#python py/main.py apex=true
HYDRA_FULL_ERROR=1 python py/main.py
#HYDRA_FULL_ERROR=1 python py/main.py apex=true arch=resnet50
# python resnet50 --apex
# python mobilenetv2 --apex
# python efficienetnetb~