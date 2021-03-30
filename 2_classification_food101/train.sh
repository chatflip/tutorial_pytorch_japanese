#!/bin/bash
# poetry shell
python py/main.py -m apex=true arch=mobilenet_v2 epochs=10
python py/main.py -m arch=efficientnet_b0,efficientnet_b3,efficientnet_b5,efficientnet_b7 epochs=3
