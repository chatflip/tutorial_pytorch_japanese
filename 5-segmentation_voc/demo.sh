#!/bin/bash
source activate pt17
python py/demo_webcam.py --weight_path weight/voc2012_DeepLabV3Plus_timm-efficientnet-b0_256_256.pth
python py/demo_webcam.py --weight_path weight/voc2012_UnetPlusPlus_timm-efficientnet-b4_512_512.pth --inference_height 512 --inference_width 512
