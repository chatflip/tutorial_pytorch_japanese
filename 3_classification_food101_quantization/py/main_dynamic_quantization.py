import os

import torch
import torch.nn as nn

from args import opt
from model import mobilenet_v2
import torch.utils.mobile_optimizer as mobile_optimizer


def load_weight(model, weight_name):
    assert os.path.isfile(weight_name), "don't exists weight: {}".format(weight_name)
    print("use pretrained model : %s" % weight_name)
    param = torch.load(weight_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(param)
    return model


if __name__ == '__main__':
    args = opt()
    model = mobilenet_v2(pretrained=False, num_classes=args.num_classes)
    weight_name = 'weight/food101_mobilenetv2_best.pth'
    model = load_weight(model, weight_name)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    saved_weight = 'weight/food101_mobilenetv2_dynamic_quantize.pth'
    torch.save(quantized_model.state_dict(), saved_weight)
    saved_script = 'weight/food101_mobilenetv2_dynamic_quantize_script.pt'
    optimized_model = mobile_optimizer.optimize_for_mobile(torch.jit.script(quantized_model))
    torch.jit.save(optimized_model, saved_script)
