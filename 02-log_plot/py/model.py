# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018

@author: okayasu.k
require pytorch 0.4.0
        torchvision 0.2.1
"""

import torch.nn as nn

#ネットワークを別プログラムに定義
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            #BxCxWxH
            #B,1,32,32 → B,6,28,28
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            #B,6,28,28 → B,6,28,28
            nn.ReLU(inplace=True),
            #B,6,28,28 → B,6,14,14
            nn.MaxPool2d(kernel_size=2, stride=0),
            #B,6,14,14 → B,16,10,10
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            #B,16,10,10 → B,16,10,10
            nn.ReLU(inplace=True),
            #B,16,10,10 → B,16,5,5
            nn.MaxPool2d(kernel_size=2, stride=0,padding=0),
        )
        self.classifier = nn.Sequential(
            #B,16×5×5 → B,120
            nn.Linear(16 * 5 * 5, 120),
            #B,120 → B,120
            nn.ReLU(inplace=True),
            #B,120 → B,84
            nn.Linear(120, 84),
            #B,84 → B,84
            nn.ReLU(inplace=True),
            #B,84 → B,10
            nn.Linear(84, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        #B,16,5,5 → 16×5×5
        x = x.view(x.size(0), 16 * 5 * 5)
        x = self.classifier(x)
        return x
