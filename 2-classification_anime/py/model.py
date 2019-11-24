import torch
import torch.nn as nn
import torchvision


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # pool1
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # pool2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # pool5
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # fc7
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # fc8
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alex(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_zoo_param = torchvision.models.alexnet(pretrained=True)
        model.features = model_zoo_param.features
        model.classifier[1].weight = model_zoo_param.classifier[1].weight
        model.classifier[1].bias = model_zoo_param.classifier[1].bias
        model.classifier[4].weight = model_zoo_param.classifier[4].weight
        model.classifier[4].bias = model_zoo_param.classifier[4].bias
    return model
