import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from baselines.resnet import BasicBlock, LambdaLayer, _weights_init

class ModifiedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ModifiedResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        kernel_size = int(out3.shape[3])
        out = F.avg_pool2d(out3, kernel_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out1, out2, out3
    

def modified_resnet20():
    return ModifiedResNet(BasicBlock, [3, 3, 3])


def modified_resnet32():
    return ModifiedResNet(BasicBlock, [5, 5, 5])

def modified_resnet44():
    return ModifiedResNet(BasicBlock, [7, 7, 7])


def modified_resnet56():
    return ModifiedResNet(BasicBlock, [9, 9, 9])

def modified_resnet110():
    return ModifiedResNet(BasicBlock, [18,18,18])
       