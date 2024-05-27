from __future__ import absolute_import

'''
This file is from: https://raw.githubusercontent.com/bearpaw/pytorch-classification/master/models/cifar/resnet.py
by Wei Yang
'''
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['resnet']

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)    
                           #卷积核为3*3，输入输出通道暂定的卷积层

def res_cellA(in_planes, out_planes, stride=1):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
def res_cellB(in_planes, out_planes, stride=1):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
def res_cellC(in_planes, out_planes, stride=1):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=5,padding=2, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
                )
class se_res_cellB(nn.Module):
    expansion = 1         #expansion的作用？

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(se_res_cellB, self).__init__()       #这句话有什么用？会不会造成误导
        self.res_cellB1=res_cellB(inplanes,planes,stride)
        self.se_layer=SELayer(planes)

    def forward(self, x):
        out = self.res_cellB1(x)
        out = self.se_layer(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1         #expansion的作用？

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()       #这句话有什么用？会不会造成误导
        self.res_cellB1=res_cellB(inplanes,planes,stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_cellB2=res_cellB(planes,planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.res_cellB1(x)
        out = self.relu1(out)

        out = self.res_cellB2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual        #残差块内自己的连接，由块的输入到块输出的残差
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual


        return out


class ResNet_map(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet_map, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        downsample1 = nn.Sequential(
                nn.Conv2d(8, 16,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(16),
            )
        downsample2 = nn.Sequential(
                nn.Conv2d(16, 32,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32),
            )
        self.relu = nn.ReLU(inplace=True)
        self.BasicBlock11_s = BasicBlock(8, 8)
        self.BasicBlock12_s = BasicBlock(8, 8)
        self.BasicBlock13_s = BasicBlock(8, 8)
        self.BasicBlock21 = BasicBlock(8, 16,2,downsample1)
        self.BasicBlock22 = BasicBlock(16, 16,stride=1)
        self.BasicBlock23 = BasicBlock(16, 16,stride=1)
        self.BasicBlock31 = BasicBlock(16, 32,2,downsample2)
        self.BasicBlock32 = BasicBlock(32, 32,stride=1)
        self.BasicBlock33 = BasicBlock(32, 32,stride=1)
#        self.dropout = nn.Dropout(0.5)
#        self.new_conv=conv3x3(64, 64)
        self.avgpool = nn.AvgPool2d((2,2))
        self.fc = nn.Linear(64 , num_classes)  

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.BasicBlock11_s(x)  # 32x32
        x = self.BasicBlock12_s(x) 
        x = self.BasicBlock13_s(x) 
        x = self.BasicBlock21(x) 
        x = self.BasicBlock22(x) 
        x = self.BasicBlock23(x) 
        x = self.BasicBlock31(x) 
        x = self.BasicBlock32(x) 
        x = self.BasicBlock33(x) 
#        x = self.new_conv(x)

        x = self.avgpool(x)
#        x=dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
#         x=x*8
        x = F.softmax(x,dim=1)
    
        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)