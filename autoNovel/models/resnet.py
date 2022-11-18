from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import  transforms
import pickle
import os.path
import datetime
import numpy as np
from torchsummary import summary

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_labeled_classes=5, num_unlabeled_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)# conv 2d taking 3 filters and outputing 64
        self.bn1      = nn.BatchNorm2d(64)# batch normalization
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head1 = nn.Linear(512*block.expansion, num_labeled_classes)# the first head
        self.head2 = nn.Linear(512*block.expansion, num_unlabeled_classes)# the second head

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)# output a list of size of 2 containing either [1,1] for first make layer and [2,1] for the rest
        layers = []# empty list
        for stride in strides:# i am looping over my list where i am using different strides for each block.
            layers.append(block(self.in_planes, planes, stride))# create a block with specific stride
            self.in_planes = planes * block.expansion# block expansion is set to 1. 
        return nn.Sequential(*layers)

    def forward(self, x):
        # x has size of (256,3,32,32)
        out = F.relu(self.bn1(self.conv1(x)))# convelution with batch noram then relu. output has size of (256,64,32,32)
        out = self.layer1(out)# output shape (256,64,32,32)
        out = self.layer2(out)# output shape (256,128,16,16)
        out = self.layer3(out) # output has shape of (256,256,8,8)
        out = self.layer4(out) # output has shape of (256,256,4,4)
        out = F.avg_pool2d(out, 4) # average pooling with kernel size 4. the output has shape of (256,512,1,1)
        out = out.view(out.size(0), -1)# reshapping to specific size. the output has shape of (256,512)
        out = F.relu(out) #add ReLU to benifit ranking. The output has size of (256,512) 
        out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2, out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)# he sets it to stride 1 by force 
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()# this is just to do shortcuts
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion*planes:# if stride isnot equal 1 or input layers not equal ouput layers(EX: you inputs64 and output 3)
            self.shortcut = nn.AvgPool2d(2) # the shortcut contains an average pooling.
            # Applies a 2D average pooling over an input signal composed of several input planes. You are using kernel size 2
            # check https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html for more details
            if in_planes != self.expansion*planes:
                self.is_padding = 1 # set the padding to be equal 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))# convlution batch norm then relu
        out = self.bn2(self.conv2(out))# convelution batch normal 

        if self.is_padding: # in the case of inplanes is equal to planes*expansion
            shortcut = self.shortcut(x)# shortcut is avgpooling. I have shape of (256,64,16,16)
            out += torch.cat([shortcut,torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)],1)# the size was 256,128,16,16 but became (256,128,16,16)
            # you are concating shortcut with a tensor of zeros same size as the shortcut along 1 direciton
        else:
            out += self.shortcut(x) # in casse padding is false you just set put a shortcut to the output
            # in this case you assume that you are using the shortcut of the sequantial

        out = F.relu(out)# activation function 
        return out # return

if __name__ == '__main__':

    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    num_labeled_classes = 10
    num_unlabeled_classes = 20
    model = ResNet(BasicBlock, [2,2,2,2],num_labeled_classes, num_unlabeled_classes)
    model = model.to(device)
    print(model)
    print(summary(model, (3,32,32),batch_size=256))
    y1, y2,y3 = model(Variable(torch.randn(256,3,32,32).to(device)))
    print(y1.size(), y2.size(),y3.size())
    for name, param in model.named_parameters(): 
        print(name)
