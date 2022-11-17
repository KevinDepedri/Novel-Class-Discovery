from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import  transforms
import pickle
import os
import os.path
import datetime
import numpy as np
from data.rotationloader import DataLoader, GenericDataset
from utils.util import AverageMeter, accuracy
from models.resnet import BasicBlock
from tqdm import tqdm
import shutil
# So 
'''
So ragazee, this is a resnet that isnot the normal reset. it is a bit trick,
Mainy difference is that in the skip connections some times he put 
average pooling which is not part of the archiecture of rest. 
I am still discovering everything 
'''
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # block has
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if is_adapters:# wonot be used as adapters is set to 0 
            self.parallel_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if is_adapters:# i donot understnad the use of id_adapters. I need to check the paper more to see if he mentions is
            out = F.relu(self.bn1(self.conv1(x)+self.parallel_conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
# training function
def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args):
    loss_record = AverageMeter()# what is average meter ??? go to utils files it is an object
    #  it saves the value, average ,sum and count
    acc_record = AverageMeter()
    exp_lr_scheduler.step()# putting this here keeps making a warning. we could move it
    model.train()# set the model in the train mood.
    for batch_idx, (data, label) in enumerate(tqdm(dataloader(epoch))):# iterating using tqdm  with enum 
        data, label = data.to(device), label.to(device)# moving data to gpu
        optimizer.zero_grad()# zeroing the gradient
        output = model(data)# passing data to the model
        loss = criterion(output, label)# cross entrop loss between predicting output and labels of the annotations
        # calculating the loss the criteria in here is CrossEntropyLoss
        # the idea or main concept in here that you rotate images and accordingly try to predict how it was rotation 
        # accordingly. Maybe i missunderstood sthg. let continue and see. ? 
        # measure accuracy and record loss
        acc = accuracy(output, label)# the accuracy function is very very annoying
        acc_record.update(acc[0].item(), data.size(0))
        loss_record.update(loss.item(), data.size(0))# it is very annoying

        # compute gradient and do optimizer step
        optimizer.zero_grad()# you are  doing zero gradients
        loss.backward()# doing backwardstep to calculate the loss
        optimizer.step()# tell the optimizer do step boy.

    print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

    return loss_record

def test(model, device, dataloader, args):
    acc_record = AverageMeter()
    model.eval()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader())):
        data, label = data.to(device), label.to(device)
        output = model(data)
     
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), data.size(0))

    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Rot_resNet')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')# just the batch size  when you donot pass it it use default value
    parser.add_argument('--no_cuda', action='store_true', default=False,
                                    help='disables CUDA training')# incaseee you are trainining without cuda
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')# passed to the dataloader.
    parser.add_argument('--seed', type=int, default=1,
                                    help='random seed (default: 1)')# specific seed is set. maybe we should consider this point in our experiments
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')# how many epochs to run 
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')# the learning rate 
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')# the momentum value use
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')# name of the dataset
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')# location of the data set. if you donot pass it it use default
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')# locaiton to save the experiments
    parser.add_argument('--model_name', type=str, default='rotnet')#rotnet arhiecture used.

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()# allow using of cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)# set the torch seed to specific value

    runner_name = os.path.basename(__file__).split(".")[0]# returns file name . for example the file is called selfsupervised_learning.py,
    # it returned supervised_leanring
    model_dir= os.path.join(args.exp_root, runner_name)# string for new directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)# if the directory is not made, we create it
        # so in general we expect that inside the experiments we should find 3 kinds of folder
        # 1 for the superverised, semi supervised and autonovel discovery
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) # so the saved weight will be turned into
    # rotnet.pth but why when i open I see rotnet_cifar10 ??? interesting interesting. 

    # then ext part has been commented in the rotation laoder. check if needed.
    dataset_train = GenericDataset(
        dataset_name=args.dataset_name,
        split='train',
        dataset_root=args.dataset_root
       )
    dataset_test = GenericDataset(
        dataset_name=args.dataset_name,
        split='test',
        dataset_root=args.dataset_root
        )

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)
    # data is laoded
    global is_adapters # i donot get the use of it but we know it is global variable and it is set to 0
    is_adapters = 0 # it is a global variable and it is set to 0 
    # what happens if I change its value ????
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=4)# it is trying to predict the rotation
    # you have only 1 head in the end that takes 512 and turns it to 4 classes related to the 4 rotations that we have
    # 0,90,180,270
    # fun fact until now i do not understand the use of block expansion. he sets it to 1
    model = model.to(device)# sending it to cude. 

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
    # setting weight decay momentum with nestrov using stochastic gradient descents algorithim. 
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    # lr scheduler it provides several methods to adjust the learning rate based on the number of epochs.
    # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    # milestones  List of epoch indices. Must be increasing.
    # gamma Multiplicative factor of learning rate decay. Default: 0.1.
    criterion = nn.CrossEntropyLoss() # normal cross entropy loss 

    best_acc = 0 # set the best accuracy to 0 
    for epoch in range(args.epochs +1):
        loss_record = train(epoch, model, device, dloader_train, optimizer, exp_lr_scheduler, criterion, args)# training step
        acc_record = test(model, device, dloader_test, args)# testing step
        
        is_best = acc_record.avg > best_acc # compare average accurayc with best accuracy
        best_acc = max(acc_record.avg, best_acc)# choose the maximum
        if is_best:
            torch.save(model.state_dict(), args.model_dir)# save the model 

if __name__ == '__main__':
    main()
