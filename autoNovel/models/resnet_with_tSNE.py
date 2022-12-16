from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, models
import pickle
import os.path
import datetime
import numpy as np
from torchsummary import summary
from utils.additional_classes import t_SNE


class resnet_sim(t_SNE, nn.Module):
    def __init__(self, num_labeled_classes=5, num_unlabeled_classes=5):

        t_SNE.__init__(self)
        nn.Module.__init__(self)

        self.encoder = models.__dict__['resnet18']()  # Initializing ResNet18 by pytorch
        self.encoder.fc = nn.Identity()  # Replace the fully connected layer with an identity
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()  # Remove the max pool layer
        self.head1 = nn.Linear(512, num_labeled_classes)  # First head: to classify known classes
        self.head2 = nn.Linear(512, num_unlabeled_classes)  # Second head: to classify unknown classes

    def _forward_impl(self, x):
        out = self.encoder(x)

        # Compute features
        features = torch.flatten(out, 1)
        for sample in range(features.shape[0]):
            self.feature_vectors.loc[len(self.feature_vectors)] = features[sample].detach().cpu().numpy()

        out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2, out

    # Push labels and call the forward implementation
    def forward(self, x: list) -> Tensor:
        self.push_labels(x[1].cpu())  # .item()
        return self._forward_impl(x[0])

# Initialization of a ResNet architecture built to perform classification using two different heads, one head is used
# to classify labeled sampled, while one is used to classify unlabeled samples.
class ResNet(t_SNE, nn.Module):
    def __init__(self, block, num_blocks, num_labeled_classes=5, num_unlabeled_classes=5):

        t_SNE.__init__(self)
        nn.Module.__init__(self)

        self.in_planes = 64
        # Initial Convolution + BatchNormalization
        self.conv1    = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        # Append ResNet18 layers
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Now we extend the (self-supervised) pre-trained network with a classification head (head1) which will be used
        # to perform a supervised fine-tuning over the labeled samples. This new head brings us from a dimension R-d
        # (feature mapping obtained from layer4 output) to dimension R-Cl (equal to the number of labeled classes Cl).
        # This is done implementing as a single linear layer followed by a softmax function (present in the training).
        # Only the last layer4 + this new head1 are fine-tuned on the labelled dataset Dl in order to learn a classifier
        # for the Cl known classes without over-fitting all the NN. This is done using Cl labels and optimizing CE loss
        self.head1 = nn.Linear(512*block.expansion, num_labeled_classes)  # First head: to classify known classes

        # Furthermore, we extend the (self-supervised) pre-trained network with another classification head (head2)
        # which will be used in the next-step for the Transfer-Learning via Rank Statistics of the unsupervised samples.
        # Before training this head we will need to compute the pairwise labels Sij, which tells if two unsupervised
        # images are similar (thus, they probably belong to the same class). This head brings us from a dimension R-d
        # (feature mapping obtained from layer4 output) to dimension R-Cu (equal to the number of unlabeled classes Cu).
        # Once the labels Sij have been obtained, we use them as pseudo-labels to train a comparison function for the
        # unlabelled data. In order to do this, we use the new head2 to extract a new descriptor vector optimized for
        # the unlabelled data. As before, the head is composed of a linear layer followed by a softmax in the training.
        # Finally, we will be able to compute the new descriptor for two samples, and then multiply one by the other,
        # the result is a score function which tells us if the two samples belong to the same class or not.
        self.head2 = nn.Linear(512*block.expansion, num_unlabeled_classes)  # Second head: to classify unknown classes

    def _make_layer(self, block, planes, num_blocks, stride):
        # Compute a strides list for the different blocks. Use the input argument stride for the first block
        # (this allows to reduce the dimension if it is >1), then use stride=1 for all the other layers in the block
        strides = [stride] + [1]*(num_blocks-1)
        # Define an empty list of layers
        layers = []
        # Each layer is composed of a set of blocks with the previously defined strides
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Block expansion is set to 1 in the BasicBlock class
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Compute the output of the NN, x has an initial size of (256,3,32,32)
        # First perform convolution with batch norm then relu
        out = F.relu(self.bn1(self.conv1(x)))  # Output has size of (256,64,32,32)
        out = self.layer1(out)  # Output shape (256,64,32,32)
        out = self.layer2(out)  # Output shape (256,128,16,16)
        out = self.layer3(out)  # Output has shape of (256,256,8,8)
        out = self.layer4(out)  # Output has shape of (256,512,4,4)
        out = F.avg_pool2d(out, 4)  # Average pooling with kernel size 4. the output has shape of (256,512,1,1)

        # FIXME: Should features be computed here?

        out = out.view(out.size(0), -1)  # Reshaping to specific size. the output has shape of (256,512)
        out = F.relu(out)  # Add ReLU to benefit ranking. The output has size of (256,512)

        # Compute features
        features = torch.flatten(out, 1)
        for sample in range(features.shape[0]):
            self.feature_vectors.loc[len(self.feature_vectors)] = features[sample].detach().cpu().numpy()

        out1 = self.head1(out)  # Compute output of head 1 to classify known classes
        out2 = self.head2(out)  # Compute output of head 2 to classify unknown classes
        return out1, out2, out  # Return the output of each head and the output before the heads

    # Push labels and call the forward implementation
    def forward(self, x: list) -> Tensor:
        self.push_labels(x[1].cpu())  # .item()
        return self._forward_impl(x[0])


# Definition of the BasicBlock of ResNet
class BasicBlock(nn.Module):
    # Define the expansion to 1, it means that the dimension will never be increased moving from block to block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Perform a first convolution with the input stride, in this case if it is > 1 then we will reduce the dimension
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # The second convolution has a fixed stride of 1 to avoid compressing more the data
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Define the shortcut that just propagates forward the input value
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        # if stride is not equal 1 or if input layers are not equal to output layers (need expansion>1 for that)
        if stride != 1 or in_planes != self.expansion*planes:
            # The shortcut contains an average pooling using a kernel 2x2. This is done to reduce the dimension
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion*planes:
                # Set the padding to be equal 1
                self.is_padding = 1

    def forward(self, x):
        # Perform convolution, batch norm and then relu
        out = F.relu(self.bn1(self.conv1(x)))
        # Perform again convolution and batch norm
        out = self.bn2(self.conv2(out))

        # If in_planes is different to expansion*planes
        if self.is_padding:
            # The result of the shortcut is the avg-pooling.
            shortcut = self.shortcut(x)
            # Now the output is incremented by the concatenation of shortcut with a tensor of zeros with the same size
            out += torch.cat([shortcut, torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)], 1)

        # Otherwise, if in_planes is different to planes*expansion
        else:
            # In this case the output is just incremented by the value of shortcut, we just use the sequential shortcut
            out += self.shortcut(x)

        # Apply the ReLu activation function and return the result
        out = F.relu(out)
        return out


if __name__ == '__main__':

    a = resnet_sim()
    print(a)
    b = ResNet(BasicBlock, [2, 2, 2, 2])
    print(b)

    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    num_labeled_classes = 10
    num_unlabeled_classes = 20
    # model = ResNet(BasicBlock, [2, 2, 2, 2], num_labeled_classes, num_unlabeled_classes)
    model= resnet_sim( num_labeled_classes, num_unlabeled_classes).cuda()
    # model = model
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #        print(name, param.data)
    ssl='Barlow_twins'
    if ssl =='Barlow_twins':
            state_dict = torch.load('trained_models/cifar10/barlow_twins/barlow-cifar10-otu5cw89-ep=999.ckpt', map_location="cpu")["state_dict"]
            for l in list(state_dict.keys()):
                if "classifier" in l or 'projector' in l :
                    del state_dict[l]
    for k in list(state_dict.keys()):
                if "encoder" in k:
                    state_dict[k.replace("encoder", "backbone")] = state_dict[k]
                if "backbone" in k:
                    state_dict['encoder.'+k.replace("backbone.", "")] = state_dict[k]
                del state_dict[k]
    # for k in list(state_dict.keys()):
	# print(k)
    # print(state_dict)
    model.load_state_dict(state_dict, strict=False)
    # print(model.parameters())
    # for param in model.parameters():
    #   print(param.data)
    for name, param in model.named_parameters():
        print(name, param.data)
    # print(model)
    # print(summary(model, (3, 32, 32), batch_size=256))
    # y1, y2, y3 = model(Variable(torch.randn(256, 3, 32, 32).to(device)))
    # print(y1.size(), y2.size(), y3.size())
    # for name, param in model.named_parameters():
    #     print(name)
