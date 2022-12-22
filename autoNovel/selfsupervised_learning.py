from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import pickle
import os
import os.path
import datetime
import numpy as np
from data.rotationloader import DataLoader, GenericDataset
from data.rotation_loader_mnisit import DataLoader_mnisit, GenericDataset_mnisit
from data.rotation_loader_mnisit_basline import DataLoader_mnisit_baseline, GenericDataset_mnisit_basline

from utils.util import AverageMeter, accuracy,seed_torch
from models.resnet import BasicBlock
from tqdm import tqdm
import shutil
import wandb
from torch.utils.data import ConcatDataset

global logging_on

'''
# Self supervised learning (as from section 2.1 of AutoNovel paper) - part 1
# Here the model is trained with self-supervision on the union of Dl and Du

A modified Resnet is used, where sometimes in the skip connections we have average pooling (not part of original ResNet)
In the paper it is not mention how RotNet is built. In the majority of the case RotNets use AlexNet and not ResNet18
GitHub of the RotNet paper: https://github.com/gidariss/FeatureLearningRotNet
'''


class resnet_sim(nn.Module):
    def __init__(self, num_labeled_classes=5):
        super(resnet_sim, self).__init__()
        self.encoder = models.__dict__['resnet18']()  # Initializing ResNet18 by pytorch
        self.encoder.fc = nn.Identity()  # Replace the fully connected layer with an identity
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()  # Remove the max pool layer
        self.head1 = nn.Linear(512, num_labeled_classes)  # First head: to classify known classes

    def forward(self, x):
        out = self.encoder(x)
        out1 = self.head1(out)
        return out1


# Initialization of a ResNet architecture built to perform self-supervised learning as RotNet. It has only one output
# heads with 4 possible output classes. The output classes are the 4 possible rotations (0, 90, 180, 270 degrees)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # Initial Convolution + BatchNormalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Append ResNet18 layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Implement a final linear layer to classify between the given classes. This head will be used only to perform
        # this unsupervised classification task (RotNet), and will be removed in the next task (Supervised training)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # If is_adapters is true then add a parallel_convolution layer
        if is_adapters:  # Not used since adapters is set to 0
            self.parallel_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        # Compute a strides list for the different blocks. Use the input argument stride for the first block
        # (this allows to reduce the dimension if it is >1), then use stride=1 for all the other layers in the block
        strides = [stride] + [1] * (num_blocks - 1)
        # Define an empty list of layers
        layers = []
        # Each layer is composed of a set of blocks with the previously defined strides
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Block expansion is set to 1 in the BasicBlock class
        return nn.Sequential(*layers)

    def forward(self, x):
        # Compute the output of the NN
        # If is adapters is true consider also the parallel convolution in the computation of the output
        if is_adapters:  # TODO:How is is_adapters used and defined??
            out = F.relu(self.bn1(self.conv1(x) + self.parallel_conv1(x)))
        # Otherwise consider just the previous layers
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        # Compute the output through all the ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Training function of ResNet
def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args):
    # Define two instances of AverageMeter to compute and store the average and current values of the loss and
    # of the accuracy during the training procedure (see util.py)
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    print(logging_on)
    # Set the model in the training mode
    model.train()
    # Iterate through the dataloader using tqdm to print a graphic progress bar
    for batch_idx, (data, label) in enumerate(tqdm(dataloader(epoch))):
        # Move both data and label to gpu, the data has input (256,3,32,32)
        data, label = data.to(device), label.to(device)
        # Zero the gradient of the optimizer to remove previously computed values
        optimizer.zero_grad()
        # Compute the output of the model for the input data. Output dimension (256,4)
        output = model(data)

        # Compute cross entropy loss between the predicted output and labels of the annotation. This is done using the
        # criterion parameter, the idea is that you rotate images and try to predict how it was rotated accordingly
        loss = criterion(output, label)

        # Compute the accuracy using the accuracy() function from the file utils.py In this call the argument 'topk' is
        # not passed, so it is set to default value of (1,). We are just computing the accuracy of the prediction
        acc = accuracy(output, label)

        # Update the accuracy and loss AverageMeter instances with the values just computed
        acc_record.update(acc[0].item(), data.size(0))  # data.size(0) is the batch size
        loss_record.update(loss.item(), data.size(0))

        # Zero the gradient of the optimizer, back-propagate the loss and perform an optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Perform a step on the input exp_lr_scheduler (scheduler used to define the learning rate)
    exp_lr_scheduler.step()

    # Print the result of the training procedure
    print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
    if logging_on:
        wandb.log({"epoch": epoch, "Total_average_loss": loss_record.avg,
                   "Head_1_training_accuracy": acc_record.avg, "lr": exp_lr_scheduler.get_last_lr()[0]}, step=epoch)
    return loss_record


def test(model, device, dataloader, epoch):
    # Define an instance of AverageMeter to compute and store the average and current values of the accuracy
    acc_record = AverageMeter()
    # Put the model in evaluation mode
    model.eval()
    # Iterate through the dataloader using tqdm to print a graphic progress bar
    for batch_idx, (data, label) in enumerate(tqdm(dataloader())):
        # Move both data and label to gpu, the data has input (256,3,32,32)
        data, label = data.to(device), label.to(device)
        # Compute the output of the model for the input data. Output dimension (256,4)
        output = model(data)

        # Compute the accuracy using the accuracy() function from the file utils.py
        # Also in this call the argument 'topk' is not passed, so it is set to default value of (1,)
        acc = accuracy(output, label)
        # Update the accuracy AverageMeter object with the values just computed
        acc_record.update(acc[0].item(), data.size(0))

    # Print the result of the testing procedure
    print('Test Acc: {:.4f}'.format(acc_record.avg))
    if logging_on:
        wandb.log({"epoch": epoch, "Head_1_val_accuracy": acc_record.avg}, step=epoch)
    return acc_record


def main():
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='Rot_resNet')

    # Add to the parser the argument: 'batch_size' with a default value of 64. It is used in the dataloader
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # Add to the parser the argument: 'no_cuda' with a default value of False. If it is True we train without cuda
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

    # Add to the parser the argument: 'num_workers' with a default value of 4. It is used in the dataloader
    parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')

    # Add to the parser the argument: 'seed' with a default value of 1. It is used in the dataloader
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # Add to the parser the argument: 'epochs' with a default value of 200. It is used in the training procedure
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')

    # Add to the parser the argument: 'lr' with a default value of 0.1. It is used in the training procedure
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')

    # Add to the parser the argument: 'momentum' with a default value of 0.9. It is used in the training procedure
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

    # Add to the parser the argument: 'dataset_name' with a default value of cifar10. It is used in the dataset
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar10c, cifar100, svhn')

    # Add to the parser the argument: 'dataset_root' with as default path the cifar10 path. It is used in the dataset
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')

    # Add to the parser the argument: 'exp_root' with as default path the experiment path. It is used to save result
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')

    # Add to the parser the argument: 'model_name' with a default value of 'rotnet'. It is used to create directory
    parser.add_argument('--model_name', type=str, default='rotnet')

    # Extract the args and make them available in the args object
    args = parser.parse_args()

    # Define if cuda can be used and initialize the device used by torch. Furthermore, specify the torch seed
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.manual_seed(args.seed)
    seed_torch(args.seed)

    # Returns the current file name. In this case file is called selfsupervised_learning.py
    runner_name = os.path.basename(__file__).split(".")[0]
    # Define the name of the wanted directory as 'experiment root' + 'name of current file'
    model_dir = os.path.join(args.exp_root, runner_name)
    # If the previously defined directory does not exist, them create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if logging_on:
        wandb.login()  # 4619e908b2f2c21261030dae4c66556d4f1f3178
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "dataset": args.dataset_name,
            "momentum": args.momentum,
            "epochs": args.epochs,
        }
        wandb.init(project="trends_project", entity="mhaggag96", config=config, save_code=True)
    # Define the name of the path to save the trained model
    args.model_dir = model_dir + '/' + '{}.pth'.format(args.model_name)
    if (args.dataset_name=="mnisit"):
        # CUDA_VISIBLE_DEVICES=0 python selfsupervised_learning.py --dataset_name mnisit --model_name rotnet_mnisit_MIXMIX 
        dataset_name_1=GenericDataset_mnisit('mnisit',split='train')
        dataset_name_2=GenericDataset_mnisit('mnisitm',split='train')
        dataset = ConcatDataset([dataset_name_1,dataset_name_2])
        dloader_train = DataLoader_mnisit(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)
        dataset_name_1=GenericDataset_mnisit('mnisit',split='test')
        dataset_name_2=GenericDataset_mnisit('mnisitm',split='test')
        dataset = ConcatDataset([dataset_name_1,dataset_name_2])
        dloader_test = DataLoader_mnisit(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)
    elif (args.dataset_name=="mnisit_base"):
            # CUDA_VISIBLE_DEVICES=0 python selfsupervised_learning.py --dataset_name mnisit_base --model_name rotnet_mnisit_only 
        dataset_name_1=GenericDataset_mnisit_basline('mnisit',split='train')
        dloader_train = DataLoader_mnisit_baseline(
            dataset=dataset_name_1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)
        dataset_name_1=GenericDataset_mnisit_basline('mnisit',split='test')
        dloader_test = DataLoader_mnisit_baseline(
            dataset=dataset_name_1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True)
    else:
    # Create a torch dataset for the train and test data. Full comments available in the rotation loader section
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

        # Create a torch dataloader for the train and test data. Full comments available in the rotation loader section
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

    # Define a global variable is_adapters. It is initialized with a value of zero by default.
    global is_adapters  # TODO: Does anyone know what is the use of this??? what happens if I change its value ????
    is_adapters = 0

    # Call the previously defined class to instantiate a ResNet architecture with 4 block (each one with two layers).
    # This NN will be used to predict the rotation of the examples coming from the dataloader, for this reason the
    # number of classes will be 4, for the four possible rotation (0, 90, 180, 270 degrees)
    # The ResNet architecture is the one described above, while the BasicBlock is imported from resnet.py
    normal_model = True
    if normal_model:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)
    else:
        model = resnet_sim(num_labeled_classes=4)
    # Send the model to the device
    model = model.to(device)
    # print(model)
    # Instantiate SGD optimizer with input learning rate and momentum, and with pre-define weight_decay and Nesterov
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)

    # Instantiate a learning rate scheduler to adjust the learning rate based on the number of epochs. In this case it
    # is set on the MultiStepLR setting. It decays the learning rate of each parameter group by gamma once the number
    # of epoch reaches one of the specified milestones.
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    # Instantiate a standard cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # Set the best accuracy to 0
    best_acc = 0

    # Iterate through the numer of epochs defined in input
    for epoch in range(args.epochs + 1):
        # Compute the loss of the training step
        loss_record = train(epoch, model, device, dloader_train, optimizer, exp_lr_scheduler, criterion, args)
        # Compute the accuracy of the testing step
        acc_record = test(model, device, dloader_test, epoch)

        # Compare the average accuracy saved in the AverageMeter object with the best accuracy measured up to now
        is_best = acc_record.avg > best_acc
        # Update the best accuracy if the current average accuracy is greater
        best_acc = max(acc_record.avg, best_acc)
        # If the average accuracy is the best accuracy achieved up to now, then save the model in the defined directory
        if is_best:
            torch.save(model.state_dict(), args.model_dir)
    if logging_on:
        wandb.finish()


# If the name variable is equal to __main__ then run the main function and start the training and testing procedure.
if __name__ == '__main__':
    logging_on = True
    main()
