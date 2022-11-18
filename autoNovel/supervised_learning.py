import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter
from models.resnet import ResNet, BasicBlock 
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from data.svhnloader import SVHNLoader
from tqdm import tqdm
import numpy as np
import os
# 
def train(model, train_loader, labeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)# just a normal sgd with momentum
    # with weight decay 
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)#Decays the learning rate of each 
    #parameter group by gamma every step_size epochs. 
    # Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler
    criterion1 = nn.CrossEntropyLoss() # our beautiful cross entropy losss
    for epoch in range(args.epochs):
        loss_record = AverageMeter()# our average meeting for the loss. WHy no average metter for accuracy ???
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):# only use label data to train the model
            x, label = x.to(device), label.to(device)
            output1, _, _ = model(x)# returns output 1 for first head,output 2 for second head, output 3 which is outptu before heads
            # my guess i am trainingi supervised ehre so i donot care about unsupervised head.
            loss= criterion1(output1, label)
            loss_record.update(loss.item(), x.size(0))# record the loss term
            optimizer.zero_grad()# zeroing the gradients
            loss.backward()# loss backward step
            optimizer.step()# tell the optimzier to take a step 
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)# run a test 

def test(model, test_loader, args):
    model.eval() # turn model to evaluation 
    preds=np.array([])# numpy array for predictions
    targets=np.array([])# numpy array for targets 
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):# why 3 things ?git
        x, label = x.to(device), label.to(device)
        output1, output2, _ = model(x)
        if args.head=='head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return preds 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)# the learning ratte
    parser.add_argument('--gamma', type=float, default=0.5)# what is gamma? check the paper i guess it is for the schedule :P
    parser.add_argument('--momentum', type=float, default=0.9)# momentum term of optimzieer
    parser.add_argument('--weight_decay', type=float, default=1e-4)# weight decay
    parser.add_argument('--epochs', default=100, type=int)# run for 100 epochs
    parser.add_argument('--step_size', default=10, type=int)# what is the step size????
    parser.add_argument('--batch_size', default=128, type=int)# batch size
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)# number of unlabled classes
    parser.add_argument('--num_labeled_classes', default=5, type=int)# number of lableled classes
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')#datasetroot
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')# where to save the files
    parser.add_argument('--rotnet_dir', type=str, default='./data/experiments/selfsupervised_learning/rotnet_cifar10.pth')# where to find pretrained model semi supervised
    parser.add_argument('--model_name', type=str, default='resnet_rotnet')#it is called resnet-rotnet
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--mode', type=str, default='train')# lets train it
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()# cuda check available
    device = torch.device("cuda" if args.cuda else "cpu")# turn cuda to on 
    runner_name = os.path.basename(__file__).split(".")[0]# take the name supervised_learning
    model_dir= os.path.join(args.exp_root, runner_name)# create a path with name supervised learning
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)# create a folder
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) # save them as with this name resnet_rotnet 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)# assuming you are working with 10 classes
    # 5 classes unlabled 5 classes lableld. he is training 2 heads 1 for the labeled and the other for the unlabled data.

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes# total number of classes.

    state_dict = torch.load(args.rotnet_dir)# laod from pre weights
    del state_dict['linear.weight']# just used for training annotation head. size is [4,512]
    del state_dict['linear.bias']# deleted the weights. [4]
    # he is deleted the of the linear part of the rot net
    # after this operation it is as if you no longer have any weights or biases in the end. they are completely deleted
    model.load_state_dict(state_dict, strict=False)#  whether to strictly enforce that the keys in state_dict match 
    # the keys returned by this module’s state_dict() function. It has to be not stricted because they donot maatch. 
    # the model in here has 2 extra heads while the semisupervised learning module contains no heads but everything else is the same
    for name, param in model.named_parameters(): 
        if 'head' not in name and 'layer4' not in name:# in here we are setting some paramters that shouldnot be trained
            # so the heads should be trained
            # layer 4  should be trained
            # anything else will be frozen
            param.requires_grad = False
    #they only train layers after layer 4. parameters before are fixed.
    # annotation head to learn low level representation. 
    # amazing i am lost but not important haahhaa
    if args.dataset_name == 'cifar10':
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'cifar100':
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
    elif args.dataset_name == 'svhn':
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))

    if args.mode == 'train':
        train(model, labeled_train_loader, labeled_eval_loader, args)# sending to the training set
        torch.save(model.state_dict(), args.model_dir)# saving the model
        print("model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':# if i am in the mood of testing
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
    print('test on labeled classes')
    args.head = 'head1'
    test(model, labeled_eval_loader, args)# testing on the lableed data set 
