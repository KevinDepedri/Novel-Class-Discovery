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
            # dataloader contain 3 things
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
        args.head = 'head1'# we are setting that we want head number 1 to be tested and also this head is trained supervised.
        # other head until the current momemt it is useless.
        test(model, labeled_eval_loader, args)# run a test 

def test(model, test_loader, args):
    model.eval() # turn model to evaluation 
    preds=np.array([])# numpy array for predictions
    targets=np.array([])# numpy array for targets 
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):# data loader contain 3 things so everything is fine. no problem
        x, label = x.to(device), label.to(device)# the labels
        output1, output2, _ = model(x)# head 1 output and head 2 output
        # output1 has size of (128,5)
        if args.head=='head1':
            output = output1# indicating that we are using the output of head 1 only in here
        else:
            output = output2# are we training boths heads ??? no we are not 
        _, pred = output.max(1)# Returns the maximum value of all elements in the input tensor.
        # dim is set to 1 okay ??? do you understand?
        # returns a tupple of (max, max_indices)
        # we care only about max indices 
        # so my guess pred is size of 128 nyahahahahahahhaha
        targets=np.append(targets, label.cpu().numpy())# label.cpu.numpy convert tensor to numpy Returns the tensor as a NumPy ndarray.
        #Append values to the end of an array called targets,
        # after each iteration you add 128 new values in the end of the target
        preds=np.append(preds, pred.cpu().numpy())
        # very similar thing is happening in here.
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    # what is the cluster_acc? it is a function written in my utils files that I am trying to understand right now.
    # what is the nemi score? sci learn function that i need to google to understand.
        # it is called normalized_mutual_info_score on sci learn
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
        # Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to scale 
        # the results between 0 (no mutual information) and 1 (perfect correlation). In this function, mutual 
        # information is normalized by some generalized mean of H(labels_true) and H(labels_pred)), defined by the average_method.
        # This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values 
        # won’t change the score value in any way.
        
    ############################################################################################################################
    # what is the ari score? sci learn function that i need to google to understand.
        # it is called adjusted_rand_score on sci learn
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
        # Rand index adjusted for chance.
        # The Rand Index computes a similarity measure between two clusterings by considering all pairs of
        # samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
        # The raw RI score is then “adjusted for chance” into the ARI score using the following scheme:
        # ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
        # The adjusted Rand index is thus ensured to have a value close to 0.0 for random labeling independently 
        # of the number of clusters and samples and exactly 1.0 when the clusterings are identical (up to a permutation).
        # Similarity score between -1.0 and 1.0.
    # Another explaination 
        # The Adjusted Rand score is introduced to determine whether two cluster results are similar to each other. 
        # In the formula, the “RI” stands for the rand index, which calculates a similarity between two cluster results 
        # by taking all points identified within the same cluster. This value is equal to 0 when points are assigned into 
        # .clusters randomly and it equals to 1 when the two cluster results are same. 
        # This metric is used to evaluate whether dimension-reduced similarity cluster results are similar to one other.
    # What is the adjusted Rand index?
        #The adjusted Rand index is the corrected-for-chance version of the Rand index. Such a correction for chance 
        # establishes a baseline by using the expected similarity of all pair-wise comparisons between clusterings specified 
        # by a random model.
        # https://baiadellaconoscenza.com/dati/argomento/read/9864-what-is-the-adjusted-rand-index#question-0
    # What does a negative adjusted Rand index mean?
        # Negative ARI says that the agreement is less than what is expected from a random result.
        # This means the results are 'orthogonal' or 'complementary' to some extend.
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
