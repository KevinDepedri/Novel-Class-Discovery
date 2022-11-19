import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps 
from models.resnet import ResNet, BasicBlock 
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.svhnloader import SVHNLoader, SVHNLoaderMix
from tqdm import tqdm
import numpy as np
import os
## starting the monster file. the hardest file
# functions for training without incremental learning
def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # the optimizier using sg and weight decay
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)# the lr schedular
    criterion1 = nn.CrossEntropyLoss() # cross entrop loss 
    criterion2 = BCE() # this is binary cross entropy but he made the functions in the utils files.
    # we should try to understand it as we go 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()# average metter  we saw orevuiysky
        model.train()# turning on training mode
        exp_lr_scheduler.step()# each step for learning rate
        # it makes a warning in here but no problem 
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) # the takes the current epoch
        # and the ramp up length it is set to 150
        # what is w ? 
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):# you pass mix data loader in here
            # i am a bit lost here.
            # what is x and x bar ???? my guess they are both same pictures but under different augmentation
            # i am sure about this fact. so i take picture of car and augment it twice by doing RandomHorizontalFlip
            # x is the label set and x bar is the unlabeled set
            # 
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)# moving things to cuda
            output1, output2, feat = model(x)# getting output of 2 head each 128,5
            output1_bar, output2_bar, _ = model(x_bar)# getting output of 2 heads
            prob1, prob1_bar, prob2, prob2_bar=F.softmax(output1, dim=1),  F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
            # turning everything into probabilities
            mask_lb = label<args.num_labeled_classes# mask of true and false checking which labels should have labels and which shouldnot
            # feat is the layer before fully connected and it has shape of (128,512)
            rank_feat = (feat[~mask_lb]).detach()# ~ is to turn each true to false as if it is flipping
            # i am taking the all the unlabelled 
            # Returns a new Tensor, detached from the current graph.
            # The result will never require gradient.
            # rank_feat are the unlabeled features
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)# you have for example 68 unlabled point each of size of 512
            # Returns the indices that sort a tensor along a given dimension in ascending order by value.
            # for example rank_feat
                # [0.4731, 0.6749, 0.1237,  ..., 0.0071, 0.0018, 0.0450],
                # [0.4363, 0.6436, 0.1356,  ..., 0.0117, 0.0087, 0.0559],
            # when we sort we say 
                # [0, 0, 1,  ..., 1, 1, 1],
                # [1,1, 0, 0.1237,  ..., 0, 0, 0],
                # so in every picture we have feature vector flattened. we sort each value in all feature vectors and put ranking 
                # according who is bigger than you.
            # rank_idx is size of (68 by 512)
            rank_idx1, rank_idx2= PairEnum(rank_idx)# mask is set to none
            # rank_idx1 (4624,512) repeated variable. imagine that you have 68 pictures features and we repeats them 68 times
            # so all the whole variable is copied and put 64 times.
            # rank_idx2. each feature vector size 512 i repeat it 68 times so i have (68,34816)=(68,68*512). then
                # i reshape it to (-1,512) so that the final vector becomes  size of [4624, 512]
                # in other words each vector is repeated 64 times then next vector repeat 64 times etc 
                # [474, 307, 448,  ..., 370, 247, 245],
                # [474, 307, 448,  ..., 370, 247, 245], keep on repeated for 64 times
            rank_idx1, rank_idx2=rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]#you take top 5 so each has this size[4624,5]
            # what he is doing here ??? i am slicing specific amount topk. this is passed with arguments 
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            # Sorts the elements of the input tensor along a given dimension in ascending order by value.
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            # sorting the indicies
            rank_diff = rank_idx1 - rank_idx2# subtract both from each other 
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)# 1 dimension shape 4624
            # you summ alng top 5
            target_ulb = torch.ones_like(rank_diff).float().to(device) 
            # Return an array of ones with the same shape and type as a given array.
            # Returns a tensor filled with the scalar value 1

            target_ulb[rank_diff>0] = -1 # put negative 1 at locations that has 0 or negative values 
            #you are picking the unlabeled tensors using mask_lb
            # prob2 is of the second head softmax output for first augmentation
            prob1_ulb, _= PairEnum(prob2[~mask_lb]) # mask passed by none
            # prob 2 is 128,5 but since u are passing ~mask_lb so you pass unlabled data so you get
            # 68,5 prob1_ulb is the return of x1 which will be size of (4624.5)
            # prob2_bar is of the second head softmax output for first augmentation
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb]) # mask passed by none 
            # prob2_ulb shape 4624 by 5 

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])#cross entropy loss on labeled data
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb) # BCE in utils file
            # target_ulb is matrix with 1 and -1 at values with rank diff less than or = 0
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = loss_ce + loss_bce + w * consistency_loss # summing up the losses

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)# send to test set for evaluation
        print('test on unlabeled classes')
        args.head='head2'
        test(model, unlabeled_eval_loader, args)# send to test set for evaluation


def train_IL(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss() 
    criterion2 = BCE() 
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
            # we transfer them to probabilityies
            mask_lb = label < args.num_labeled_classes# select unlabled data features. 

            rank_feat = (feat[~mask_lb]).detach()

            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]# select topk between 2 samples
            
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)# we sort them. 
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)# it indicate they are exactly same so they set their label pairwise to 1.
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1# otherwise we set them to -1. they belong to different classes

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])# crosss entropy loss for supervised part 
            # all above very similar to before 
            # the next part is the different part. 
            # this is the different part for IL
            
            label[~mask_lb] = (output2[~mask_lb]).detach().max(1)[1] + args.num_labeled_classes# 

            loss_ce_add = w * criterion1(output1[~mask_lb], label[~mask_lb]) / args.rampup_coefficient * args.increment_coefficient
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)#binary cross entropy
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)# between label data or unlabled data 

            loss = loss_ce + loss_bce + loss_ce_add + w * consistency_loss

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        test(model, labeled_eval_loader, args)
        print('test on unlabeled classes')
        args.head='head2'
        test(model, unlabeled_eval_loader, args)

def test(model, test_loader, args):# normal test function you have seen similar to test in previous stages. i donot comment it
    model.eval()
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):# we donot care about index 
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)# these are things that you can input through the terminal
    # you use the parse so that when you run the file oyu can pass some inputs these are the inputs. 
    parser.add_argument('--lr', type=float, default=0.1)# the learning rate
    parser.add_argument('--gamma', type=float, default=0.1)# the gamma of the schedular
    parser.add_argument('--momentum', type=float, default=0.9)# the momenetum of the optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-4)# the weight decay cofficient fo l2 
    parser.add_argument('--epochs', default=200, type=int)# the number of epochs that we run during traininnig
    parser.add_argument('--rampup_length', default=150, type=int)# the length of rampup that is passed to ramps.py functioncts 
    parser.add_argument('--rampup_coefficient', type=float, default=50)# rampup coefficieints
    parser.add_argument('--increment_coefficient', type=float, default=0.05)# what is this ?
    parser.add_argument('--step_size', default=170, type=int)# what is the step size it is used with learnign rate schedule
    parser.add_argument('--batch_size', default=128, type=int)# the batch size used 
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)# the number of unlabled classes
    parser.add_argument('--num_labeled_classes', default=5, type=int)# the number of labled classes
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')# the root directoryyy
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')# where i save my experiements
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth')#the supervised model saved from step 2 
    parser.add_argument('--topk', default=5, type=int)# interesting what is topk? 
    parser.add_argument('--IL', action='store_true', default=False, help='w/ incremental learning')# turning on the incremental leanring feature
    parser.add_argument('--model_name', type=str, default='resnet')# the name of model
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')# the dataset what we will work on
    parser.add_argument('--seed', default=1, type=int)# specific seed that we set in the beinging
    parser.add_argument('--mode', type=str, default='train')# what mode are we working on.??? the training mode
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()# checkign if there is cuda
    device = torch.device("cuda" if args.cuda else "cpu")# setting device to cuda incase it is here else we set it to cpu
    seed_torch(args.seed)# this is a funciton in the utils folder that set alot different modules to a specific seed.
    runner_name = os.path.basename(__file__).split(".")[0]# create a folder with with a name of autonovel under experiments
    model_dir= os.path.join(args.exp_root, runner_name)# join the path
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)# create the folder
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name) # the name that we will save the model with 

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    # we have 2 heads , label head and unlabled head

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes # total number of classes.

    if args.mode=='train':# if we are in training mood.
        state_dict = torch.load(args.warmup_model_dir)# load the training weights for supervised learning part 
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters(): 
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False# anything before layer 4 is fixed 
    # we fix paramters before layer 4
    # this will be very tricky many loaders so we better understand what is each loader doing. 
    if args.dataset_name == 'cifar10':
        mix_train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        # you ahve both labeled and unlabled dataset 
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        # labeled dataset only
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        # unlabled evaluation set
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        # unlabled test set
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        # labled test set
        all_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
        # contains both lableld and unlabled
    elif args.dataset_name == 'cifar100':
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'svhn':
        mix_train_loader = SVHNLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, target_list = range(args.num_labeled_classes))
        unlabeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
     # entering training mode next 
    if args.mode == 'train':
        if args.IL:# this is code set to say hey do you want work with incremental learning or not 
            save_weight = model.head1.weight.data.clone()# cloning the weights of the head 1 from supervised learning size 5 by 512 
            save_bias = model.head1.bias.data.clone()# cloning the bias of the head 1 from supervised learning 
            model.head1 = nn.Linear(512, num_classes).to(device)# replacing head from just 5 to 10 so now the size is 10,512 and all values are zero
            model.head1.weight.data[:args.num_labeled_classes] = save_weight# intialize half matrix with the weights available previously
            # from the supervised learning
            model.head1.bias.data[:] = torch.min(save_bias) - 1. # the bias for the 10 classes is set to certain value
            # the min of previous bias and we subtract from it -1
            train_IL(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)
        else:
            train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)# traiining without incrmenetal learning
        torch.save(model.state_dict(), args.model_dir)# saving the model. 
        print("model saved to {}.".format(args.model_dir))
    else:# entering evaluation mode 
        print("model loaded from {}.".format(args.model_dir))
        if args.IL:
            model.head1 = nn.Linear(512, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_dir))

    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader, args)
    if args.IL:
        print('test on unlabeled classes (test split)')
        test(model, unlabeled_eval_loader_test, args)
        print('test on all classes (test split)')
        test(model, all_eval_loader, args)
    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    test(model, unlabeled_eval_loader, args)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args)