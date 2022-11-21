from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
# from sklearn.utils.linear_assignment_ import linear_assignment
from utils.linear_assignment_ import linear_assignment
# check https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment the second comment
# 
import random
import os
import argparse
#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    # ytrue is an array of size 5000 containing all the test data
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size # check that predict and true has the same dimensions
    D = max(y_pred.max(), y_true.max()) + 1 #extrats value 5 Extract the size of the matrix. 
    # extract maximum from ypred and extract maximum from ytrue and pick the max of both then add 1
    w = np.zeros((D, D), dtype=np.int64)# create a matrix D by d filled with zeros
    # imagine it as if it is D arrays each array contain D elements
    for i in range(y_pred.size):# for the full size of the test set.
        w[y_pred[i], y_true[i]] += 1# he is just filling up the matrix
        # this matrix is what? THE AMAZING CONFUSION MATRIXXX. It took me sometimes to figure it out.
        # think of it  on x axis you have y predict class
        # on y axsis you have the y true classes.
        # so every item i predicted class 1 and it is class 1 i add 1 
    # 
    ind = linear_assignment(w.max() - w)# i donot quite get why he is doing in here??? why subtract the maxx? 
    # he takes a matrix that is called w and apply on it the max operation.  so you get the maximum number in all of the matrices.
    # so for example if your matrix contain biggeest number equal to 10, he subtract 10 out of everything else and input it to linear
    # linear assigment
    # it returns The pairs of (row, col) indices in the original array giving the original ordering. 
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size # he bring out the solution with least cost and we add all the cost
    # what does this mean???
    # jacopo the genius figured it out. ask him. i give him some credit. 
    # the hungarian algorithim is about find least cost to a problem
    # imagine you have 3 people sick in different places in trento  and there are 3 ambulences in trento
    # which ambuluence go to which person? there is a different expense for every ambulence if it go to specific person
    # so ambulence A if go to person 1, cost is 100.if go to person 2, cost is 600.if go to person 3, cost is 30.
    # similar situation for the other ambulences. 
    # hungarian algorithim will give you solution to which ambulence go to which person to decrease the cost while being computationally not expensive.
    # in this case you have predicted labels and true label and you are trying to say how much do we assign this cluster this specific label(y_true).
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):# setting all values to zero
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n# n is number of the batch
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
# I donot get it how he is calcuating stuff in here 
class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        # make sure that everything has the same size
        # simi is a 1d vector
        # are these tensors or what exactly??? yes they are all tensors 
        P = prob1.mul_(prob2)# output is (4624,5) so 
        # i have some probabilities from first picture
        # i have some probabilties from second picture
        # we can say i am multiply probability matrix for all picture in first augmentation with first probebiltiy tensor for second augmentation
        # you do that for all X matrix you have.
        P = P.sum(1)# sum all probabilities in each vector (4624)
        # you have a vector of size 4624
        # you mutlmiply simi by P then you add to it 
        # simi.eq(-1).type_as(P) reutnra tensor with 1 and zero at location where tnesor has value of -2
        # you mulptily by simi then you add to it this tensor
        # simi.eq(-1) returns the places that are equal to -1 in form for boolean 
        # as type turn it into 1 and 0
        # you add 1 to places that are not similar (have the value -1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()# has size of 4624
        return neglogP.mean()# you calculate the mean 
def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    # x is a tensor of size 68,512 
    #68 indicate number of pictures
    # in the end what you are passing is the ranking of the features 
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    # Repeats this tensor along the specified dimensions. this function copies the tensor’s data.
    # you passs inside the number of times to repeat this tensor along each dimension
    # the 1 indicate the dimension that you repeat the data on 
    # you have 68 instance each instance so you will repeat the data 68*68
    # end output will be size of (4624,512)=(68*68,512)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    # i repeating again this time i want to repeat 1 time along axis of
    # so what i do is that I take 1 tensor size 512 and i repeat it 64 times 
    # so size is (68,34816)
    if mask is not None:# this part will be skipped  as mask is none
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():# donot use gradients in this area
        maxk = max(topk)# if topk(1,) so maxk is set to 1
        batch_size = target.size(0)# set to 256 which indicate my batch size 
        # output is a tensor (256,4)
        _, pred = output.topk(maxk, 1, True, True)#shape of pred is (1,256), pred is the index we are returning the index
        # topk Returns the k largest elements of the given input tensor along a given dimension.
        # torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
        # k the k in “top-k” which is topk
        # dim the dimension to sort along which is 1
        # this means that pred has size of (256,1) each element has value of what does network 
        # predicts roatation here. so for first sample it predict 90 second sample it predice 180
        # etc etc
        
        pred = pred.t()# shape is changed to (1,256)
        # we do the transpose operation.
        # Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # target has shape of 256 so we make target.view to turn it to (1,256)
        correct = pred.eq(target.view(1, -1).expand_as(pred))# [1,256]
        # Expand this tensor to the same size as other. self.expand_as(other) is equivalent to self.expand(other.size()).
        # i donot get why in here he uses the expand operations but for semi supervised learning , it doesnot patter this operation
        # pred.eq Computes element-wise equality
        # it has output shape of (1,256) it is filled with true and falses. indicating where  predict and target where correct 
        res = []
        for k in topk:# k is equal to 1
            # you turn true and false matrix into numbers and you sum it all
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)# you reshape to (256) and you summ them all
            res.append(correct_k.mul_(100.0 / batch_size))# divided by batch size while mutlipl by 100
        # it is like you are averaging and calculating the accuracy.
        return res
# setting seed to something very specfic 
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# for user input i guess
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')