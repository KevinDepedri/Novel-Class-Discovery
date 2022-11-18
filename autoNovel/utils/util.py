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
    assert y_pred.size == y_true.size # check that predict and true has teh same dimensions
    D = max(y_pred.max(), y_true.max()) + 1 #extrats value 5 Extract the size of the matrix. 
    # extract maximum from ypred and extract maximum from ytrue and pick the max of both then add 1
    w = np.zeros((D, D), dtype=np.int64)# create a matrix D by d filled with zeros
    # imagine it as if it is D arrays each array contain D elements
    for i in range(y_pred.size):# for the full size of the test set.
        w[y_pred[i], y_true[i]] += 1# he is just filling up the matrix
        # this matrix is what? THE AMAZING CONFUSION MATRIXXX. It took me sometimes to figure it out.
    # 
    ind = linear_assignment(w.max() - w)# i donot quite get why he is doing in here??? why subtract the maxx? 
    # he takes a matrix that is called w and apply on it the max operation.  so you get the maximum number in all of the matrices.
    # so for example if your matrix contain biggeest number equal to 10, he subtract 10 out of everything else and input it to linear
    # assigment
    # it returns The pairs of (row, col) indices in the original array giving the original ordering. 
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size # he bring out the solution with least cost and we add all the cost
    # what does this mean???
    # why is hungarian algorithim used on the confusion matrix? is it a confusion matrix??? maybe i should ask zhong.

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
# why doesnot he just use the module of binary cross entropy? My guess it is different in here than normal case
# slowly we need to check it.
class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        # make sure that everything has the same size
        # are these tensors or what exactly???
        P = prob1.mul_(prob2)
        P = P.sum(1)# sum along axis
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()
# slowly we need to check it later
def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)# if topk(1,) so maxk is set to 1
        batch_size = target.size(0)# set to 256
        # output is a tensor (256,4)
        _, pred = output.topk(maxk, 1, True, True)#shape of pred is (1,256), pred is the index we are returning the index
        # Returns the k largest elements of the given input tensor along a given dimension.
        # torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
        # k the k in “top-k”
        # dim the dimension to sort along
        pred = pred.t()# shape is (1,256) # why didnot it change shape after transpose? double check ?
        # we do the transpose operation.
        # Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # target has shape of 256 so we make target.view to turn it to (1,256)
        correct = pred.eq(target.view(1, -1).expand_as(pred))# [1,256]
        # Expand this tensor to the same size as other. self.expand_as(other) is equivalent to self.expand(other.size()).
        # i donot get why in here he uses the expand operations but for semi supervised learning , it doesnot patter
        # pred.eq Computes element-wise equality
        # it has output shape of 1,256



        res = []
        for k in topk:# k is equal to 1
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)# you reshape to (256) and you summ them all
            res.append(correct_k.mul_(100.0 / batch_size))# divided by batch size while mutlipl by 100
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