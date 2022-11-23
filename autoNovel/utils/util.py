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
import random
import os
import argparse
# The following module has been removed from sklearn, therefore it will be imported from a local file
# Check second comment at: https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment
# from sklearn.utils.linear_assignment_ import linear_assignment
from utils.linear_assignment_ import linear_assignment


#######################################################
# Evaluate Criterion
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
    # Check and ensure that predict and true arrays have the same dimensions
    assert y_pred.size == y_true.size
    # Extract maximum value from ypred and from ytrue, then pick the max between these two, finally add 1.
    D = max(y_pred.max(), y_true.max()) + 1
    # Create a matrix with size DxD and filled with zeros, imagine it as if it is D arrays each array contain D elements
    w = np.zeros((D, D), dtype=np.int64)

    # For each prediction performed in the test-set
    for i in range(y_pred.size):
        # Fill the matrix cell on the intersection between the prediction (row) and the true label (column) for that
        # specific iteration with a '1'
        w[y_pred[i], y_true[i]] += 1
        # this matrix is what? THE AMAZING CONFUSION MATRIX. It took me sometimes to figure it out.
        # think of it on x-axis you have y predict class
        # on y-axis you have the y true classes.
        # so every item I predicted class 1, and it is class 1, I add 1

    # Apply Hungarian algorith (a combinatorial optimization algorithm used to solve assignment problems). In this case
    # the algorith is applied since we are working on two vectors: y_true and y_pred.
    # The vector y_true is a list of numbers where each number specify the true label of a given example.
    # The vector y_pred is a list of numbers where each number specify the class that has been assigned to a given
    # example by our NN. As seen in the paper, the heads have been created with a number of output dimension equal to
    # the number of classes (supervised classes for head1, unsupervised classes for head2). Then a torch.max function
    # has been applied in the testing phase to extract the class with the highest probability for each sample.
    # The point is that the classes predicted by the NN (y_pred) will probably be linked with numeric label that is
    # different from the numeric label used in the dataset (y_true).
    # For this reason we apply the Hungarian algorithm, to optimize the combinatorial problem and associate each class
    # with its correct numeric label)
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    # I don't quite get why he is doing in here??? why subtract the max?
    # he takes a matrix that is called w and apply on it the max operation.  so you get the maximum number in all the
    # matrices. so for example if your matrix contain biggest number equal to 10, he subtracts 10 out of everything else
    # and input it to linear assigment
    # it returns The pairs of (row, col) indices in the original array giving the original ordering.

    # he brings out the solution with the least cost, and we add all the cost. What does this mean???
    # Jacopo the genius figured it out. ask him. I give him some credit.
    # the hungarian algorithm is about find the least cost to a problem
    # imagine you have 3 people sick in different places in Trento and there are 3 ambulance in Trento
    # which ambulance go to which person? there is a different expense for every ambulance if it goes to specific person
    # so ambulance A if go to person 1, cost is 100. if you go to person 2, cost is 600. if you go to person 3, cost is
    # 30. similar situation for the other ambulance.

    # hungarian algorithm will give you solution to which ambulance go to which person to decrease the cost while
    # being computationally not expensive. In this case you have predicted labels and true label and you are trying to
    # say how much do we assign this cluster this specific label(y_true).


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        # Initialize the object calling the method resnet()
        self.reset()

    def reset(self):
        # Set all values to zero
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Update the values when a new value (val) and the number of batches (n) is given
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BCE(nn.Module):
    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),
                                                                                            str(len(prob2)),
                                                                                            str(len(simi)))
        # make sure that everything has the same size
        # simi is a 1d vector
        P = prob1.mul_(prob2)  # multiply(4624,5) by another matrix same size
        # i have some probabilities from first picture
        # i have some probabilities from second picture
        # we can say i am multiply probability matrix for all picture in first augmentation
        # with first probability tensor for second augmentation
        # tensor for second augmentation you do that for all X matrix you have.
        P = P.sum(1)  # sum along axis 1 so you have size fo 4624
        # you have a vector of size 4624
        # you multiply simi by P then you add to it
        # simi has shape as P 
        # simi.eq(-1).type_as(P) return a tensor with 1 and zero at location where tensor has value of -2
        # you multiply by simi then you add to it this tensor
        # simi.eq(-1) returns the places that are equal to -1 in form for boolean
        # as type turn it into 1 and 0
        # you add 1 to places that are not similar (have the value -1
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()  # has size of 4624
        return neglogP.mean()  # you calculate the mean


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in the tensor x. In this example x is of size (68,512), meaning that for 68
    # examples we have 512 features, which are sorted and represented by their ranking index due to 'torch.argsort()'

    # Ensure that the dimension of the input tensor is 2
    assert x.ndimension() == 2, 'Input dimension must be 2'

    # Each value specifies how many times the tensor will be repeated along that dimension. In this case we have a
    # tensor of dimension 2, so we need to specify the wanted repetitions along these two dimensions. Here we have
    # repeated the first dimension by itself (e.g., if it is 68 it will be repeated 68 times), while the second
    # dimension is repeated only once (meaning that it is not repeated at all)
    # x1 = x.repeat(x.size(0), 1)  
    x1 = x.repeat(x.size(0),1)# Example: output tensor (68*68,512) = (4624,512)

    # Apply the same function to repeat the second dimension (512) a number of times equal to the first dimension (68
    # in this example), the result is a tensor (68,34816).
    # Then resize the tensor in a way that it has a number of columns equal to the 1-st dimension of the previous
    # tensor, and the number of rows necessary to store all the values. The result is a tensor (68,34816).
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))

    # If we input a specific mask
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Use no_grad to avoid updating gradients in this function
    with torch.no_grad():
        # Extract the maximum value k from the tuple topk
        maxk = max(topk)  # If we have default input topk=(1,) then variable maxk will take value 1
        # Extract the batch size from the target first dimension (256 in this case, since target is a (256,4) tensor)
        batch_size = target.size(0)

        # Use torch.topk to return the k-largest elements (and its index if required) over the specified dimension of
        # the given tensor. 'Largest' used to choose largest or the smallest, 'Sorted' used to sort them once returned
        # In this case we analyze the first dimension of the output tensor (256,4), we return the index tensor (256,1),
        # This means that for each of the 256 samples we choose as prediction the rotation with the highest probability
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)

        # Transpose the pred tensor from (256,1) to (1,256)
        ### Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        pred = pred.t()

        # Target tensor has shape of 256, so we use target.view to turn it to (1,256) shape to match the pred tensor.
        # Then we compute the tensor equality (composed of true/false). Indicating where predictions where correct
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (1, -1) means use 1 row, distribute over columns
        ### Expand this tensor to the same size as other. self.expand_as(other) is equivalent to self.expand(other.size()).
        ### i donot get why in here he uses the expand operations but for semi supervised learning, it doesnot patter this operation

        # Initialize an empty list
        res = []
        # Iterate along k, as seen k is equal to 1
        for k in topk:
            # Take a tensor with k values, reshape it to (256), use float to convert it to numbers, them sum them up,
            # the result value is equal to the total number of correct predictions
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # Now divide the correct number of predictions by the batch_size and multiply by 100 to get the % accuracy
            res.append(correct_k.mul_(100.0 / batch_size))
        # Having k=1 is like averaging and computing the accuracy.
        return res


# Setting seed to something very specific
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
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
