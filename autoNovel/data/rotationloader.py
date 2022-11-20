from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv
from tqdm import tqdm

from pdb import set_trace as breakpoint

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None, dataset_root=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        # not my comment
        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the 
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name=='cifar10':
            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]] # [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]] # [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
            # I ran the codes above and I got these values.
            # they are different from values in cifar_datalaoders values in the other file. 
            # I googled and this are the true mean and std of cifar 10 dataset
            # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []# list of transformer that will be impleented 
            if (split != 'test'):# you are entering if you are in the training conditions
                transform.append(transforms.RandomCrop(32, padding=4)) # Crop the given image at a random location.
                # If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively
                transform.append(transforms.RandomHorizontalFlip()) # Horizontally flip the given image randomly with a given probability
                # default is 0.5. 
            transform.append(lambda x: np.asarray(x))# add transformaiton of convert input into array
            self.transform = transforms.Compose(transform) # Composes several transforms together. 
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, train=self.split=='train',
                download=True, transform=self.transform)# what is this???
            # object.__dict__ is A dictionary or other mapping object used to store an object’s (writable) attributes.
            # datasets.__dict__['CIFAR10'] ---> <class 'torchvision.datasets.cifar.CIFAR10'>
            # you are telling him andiammmoooo download me dataset put in this path and do the following transformaiton.
        elif self.dataset_name=='cifar100':
            self.mean_pix = [x/255.0 for x in [129.3, 124.1, 112.4]]
            self.std_pix = [x/255.0 for x in [68.2, 65.4, 70.4]]
            # very similar thing where he is using the normalizing stuff.
            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, train=self.split=='train',
                download=True, transform=self.transform)
        elif self.dataset_name=='svhn':
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the SVHN dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))# why no horitonzal flip with svhn???i have no explanation for this
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, split=self.split,
                download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dataset_name))
# applying the 2 main methods that should be implemented for any dataset class by ptorch
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):# they idea is here is he take tensor multiply by std and add to it the mean
    # lets wait and see where he is using this in the code
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def rotate_img(img, rot):# this function is used to rotat the image.
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
        # flipud Reverse the order of elements along axis 0 (up/down).
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

# class of data loader
class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset# the dataset
        self.shuffle = shuffle# flag for shufling
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset) # what is the epoch size?number of iterations?
        # i donot know what is this but he doesnot use it with cifar 10,100 svhn case. it is passed as none
        self.batch_size = batch_size
        self.unsupervised = unsupervised# flag to indicate unsupervised
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix# they are set accordingly to their value from dataset object
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])#turning into tensors and transfomring
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])# turn back to normal numpy array
        # relatively he doesnot use inv_transform anywhere else .

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)# the size of the data set
                # we are making index from 0-4999 
                img0, _ = self.dataset[idx]# i donot care soo much about the label in here. I am accessing the image
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90).copy()),
                    self.transform(rotate_img(img0, 180).copy()),
                    self.transform(rotate_img(img0, 270).copy())
                ]# it is al ist containing all the rotaiton image applied to it the transform
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels# turn list into a stack
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)# make sure batch size is equal to 2 ? whyyy??because i guess load function returns a stack of roated dimensiaon and roation labels
                #  returns a stack and rotation labels
                batch_size, rotations, channels, height, width = batch[0].size()# you accessesing the stack
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])# changing the shape
                batch[1] = batch[1].view([batch_size*rotations])# changing the shape
                # Returns a new tensor with the same data as the self tensor but of a different shape.
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            # collate_fn is called with a list of data samples at each time. 
            # It is expected to collate the input samples into a batch for yielding from the data loader iterator.
            _collate_fun = default_collate
            # Function that takes in a batch of data and puts the elements within the batch into a tensor with
            # an additional outer dimension - batch size. The exact output type can be a torch.Tensor,
            # a Sequence of torch.Tensor, a Collection of torch.Tensor, or left unchanged, depending on the input type. 

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)#tnt is a library imported above
        ##Dataset which loads data from a list using given function.
        # load=function which loads the data.i-th sample is returned by `load(elem_list[i])`. By default `load`
        #    is identity i.e, `lambda x: x`
        # 
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)# you returning directly the data loader
            # torch.utils.data.DataLoader(batch_size=1, shuffle=None,num_workers=0, collate_fn=None)
            # collate – merges a list of samples to form a mini-batch of Tensor(s). 
              # Used when using batched loading from a map-style dataset.
            # num_workers  how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. 
        # summary this part is not so important. just all you need to know is that he is returning the datalaoder e basta.
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
    
# this should be removed for later I was just testing to know what is happening inside to make sure i understood everything 
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset_train = GenericDataset(
        dataset_name='cifar10',
        split='train',
        dataset_root='./data/datasets/CIFAR/'
       )
    print(len(dataset_train)) # the length for cifar is 50k
    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=64,
        num_workers=4,
        shuffle=True)
    iterator=iter(dloader_train(0))
    inputs, classes = next(iterator)
    print(inputs[1])
    print(inputs[1].shape) #3*32*32
    print(type(inputs[1])) #torch tensor
    print(len(inputs)) #256
    print(len(classes)) #256
    print((classes)) #256
    # plt.imshow( inputs[0].permute(1, 2, 0)  )
    # plt.show()