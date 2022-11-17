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
            # they are different from values in cifar_datalaoders. I need to understand where did the values of datalaoder come from
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
            # you are telling him andiammmoooo download me dataset putin this path and do the following transformaiton.
            # we learn that are in the form of numpy arrays
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
                transform.append(transforms.RandomCrop(32, padding=4))# why no horitonzal flip with svhn???
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, split=self.split,
                download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dataset_name))

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)# applying the 2 main methods that should be implemented for any dataset

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

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset) # what is the epoch size?number of iterations? maybe?
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix# they are set accordingly
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])#turning into tensors and transfomring
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])# turn back to normal numpy array

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90).copy()),
                    self.transform(rotate_img(img0, 180).copy()),
                    self.transform(rotate_img(img0, 270).copy())
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size