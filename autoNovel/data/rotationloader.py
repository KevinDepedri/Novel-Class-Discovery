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
import sys
import csv
from tqdm import tqdm
from pdb import set_trace as breakpoint


class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None, dataset_root=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        # The num_imgs_per_cats input argument specifies the number of training examples per category that would be
        # used. This input argument was introduced in order to be able to use less annotated examples than what are
        # available in a semi-supervised experiment. By default, all the available training examples per category are
        # being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        # If the dataset is cifar10 then apply the following normalization
        if self.dataset_name == 'cifar10':
            self.mean_pix = [x / 255.0 for x in
                             [125.3, 123.0, 113.9]]  # [0.4913725490196078, 0.4823529411764706, 0.4466666666666667]
            self.std_pix = [x / 255.0 for x in
                            [63.0, 62.1, 66.7]]  # [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
            # TODO: these above are the correct values for cifar, verify why in the CifarLoader file they are different
            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            # Build a transform as a list
            transform = []
            # If we are in the training split
            if split != 'test':
                # Perform a random crop of 32x32 with a padding of 4, if sequence provided it is [left,top,right,bottom]
                transform.append(transforms.RandomCrop(32, padding=4))
                # Horizontally flip the given image randomly with a given probability (default is 0.5)
                transform.append(transforms.RandomHorizontalFlip())
            # Convert the input into a numpy array
            transform.append(lambda x: np.asarray(x))

            # Compose the previously defined transform
            self.transform = transforms.Compose(transform)

            # call dataset.__dict__['CIFAR10'], put CIFAR10 dataset in the dataset_root directory (if it is not in the
            # directory then download it since download=True), create a train split and apply the given transform.
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, train=self.split == 'train', download=True, transform=self.transform)

        # If the dataset is cifar100 then apply the following procedure (same as above)
        elif self.dataset_name == 'cifar100':
            self.mean_pix = [x / 255.0 for x in [129.3, 124.1, 112.4]]
            self.std_pix = [x / 255.0 for x in [68.2, 65.4, 70.4]]
            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if split != 'test':
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, train=self.split == 'train', download=True, transform=self.transform)

        # If the dataset is svhn then apply the following procedure (similar as above)
        elif self.dataset_name == 'svhn':
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the SVHN dataset')

            transform = []
            if split != 'test':
                transform.append(transforms.RandomCrop(32, padding=4))
                # We have no horizontal flip for svhn
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                dataset_root, split=self.split,
                download=True, transform=self.transform)

        # If the dataset is none of the ones above, then rise an error
        else:
            raise ValueError('Not recognized dataset {0}'.format(dataset_name))

    # Define the 2 main methods needed for any dataset class in pytorch
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)


# Invert the normalization procedure, used in the inverse transform in the DataLoader class
class Denormalize(object):
    # Take the input tensor multiply by its dataset std and add to it the mean of its dataset
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# Function used to rotate the input image according to the input rotation.
def rotate_img(img, rot):
    # If rot = 0 then apply a 0 degrees rotation and return the rotated image
    if rot == 0:
        return img

    # If rot = 1 then apply a 90 degrees rotation and return the rotated image
    elif rot == 90:
        # Use numpy to reverse the order of elements along axis 0 (up/down)
        return np.flipud(np.transpose(img, (1, 0, 2)))

    # If rot = 2 then apply a 180 degrees rotation and return the rotated image
    elif rot == 180:
        # Use numpy to reverse the order of elements along axis 1 (left/right)
        return np.fliplr(np.flipud(img))

    # If rot = 3 then apply a 270 degrees rotation and return the rotated image
    elif rot == 270:
        # Use numpy to reverse the order of elements along axis 0 (up/down)
        return np.transpose(np.flipud(img), (1, 0, 2))

    # If rot is none of he above values, then rise error
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
        self.dataset = dataset  # Dataset name
        self.shuffle = shuffle  # Flag to enable shuffling
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)  # Used to generate the seed
        self.batch_size = batch_size  # Batch size
        self.unsupervised = unsupervised  # Flag to indicate unsupervised
        self.num_workers = num_workers  # Num of workers

        # Perform normalization accordingly to the statistical values from the dataset object
        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        # Build the standard transform (turn input image to tensor and apply the normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        # Build the inverse transform to go back to a normal numpy array (never used in the code)
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        # Compute and apply a random seed
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # If the dataloader is working in unsupervised mode
        if self.unsupervised:
            # Define a loader function that given the index of an image returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., either one between [0,1,2,3] for [0,90,180,270] degrees rotation.
            def _load_function(idx):
                # Update the index as the module obtained dividing the idx by the size of the dataset
                idx = idx % len(self.dataset)
                # Get the image with the computed index from the downloaded dataset, do not consider the label
                img0, _ = self.dataset[idx]
                # Define the list of transform necessary to get the 4 rotation of the chosen image
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0, 90).copy()),
                    self.transform(rotate_img(img0, 180).copy()),
                    self.transform(rotate_img(img0, 270).copy())
                ]
                # Build a rotation table of LongTensors and turn the transform list into a stack, then return both
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels

            def _collate_fun(batch):
                # Collate the input samples into a batch for yielding from the data loader iterator
                batch = default_collate(batch)
                # Check that the length of the batch is equal to 2, otherwise rise error
                # TODO: Needed since Load function returns a stack of rotated dimension and rotation labels?
                assert (len(batch) == 2)
                #  returns a stack and rotation labels
                batch_size, rotations, channels, height, width = batch[0].size()  # you accessioning the stack
                batch[0] = batch[0].view([batch_size * rotations, channels, height, width])  # changing the shape
                batch[1] = batch[1].view([batch_size * rotations])  # changing the shape
                # Returns a new tensor with the same data as the self tensor but of a different shape.
                return batch

        # Otherwise, if the dataloader is working in supervised mode
        else:
            # Define a loader function that given the index of an image it returns the image and its categorical label
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

        # Create a torchnet dataset using the previously define load function
        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=_load_function)
        # TorchNet is a library imported used to implement a dataset which loads data from a list using given function.
        # The function 'load' is used to load the i-th data sample returned by `load(elem_list[i])`

        # Build a torchnet dataloader that is based on the previously define torchnet dataset
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        # Here the collate_fun merges a list of samples to form a mini-batch of Tensor(s). Useful when using batched
        # loading from a map-style dataset. If num_workers is 0 means that the data will be loaded in the main process.
        return data_loader

    # If the class is called, then use the method get_iterator define above
    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
