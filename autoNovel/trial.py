# useless file just for deleting it
from data.cifarloader import CIFAR10Loader, CIFAR100Loader,CIFAR10LoaderMix
from tqdm import tqdm

if __name__ == '__main__':
        # labeled_train_loader = CIFAR10Loader(root='./data/datasets/CIFAR/', batch_size=128, split='train', aug='once', shuffle=True, target_list = range(5))
        # labeled_eval_loader = CIFAR10Loader(root='./data/datasets/CIFAR/', batch_size=128, split='test', aug=None, shuffle=False, target_list = range(5))
        mix_train_loader = CIFAR10LoaderMix(root='./data/datasets/CIFAR/', batch_size=128, split='train', aug='twice', shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10))
        print("The size of train loader is -->"+str(len(mix_train_loader)))# we have 391 batches
        iterator=iter(mix_train_loader)
        inputs = next(iterator)
        # print(type(inputs))# the type is list
        # print(len(inputs))# the len is 3
        #
        # print((inputs[0]))#the picture
        # so in here we have 2 pictures
        # but why 2 pictures ????
        print('the shape of inputs[0][0] is '+str(len(inputs[0])))# this is a list of size 2
        print('the shape of inputs[0][0] is '+str(inputs[0][0].shape))# tensor with size of 128,3,32,32
        print(type(len(inputs[0])))#int
        import matplotlib.pyplot as plt
        import numpy as np 
        plt.imshow(np.transpose(inputs[0][0][0].numpy(), (1, 2, 0)))
        plt.show()
        plt.imshow(np.transpose(inputs[0][1][0].numpy(), (1, 2, 0)))
        plt.show()

        #
        # print((inputs[1]))#the labels
        # print(len(inputs[1]))#len is 128
        # print(type(len(inputs[1])))#int
        # #
        # print((inputs[2]))#the indices
        # print(len(inputs[2]))#len is 128
        # print(type(len(inputs[2])))#int
    #     import os
    #     import sys
    #     if sys.version_info[0] == 2:
    #             import cPickle as pickle
    #     else:
    #             import pickle
    #     train_list = [
    #     ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    #     ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    #     ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    #     ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    #     ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    # ]# different files for training
    #     test_list = [
    #     ['test_batch', '40351d587109b95175f43aff81a1287e'],
    # ]#this file is the test file 
    #     meta = {
    #     'filename': 'batches.meta',
    #     'key': 'label_names',
    #     'md5': '5ff9c542aee3614f3951f8cda6e48888',
    # }# the meta file contains the labels in general capittooooo ragaaa??
    #     downloaded_list = []# empty list
    #     downloaded_list.extend(train_list)# put in a list
    #     downloaded_list.extend(test_list)# extend is just like append but it places items in begining of list
    #     root='./data/datasets/CIFAR/'
    #     base_folder='cifar-10-batches-py'
    #     data = []# two empty lists
    #     targets = []# two empty lists
    #     for file_name, checksum in downloaded_list:
    #         file_path = os.path.join(root,base_folder, file_name)# join ./data/datasets/CIFAR/ with cifar-10-batches-py with file name
    #         # you get sthg like ./data/datasets/CIFAR/cifar-10-batches-py/data_batch_1
    #         with open(file_path, 'rb') as f:# we are opening the file 
    #             if sys.version_info[0] == 2:
    #                 entry = pickle.load(f)
    #             else:
    #                 entry = pickle.load(f, encoding='latin1')# loading a pickle files
    #                 # it has under it data,labels, file names
    #             data.append(entry['data'])# open the data list and add to it the list of data.
    #             # entry[data] is numpy array of size of (10000,3072) this mean in this batch we have 10k pictures
    #             if 'labels' in entry:# in case wefine lables in entry.
    #                 # why wouldnot we find labels ???? 
    #                 # what is the fine_labels ???
    #                 targets.extend(entry['labels'])# add to the targets the labels which is a list of size 10k
    #             else:
    #                 # i donto understand what is happening in here but the code was always here
    #                 #  self.targets.extend(entry['coarse_labels'])
    #                 targets.extend(entry['fine_labels'])