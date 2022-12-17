import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision import datasets, models
import torchvision.transforms as T

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import random
import time

from PIL import Image

import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from data.utils import download_url, check_integrity

# from data.cifarloader_unbalanced import CIFAR10
# The CIFAR10 class has been copied here to avoid using the above line which create a circular dependency error
class CIFAR10(Dataset):# this is class dataset
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    # all the following stuff are training variables
    base_folder = 'cifar-10-batches-py'# name of the folder that you should have
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"# you get from this website
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    # you can find all the following files in following directory after running the first script download_pretrained_models_dataset.sh
    # --> autoNovel/data/datasets/CIFAR/cifar-10-batches-py
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]# different files for training

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]#this file is the test file
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }# the meta file contains the labels in general capittooooo ragaaa??

    def __init__(self, root, split='train+test',
                 transform=None, target_transform=None,
                 download=False, target_list=range(5)):
        # root is the directory where the data set is located. which is in this case ./data/datasets/CIFAR/ capitto ?
        #you pass the transformations that you want to do, download is to downlaod ddata set
        # target list is the range of the labels that you have. for example for labeled you are passing labels from0 to 4 and
        self.root = os.path.expanduser(root)
        self.transform = transform# group of transformations
        self.target_transform = target_transform# it is a range
        if download:# if you turn on the download option, it start downloading everything
            self.download()

        if not self._check_integrity(): # funcitons used to check if the data is available or not
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        downloaded_list = []# empty list
        # split is a paramter passed in the begining of the intializing of the function
        if split=='train':
            downloaded_list = self.train_list# global bariables containing the files names of training
        elif split=='test':
            downloaded_list = self.test_list # global bariables containing the files names of test
        elif split=='train+test':
            downloaded_list.extend(self.train_list)# put in a list
            downloaded_list.extend(self.test_list)# extend is just like append but it places items in begining of list
            # in here we can say that downlaoded list has the test items first then training items

        self.data = []# two empty lists
        self.targets = []# two empty lists

        # now load the picked numpy arrays
        # each item in downlaoded list contain the file name and code. I donot understand this code
        # but he is using it
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)# join ./data/datasets/CIFAR/ with cifar-10-batches-py with file name
            # you get sthg like ./data/datasets/CIFAR/cifar-10-batches-py/data_batch_1
            with open(file_path, 'rb') as f:# we are opening the file
                if sys.version_info[0] == 2:# this part is due to fact that according to version of sys
                    # we import different libraries of pickle.
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')# loading a pickle files
                self.data.append(entry['data'])# open the data list and add to it this entry
                # this is a numpy array of size of 10000,3072 indicating that we have 10k pictures with size of each array
                # 3072
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    # I donot understand which kind of files have fine labels instead of labels
                    # atleast by debugging cifar 10 they all had fine labels thing
                    #  the comment code was always here
                    #  self.targets.extend(entry['coarse_labels'])
                    self.targets.extend(entry['fine_labels'])
        # you take the data the list containing in each entry 1000,3072
        # you reshape it
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)# when you do-1 you flatten it then you turn it to 3 by 32 by 32
        # so you remember i said that the size of 1 numpy array was (10000,3072) it is turned to (10000,3,32,32) for example
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to Heght*width*color
        self._load_meta()# calling function of load meta check it it is heavily commented

        ind = [i for i in range(len(self.targets)) if self.targets[i] in target_list]
        # the ind in here is used  as following
        # for length of targets which contains the labels. it is a list
        # if targets is within targetlist(do you remember target list is where we say i want to be between class 0 to 5 )
        self.data = self.data[ind]# we are slicing the data to contain only the things in my targetlist
        self.targets = np.array(self.targets)# turning it to numpy array to slice it
        self.targets = self.targets[ind].tolist()# turning it back to list
        # in the end you have data and targets containing data and labels

    def _load_meta(self):#opening the pickle file
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')# loading the data
            # {'num_cases_per_batch': 10000,
            # 'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_vis': 3072}
            self.classes = data[self.meta['key']]# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        #{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

        # this commented part bellow was always in the code
        #  x = self.class_to_idx
        #  sorted_x = sorted(x.items(), key=lambda kv: kv[1])
        #  print(sorted_x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):# used to check if the files are their in this case
        # no need to redownload stuff
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):# download teh files
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CustomCIFAR10(Dataset):
    def __init__(self, root, split='train+test', transform=None, target_list=range(5), download=False,
                 remove_dict: dict = None, remove_lst: list = None):
        # OLD LINE
        # self.cifar10 = datasets.CIFAR10(root='./CIFAR', download=download, train=train, transform=T.ToTensor())
        # NEW LINE
        self.transform = transform  # This will not be passed to CIFAR10 since it will be applied here in CustomCIFAR10
        self.cifar10 = CIFAR10(root=root, split=split, target_list=target_list, download=download)

        wanted_format_input_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        assert remove_lst is None or remove_dict is None, "Cannot use both remove_lst and remove_dict together, " \
                                                          "instantiate only one of them"
        if remove_dict is not None:
            assert remove_dict.keys() == wanted_format_input_dict.keys(), "Input remove_dict need to have all and " \
                                                                          "only the keys from 0 to 9"
            underThreshold = True
            threshold = int(len(self.cifar10.data)/len(target_list))
            for key in remove_dict.keys():
                if remove_dict[key] > threshold:
                    underThreshold = False
                    break
            assert underThreshold is True, f"With this input setup of split+target_list the remove_dict can have " \
                                           f"at most values of {threshold} for each of the following " \
                                           f"keys: {list(target_list)}"

        self.data = self.cifar10.data
        self.targets = self.cifar10.targets
        self.used_classes = target_list
        self.remove_list = remove_lst
        self.remove_dictionary = remove_dict

        if self.remove_list is not None:
            print(f"Going to remove {len(self.remove_list)} samples as from list")
            self.final_data, self.final_targets = self.__remove_from_position__()

        elif self.remove_dictionary is not None:
            print(f"Going to remove {sum([self.remove_dictionary[x] for x in self.used_classes])} "
                  f"samples as from dict")
            self.final_data, self.final_targets = self.__randomly_remove_from_class__()

        elif self.remove_list is None and self.remove_dictionary is None:
            self.final_data, self.final_targets = self.data, self.targets
            print("No samples removed")

        print(f"Total number of samples and labels in CustomCIFAR10 dataset: "
              f"{len(self.final_data), len(self.final_targets)}")

    def __getitem__(self, index):
        img, target = self.final_data[index], self.final_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # Implement target_transform only if needed (not needed in AutoNovel)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.final_data)

    def __remove_from_position__(self):
        """"Removes samples in that specific index position of the dataset, regardless to the class they belong to"""
        data = np.delete(self.data, self.remove_list, axis=0)
        targets = np.delete(self.targets, self.remove_list, axis=0)
        print(f"Removed {len(self.remove_list)} samples")
        return data, targets

    def __randomly_remove_from_class__(self):
        """Removes the specified number of samples for each class, choosing random samples to remove inside the class"""

        # Define a dict with a key for each used class, it will store the index position of samples for each class
        self.class_sample_position = {x: [] for x in self.used_classes}
        # For each sample in the dataset(targets), save the index of that sample in the dataset under the dict
        # key for that class(class_label)
        for idx, class_label in enumerate(self.targets):
            self.class_sample_position[class_label].append(idx)
        # Define a copy of the dict since we will need to remove elements from this new dict to avoid to
        # random-chose an element twice (see below)
        class_sample_position = self.class_sample_position

        # Define a dict with a key for each used class, to store the index position of samples to remove for each class
        self.class_removed_sample_position = {x: [] for x in self.used_classes}
        # For each used classes
        for current_class in self.used_classes:
            # For the number of samples to remove for that class (given in input and stored in self.remove_dictionary)
            for _ in range(self.remove_dictionary[current_class]):
                # Choose a random sample index between the available index for that class
                random_idx = random.choice(class_sample_position[current_class])
                # Add the selected sample index to the list of samples index that we want to remove for that class
                self.class_removed_sample_position[current_class].append(random_idx)
                # Remove that sample index from the available index for that class, in this way it cannot be chosen
                # twice leading to errors
                class_sample_position[current_class].remove(random_idx)

        # Define remove_list as an empty list, here we will store the index of all the samples to remove, for all
        # the classes without distinction
        self.remove_list = []
        # For each used classes
        for current_class in self.used_classes:
            # Extend the remove_list adding the indexes of the sample to remove for each one of the classes of CIFAR-10
            self.remove_list.extend(self.class_removed_sample_position[current_class])

        # Remove the elements using the __remove_from_position__ method, then return data and labels
        data, targets = self.__remove_from_position__()
        return data, targets

class t_SNE(object):
    def __init__(self):
        # Store of features, labels and joint dataframe of the two
        self.feature_vectors = pd.DataFrame(columns=[f'Feature_{x}' for x in range(0, 512)])
        self.feature_vectors_labels = []
        self.feature_label_dataframe = None

        # # Computation of t-SNE and full dataframe with labels, t-sne and features
        self.num_classes = None
        self.tsne = None
        self.tsne_results = None
        self.feature_label_tsne_dataframe = None

    def get_feature_label_dataframe(self, print_df=False):
        assert len(self.feature_vectors_labels) == len(self.feature_vectors), "Number of features and of labels do " \
                                                                              "not correspond"
        self.feature_label_dataframe = self.feature_vectors.copy()
        self.feature_label_dataframe.insert(0, 'Label', self.feature_vectors_labels)
        if print_df:
            print(self.feature_label_dataframe)
        return self.feature_label_dataframe

    def compute_and_plot_2d_t_sne(self, print_df=False, plot_path='my_plot', verbose=1, perplexity=1, n_iter=300):
        self.get_feature_label_dataframe(print_df=False)

        time_start = time.time()
        self.tsne = TSNE(n_components=2, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
        self.tsne_results = self.tsne.fit_transform(self.feature_vectors.values)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        self.feature_label_tsne_dataframe = self.feature_label_dataframe.copy()
        self.num_classes = len(set(self.feature_label_tsne_dataframe["Label"]))
        self.feature_label_tsne_dataframe.insert(loc=1, column='tsne-d-one', value=self.tsne_results[:, 0])
        self.feature_label_tsne_dataframe.insert(loc=2, column='tsne-d-two', value=self.tsne_results[:, 1])
        if print_df:
            print(self.feature_label_tsne_dataframe)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-d-one", y="tsne-d-two",
            hue="Label",
            palette=sns.color_palette("tab10", self.num_classes),
            data=self.feature_label_tsne_dataframe,
            legend="full",
            alpha=0.3
        )
        plt.savefig(plot_path)
        print(f"tSNE plot saved in : {plot_path}")
        return self.feature_label_tsne_dataframe

    # Used in NN.forward() to push labels in the t-SNE class
    def push_labels(self, labels_list):
        for x in labels_list:
            self.feature_vectors_labels.append(x.item())

    # Used after the plot of a t-SNE to clear the variables for the next plot
    def __init_tsne__(self):
        self.feature_vectors = pd.DataFrame(columns=[f'Feature_{x}' for x in range(0, 512)])
        self.feature_vectors_labels = []
        self.feature_label_dataframe = None
        self.num_classes = None
        self.tsne = None
        self.tsne_results = None
        self.feature_label_tsne_dataframe = None
