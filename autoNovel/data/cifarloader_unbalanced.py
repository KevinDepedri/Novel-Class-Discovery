from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import random
import torch
import torch.utils.data as data
# from .utils import download_url, check_integrity
# from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler
# from .concat import ConcatDataset
from data.utils import download_url, check_integrity
from data.utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler
from data.concat import ConcatDataset
import torchvision.transforms as transforms
from utils.additional_classes import CustomCIFAR10

class CIFAR10(data.Dataset):# this is class dataset
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

def CIFAR10Data(root, split='train', aug=None, target_list=range(5), download=False,
                remove_dict: dict = None, remove_lst: list = None):
    # If we have no augmentation just transform to tensor and normalize
    if aug == None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),

        ])

    # If we have one augmentation then random crop and horizontal flip the image, then move to tensor and normalize
    elif aug == 'once':  # Used in supervised_learning.py
        # transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),  # Random cropping while padding
        #     transforms.RandomHorizontalFlip(),  # Perform a random horizontal flip
        #     transforms.ToTensor(),  # Turn the image to tensor
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using mean and std
        #     # TODO: where this STD comes from? the std one are not the same to the computed ones as in RotationLoader
        # ])
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),  # Random cropping while padding
            transforms.RandomHorizontalFlip(),  # Perform a random horizontal flip
            transforms.ToTensor(),  # Turn the image to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # TODO: where this STD comes from? the std one are not the same to the computed ones as in RotationLoader
        ])
        # values from mean [0.4913725490196078, 0.4823529411764706, 0.4466666666666667] are the same
        # values from std [0.24705882352941178, 0.24352941176470588, 0.2615686274509804] but the std is different

    # If we have two augmentations
    elif aug == 'twice':  # Used in auto_novel.py
        transform = TransformTwice(transforms.Compose([
            RandomTranslateWithReflect(4),  # Random translate and reflect
            transforms.RandomHorizontalFlip(),  # Perform a random horizontal flip
            transforms.ToTensor(),  # Turn the image to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using mean and std
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ]))

    # OLD LINE
    # dataset = CIFAR10(root=root, split=split, transform=transform, target_list=target_list)
    # NEW LINE
    dataset = CustomCIFAR10(root=root, split=split, transform=transform, target_list=target_list, download=download,
                            remove_dict=remove_dict, remove_lst=remove_lst)

    return dataset

# Used in supervised_learning.py and auto_novel.py
def CIFAR10Loader(root, batch_size, split='train',  aug=None, shuffle=True, target_list=range(5), num_workers=2,
                  download=False, remove_dict: dict = None, remove_lst: list = None):
    # Called in supervised learning, target_list contains range with the indexes for the labeled classes.
    # In the paper target_list=range(5), in this way the classes with labels will be [0,1,2,3,4]
    # Instantiate the dataset (for supervised learning case the augmentation parameter is set to 'once')
    dataset = CIFAR10Data(root, split, aug, target_list, download, remove_dict, remove_lst)
    # Define the dataloader with the given parameters and return it
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

# Used only in auto_novel.py
def CIFAR10LoaderMix(root, batch_size, split='train', num_workers=2, aug=None, shuffle=True, labeled_list=range(5),
                     unlabeled_list=range(5, 10), new_labels=None, download=False,
                     remove_dict: dict = None, remove_lst: list = None):
    # First choose the type of augmentation between none, once and twice
    if aug==None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    # Type of augmentation used in the mix_train_loader in the auto_novel.py file
    elif aug == 'twice':
        # Build the transform used to augment the examples
        transform = TransformTwice(transforms.Compose([
            # Translate the image vertically and horizontally by n pixels chosen in the interval between 0 and
            # max_translation. Then fill the uncovered blank area with reflect padding
            RandomTranslateWithReflect(max_translation=4),
            transforms.RandomHorizontalFlip(),  # Perform a random horizontal flip
            transforms.ToTensor(),  # Turn the image to tensor
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using mean and std
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]))

    # Define a labeled dataset calling the CIFAR10 class, we pass target_list=labeled_list to choose the first 5 classes
    # OLD LINE
    # dataset_labeled = CIFAR10(root=root, split=split, transform=transform, target_list=labeled_list)
    # NEW LINE
    dataset_labeled = CustomCIFAR10(root=root, split=split, transform=transform, target_list=labeled_list,
                                    download=download, remove_dict=remove_dict, remove_lst=remove_lst)


    # Each dataset_labeled[i] is a tuple containing 3 things:
    # - A tuple containing 2 tensor of (3,32,32) which are the original image and its augmentation
    # - A class label as an integer
    # - The index of the picture in the original dataset (CIFAR10)

    # Define an unlabeled dataset calling the CIFAR10 class, target_list=unlabeled_list to choose the last 5 classes
    # OLD LINE
    # dataset_unlabeled = CIFAR10(root=root, split=split, transform=transform, target_list=unlabeled_list)
    # NEW LINE
    dataset_unlabeled = CustomCIFAR10(root=root, split=split, transform=transform, target_list=unlabeled_list,
                                      download=download, remove_dict=remove_dict, remove_lst=remove_lst)

    # If we have some addition input labels they are applied and used as targets
    if new_labels is not None:
        dataset_unlabeled.targets = new_labels

    # Now we have 2 dataset one for labeled and one for unlabeled data, here:
    # - dataset_labeled.data has size of (25000, 32, 32, 3)
    # - dataset_labeled.targets has size of 25000
    # - dataset_unlabeled.data has size of (25000, 32, 32, 3)
    # - dataset_unlabeled.targets has size of 25000
    # These two datasets are concatenated in a two unique dataset (one for targets and one for data), now we have:
    # - dataset_unlabeled.data has size of (50000, 32, 32, 3)
    # - dataset_unlabeled.targets has size of 50000
    # Of this dataset the first half is labeled while the second half is unlabeled
    dataset_labeled.targets = np.concatenate((dataset_labeled.targets, dataset_unlabeled.targets))
    dataset_labeled.data = np.concatenate((dataset_labeled.data, dataset_unlabeled.data), 0)

    # Instantiate a dataloader for that dataset with the specific batch_size and enabling shuffle to mix the order of
    # the labeled and the unlabeled samples. Finally return this loader
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR10LoaderTwoStream(root, batch_size, split='train', num_workers=2, aug=None, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10), unlabeled_batch_size=64):
    dataset_labeled = CIFAR10Data(root, split, aug, labeled_list)
    dataset_unlabeled =  CIFAR10Data(root, split, aug, unlabeled_list)
    dataset = ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled))
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size)
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader

class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        #  'key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

def CIFAR100Data(root, split='train', aug=None, target_list=range(80)):
    if aug==None:
        transform = transforms.Compose([# for test you donot do any crop or horizontal flip you enter here with test set
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])
    elif aug=='once':# enter into here during supervised learning
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),# random crop with padding for all sides
            transforms.RandomHorizontalFlip(),# flipp
            transforms.ToTensor(),# turn to tensor
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),# i donot understand from where he got this values
        ])
    elif aug=='twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ]))
    dataset = CIFAR100(root=root, split=split, transform=transform, target_list=target_list)
    return dataset

# used with cifar 100 in supervised learning
def CIFAR100Loader(root, batch_size, split='train', num_workers=2,  aug=None, shuffle=True, target_list=range(80)):
    dataset = CIFAR100Data(root, split, aug,target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)# returns the data laoder
    return loader

def CIFAR100LoaderMix(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(80), unlabeled_list=range(90, 100)):
    dataset_labeled = CIFAR100Data(root, split, aug, labeled_list)
    dataset_unlabeled = CIFAR100Data(root, split, aug, unlabeled_list)
    dataset_labeled.targets = np.concatenate((dataset_labeled.targets,dataset_unlabeled.targets))
    dataset_labeled.data = np.concatenate((dataset_labeled.data,dataset_unlabeled.data),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def CIFAR100LoaderTwoStream(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(80), unlabeled_list=range(90, 100), unlabeled_batch_size=32):
    dataset_labeled = CIFAR100Data(root, split, aug, labeled_list)
    dataset_unlabeled = CIFAR100Data(root, split, aug, unlabeled_list)
    dataset = ConcatDataset((dataset_labeled, dataset_unlabeled))
    labeled_idxs = range(len(dataset_labeled))
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size)
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader


if __name__ == "__main__":
    import argparse

    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Get all the needed input arguments
    parser.add_argument('--lr', type=float, default=0.1)  # Learning rate of optimizer
    parser.add_argument('--gamma', type=float, default=0.1)  # Gamma of the learning rate scheduler
    parser.add_argument('--momentum', type=float, default=0.9)  # Momentum term of optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # Weight decay of optimizer
    parser.add_argument('--epochs', default=200, type=int)  # Number of epochs
    parser.add_argument('--rampup_length', default=150, type=int)  # Ramp-up length passed to ramps.py function
    parser.add_argument('--rampup_coefficient', type=float, default=50)  # Ramp-up coefficient
    parser.add_argument('--increment_coefficient', type=float, default=0.05)  # Incremental learning coefficient
    parser.add_argument('--step_size', default=170, type=int)  # Step size of learning rate scheduler
    parser.add_argument('--batch_size', default=128, type=int)  # Batch size
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)  # Number of unlabeled classes
    parser.add_argument('--num_labeled_classes', default=5, type=int)  # Number of labeled classes
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')  # Dataset root directory
    parser.add_argument('--exp_root', type=str,
                        default='./data/experiments/')  # Directory to save the resulting files
    parser.add_argument('--warmup_model_dir', type=str,
                        default='./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth')  # Directory to find the supervised pretrained model
    parser.add_argument('--topk', default=5, type=int)  # Number of top elements that we want to compare
    parser.add_argument('--IL', action='store_true', default=False,
                        help='w/ incremental learning')  # Enable/Disable IL
    parser.add_argument('--model_name', type=str, default='resnet')  # Name of the model
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='options: cifar10, cifar100, svhn')  # Name of the used dataset
    parser.add_argument('--seed', default=1, type=int)  # Seed to use
    parser.add_argument('--mode', type=str, default='train')  # Mode: train or test
    logging_on = False  # Variable to stop logging when we do not want to log anything

    # Extract the args and make them available in the args object
    args = parser.parse_args()
    # Define if cuda can be used and initialize the device used by torch. Furthermore, specify the torch seed
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    # Remove all from here
    print("BUILD CUSTOM DATASET")
    unbalanced = False
    if unbalanced:
        sample_per_class_to_remove_dictionary = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    else:
        sample_per_class_to_remove_dictionary = None

    labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                        aug=None, shuffle=False, target_list=range(args.num_labeled_classes),
                                        remove_dict=sample_per_class_to_remove_dictionary)
    print("CUSTOM DATASET BUILT")
    # to here
