####### IMPORTS
### torhc imports
import torch
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torchnet as tnt

from .Mnisit_M_loading import MNISTM
from .Mnisit_loading import MNIST_DS
import numpy as np 
import random
import matplotlib.pyplot as plt
import os 
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
    
class GenericDataset_mnisit(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None, dataset_root=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.random_sized_crop = random_sized_crop
        self.num_imgs_per_cat = num_imgs_per_cat
        # self.mean_pix = [x / 255.0 for x in [67.6291, 68.0080, 63.3630]]# this is the std and mean calculated for the new dataset
        # self.std_pix = [x / 255.0 for x in [67.1277, 66.0482, 67.1296]]
        if self.dataset_name == 'mnisit':
            transform = []
            if split != 'test':
                # Perform a random crop of 32x32 with a padding of 4, if sequence provided it is [left,top,right,bottom]
                # transform.append(transforms.Resize(size=(32,32)))
                transform.append(transforms.RandomCrop(32, padding=4))
                # Horizontally flip the given image randomly with a given probability (default is 0.5)
            else:
                transform.append(transforms.Resize(size=(32,32)))
            
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data =MNIST_DS( 'data/datasets/MNISIT/',train=split=="train",download=False, transform=self.transform)
            flag=(self.data.targets<5).nonzero().squeeze(-1)
            self.data = torch.utils.data.Subset(self.data, flag)
        elif self.dataset_name == 'mnisitm':            
            transform = []
            if split != 'test':
                # Perform a random crop of 32x32 with a padding of 4, if sequence provided it is [left,top,right,bottom]
                # transform.append(transforms.Resize(size=(32,32)))
                transform.append(transforms.RandomCrop(32, padding=4))

                # Horizontally flip the given image randomly with a given probability (default is 0.5)
            else:
                transform.append(transforms.Resize(size=(32,32)))
            
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data =MNISTM('data/datasets/MNISIT_M/', download=False, train=split=="train", transform=self.transform)
            flag=(self.data.targets>=5).nonzero().squeeze(-1)
            self.data = torch.utils.data.Subset(self.data, flag)
        else:
                raise ValueError('Not recognized dataset {0}'.format(dataset_name))
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)
class DataLoader_mnisit(object):
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
        self.mean_pix = [x / 255.0 for x in [57.3801, 57.7565, 53.2447]]# this is the std and mean calculated for the new dataset
        self.std_pix = [x / 255.0 for x in [69.0935, 68.2257, 68.3474]]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_pix, std=self.std_pix)
        ])

    def get_iterator(self, epoch=0):
        # Compute and apply a random seed
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0, 90).copy()),
                    self.transform(rotate_img(img0, 180).copy()),
                    self.transform(rotate_img(img0, 270).copy())
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels

            def _collate_fun(batch):
                batch = default_collate(batch)
                assert (len(batch) == 2)
                batch_size, rotations, channels, height, width = batch[0].size()  # you accessioning the stack
                batch[0] = batch[0].view([batch_size * rotations, channels, height, width])  # changing the shape
                batch[1] = batch[1].view([batch_size * rotations])  # changing the shape
                return batch

        else:
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label

            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size), load=_load_function)


        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader
    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
    
def get_mean_and_std(dataset):
        '''Compute the mean and std value of dataset.'''
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += torch.Tensor.float(inputs[:,:,:,i]).mean()
                std[i] += torch.Tensor.float(inputs[:,:,:,i]).std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    seed_torch(1)
    dataset_name_1=GenericDataset_mnisit('mnisit',split='test')
    print((dataset_name_1[0][0].shape))
    print((dataset_name_1[0][1]))
    dataset_name_2=GenericDataset_mnisit('mnisitm',split='test')
    print((dataset_name_2[0][0].shape))
    print((dataset_name_2[0][1]))
    dataset = ConcatDataset([dataset_name_1,dataset_name_2])

    dloader_train = DataLoader_mnisit(
            dataset=dataset,
            batch_size=64,
            num_workers=5,
            shuffle=True)

    iterator=iter(dloader_train())
    inputs = next(iterator)
    # print(get_mean_and_std(dataset))
    print(inputs[0].shape)
    imshow(torchvision.utils.make_grid(inputs[0]))
