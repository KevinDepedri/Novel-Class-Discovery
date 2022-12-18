import torchvision.transforms as transforms
from  .utils import TransformTwice
from torch.utils.data import ConcatDataset
import torch
import torch.utils.data as data
import numpy as np 
from .Mnisit_M_loading import MNISTM
from .Mnisit_loading import MNIST_DS
import random
import os
import torchvision
class MNISIT_MIX(data.Dataset):
    def __init__(self, dataset_name, split, transform):
        self.dataset_name = dataset_name
        if self.dataset_name == 'mnisit':
            self.data =MNIST_DS( 'data/datasets/MNISIT/',train=split=="train",download=False, transform=transform)
            flag=(self.data.targets<5).nonzero().squeeze(-1)
            self.data = torch.utils.data.Subset(self.data, flag)
        elif self.dataset_name == 'mnisitm':            
            self.data =MNISTM('data/datasets/MNISIT_M/', download=False, train=split=="train", transform=transform)
            flag=(self.data.targets>=5).nonzero().squeeze(-1)
            self.data = torch.utils.data.Subset(self.data, flag)
        else:
                raise ValueError('Not recognized dataset {0}'.format(dataset_name))
    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label),index
    def __len__(self):
        return len(self.data)

def MNISITData(split='train', aug=None, number_of_classes=5,catego='labeled'):
    mean_pix = [x / 255.0 for x in [57.3801, 57.7565, 53.2447]]# this is the std and mean calculated for the new dataset
    std_pix = [x / 255.0 for x in [69.0935, 68.2257, 68.3474]]
    if aug==None:
        transform = transforms.Compose([
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean_pix, std_pix),
        ])

    elif aug == 'once':  # Used in supervised_learning.py
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Random cropping while padding
            transforms.ToTensor(),  # Turn the image to tensor
            transforms.Normalize(mean_pix, std_pix),
        ])

    elif aug == 'twice':  
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),  
            transforms.Normalize(mean_pix, std_pix),
            ]))
    dataset_name_1=MNISIT_MIX('mnisit',split=split,transform=transform)
    dataset_name_2=MNISIT_MIX('mnisitm',split=split,transform=transform)
    if number_of_classes==5:
        if catego=="labeled":
            dataset = MNISIT_MIX('mnisit',split=split,transform=transform)
        elif catego=="unlabeled":
            dataset = MNISIT_MIX('mnisitm',split=split,transform=transform)
    else:
        dataset = ConcatDataset([dataset_name_1,dataset_name_2])
    return dataset
def MNISITLoader(batch_size, split='train', num_workers=2,  aug=None, shuffle=True, catego='labeled',number_of_classes=5):
    dataset = MNISITData(split, aug, number_of_classes,catego)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
def MNISITLoaderMix(batch_size, split='train',num_workers=2, aug=None, shuffle=True,catego='labeled',number_of_classes=5):
    dataset = MNISITData(split, aug, number_of_classes,catego)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
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
    import matplotlib.pyplot as plt
    seed_torch(1)
    # x=MNISITLoader(64,'train',2,'twice' ,shuffle=True,number_of_classes=5)
    x=MNISITLoaderMix(64,'train',2,'twice' ,shuffle=True, number_of_classes=10)
    def imshow(img):
        npimg = img.numpy()
        plt.imshow((np.transpose(npimg, (1, 2, 0))* 255).astype(np.uint8))
        plt.show()
    iterator=iter(x)
    inputs = next(iterator)
    # print(inputs[0][0].shape)
    # print(torch.sum((inputs[0][0][0]==inputs[0][0][0]).int()))
    imshow(torchvision.utils.make_grid(inputs[0][0]))
    imshow(torchvision.utils.make_grid(inputs[0][1]))