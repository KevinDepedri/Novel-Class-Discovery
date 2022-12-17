####### IMPORTS
### torhc imports
import torch
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset

## 
import os 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchtext
import random
class MNISTM(VisionDataset):
    

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join('data/datasets/MNISIT_M/', data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join('data/datasets/MNISIT_M/', self.training_file)) and
                os.path.exists(os.path.join('data/datasets/MNISIT_M/', self.test_file)))

    def download(self):
        """Download the MNIST-M data."""
        raw_folder  =   ['data/datasets/MNISIT_M/mnist_m_train.pt.tar.gz',
                        'data/datasets/MNISIT_M/mnist_m_test.pt.tar.gz']
    
        url         =   ['https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
                     'https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz']
        torchtext.utils.download_from_url(url[0],raw_folder[0])
        torchtext.utils.extract_archive(raw_folder[0])
        os.remove(raw_folder[0])
        ##
        torchtext.utils.download_from_url(url[1],raw_folder[1])
        torchtext.utils.extract_archive(raw_folder[1])
        os.remove(raw_folder[1])


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
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
    source_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])   
    mnistm_ds = MNISTM('data/datasets/MNISIT_M/', download=False, train=False, transform=source_transform)
    print(mnistm_ds)