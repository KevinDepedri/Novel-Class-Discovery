from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import torchtext
import torch
import os
import torchvision
import random
from torchvision.datasets import VisionDataset
# from utils.util import seed_torch
from PIL import Image
import torchvision.transforms as transforms

class MNIST_DS(VisionDataset):
    training_file = "tensor_dataset_train.pt"
    test_file = "tensor_dataset_test.pt"
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
    def __init__(self, root, train=True, transform=None, target_transform=None,download=False):
        """Init MNIST_DS dataset."""
        super(MNIST_DS, self).__init__(root,transform=transform, target_transform=target_transform)

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
        self.data, self.targets = torch.load(os.path.join('data/datasets/MNISIT/', data_file))
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
        return (os.path.exists(os.path.join('data/datasets/MNISIT/', self.training_file)) and
                os.path.exists(os.path.join('data/datasets/MNISIT/', self.test_file)))
    def download(self):
        """Download the MNIST data."""
        raw_folder  =   ['data/datasets/MNISIT/train-images-idx3-ubyte.gz',
                            'data/datasets/MNISIT/train-labels-idx1-ubyte.gz',
                            'data/datasets/MNISIT/t10k-images-idx3-ubyte.gz',
                            'data/datasets/MNISIT/t10k-labels-idx1-ubyte.gz']
            
        to_path     =   ['data/datasets/MNISIT/train-images-idx3-ubyte',
                            'data/datasets/MNISIT/train-labels-idx1-ubyte',
                            'data/datasets/MNISIT/t10k-images-idx3-ubyte',
                            'data/datasets/MNISIT/t10k-labels-idx1-ubyte']
            
        url         =   ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
        for i in range(len(raw_folder)):
                torchtext.utils.download_from_url(url[i],raw_folder[i])
                torchtext.utils.extract_archive(raw_folder[i], to_path[i])
                os.remove(raw_folder[i])
        seed_torch(1)
        home='data/datasets/MNISIT/'
        mndata = MNIST(home,return_type='numpy')
        images, labels = mndata.load_training()
        new_images=np.reshape(images,(images.shape[0],28, 28,1))
        dataset=self.create_mnistm(new_images).astype(np.uint8)
        torch.save((torch.from_numpy(dataset),torch.from_numpy(labels)),home+'tensor_dataset_train.pt')
        # Testing
        mndata = MNIST(home,return_type='numpy')
        images, labels = mndata. load_testing()
        new_images=np.reshape(images,(images.shape[0],28, 28,1))
        dataset=self.create_mnistm(new_images).astype(np.uint8)
        torch.save((torch.from_numpy(dataset),torch.from_numpy(labels)),home+'tensor_dataset_test.pt')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    def create_mnistm(self,X):
        X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
        X_ = np.concatenate([X,X,X],axis=-1)
        return X_
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
    mnistm_ds = MNIST_DS(root='data/datasets/MNISIT/',train=True, transform=source_transform, download=False)
    print(mnistm_ds)
