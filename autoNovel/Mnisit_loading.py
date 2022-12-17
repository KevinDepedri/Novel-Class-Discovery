from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import torchtext
import torch
import os
from utils.util import seed_torch
def create_mnistm(X):
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    X_ = np.concatenate([X,X,X],axis=-1)
    return X_
if __name__ == '__main__':
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
    dataset=create_mnistm(new_images).astype(np.uint8)
    torch.save((torch.from_numpy(dataset),torch.from_numpy(labels)),home+'tensor_dataset_train.pt')
    # Testing
    mndata = MNIST(home,return_type='numpy')
    images, labels = mndata. load_testing()
    new_images=np.reshape(images,(images.shape[0],28, 28,1))
    dataset=create_mnistm(new_images).astype(np.uint8)
    torch.save((torch.from_numpy(dataset),torch.from_numpy(labels)),home+'tensor_dataset_test.pt')
