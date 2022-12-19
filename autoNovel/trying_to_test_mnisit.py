from data.rotation_loader_mnisit_basline import DataLoader_mnisit_baseline, GenericDataset_mnisit_basline
import random
import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from data.MNISIT_loader import MNISITLoader_main, MNISITLoader

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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
def imshow(img):
        npimg = img.numpy()
        f = plt.figure(figsize=(10, 6))
        ax = f.add_subplot()

        plt.imshow((np.transpose(npimg, (1, 2, 0))* 255).astype(np.uint8))
        ax.axis(False)
        ax.set_position([0, 0, 1, 1])
        plt.savefig('images/Mnisit.png')
        plt.show()
if __name__ == '__main__':
    seed_torch(1)
    # dataset_name=GenericDataset_mnisit_basline('mnisit',split='train')


    # dloader_train = DataLoader_mnisit_baseline(
    #             dataset=dataset_name,
    #             batch_size=20,
    #             num_workers=5,
    #             shuffle=True)
    # dataset_name_1=GenericDataset_mnisit_basline('mnisit',split='test')
    # dloader_test = DataLoader_mnisit_baseline(
    #             dataset=dataset_name_1,
    #             batch_size=20,
    #             num_workers=6,
    #             shuffle=True)
    # iterator=iter(dloader_test())
    # inputs = next(iterator)
    # # print(get_mean_and_std(dataset_name))
    # print(inputs[0].shape)
    # imshow(torchvision.utils.make_grid(inputs[0]))
    x=MNISITLoader_main(32,'train',2,None ,shuffle=True,catego='unlabeled',number_of_classes=10)
    # x=MNISITLoader(32,'train',2,None ,shuffle=True,catego='unlabeled',number_of_classes=10)

    iterator=iter(x)
    inputs = next(iterator)
    # print(inputs[0][0].shape)
    # print(torch.sum((inputs[0][0][0]==inputs[0][0][0]).int()))
    imshow(torchvision.utils.make_grid(inputs[0]))
    # imshow(torchvision.utils.make_grid(inputs[0][1]))