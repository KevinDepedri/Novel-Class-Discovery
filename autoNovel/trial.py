# useless file just for deleting it
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from tqdm import tqdm

if __name__ == '__main__':
        labeled_train_loader = CIFAR10Loader(root='./data/datasets/CIFAR/', batch_size=128, split='train', aug='once', shuffle=True, target_list = range(5))
        labeled_eval_loader = CIFAR10Loader(root='./data/datasets/CIFAR/', batch_size=128, split='test', aug=None, shuffle=False, target_list = range(5))
        iterator=iter(labeled_train_loader)
        inputs = next(iterator)
        print(len(inputs))# the len is 3