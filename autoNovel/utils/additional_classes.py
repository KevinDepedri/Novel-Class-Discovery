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

class CustomCIFAR_10(Dataset):
    def __init__(self, remove_lst: list = None, remove_dict: dict = None, train=True, download=False):
        self.cifar10 = datasets.CIFAR10(root='./CIFAR', download=download, train=train, transform=T.ToTensor())

        wanted_format_input_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        assert remove_lst is None or remove_dict is None, "Cannot use both remove_lst and remove_dict together, " \
                                                          "instantiate only one of them"
        assert remove_dict.keys() == wanted_format_input_dict.keys(), "Input remove_dict need to have all keys " \
                                                                      "from 0 to 9"

        under5000 = True
        for key in remove_dict.keys():
            if remove_dict[key] > 5000:
                under5000 = False
                break
        assert under5000 is True, "Input remove_dict need to have values lower than 5000 for each key"

        self.data = self.cifar10.data
        self.targets = self.cifar10.targets
        self.remove_list = remove_lst
        self.remove_dictionary = remove_dict

        if self.remove_list is not None:
            print(f"Going to remove {len(self.remove_list)} samples as from list")
            self.final_data, self.final_targets = self.__remove_from_position__()

        elif self.remove_dictionary is not None:
            print(f"Going to remove {sum([self.remove_dictionary[x] for x in range(len(self.remove_dictionary))])} "
                  f"samples as from dict")
            self.final_data, self.final_targets = self.__randomly_remove_from_class__()

        elif self.remove_list is None and self.remove_dictionary is None:
            self.final_data, self.final_targets = self.data, self.targets
            print("No samples removed")

        print(f"Total number of samples and labels in the Custom_CIFAR_10 dataset: "
              f"{len(self.final_data), len(self.final_targets)}")

    def __getitem__(self, index):
        data, target = self.final_data[index], self.final_targets[index]
        return data, target  # index

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

        # Define a dict with a key for each class (10 classes on CIFAR-10), it will store the index position of
        # samples for each class
        self.class_sample_position = {x: [] for x in range(10)}
        # For each sample in the dataset(targets), save the index of that sample in the dataset under the dict
        # key for that class(class_label)
        for idx, class_label in enumerate(self.targets):
            self.class_sample_position[class_label].append(idx)
        # Define a copy of the dict since we will need to remove elements from this new dict to avoid to
        # random-chose an element twice (see below)
        class_sample_position = self.class_sample_position

        # Define a dict with a key for each class (10 classes on CIFAR-10), it will store the index position of
        # samples to remove for each class
        self.class_removed_sample_position = {x: [] for x in range(10)}
        # For each of the 10 classes of CIFAR-10
        for current_class in range(10):
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
        # For each of the 10 classes of CIFAR-10
        for current_class in range(10):
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

    def compute_and_plot_2d_t_sne(self, print_df=False, plot_name='my_plot', verbose=1, perplexity=1, n_iter=300):
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
        picture_save_path = 'tSNE_plots/' + plot_name + '.png'
        plt.savefig(picture_save_path)
        print(f"tSNE plot saved as : {plot_name + '.png'}")
        return self.feature_label_tsne_dataframe

    # Used in NN.forward() to push labels in the t-SNE class
    def push_labels(self, labels_list):
        for x in labels_list:
            self.feature_vectors_labels.append(x.item())
