# AutoNovel

This is **[Automatically Discovering and Learning New Visual Categories with Ranking Statistics, ICLR 2020](http://www.robots.ox.ac.uk/~vgg/research/auto_novel/)** paper by 
<br>[Kai Han*](http://www.hankai.org), [Sylvestre-Alvise Rebuffi*](http://www.robots.ox.ac.uk/~srebuffi/), [Sebastien Ehrhardt*](), [Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi/), [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/)<br> that is used to experiment on it and understanding the implementation of the code. 

## Dependencies

All dependencies are included in `environment.yml`. To install, run

```shell
conda env create -f environment.yml
```

(Make sure you have installed [Anaconda](https://www.anaconda.com/) before running.)

Then, activate the installed environment by

```
conda activate auto_novel
```

## Overview

We provide code and models for our experiments on CIFAR10, CIFAR100, SVHN,:

- Code for self-supervised learning
- Code for supervised learning
- Code for novel category discovery
- Our trained models and all other required pretrained models

## Data preparation

By default, we save trained models in `./data/experiments/` (soft link is suggested). You may also use any other directories you like by setting the `--dataset_root` argument to `/your/data/path/`, and the `--exp_root` argument to `/your/experiment/path/` when running all experiments below. 

- For CIFAR-10, CIFAR-100, and SVHN, simply download the datasets and put into `./data/datasets/`.
Our code for step of self supervised learning is based on the official code of the [RotNet paper](https://arxiv.org/pdf/1803.07728.pdf).

## Step 1: Download datasets and weights

1. Clone the repository in whatever directory you want. 

2. Move to autoNovel directory and running the following directory and everything will be downloaded starting data sets and preweights of previous trained models.

   ```
   sh scripts/download_pretrained_models_dataset.sh
   
   ```

   

## Step 2: Joint training for novel category discovery

### Novel category discovery on CIFAR10/CIFAR100/SVHN

```shell
# Train on CIFAR10
CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth
CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth resnet_IL_cifar10_bce

# Train on CIFAR100
CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_cifar100.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar100.pth

# Train on SVHN
CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_svhn.sh ./data/datasets/SVHN/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_svhn.pth
```

To train in the Incremental Learning (IL) mode, replace ``auto_novel_{cifar10, cifar100, svhn}.sh`` in the above commands by ``auto_novel_IL_{cifar10, cifar100, svhn}.sh``.



### Evaluation on novel category discovery

To run our code in evaluation mode, set the `--mode` to `test`. 

```shell
# For CIFAR10
CUDA_VISIBLE_DEVICES=0 python auto_novel.py --mode test --dataset_name cifar10 --model_name resnet_cifar10 --exp_root ./data/experiments/pretrained/

# For CIFAR100
CUDA_VISIBLE_DEVICES=0 python auto_novel.py --mode test --dataset_name cifar100 --model_name resnet_cifar100 --exp_root ./data/experiments/pretrained/ --num_labeled_classes 80 --num_unlabeled_classes 20 

# For SVHN
CUDA_VISIBLE_DEVICES=0 python auto_novel.py --mode test --dataset_name svhn --model_name resnet_svhn --exp_root ./data/experiments/pretrained/ --dataset_root ./data/datasets/SVHN
```

To perform the evaluation in the Incremental Learning (IL) mode, add in the above commands the argument ``--IL`` and replace the model name``resnet_{cifar10, cifar100, svhn}`` by ``resnet_IL_{cifar10, cifar100, svhn}``.

### Citation

If this work is helpful for your research, please cite our paper.

```
@inproceedings{Han2020automatically,
author    = {Kai Han and Sylvestre-Alvise Rebuffi and Sebastien Ehrhardt and Andrea Vedaldi and Andrew Zisserman},
title     = {Automatically Discovering and Learning New Visual Categories with Ranking Statistics},
booktitle = {International Conference on Learning Representations (ICLR)},
year      = {2020}
}
```

# AutoNovel experiments

## Experiment 1 (Using different SSL techniques)
To test the results obtained by applying different SSL .....

## Experiment 2 (Domain Shift)
..............

## Experiment 3 (Unbalanced Classes) - Supported only for CIFAR-10
This experiments allow to train and thest a model using a custom number of samples for each class of CIFAR10.

We performed this experiment to see how the model performs in cases where the number of labeled samples is very low (1/10 of the unlabeled samples), and in the opposite cases, where the number of unlabeled samples is equalt to 1/10 of the labeled samples.

## Plotting t-SNE for any experiment 
The t-distributed Stochastic Neighbor Embedding is a statistical tool that allows to represent high dimensional samples into a low dimensional space relying on a statistical algorithm. Due to its stochastic nature this algorithm leads to different output for each run, also if the input data and the used parameters are exactly the same.

We used the t-SNE plots to show how much the features learned by our models are effective. They allow us to see how the samples belonging to different categories are clustered. Ideally, we would like to see compact cluster well distatiented between them. This condition would point that our model learn some good features which allows to distinguish between samples coming from different classes in an efficient way.

To plot the t-SNE for your model follow the ensuing procedure (steps using CIFAR-10):

1. Train your model until the end of the 'AutoNovel-step' and store the weights of your model

2. Open the file ``auto_novel_for_tSNE.py``
   - If your model has been trained using the ResNet defined by the authors, then be sure that at line 448 the parameter ``New_resnet = False``
   - If your model has been trained using a standard ResNet, then be sure that at line 448 the parameter ``New_resnet = True``

3. Put the weights of your model into the path ``data/experiments/auto_novel_for_tSNE/name_of_you_model.pth``

4. Depending on the Incremental-Learning (IL) setting that you used to train your model:
- If IL enabled -> run ``auto_novel_IL_cifar10_tSNE.sh`` through cmd using the following line of code (change the parameter ``name_of_you_model``):
```shell
   CUDA_VISIBLE_DEVICES=0 sh ``scripts/auto_novel_IL_cifar10_tSNE.sh`` ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth name_of_you_model
```

- If IL disabled -> run ``auto_novel_no_IL_cifar10_tSNE`` through cmd using the following line of code (change the parameter ``name_of_you_model``):
```shell
   CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_no_IL_cifar10_tSNE.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth name_of_you_model 
```
   
5. The produced plots will be stored in the folder ``tSNE_plots/name_of_you_model``

6. If you are working on a dataset different from CIFAR-10, or if other changes have been applied on the training procedure, then apply the due changes also to the py file ``auto_novel_for_tSNE.py`` and to the lunch sh file ``auto_novel_IL_cifar10_tSNE.sh`` or ``auto_novel_no_IL_cifar10_tSNE.sh``
