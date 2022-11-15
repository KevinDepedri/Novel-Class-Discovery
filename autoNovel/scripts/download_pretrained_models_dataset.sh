#!/usr/bin/env bash


path="../../data_sets/autonovel/"
path_2="../../data_sets/CIFAR/"
path_3="../../data_sets/SVHN/"

mkdir -p $path
mkdir -p $path_2
mkdir -p $path_3


cd $path_2
# downloading the cifar datasets
wget  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz && rm cifar-10-python.tar.gz

wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz && rm cifar-100-python.tar.gz
# downloading the SVHN datasets
cd ../SVHN
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

# downloading the preweights for supervised and semi-supervised
cd ../autonovel
wget http://www.robots.ox.ac.uk/~vgg/research/auto_novel/asset/pretrained.zip

unzip pretrained.zip && rm pretrained.zip

cd ../../Trends_projects/autoNovel