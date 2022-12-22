# Trends_projects

Novel Class Discovery objective is to classify samples from two disjoint but similar dataset, one of which is labeled and the other unlabaled.
Here, we present some methods that tackle this task and our contributions: first, we further commented the original code, then we ran experiments to check the behaviors of these models in different settings.

This repository is part of the student project for "Application and Trends of Computer Vision" course, from University of Trento.

We worked with the two following architectures:
* AutoNovel
* UNO

This readme file presents all the experiments that we have carried out for both the architectures.
Inside each folder it is possible to find the original readme file of that specific architecture, which explains its design and how to run the base code.

All the weights of the trained model resulting from performed experiments can be found [here](https://drive.google.com/drive/u/1/folders/1H0lRtJJ7G6jjA_u0GSGxPQtZjSBDQSHm)

****
# Experiments performed on AutoNovel

## Experiment 1 (Loss terms ablation study)

Useful to understand the impact of different loss terms on the performance of the model.

1. Open ``Autonovel.py`` file

2. Comment any of the loss term to ignore it during training

3. Run ``autonovel-step`` using the pre-trained weights of the ``supervised_learning-step``

## Experiment 2 (Change Topk)

Useful to understand the impact of different ``topk`` values on the performance of the model.

1. Open ``scripts/auto_novel_IL_cifar10.sh``

2. Change ``topk`` parameter to the desired value

3. Train the model


## Experiment 3 (Remove SSL)

Useful to understand the impact of the self-supervised learning step on the performance of the model.

1. Open ``supervised_learning.py`` file

2. Comment lines 301-310 

  ```python
      model.load_state_dict(state_dict, strict=False)
      for name, param in model.named_parameters():
          # If the parameter under analysis does not belong to 'head' (one of the two heads) or to 'layer4' (features
          # layer before the two heads), then freeze that parameter. In this way we are ensuring that all the parameters
          # will be frozen except for the two heads and the features layer, which we want to train.
          if 'head' not in name and 'layer4' not in name:
              param.requires_grad = False 
  ```

3. Run ``supervised_learning.py`` 

4. Run ``autonovel.py`` 


## Experiment 4 (Using different SSL techniques)

Useful to understand the impact of different self-supervised learning techniques on the performance of the model.

The step of self-supervised learning in the original paper is based on the official code of the [RotNet paper](https://arxiv.org/pdf/1803.07728.pdf).
Here, we try to use other SSL models exploiting [solo-learn](https://github.com/vturrisi/solo-learn), which is a library of self-supervised methods.

The weights of the trained self-supervised models are given by solo-learn library.


1. Run the following command to download the weights

```bash
sh scripts/load_SSL_weights.sh
```

2. Open ``supervised_learning.py`` to find the following SSL method: Barlow_twins, simsiam, supcon, swav, vibcreg, vicreg, wmse

3. There are 2 flags called ``New_SSL_methods`` and ``New_Resnet_config`` at line 229 and 230, respectively. 
   - ``New_SSL_methods`` : True when using SSL methods other than RotNet
   - ``New_Resnet_config`` : True 
   
    This flag is used to indicate if you use the Autonovel Resnet or Resnet architecture similar to solo learn. There is a small difference in the performance between these two. When you load any of the Sololearn methods, we automatically use the similar Resnet architecture. When we are using rotnet, we leave it to the user to decide what to do. Setting to False means that we are using the Autonovel Resnet architecture.

4. Set variable called ``ssl``  in line 237 to the preferred method using one of these keys: Barlow_twins, simsiam, supcon, swav, vibcreg, vicreg, wmse 

5. Run ``supervised_learning.py`` 

6. Run ``autonovel.py`` 

## Experiment 5 (Domain Shift)

Useful to understand the impact of domain shift on the performance of the model.

### Cifar-10 with domain shift experiment

Here, Cifar10 dataset has been automatically corrupted using Gaussian Noise.


1. a. Run ``selfsupervised_learning`` and ``supervised_learning.py`` with the Cifar10-C by passing ``cifar10c`` as dataset_name
   OR
   b. Download weights of self-supervised and supervised training, by running ``sh scripts/download_cifar_c_weights.sh``

2. Run

``bash
CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_cifar10c.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/cifar_c/ ./data/experiments/cifar_c/supervised_learning/resnet_rotnet_cifar10_c.pth
``

To evaluate run:

``bash
CUDA_VISIBLE_DEVICES=0 python auto_novel_cifar10_c.py --mode test --dataset_name cifar10 --model_name resnet_IL_cifar10_c --exp_root ./data/experiments/pretrained/
``

### Mnist with domain shift experiment

Here, Mnist and Mnist-M are exploited. 
The dataset will be automatically downloaded by running the code.

1. a. Run ``selfsupervised_learning`` and ``supervised_learning.py`` with the Mnist-M by passing ``mnisit`` to load a dataset containing the first 5 classes from the   original Mnist dataset and the second 5 classes from the Mnist-M dataset
   OR
   b. Download weights of self-supervised and supervised training, by running ``sh scripts/download_mnisit_weights.sh`` 

2. Run 
```bash
# Train on original mnist
CUDA_VISIBLE_DEVICES=0 sh scripts/autonovel_IL_mnisit_mix.sh ./data/datasets/MNISIT/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_mnisit_baseline.pth resnet_IL_minsiit_baseline mnisit_baseline
# Train on novel mnist
CUDA_VISIBLE_DEVICES=0 sh scripts/autonovel_IL_mnisit_mix.sh ./data/datasets/MNISIT/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_mnisit_MIX.pth resnet_IL_minst_mix mnisit
```

To evaluate use this command

```bash
# for mnist
CUDA_VISIBLE_DEVICES=0 python auto_novel.py --mode test --dataset_name mnisit --model_name resnet_IL_minst_mix --exp_root ./data/experiments/pretrained/
# for mnist Base
CUDA_VISIBLE_DEVICES=0 python auto_novel.py --mode test --dataset_name mnisit_baseline --model_name resnet_IL_minsiit_baseline --exp_root ./data/experiments/pretrained/
```

## Experiment  6 (Unbalanced Classes) - Supported only for CIFAR-10

Useful to understand the impact of unbalanced domain on the performance of the model.

This experiment allows to train and test a model using a custom number of samples for each class of CIFAR10.

In particular, we performed this experiment to see how the model performs in case the number of labeled samples is very low (1/10 of the unlabeled samples), and in the opposite case, where the number of unlabeled samples is equal to 1/10 of the labeled samples.

The experiment is based on a custom version of the CIFAR10 dataset called CustomCIFAR10. This takes in input all the usual parameters of the CIFAR10 dataset and a ``remove_dict``. This parameters allow to give in input a dictionary, which specifies how many samples we want to be removed for each class. The dictionary need to follow this format: ``remove_dict={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 4500, 6: 4500, 7: 4500, 8: 4500, 9: 4500}``. In this previous example we are removing 0 samples for each class from 0 to 4, while we are removing 4500 samples for each class from 5 to 9. The specified number of samples is removed randomly from the corrsponding class.

To run your own unbalanced experiment, follow the ensuing procedure:

1. Run ``selfsupervised_learning-step`` and store the weights of your model (refer to the of the ``readme`` file for the full procedure to follow for SSL training)

2. Open the file ``unbalanced_supervised_learning.py``
   - At line 191 turn ``logging_on`` to True if you need to log the data to WandB, otherwise check it to be False
   - At line 224 turn ``New_SSL_methods`` to True if you have used a different SSL techinque (see Experiment 1), in that case, specify at line 235 which model you want to load. Otherwise check it to be False
   - At line 310 verify that ``unbalanced`` is set to True to use the unbalanced version of CustomCIFAR10
   - At line 312 define your own ``remove_dict`` that will be applied to CustomCIFAR10

3. From cmd run the following line of code to perform the supervised_learning (change the parameters ``name_of_your_input_model`` and ``name_of_your_output_model``):
   ```shell
      CUDA_VISIBLE_DEVICES=0 python unbalanced_supervised_learning.py --ssl_weights_dir ./data/experiments/...../name_of_your_input_model.pth --model_name  name_of_your_output_model --new_resnet
   ```
   The flag ``new_resnet`` is used to turn on its respective option
      - Do not use the flag ``--new_resnet`` if your model has been trained using the ResNet defined by the AutoNovel authors. Use that flag if your model has been trained using a standard ResNet (as from ResNet original paper)

4. Your trained model weights will be stored in ``data/experiments/unbalanced_supervised_learning/name_of_your_output_model.pth``

5. Once the ``unbalanced_supervised_learning-step`` is finished, open the file ``unabalanced_auto_novel.py``
   - At line 410 turn ``logging_on`` to True if you need to log the data to WandB, otherwise check it to be False
   - At line 430 turn ``New_Resnet`` to True if you used a standard ResNet, check it to be False if you used the ResNet defined by the authors
   - At line 486 turn ``unbalanced`` to True to use the unbalanced version of CustomCIFAR10
   - At line 488 define your own ``remove_dict`` that will be applied to CustomCIFAR10

6. Depending on the Incremental-Learning (IL) setting that you want to use to train your model:
   - If IL enabled -> run ``auto_novel_IL_cifar10_unbalanced.sh`` through cmd using the following line of code (change the parameters ``name_of_your_input_model`` and ``name_of_your_output_model``):
   ```shell
      CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10_unbalanced.sh ./data/experiments/...../name_of_your_input_model.pth name_of_your_output_model
   ```

   - If IL disabled -> run ``auto_novel_cifar10_unbalanced`` through cmd using the following line of code change the parameters ``name_of_your_input_model`` and ``name_of_your_output_model``):
   ```shell
      CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_cifar10_unbalanced.sh ./data/experiments/...../name_of_your_input_model.pth name_of_your_output_model
   ```

7. Your trained model weights will be stored in ``data/experiments/unbalanced_auto_novel/name_of_your_output_model.pth``

## Experiment 7 (Different number of unlabeled/labeled classes)

Useful to understand the impact of unbalanced domain with different number of labeled and unlabeled classes on the performance of the model.

1. run ``supervised_learning.py`` specifying the number of labeled and unlabaled data with ``--num_labeled_classes <num_lab> --num_unlabeled_classes <num_unlab>``
2. run 

 ```shell
 auto_novel.py --dataset_name cifar10 --warmup_model_dir <checkpoint supervised model> --dataset_root <dataset directory> --model_name <model_name>  --num_labeled_classes <num_lab> --num_unlabeled_classes <num_unlab> --lr 0.1  --gamma 0.1 --weight_decay 1e-4 --step_size 170 --batch_size 128 --epochs 200 --rampup_length 50 --rampup_coefficient 5.0 --seed 0 --mode train
```

## Plotting t-SNE graph for any experiment
The t-distributed Stochastic Neighbor Embedding is a statistical tool that allows to represent high dimensional samples into a low dimensional space relying on a statistical algorithm. Due to its stochastic nature this algorithm leads to different output for each run, also if the input data and the used parameters are exactly the same.

We used the t-SNE plots to show how much the features learned by our models are effective. They allow us to see how the samples belonging to different categories are clustered. Ideally, we would like to see compact cluster well distatiented between them. This condition would point that our model learnt some good features which allows to distinguish between samples coming from different classes in an efficient way.

To plot the t-SNE for your model follow the ensuing procedure (steps using CIFAR-10 as dataset):

1. Train your model until the end of the ``AutoNovel-step`` and store the weights of your model

2. Put the weights of your model into the path ``data/experiments/auto_novel_tSNE_plot/name_of_your_input_model.pth``

3. To start the generation of the t-SNE plots you will need to use the following command line from cmd (change the parameter ``name_of_your_input_model`` with the name of the model weights that you want to load from ``data/experiments/auto_novel_tSNE_plot``):
   ```shell
   CUDA_VISIBLE_DEVICES=0 python auto_novel_tSNE_plot.py --input_model_name name_of_your_input_model --new_resnet --IL 
   ```
   The two flags (``--new_resnet`` and ``--IL``) are used to turn on the respective options
      - Do not use the flag ``--new_resnet`` if your model has been trained using the ResNet defined by the AutoNovel authors. Use that flag if your model has been trained using a standard ResNet (as from ResNet original paper)
      - Use the flag ``--IL`` if your model has been trained in the ``AutoNovel-step`` using Incremental-Learning, otherwise, do not use this flag

4. The produced plots will be stored in the folder ``tSNE_plots/name_of_you_model``

5. If you are working on a dataset different from CIFAR-10, or if other changes have been applied on the training procedure, then apply the due changes also to the py file ``auto_novel_tSNE_plot.py``
***

# Experiments performed on UNO

## Experiment 1 (Different number of unlabeled/labeled classes)

Useful to understand the impact of unbalanced domain with different number of labeled and unlabeled classes on the performance of the model.

1. run ``main_pretrain.py`` specifying the number of labeled and unlabaled data with ``--num_labeled_classes <num_lab> --num_unlabeled_classes <num_unlab>``
2. run ``main_discover.py`` specifying the number of labeled and unlabaled data with ``--num_labeled_classes <num_lab> --num_unlabeled_classes <num_unlab>``
