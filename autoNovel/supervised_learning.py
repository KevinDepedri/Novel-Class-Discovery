import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter, accuracy
from models.resnet import ResNet, BasicBlock, resnet_sim
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from data.svhnloader import SVHNLoader
from tqdm import tqdm
import numpy as np
import os
import wandb
from data.MNISIT_loader import MNISITLoader, MNISITLoaderMix

global logging_on

'''
Self supervised learning (as from section 2.1 of AutoNovel paper) - part 2
The function ηl ◦ Φ is fine-tuned on the labelled dataset Dl in order to learn a classifier for the Cl known classes,
this time using the labels yi and optimizing the standard cross-entropy (CE) loss.
Only ηl and the last macro-block of Φ (section 3) are updated in order to avoid over-fitting the representation to the 
labelled data. We freeze the first 3 block and fine-tune last block with the classifier in supervised learning setting
'''


def train(model, train_loader, labeled_eval_loader, args):
    # Instantiate SGD optimizer with input learning rate, momentum and weight_decay
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Instantiate a learning rate scheduler to adjust the learning rate based on the number of epochs. In this case it
    # is set on the StepLR setting. Decays the learning rate of each parameter group by gamma every step_size epochs.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Instantiate a standard cross entropy loss
    criterion1 = nn.CrossEntropyLoss()

    # Iterate for each epoch
    for epoch in range(args.epochs):
        # Define an instance of AverageMeter to compute and store the average and current values of the loss
        loss_record = AverageMeter()
        accuracy_record = AverageMeter()
        # Set the model in the training mode
        model.train()

        # Iterate for each batch in the dataloader
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            # Dataloader contain: x(input sample), label(sample label), idx(index of sample in the original dataset)
            # We are interested in using just the sample x and its label to perform supervised learning
            x, label = x.to(device), label.to(device)

            # Output1, Output2 and Output3 are the results of Head1, Head2, and features-layer4 respectively, since
            # we want to perform supervised training, we are just interested in the output of Head1
            output1, _, _ = model(x)

            # Compute the CE loss and update the loss AverageMeter with that value of loss
            loss = criterion1(output1, label)
            loss_record.update(loss.item(), x.size(0))
            acc = accuracy(output1, label)  # calculating the accuracy
            accuracy_record.update(acc[0].item(), x.size(0))

            # Zero the gradient of the optimizer, back-propagate the loss and perform an optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Perform a step on the input exp_lr_scheduler (scheduler used to define the learning rate)
        exp_lr_scheduler.step()

        # Print the result of the training procedure over that epoch
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        # Set the head argument to 'head1', this to ensure that we test the supervised head in the training step below
        print('test on labeled classes')
        args.head = 'head1'
        acc_H1, nmi_H1, ari_H1, acc_testing_H1 = test(model, labeled_eval_loader, args)
        if logging_on:
            wandb.log({"epoch": epoch, "Total_average_loss": loss_record.avg,
                       "Head_1_training_accuracy": accuracy_record.avg,
                       "cluster_acc_Head_1": acc_H1, "nmi_Head_1": nmi_H1, "ari_Head_1": ari_H1,
                       "testing_acc_Head_1": acc_testing_H1,
                       "lr": exp_lr_scheduler.get_last_lr()[0]}, step=epoch)


def test(model, test_loader, args):
    # Put the model in evaluation mode
    model.eval()
    # Instantiate two numpy arrays, one for predictions and oen for targets
    preds = np.array([])
    targets = np.array([])
    acc_record = AverageMeter()  # track the accuracy of the first head

    # Iterate for each batch in the dataloader
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        # Dataloader contain: X(input sample), label(sample label), index(index of sample in the original dataset)
        # We are interested in using just the sample x and its label to perform supervised learning
        x, label = x.to(device), label.to(device)

        # Output 1, Output2 and Output3 are the results of Head1, Head2, and features-layer4 respectively, we take the
        # outputs of the two heads since we are interested in testing the accuracy of one of the two
        output1, output2, _ = model(x)

        # If the argument head is 'head1' then we take as final output the result of the supervised head
        if args.head == 'head1':
            # output1 has size of (128,5), since the batch_size is 128 and the possible classes are 5
            output = output1
        # Otherwise, we take as final output the result of the unsupervised head
        else:
            output = output2

        # Returns the maximum value for each element in the input tensor, therefore we move from size (128,5) to (128)
        # Here we are not interested in the value, so we put '_' for the first term. We are interested in the second
        # term, which is the index of that value, since the index is equal to the predicted class for that input sample.
        _, pred = output.max(1)
        acc_testing = accuracy(output, label)  # calculating the accuracy
        acc_record.update(acc_testing[0].item(), x.size(0))
        # Convert tensor to numpy using 'label.cpu.numpy', then append the value in the respective numpy array
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    # Compute the accuracy metrics for the current test step
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets,
                                                                                                              preds)
    # The used metrics are:
    # -----------------------------------------------------------------------------------------------------------------
    # 1) CLUSTER ACCURACY (cluster_acc), it is a function written in utils.py files
    # -----------------------------------------------------------------------------------------------------------------
    # 2) NORMALIZED MUTUAL INFORMATION (nmi_score), it is a sci-learn function
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
    # Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to scale the results
    # between 0 (no mutual information) and 1 (perfect correlation). In this function, mutual is normalized by some
    # generalized mean of H(labels_true) and H(labels_pred)), defined by the average_method. This metric is independent
    # of the absolute values of the labels: a permutation of the class or cluster label values
    # won’t change the score value in any way.
    # -----------------------------------------------------------------------------------------------------------------
    # 3) ADJUSTED RAND SCORE (ari_score), it is a sci-learn function
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    # Rand index adjusted for chance.
    # The Rand Index computes a similarity measure between two clustering by considering all pairs of
    # samples and counting pairs that are assigned in the same or different clusters in the predicted and true
    # clustering. The raw RI score is then “adjusted for chance” into the ARI score using the following scheme:
    # ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    # The adjusted Rand index is thus ensured to have a value close to 0.0 for random labeling independently
    # of the number of clusters and samples and exactly 1.0 when the clustering are identical (up to a permutation).
    # Similarity score between -1.0 and 1.0.

    # ANOTHER EXPLANATION:
    # The Adjusted Rand score is introduced to determine whether two cluster results are similar to each other.
    # In the formula, the “RI” stands for the rand index, which calculates a similarity between two cluster results
    # by taking all points identified within the same cluster. This value is equal to 0 when points are assigned into
    # .clusters randomly, and it equals to 1 when the two cluster results are same.
    # This metric is used to evaluate whether dimension-reduced similarity cluster results are similar to one other.
    # What is the adjusted Rand index?
    # The adjusted Rand index is the corrected-for-chance version of the Rand index. Such a correction for chance
    # establishes a baseline by using the expected similarity of all pair-wise comparisons between clustering specified
    # by a random model.
    # https://baiadellaconoscenza.com/dati/argomento/read/9864-what-is-the-adjusted-rand-index#question-0
    # What does a negative adjusted Rand index mean?
    # Negative ARI says that the agreement is less than what is expected from a random result.
    # This means the results are 'orthogonal' or 'complementary' to some extend.
    # -----------------------------------------------------------------------------------------------------------------

    # Print the result of the testing procedure obtained computing the three metrics above
    print('Test cluster acc {:.4f}, nmi {:.4f}, ari {:.4f}, test accuracy {:.4f}'.format(acc, nmi, ari, acc_record.avg))
    return acc, nmi, ari, acc_record.avg


if __name__ == "__main__":
    import argparse

    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Get all the needed input arguments
    parser.add_argument('--lr', type=float, default=0.1)  # Learning rate of optimizer
    parser.add_argument('--gamma', type=float, default=0.5)  # Gamma of the learning rate scheduler
    parser.add_argument('--momentum', type=float, default=0.9)  # Momentum term of optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # Weight decay of optimizer
    parser.add_argument('--epochs', default=100, type=int)  # Number of epochs
    parser.add_argument('--step_size', default=10, type=int)  # Step size of learning rate scheduler
    parser.add_argument('--batch_size', default=128, type=int)  # Batch size
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)  # Number of unlabeled classes
    parser.add_argument('--num_labeled_classes', default=5, type=int)  # Number of labeled classes
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')  # Dataset root directory
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')  # Directory to save the resulting files
    parser.add_argument('--rotnet_dir', type=str,
                        default='./data/experiments/selfsupervised_learning/rotnet_cifar10.pth')  # Directory to find the semi-supervised pretrained model
    parser.add_argument('--model_name', type=str, default='resnet_rotnet')  # Name of the model
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='options: cifar10, cifar100, svhn')  # Name of the used dataset
    parser.add_argument('--mode', type=str, default='train')  # Mode: train or test
    # Extract the args and make them available in the args object
    args = parser.parse_args()
    logging_on = True
    if logging_on:
        wandb.login()  # 4619e908b2f2c21261030dae4c66556d4f1f3178
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "dataset": args.dataset_name,
            "unlabled_classes": args.num_unlabeled_classes,
            "labled_classes": args.num_labeled_classes,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "step_size": args.step_size,
            "mode": args.mode
        }
        wandb.init(project="trends_project", entity="mhaggag96", config=config, save_code=True)
    # Define if cuda can be used and initialize the device used by torch
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Returns the current file name. In this case file is called supervised_learning.py,
    runner_name = os.path.basename(__file__).split(".")[0]
    # Define the name of the wanted directory as 'experiment root' + 'name of current file'
    model_dir = os.path.join(args.exp_root, runner_name)
    # If the previously defined directory does not exist, them create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Define the name of the path to save the trained model

    args.model_dir = model_dir + '/' + '{}.pth'.format(args.model_name)
    '''
    Changing the New_SSL_methods will turn you from the normal rot net to 
    using other self supervised learning methods
    '''
    New_SSL_methods = True
    New_Resnet_config = True
    # model = resnet_sim(args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    if New_SSL_methods:
        model = resnet_sim(args.num_labeled_classes, args.num_unlabeled_classes).to(device)
        '''
        I am using barlow twins pre loading. you can use different kind of weights
        '''
        ssl = 'swav'
        print("We are working with this self learning method " + ssl)
        if ssl == 'Barlow_twins':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_Barlow_twins_2
            state_dict = \
            torch.load('trained_models/cifar10/barlow_twins/barlow-cifar10-otu5cw89-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        elif ssl == 'simsiam':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_simsam
            state_dict = \
            torch.load('trained_models/cifar10/simsiam/simsiam-cifar10-252e1tvw-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        elif ssl == 'supcon':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_supcon
            state_dict = \
            torch.load('trained_models/cifar10/supcon/supcon-cifar10-1w8chdt4-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        elif ssl == 'swav':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_swav
            state_dict = torch.load('trained_models/cifar10/swav/swav-2rwotcpy-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        elif ssl == 'vibcreg':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_vibcreg
            state_dict = \
            torch.load('trained_models/cifar10/vibcreg/vibcreg-cifar10-3ehq2v3f-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        elif ssl == 'vicreg':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_vicreg
            state_dict = \
            torch.load('trained_models/cifar10/vicreg/vicreg-cifar10-qx5zahvt-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        elif ssl == 'wmse':
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py   --dataset_name cifar10 --model_name resnet_rotnet_cifar10_wmse
            state_dict = \
            torch.load('trained_models/cifar10/wmse/wmse-cifar10-6z3m2p9o-ep=999.ckpt', map_location="cpu")[
                "state_dict"]
        for k in list(state_dict.keys()):
            if "encoder" in k:
                state_dict[k.replace("encoder", "backbone")] = state_dict[k]
            if "backbone" in k:
                state_dict['encoder.' + k.replace("backbone.", "")] = state_dict[k]
            del state_dict[k]
    else:
        #    Initialize ResNet architecture and also the BasicBlock, which are imported from resnet.py. Then send to cuda
        if New_Resnet_config:
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py --rotnet_dir ./data/experiments/selfsupervised_learning/rotnet_cifar10_new_config.pth  --dataset_name cifar10 --model_name resnet_rotnet_cifar10_new_config
            model = resnet_sim(args.num_labeled_classes, args.num_unlabeled_classes).to(device)
            state_dict = torch.load(args.rotnet_dir)
            del state_dict['head1.weight']  # Size of the old head was [4,512]
            del state_dict['head1.bias']  # Deleted not only weights but also the biases [4]
        else:
            # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py --rotnet_dir ./data/experiments/selfsupervised_learning/rotnet_cifar10_basicconfig.pth --dataset_name cifar10 --model_name resnet_rotnet_cifar10_basicconfig
            model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
            # Default inputs assume that we are working with 10 classes of which: 5 classes unlabeled 5 classes labeled.
            # We have two heads in this ResNet model, head1 for the labeled and head2 for the unlabeled data.
            # Load the weights for the ResNet model from the self-supervised previously trained model (selfsupervised_learning.py)
            state_dict = torch.load(args.rotnet_dir)
            # Delete the old linear head parameters. It was used just to perform semi-supervised learning (to predict rotation)
            del state_dict['linear.weight']  # Size of the old head was [4,512]
            del state_dict['linear.bias']  # Deleted not only weights but also the biases [4]
            # After this operation we no longer have any weights or biases in the end. They are completely deleted, we are ready
            # to learn the weights for the two new heads.
        # Apply the loaded weights to the model, we do not strictly enforce that the keys in state_dict match since the old
        # model has one head that was removed, while the new model has two new heads. Therefore, hey cannot fully match
    model.load_state_dict(state_dict, strict=False)

    # # Compute the total number of classes
    # # Iterate through all the parameters of the new model
    for name, param in model.named_parameters():
        # If the parameter under analysis does not belong to 'head' (one of the two heads) or to 'layer4' (features
        # layer before the two heads), then freeze that parameter. In this way we are ensuring that all the parameters
        # will be frozen except for the two heads and the features layer, which we want to train.
        if 'head' not in name and 'layer4' not in name:
            param.requires_grad = False
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # If the dataset argument is 'cifar10' then use its apposite loader, see cifarloader.py for full explanation
    if args.dataset_name == 'cifar10':
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                            aug=None, shuffle=False, target_list=range(args.num_labeled_classes))
    # Otherwise, if the dataset argument is 'cifar100' then use its apposite loader
    elif args.dataset_name == 'cifar100':
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                             aug=None, shuffle=False, target_list=range(args.num_labeled_classes))
    # Otherwise, if the dataset argument is 'svhn' then use its apposite loader
    elif args.dataset_name == 'svhn':
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                          aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                         aug=None, shuffle=False, target_list=range(args.num_labeled_classes))
    elif args.dataset_name == 'mnisit':
        labeled_train_loader = MNISITLoader(batch_size=args.batch_size, split='train',
                                          aug='once', shuffle=True,number_of_classes=5 )
        labeled_eval_loader = MNISITLoader(batch_size=args.batch_size, split='test',
                                         aug=None, shuffle=False, number_of_classes=5)
        # CUDA_VISIBLE_DEVICES=0 python supervised_learning.py --rotnet_dir ./data/experiments/self_super_mnisi/rotnet_mnisit_MIXMIX.pth --dataset_name mnisit --model_name resnet_rotnet_mnisit_MIX

    # Finally, if the mode argument is 'train', then run the training procedure
    if args.mode == 'train':
        # Perform the training over the defined model
        train(model, labeled_train_loader, labeled_eval_loader, args)
        # Save the model to the specified path
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    # Otherwise, if the mode argument is 'test', then run the testing procedure
    elif args.mode == 'test':
        # Load the model from the specified path
        print("model loaded from {}.".format(args.model_dir))
        # Perform the testing over the loaded model
        model.load_state_dict(torch.load(args.model_dir))
    # In the end, test the model using head1 over the labeled dataloader
    print('test on labeled classes')
    args.head = 'head1'
    test(model, labeled_eval_loader, args)
    if logging_on:
        wandb.finish()
