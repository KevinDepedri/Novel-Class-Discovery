import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, accuracy
from utils import ramps
from models.resnet import ResNet, BasicBlock, resnet_sim
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.svhnloader import SVHNLoader, SVHNLoaderMix
from tqdm import tqdm
import numpy as np
import os
import wandb
import random

# global current_epoch
# Auto-novel training without incremental learning (IL)
global logging_on


def train(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    # Instantiate SGD optimizer with input learning rate, momentum and weight_decay
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Instantiate a learning rate scheduler to adjust the learning rate based on the number of epochs. In this case it
    # is set on the StepLR setting. Decays the learning rate of each parameter group by gamma every step_size epochs.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Define two losses that share the same embedding, which is the output of the features layer (layer4). The two
    # losses are applied jointly instead of sequentially to avoid the phenomenon known as catastrophic forgetting.
    # Instantiate a standard cross entropy loss for the labeled data classification
    criterion1 = nn.CrossEntropyLoss()
    # Instantiate a binary cross entropy loss for the unlabeled data classification
    criterion2 = BCE()

    # Iterate for each epoch
    for epoch in range(args.epochs):
        # Define an instance of AverageMeter to compute and store the average and current values of the loss
        # current_epoch = epoch# setting the global variable to the current epoch 
        loss_record = AverageMeter()
        loss_record_CEL = AverageMeter()  # average meter to follow the cross entropy loss
        loss_record_BCE = AverageMeter()  # average meter to follow the binary cross entropy loss
        loss_record_CON_1 = AverageMeter()  # average meter to follow the first parameter of consistency loss
        loss_record_CON_2 = AverageMeter()  # average meter to follow the second parameter of consistency loss
        loss_record_CON_total = AverageMeter()  # average meter to follow the total of consistency loss
        acc_record = AverageMeter()  # track the accuracy of the first head

        # Set the model in the training mode
        model.train()

        # Define the rump-up function that will be used as multiplier of the consistency loss.
        # Here by default the ramp-up function is a sigmoid while the ramp-up  length it is set to 150
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

        # Iterate for each batch in the dataloader
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            # The MixDataLoader contains: x(input sample), x_bar(augmented input sample), label(sample label),
            # idx(index of sample in the original dataset)
            # We are interested in all of them except for idx which is never used.
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)

            # Output1, Output2 and feat are the results of Head1, Head2, and features-layer4 for the base samples
            output1, output2, feat = model(x)  # Outputs of the two heads are each (128,5)
            # Output1_bar, Output2_bar are the results of Head1, Head2, and features-layer4 for the augmented samples
            output1_bar, output2_bar, _ = model(x_bar)  # getting output of 2 heads

            # Apply softmax on all the computed outputs to turn everything into probabilities
            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), \
                                                 F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)

            # Compute a tensor mask of true and false with the size of the batch size. Here, for each example in the
            # batch_size we will have a 'true' value when its label has a value lower than the num_labeled_classes (5),
            # this means that the label is linked with a labeled example. Otherwise, it takes value 'false', meaning
            # that this label is linked with an unlabeled example, which should be without label.
            # The mask will then be used in the next line of code to remove the labeled samples from our samples batch
            mask_lb = label < args.num_labeled_classes

            # Apply the tensor mask to feat, which is the features layer before the heads, and it has shape of (128,512)
            # Applying the tensor mask is equivalent to slicing feat, taking only its vectors which are associated with
            # a 'True' tensor value in the tensor mask. The result is that we are taking all the unlabeled data
            # Now, we apply 'detach' to obtain a Tensor, detached from the current graph since we will not require
            # gradient. The result 'rank_feat' are the features of the unlabeled samples that we want to compare between
            # different unlabeled samples to define if they belong to the same class or not.
            # After applying this operation we are left with a tensor of shape approximately (60-70, 512), this because
            # each batch size is composed of 128 examples, and of these more or less half will be labeled. The labeled
            # samples are removed. Therefore, we are left with more or less half of the batch_size.
            rank_feat = (feat[~mask_lb]).detach()  # Here ~ is to turn each true to false, as if it is flipping
            # rank_feat are the subset of unlabeled features for first augmentation
            # rank_feat becomes z_{i}^{u}
            # we rank the values in vector z_{i}^{u} by magnitude.

            # Now we sort internally, in descending order, each vector which has survived to the splicing. In this way
            # each vector which is linked with an unsupervised sample will have all its components ordered from
            # the biggest to the smallest. Then we convert the tensor values to indexes. We are left with sorted indexes
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)  # Example: rank_idx has dimension (68,512)
            # for example :
            # a =tensor([[8.8466e-01, 7.5498e-01, 5.3452e-01, 2.3279e-01, 4.1647e-01],
            #     [5.7411e-02, 2.8372e-04, 1.0783e-02, 5.9285e-01, 4.2223e-01],
            #     [6.2338e-01, 2.0106e-02, 6.4205e-01, 9.2479e-01, 4.8820e-01],
            #     [1.1800e-01, 6.9137e-01, 6.6532e-01, 5.5088e-01, 1.9753e-01]])
            # when we sort we say in descending order
            # tensor([[0, 1, 2, 4, 3],
            #     [3, 4, 0, 2, 1],
            #     [3, 2, 0, 4, 1],
            #     [1, 2, 3, 4, 0]])

            # Generate two matrices
            rank_idx1, rank_idx2 = PairEnum(rank_idx)  # mask is set to none
            # rank_idx1 (4624,512) repeated variable. imagine that you have 68 pictures features, and we repeat them
            # 68 times so all the whole variable is copied and put 64 times.
            # rank_idx2. each feature vector size 512 I repeat it 68 times, so I have (68,34816)=(68,68*512). then
            # i reshape it to (-1,512) so that the final vector becomes  size of [4624, 512]
            # if you don't understand the things above ignore it and look at example to see the kind of output
            # returned in here. Example if you have the following tensor x
            #      x =tensor([[1, 2,3, 4, 5],
            #        [6, 7, 8, 9, 10],
            #        [11, 12, 13, 14, 15]]) it is tensor 3,5
            # x1,x2 =PairEnum(x)
            # x1 will be size of (9,5) and it will look like this
            # x1 =tensor([[1, 2,3, 4, 5]image1,
            #        [6, 7, 8, 9, 10]image2,
            #        [11, 12, 13, 14, 15]image3,
            #        [1, 2,3, 4, 5]image,
            #        [6, 7, 8, 9, 10],
            #        [11, 12, 13, 14, 15],
            #        [1, 2,3, 4, 5],
            #        [6, 7, 8, 9, 10],
            #        [11, 12, 13, 14, 15]])
            # x2 will be size of (9,5) and it looks like this
            # x2 =tensor([[1, 2,3, 4, 5]image1,
            #             [1, 2,3, 4, 5]image1,
            #             [1, 2,3, 4, 5]image1,
            #             [6, 7, 8, 9, 10]image2,
            #             [6, 7, 8, 9, 10]image2,
            #             [6, 7, 8, 9, 10]image2,
            #             [11, 12, 13, 14, 15],
            #             [11, 12, 13, 14, 15],
            #             [11, 12, 13, 14, 15]])

            # Slice the rank_idx previously computed and take the top-5 elements. Now the tensor has this size (4624,5)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]

            # Sorts the elements of the input tensor along a given dimension in ascending order by value on both tensors along dimension 1
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            # Compute the difference between the previously computed sorted rank. This matrix to have alot of 0s and some positive and some negative numbers.
            rank_diff = rank_idx1 - rank_idx2
            # Sum the elements of the input tensor along a given dimension, in this case dimension 1 (which as size 5),
            # applying abs operation accord each variable in the matrix, so you shouldn't have any negative number
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)  # Example: output tensor (1,4624)

            # Instantiate an array of ones (1) with the same shape and type as a given array (1,4624)
            target_ulb = torch.ones_like(rank_diff).float().to(device)  # Example: tensor (1,4624)

            # Change the ones (1) to minus one (-1) if the rank_difference in that position is greater than 0, it means
            # that the two compared features has not the same values. We want to keep track of that since it means that
            # it is less probable that the two samples belong to the same class
            # Therefore, target_ulb is a matrix of 1 and -1 if Sij: 1->similar; -1->dissimilar; 0->unknown(ignore)
            target_ulb[rank_diff > 0] = -1
            # target_ulb is Sij. It is defined as Sij = 1 {topk(Φ(xu i )) = topk(Φ(xu j ))}

            # Extract the probabilities of the unlabeled sample computed by the second head for the base sample
            # We use the tensor mask_lb to keep only the probabilities for the unlabeled sample, cutting out the others
            # Then pass the resulting tensor to PairEnum which leads to a tensor much bigger, in this example (4624,5)
            prob1_ulb, _ = PairEnum(prob2[~mask_lb])  # Example: tensor (62,5) --> (4624,5)

            # Do the same as above. Extract the probabilities of the unlabeled sample computed by the second head for
            # the first augmentation, then slice using the tensor mask, finally apply PairEnum
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])  # Example: tensor (62,5) --> (4624,5)

            # Compute the Cross entropy (CE) loss over the labeled sample (using the tensor mask to slice them)
            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            acc = accuracy(output1[mask_lb], label[mask_lb])  # calculating the accuracy
            acc_record.update(acc[0].item(), x.size(0))
            # Compute the Binary Cross entropy (BCE) loss over the labeled sample (using the tensor mask to slice them)
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            # Compute the Mean Squared error (MSE) loss used as consistency loss between base samples and augmented ones

            # consistency loss start
            consistency_loss_c1 = F.mse_loss(prob1, prob1_bar)
            consistency_loss_c2 = F.mse_loss(prob2, prob2_bar)
            consistency_loss = consistency_loss_c1 + consistency_loss_c2
            # consistency loss end

            # consistency_loss =F.mse_loss(prob1, prob1_bar)+F.mse_loss(prob2, prob2_bar)
            # Add up, apply weights and compute the final loss. Then update the loss AverageMeter with that value
            # original loss
            # alpha = random.uniform(0, 1)
            # beta  = 1-alpha
            loss = loss_ce + loss_bce + w * consistency_loss
            loss_record.update(loss.item(), x.size(0))
            loss_record_CEL.update(loss_ce.item(), x.size(0))
            loss_record_BCE.update(loss_bce.item(), x.size(0))
            loss_record_CON_1.update(consistency_loss_c1.item(), x.size(0))
            loss_record_CON_2.update(consistency_loss_c2.item(), x.size(0))
            loss_record_CON_total.update(consistency_loss.item(), x.size(0))
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

        # Set the head argument to 'head2', this to ensure that we test the unsupervised head in the training step below
        print('test on unlabeled classes')
        args.head = 'head2'
        acc_H2, nmi_H2, ari_H2, _ = test(model, unlabeled_eval_loader, args)
        # Print the result of the testing procedure obtained computing the three metrics above
        if logging_on:
            wandb.log({"epoch": epoch, "Total_average_loss": loss_record.avg, "Cross_entropy_loss": loss_record_CEL.avg,
                       "Binary_cross_entropy_loss": loss_record_BCE.avg,
                       "Consistency_loss_part_a": loss_record_CON_1.avg,
                       "Consistency_loss_part_b": loss_record_CON_2.avg,
                       "Consistency_loss_total": loss_record_CON_total.avg,
                       "Head_1_training_accuracy": acc_record.avg,
                       "cluster_acc_Head_1": acc_H1, "nmi_Head_1": nmi_H1, "ari_Head_1": ari_H1,
                       "testing_acc_Head_1": acc_testing_H1,
                       "cluster_acc_Head_2": acc_H2, "nmi_Head_2": nmi_H2, "ari_Head_2": ari_H2,
                       "lr": exp_lr_scheduler.get_last_lr()[0]}, step=epoch)


def train_IL(model, train_loader, labeled_eval_loader, unlabeled_eval_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        loss_record_CEL = AverageMeter()  # average meter to follow the cross entropy loss
        loss_record_BCE = AverageMeter()  # average meter to follow the binary cross entropy loss
        loss_record_CON_1 = AverageMeter()  # average meter to follow the first parameter of consistency loss
        loss_record_CON_2 = AverageMeter()  # average meter to follow the second parameter of consistency loss
        loss_record_CON_total = AverageMeter()  # average meter to follow the total of consistency loss
        loss_record_IL = AverageMeter()  # average meter to follow the incremental learning

        acc_record = AverageMeter()  # track the accuracy of the first head
        model.train()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(
                output2, dim=1), F.softmax(output2_bar, dim=1)
            # we transfer them to probabilities
            mask_lb = label < args.num_labeled_classes  # select unlabeled data features.

            rank_feat = (feat[~mask_lb]).detach()

            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]  # select topk between 2 samples

            rank_idx1, _ = torch.sort(rank_idx1, dim=1)  # we sort them.
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)  # it indicates they are exactly same so they set
            # their label pairwise to 1.
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1  # otherwise we set them to -1. they belong to different classes

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])  # crosses entropy loss for supervised part
            # all above very similar to before 

            acc = accuracy(output1[mask_lb], label[mask_lb])  # calculating the accuracy
            acc_record.update(acc[0].item(), x.size(0))
            # mask_lb used to access unlabeled stuff.
            # (output2[~mask_lb]).detach().max(1)[1] you have 62*5 tensor you removed gradient return the biggest tensor,
            # but I want to return the index of biggest not the values, so I expect to have values between 0 and 5
            # then I add 5, so I get the new labeled of unlabeled class to be
            label[~mask_lb] = (output2[~mask_lb]).detach().max(1)[1] + args.num_labeled_classes  # 5 +

            # calculating loss entropy between output 1 and labels pseudo for the unlabelled data we use the
            # pseudo-labels ˆ yu i , which are generated on-the-fly from the head ηu at each forward pass
            loss_ce_add = w * criterion1(output1[~mask_lb],
                                         label[~mask_lb]) / args.rampup_coefficient * args.increment_coefficient

            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)  # binary cross entropy
            consistency_loss_c1 = F.mse_loss(prob1, prob1_bar)
            consistency_loss_c2 = F.mse_loss(prob2, prob2_bar)
            consistency_loss = consistency_loss_c1 + consistency_loss_c2
            loss = loss_ce + loss_bce + loss_ce_add + w * consistency_loss

            loss_record.update(loss.item(), x.size(0))
            loss_record_IL.update(loss_ce_add.item(), x.size(0))
            loss_record_CEL.update(loss_ce.item(), x.size(0))
            loss_record_BCE.update(loss_bce.item(), x.size(0))
            loss_record_CON_1.update(consistency_loss_c1.item(), x.size(0))
            loss_record_CON_2.update(consistency_loss_c2.item(), x.size(0))
            loss_record_CON_total.update(consistency_loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        exp_lr_scheduler.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        acc_H1, nmi_H1, ari_H1, acc_testing_H1 = test(model, labeled_eval_loader, args)
        print('test on unlabeled classes')
        args.head = 'head2'
        acc_H2, nmi_H2, ari_H2, _ = test(model, unlabeled_eval_loader, args)
        if logging_on:
            wandb.log({"epoch": epoch, "Total_average_loss": loss_record.avg, "Cross_entropy_loss": loss_record_CEL.avg,
                       "Binary_cross_entropy_loss": loss_record_BCE.avg,
                       "Consistency_loss_part_a": loss_record_CON_1.avg,
                       "Consistency_loss_part_b": loss_record_CON_2.avg,
                       "Consistency_loss_total": loss_record_CON_total.avg,
                       "Head_1_training_accuracy": acc_record.avg,
                       "cluster_acc_Head_1": acc_H1, "nmi_Head_1": nmi_H1, "ari_Head_1": ari_H1,
                       "testing_acc_Head_1": acc_testing_H1,
                       "cluster_acc_Head_2": acc_H2, "nmi_Head_2": nmi_H2, "ari_Head_2": ari_H2,
                       "lr": exp_lr_scheduler.get_last_lr()[0],
                       "incremental_loss": loss_record_IL.avg}, step=epoch)


def test(model, test_loader, args):
    # Put the model in evaluation mode
    model.eval()
    # Instantiate two numpy arrays, one for predictions and oen for targets
    preds = np.array([])
    targets = np.array([])
    acc_record = AverageMeter()  # track the accuracy of the first head
    # loss_record = AverageMeter()

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
            # label-=args.num_labeled_classes

        # Returns the maximum value for each element in the input tensor, therefore we move from size (128,5) to (128)
        # Here we are not interested in the value, so we put '_' for the first term. We are interested in the second
        # term, which is the index of that value, since the index is equal to the predicted class for that input sample.
        _, pred = output.max(1)

        # Convert tensor to numpy using 'X.cpu.numpy', then append the value in the respective numpy array
        if args.head == 'head1':
            acc_testing = accuracy(output, label)  # calculating the accuracy
            acc_record.update(acc_testing[0].item(), x.size(0))
        else:
            acc_testing = 0
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    # Compute the accuracy metrics for the current test step, see supervised_learning.py for full explanation
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets,
                                                                                                              preds)

    print('Test cluster acc {:.4f}, nmi {:.4f}, ari {:.4f}, test accuracy {:.4f}'.format(acc, nmi, ari, acc_record.avg))
    return acc, nmi, ari, acc_record.avg


if __name__ == "__main__":
    import argparse

    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Get all the needed input arguments
    parser.add_argument('--lr', type=float, default=0.1)  # Learning rate of optimizer
    parser.add_argument('--gamma', type=float, default=0.1)  # Gamma of the learning rate scheduler
    parser.add_argument('--momentum', type=float, default=0.9)  # Momentum term of optimizer
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # Weight decay of optimizer
    parser.add_argument('--epochs', default=200, type=int)  # Number of epochs
    parser.add_argument('--rampup_length', default=150, type=int)  # Ramp-up length passed to ramps.py function
    parser.add_argument('--rampup_coefficient', type=float, default=50)  # Ramp-up coefficient
    parser.add_argument('--increment_coefficient', type=float, default=0.05)  # Incremental learning coefficient
    parser.add_argument('--step_size', default=170, type=int)  # Step size of learning rate scheduler
    parser.add_argument('--batch_size', default=128, type=int)  # Batch size
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)  # Number of unlabeled classes
    parser.add_argument('--num_labeled_classes', default=5, type=int)  # Number of labeled classes
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')  # Dataset root directory
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')  # Directory to save the resulting files
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth')  # Directory to find the supervised pretrained model
    parser.add_argument('--topk', default=5, type=int)  # Number of top elements that we want to compare
    parser.add_argument('--IL', action='store_true', default=False, help='w/ incremental learning')  # Enable/Disable IL
    parser.add_argument('--model_name', type=str, default='resnet')  # Name of the model
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')  # Name of the used dataset
    parser.add_argument('--seed', default=1, type=int)  # Seed to use
    parser.add_argument('--mode', type=str, default='train')  # Mode: train or test
    logging_on = True  # Variable to stop logging when we do not want to log anything

    # Extract the args and make them available in the args object
    args = parser.parse_args()
    # Define if cuda can be used and initialize the device used by torch. Furthermore, specify the torch seed
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    # Returns the current file name. In this case file is called auto_novel.py,
    runner_name = os.path.basename(__file__).split(".")[0]
    # Define the name of the wanted directory as 'experiment root' + 'name of current file'
    model_dir = os.path.join(args.exp_root, runner_name)
    # If the previously defined directory does not exist, them create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Define the name of the path to save the trained model
    args.model_dir = model_dir + '/' + '{}.pth'.format(args.model_name)

    New_resnet = True
    # CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_cifar10_Barlow_twins_2.pth resnet_IL_cifar10_Barlow_twins_2
    if New_resnet:
        # CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_cifar10_new_config.pth resnet_IL_cifar10_new_config
        model = resnet_sim(args.num_labeled_classes, args.num_unlabeled_classes).to(device)
        # another run
        # CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_cifar10_simsam_2.pth resnet_IL_cifar10_simsam_2
        # another run
        # CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_cifar10_supcon.pth resnet_IL_cifar10_supcon_2
        # CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_cifar10_swav.pth resnet_IL_cifar10_swav_2

    else:
        # Initialize ResNet architecture and also the BasicBlock, which are imported from resnet.py. Then send to cuda
        # CUDA_VISIBLE_DEVICES=0 sh scripts/auto_novel_IL_cifar10.sh ./data/datasets/CIFAR/ ./data/experiments/ ./data/experiments/supervised_learning/resnet_rotnet_cifar10_basicconfig.pth resnet_IL_cifar10_basic_config
        model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    # Default inputs assume that we are working with 10 classes of which: 5 classes unlabeled 5 classes labeled.
    # We have two heads in this ResNet model, head1 for the labeled and head2 for the unlabeled data.

    # Compute the total number of classes
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    # to login into wandb this is the password
    if logging_on:
        wandb.login()  # 4619e908b2f2c21261030dae4c66556d4f1f3178
        config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "dataset": args.dataset_name,
            "unlabled_classes": args.num_unlabeled_classes,
            "labled_classes": args.num_labeled_classes,
            "topk": args.topk,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "rampup_length": args.rampup_length,
            "rampup_coefficient": args.rampup_coefficient,
            "increment_coefficient": args.increment_coefficient,
            "step_size": args.step_size,
            "IL": args.IL,
            "mode": args.mode
        }
        wandb.init(project="trends_project", entity="mhaggag96", config=config, save_code=True)

    # If we are in training mode
    if args.mode == 'train':
        # Load the weights for the ResNet model from the supervised previously trained model (supervised_learning.py)
        state_dict = torch.load(args.warmup_model_dir)
        # Apply the loaded weights to the model, we do not strictly enforce that the keys in state_dict match since old
        # model has one head that was removed, while the new model has two new heads. Therefore, hey cannot fully match
        model.load_state_dict(state_dict, strict=False)
        # Iterate through all the parameters of the new model
        for name, param in model.named_parameters():
            # If the parameter under analysis does not belong to 'head' (one of the two heads) or to 'layer4' (features
            # layer before the two heads), then freeze that parameter. In this way we are ensuring that all the
            # parameters will be frozen except for the two heads and the features layer, which we want to train.
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False  # anything before layer 4 is fixed

    # If the dataset argument is 'cifar10' then use its apposite loader, see cifarloader.py for full explanation
    if args.dataset_name == 'cifar10':
        # Main loader used to train the model. It is mixed (labeled+unlabeled data) and it has a double augmentation.
        # It is in format of ((picture1,picture2),label,index) where picture 1 is the original example, picture 2 is
        # the augmentation of the original example, label is the label for both the samples and index is the position
        # of that specific example in the original dataset
        mix_train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                            aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes),
                                            unlabeled_list=range(args.num_labeled_classes, num_classes))

        # Loader never used
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='once', shuffle=True, target_list=range(args.num_labeled_classes))

        # Unlabeled loader only, used for the evaluation over unlabeled samples
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None, shuffle=False,
                                              target_list=range(args.num_labeled_classes, num_classes))

        # unlabeled evaluation set
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                   aug=None, shuffle=False,
                                                   target_list=range(args.num_labeled_classes, num_classes))

        # Labeled loader only, used for the evaluation over labeled samples
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                            aug=None, shuffle=False, target_list=range(args.num_labeled_classes))

        # labeled test set, contains both labeled and unlabeled
        all_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                        aug=None, shuffle=False, target_list=range(num_classes))

    # Otherwise, the dataset argument is 'cifar100' then use its apposite loader, see comments above on cifar10
    elif args.dataset_name == 'cifar100':
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes),
                                             unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                               aug=None, shuffle=False,
                                               target_list=range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                    aug=None, shuffle=False,
                                                    target_list=range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

    # Otherwise, the dataset argument is 'svhn' then use its apposite loader, see comments above on cifar10
    elif args.dataset_name == 'svhn':
        mix_train_loader = SVHNLoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice',
                                         shuffle=True, labeled_list=range(args.num_labeled_classes),
                                         unlabeled_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once',
                                          shuffle=True, target_list=range(args.num_labeled_classes))
        unlabeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None,
                                           shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                aug=None, shuffle=False,
                                                target_list=range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(args.num_labeled_classes))
        all_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                     shuffle=False, target_list=range(num_classes))

    # Finally, if the mode argument is 'train', then run the training procedure
    if args.mode == 'train':
        # Now, if the IL argument is 'True', then run the IL training procedure
        if args.IL:
            # Clone the weights and the bias of the head 1 from supervised learning size 5 by 512
            save_weight = model.head1.weight.data.clone()
            save_bias = model.head1.bias.data.clone()

            # Replace head from just 5 to 10. Now the size of the head is (10,512) and all values are zero
            model.head1 = nn.Linear(512, num_classes).to(device)

            # Initialize half matrix with the weights available previously from the supervised trained model
            model.head1.weight.data[:args.num_labeled_classes] = save_weight
            # The bias for the 10 classes in the new head is set to the minimum of previous bias -1
            model.head1.bias.data[:] = torch.min(save_bias) - 1.
            model.head1.bias.data[:args.num_labeled_classes] = save_bias

            # Finally lunch the training procedure
            train_IL(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)

        # Otherwise, train without IL
        else:
            # Perform the training over the defined model
            train(model, mix_train_loader, labeled_eval_loader, unlabeled_eval_loader, args)

        # In the end, save the model to the specified path
        torch.save(model.state_dict(), args.model_dir)  # saving the model.
        print("model saved to {}.".format(args.model_dir))

    # If the mode argument is 'test', then run the testing procedure
    else:
        print("model loaded from {}.".format(args.model_dir))
        # if the IL argument is 'True', then change the head1 to (10,512)
        if args.IL:
            model.head1 = nn.Linear(512, num_classes).to(device)
        # Load the model from the specified path
        model.load_state_dict(torch.load(args.model_dir))

    # In the end, first test the model using head1 over the labeled dataloader
    print('\n\n\nEVALUATING ON HEAD1')
    args.head = 'head1'
    print('Test on labeled classes (test split)')
    test(model, labeled_eval_loader, 'labeled_classes_test_split', args)
    if args.IL:
        print('\n\nTest on unlabeled classes (test split)')
        test(model, unlabeled_eval_loader_test, 'unlabeled_classes_test_split', args)
        print('\n\nTest on all classes (test split)')
        test(model, all_eval_loader, 'all_classes_test_split', args)

    # Then test the model using head2 over the unlabeled dataloader
    print('\n\n\nEVALUATING ON HEAD2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    test(model, unlabeled_eval_loader, 'unlabeled_classes_train_split', args)

    print('\n\ntest on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, 'unlabeled_classes_test_split', args)

    if logging_on:
        wandb.finish()
