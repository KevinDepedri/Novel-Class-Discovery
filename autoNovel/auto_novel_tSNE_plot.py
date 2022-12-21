import torch
import torch.nn as nn
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, AverageMeter, seed_torch, accuracy
from models.resnet_with_tSNE import ResNet, BasicBlock, resnet_sim
from data.cifarloader_unbalanced import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.svhnloader import SVHNLoader, SVHNLoaderMix
from tqdm import tqdm
import numpy as np
import os


def test(model, test_loader, plot_name, args):
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
        output1, output2, _ = model([x, label])

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

    # Get the feature label dataframe
    model.get_feature_label_dataframe(print_df=True)

    # Define the name of the plot
    partial_plot_path = str('tSNE_plots/' + args.input_model_name)
    full_plot_path = str(partial_plot_path + '/' + 'tSNE_' + args.head + '_' + plot_name + '.png')
    # Get the feature label tSNE dataframe and save the tSNE plot in the given directory
    model.compute_and_plot_2d_t_sne(print_df=True, plot_path=full_plot_path, verbose=1, perplexity=40, n_iter=300)

    # Initialize the tSNE module for the next plot with the current model
    model.__init_tsne__()

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
    parser.add_argument('--batch_size', default=128, type=int)  # Batch size
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)  # Number of unlabeled classes
    parser.add_argument('--num_labeled_classes', default=5, type=int)  # Number of labeled classes
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')  # Dataset root directory
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')  # Directory to save the resulting files
    parser.add_argument('--IL', action='store_true', default=False, help='enable incremental learning')  # Enable/Disable IL
    parser.add_argument('--new_resnet', action='store_true', default=False, help='enable New-ResNet? or use the one defined by authors?')  # Enable/Disable IL
    parser.add_argument('--input_model_name', type=str, default='resnet_IL_cifar10')  # Name of the model
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn')  # Name of the used dataset
    parser.add_argument('--seed', default=1, type=int)  # Seed to use

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
    args.model_dir = model_dir + '/' + '{}.pth'.format(args.input_model_name)

    # Build a new directory for the tSNE plots of the current tested model
    partial_plot_path = str('tSNE_plots/' + args.input_model_name)
    if not os.path.exists(partial_plot_path):
        os.makedirs(partial_plot_path)

    # Choose which ResNet architecture we want to run
    New_resnet = args.new_resnet
    if New_resnet:
        # Initialize the New ResNet architecture (original resnet no changes) and also the BasicBlock. Then send cuda
        model = resnet_sim(args.num_labeled_classes, args.num_unlabeled_classes).to(device)
        print("Working with New-ResNet architecture (original ResNet paper)")
    else:
        # Initialize the old ResNet architecture (changed from the authors) and also the BasicBlock. Then Send cuda
        model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
        print("Working with the ResNet architecture defined by authors")

    # Default inputs assume that we are working with 10 classes of which: 5 classes unlabeled 5 classes labeled.
    # We have two heads in this ResNet model, head1 for the labeled and head2 for the unlabeled data.
    # Compute the total number of classes
    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # If the dataset argument is 'cifar10' then use its apposite loader, see cifarloader.py for full explanation
    if args.dataset_name == 'cifar10':
        # Dictionary of the samples that we want to remove from each class. For each class (labels from 0 to 9) it is
        # possible to specify how many samples we want to be removed. They will be removed randomly in that class.
        unbalanced = False
        if unbalanced:
            train_sample_per_class_to_remove_dictionary = {0: 4500, 1: 4500, 2: 4500, 3: 4500, 4: 4500,
                                                           5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        else:
            train_sample_per_class_to_remove_dictionary = None

        # Main loader used to train the model. It is mixed (labeled+unlabeled data) and it has a double augmentation.
        # It is in format of ((picture1,picture2),label,index) where picture 1 is the original example, picture 2 is
        # the augmentation of the original example, label is the label for both the samples and index is the position
        # of that specific example in the original dataset
        mix_train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                            aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes),
                                            unlabeled_list=range(args.num_labeled_classes, num_classes),
                                            remove_dict=train_sample_per_class_to_remove_dictionary, download=False)

        # Loader never used
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='once', shuffle=True, target_list=range(args.num_labeled_classes),
                                             remove_dict=train_sample_per_class_to_remove_dictionary, download=False)

        # Unlabeled loader only, used for the evaluation over unlabeled samples - Used in test-phase4
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None, shuffle=False,
                                              target_list=range(args.num_labeled_classes, num_classes),
                                              remove_dict=train_sample_per_class_to_remove_dictionary, download=False)

        # unlabeled evaluation set - Used in test-phase2 and phase5
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                   aug=None, shuffle=False,
                                                   target_list=range(args.num_labeled_classes, num_classes),
                                                   remove_dict=None, download=False)

        # Labeled loader only, used for the evaluation over labeled samples - Used in test-phase1
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                            aug=None, shuffle=False, target_list=range(args.num_labeled_classes),
                                            remove_dict=None, download=False)

        # labeled test set, contains both labeled and unlabeled - Used in test-phase3
        all_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                        aug=None, shuffle=False, target_list=range(num_classes),
                                        remove_dict=None, download=False)

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

    # FWe are in test mode by default
    print("model loaded from {}.".format(args.model_dir))
    # if the IL argument is 'True', then change the head1 to (10,512)
    if args.IL:
        model.head1 = nn.Linear(512, num_classes).to(device)
        print("Working with Incremental-Learning enabled")
    # Load the model from the specified path
    model.load_state_dict(torch.load(args.model_dir, map_location=torch.device(device)))

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
