import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.callbacks import PretrainCheckpointCallback

from argparse import ArgumentParser
from datetime import datetime


# Initialize the basic argument parser and take input arguments
parser = ArgumentParser()
parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
parser.add_argument("--download", default=True, action="store_true", help="wether to download")
parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=5, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="UNO", type=str, help="wandb project")
parser.add_argument("--entity", default="donkeyshot21", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained checkpoint path")


class Pretrainer(pl.LightningModule):
    """Built the correct model depending on the input arguments"""
    def __init__(self, **kwargs):
        """Initialize the model"""
        # Call the initializer of the parent class
        super().__init__()
        # Store all the provided arguments under the self.hparams attribute of the current class
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # Build a MultiHeadResNet (only with the h head for labeled classes)
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            # True if the string CIFAR is present in self.hparams.dataset (Normally true since dataset=CIFAR100)
            low_res="CIFAR" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            # Number of heads not specified, only the supervised prototype head will be created
            num_heads=None,
        )

        # If a pretrained checkpoint path has been specified then load its data
        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            self.model.load_state_dict(state_dict, strict=False)

        # Add a final accuracy meter to the model
        self.accuracy = Accuracy('multiclass', num_classes=self.hparams.num_labeled_classes)

    def configure_optimizers(self):
        """Configure SGD optimizer and LR scheduler"""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Train the model"""
        # Extract images and labels from the batch
        images, labels = batch

        # Normalize the prototypes (it works on all the heads but here we have only the supervised one)
        self.model.normalize_prototypes()

        # Compute the output of the model (call to ....)
        outputs = self.model(images)

        # Compute the supervised CE loss for each output of the labeled head
        loss_supervised = torch.stack(
            [F.cross_entropy(o / self.hparams.temperature, labels) for o in outputs["logits_lab"]]
        ).mean()

        # Save the loss and the learning-rate, then log them
        results = {
            "loss_supervised": loss_supervised,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)

        # Return the supervised loss
        return loss_supervised

    def validation_step(self, batch, batch_idx):
        """Validate the model"""
        # Extract images and labels from the batch
        images, labels = batch

        # Get the output of the model only from the labeled head (call to ....)
        logits = self.model(images)["logits_lab"]
        # Extract the prediction for the dimension -1 from the supervised logits
        _, preds = logits.max(dim=-1)

        # Compute supervised CE loss and accuracy
        loss_supervised = F.cross_entropy(logits, labels)
        acc = self.accuracy(preds, labels)

        # Save the validation loss and the validation accuracy, then log them
        results = {
            "val/loss_supervised": loss_supervised,
            "val/acc": acc,
        }
        self.log_dict(results, on_step=False, on_epoch=True)

        # Return the logged values as result for this iteration
        return results


def main(args):
    # Instantiate a datamodule in 'pre-train' mode over the dataset specified by args (default = CIFAR100), this module
    # encompasses all the information about dataset, transforms, ...
    dm = get_datamodule(args, "pretrain")

    # Define the name of the current run
    run_name = "-".join(["pretrain", args.arch, args.dataset, args.comment])
    # Define an instance of Pytorch-Lightning WandB logger using the arguments of the parser and the name of the run
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    # Instantiate a model giving in input all the available arguments as a dictionary to the PreTrainer class
    model = Pretrainer(**args.__dict__)

    # Define a Pytorch-Lightning Trainer using the parameter from the Pytorch-Lightning parser. Furthermore, it will
    # send its results to the specified logger, and it will use the callback to save the model weights in the right way
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[PretrainCheckpointCallback()])
    # Fit the model over the data-module
    trainer.fit(model, dm)


# Run the code
if __name__ == "__main__":
    # Extend the basic ArgParser with a PyTorch-Lightning parser, this will introduce all the NN-training terms
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse the arguments and make them ready for the retrieval
    args = parser.parse_args()

    # Run the main giving all the arguments (basic + training)
    main(args)
