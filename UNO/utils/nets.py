import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Prototypes(nn.Module):
    """Build the Prototype module that is used as labeled-head. It takes the specified number of
    features as input, and it has a number of output neurons equal to the number of classes"""
    def __init__(self, output_dim, num_prototypes):
        # Call the initializer of the parent class
        super().__init__()

        # Built the prototype as a linear module used as classifier
        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        # Clone the weight of the prototype
        w = self.prototypes.weight.data.clone()
        # Normalize the weight along dimension 1 with a power of 2
        w = F.normalize(w, dim=1, p=2)
        # Overwrite the previous weight with the normalized weight
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    """Build the MLP block used as feature projector to reduce the feature dimension"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        # Call the initializer of the parent class
        super().__init__()

        # Add 3 layers with the specified parameters
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]

        # For each hidden layer
        for _ in range(num_hidden_layers - 1):
            # Add another set of 3 layer to the default defined layers
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]

        # Append to the layers a linear module used as classifier
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Built the MPL as a sequence of the given list of layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    """Build the MultiHead module that is used as unlabeled-head. It uses projectors based on MLP which
     allows to take the specified number of input features, pass through the specified number of hidden layers (with
     a specified number of hidden units) and return the specified number of output features < number input features.
     The output dimension of the prototype classifier is equal to the number of unlabeled samples, number heads is 5"""
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        """Initialize the MultiHead with the specified number of heads"""
        # Call the initializer of the parent class
        super().__init__()
        # Set the number of heads
        self.num_heads = num_heads

        # Initialize an MLP projector for each head
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # Initialize a prototype classifier for each head, with a number of output class equal to the unlabeled classes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        # Normalize the prototype classifiers
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        # For each prototype of the class
        for p in self.prototypes:
            # Initialize the prototype
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        # Compute the output for the given input features using the projector for the specified head
        z = self.projectors[head_idx](feats)
        # Normalize the putput over the first dimension
        z = F.normalize(z, dim=1)
        # Return the output for the normalized vector using the specified head
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadResNet(nn.Module):
    """Built the MultiHeadResNet module used as NN in the NCD task"""
    def __init__(
        self,
        arch,
        low_res,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=5,
        num_hidden_layers=1,
    ):
        # Call the initializer of the parent class
        super().__init__()

        # Build the backbone starting with the encoder, according to the given architecture (normally: ResNet-18)
        self.encoder = models.__dict__[arch]()
        # Extract the dimension of the output features before the fc layer. Or the number of input features in fc layer
        self.feat_dim = self.encoder.fc.weight.shape[1]
        # Overwrite the fc layer with an Identity layer
        self.encoder.fc = nn.Identity()
        # Modify the encoder for lower resolution
        if low_res:
            # Lower kernel size from (7x7) to (3x3), lower the stride from (2) to (1), lower padding from (3,3) to (1,1)
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Change the MaxPool2d in ad Identity layer
            self.encoder.maxpool = nn.Identity()
            # Call the function to re-initialize all the layers in the architecture
            self._reinit_all_layers()

        # Add the Head for Labeled samples specifying number of input features and number of labeled classes for the
        # prototype classifier
        self.head_lab = Prototypes(self.feat_dim, num_labeled)

        # If a number of head has been specified, then add two MultiHead with that number of heads.
        # These multi-heads are used to process the unlabeled data
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        # For each module m in the architecture
        for m in self.modules():
            # If the module is a 'Conv2d' then fill the weight tensor according to the given mode and non-linearity
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # Otherwise, if it is a 'BatchNorm2d' or a 'GroupNorm' put weight to 1 and bias to 0
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        # Normalize the labeled head prototype classifiers
        self.head_lab.normalize_prototypes()
        # If unlabeled head is present then normalize also them, otherwise getattr return False, and it is not executed
        if getattr(self, "head_unlab", False):
            # Normalize the unlabeled head prototypes classifiers
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        # Compute the normalized output logit using the labeled head for the given input features
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        # If an unlabeled head exists then
        if hasattr(self, "head_unlab"):
            # Compute the output logit using the two unlabeled heads for the given input features
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            # Update the previously defined dictionary with the new output just computed
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        # If views is a list of different images
        if isinstance(views, list):
            # Compute the features using the encoder for each element in the list views
            feats = [self.encoder(view) for view in views]
            # Compute the output of the heads for each feature vector previously computed
            out = [self.forward_heads(f) for f in feats]
            # Concatenate the feature vectors previously obtained (one for each view) in one single tensor
            out_dict = {"feats": torch.stack(feats)}

            # For each key in the dictionary of the 'outputs of the heads'
            for key in out[0].keys():
                # Stack on the previously stacked tensor, all the values for that key, for all the different views
                out_dict[key] = torch.stack([o[key] for o in out])
            # Return the concatenated dictionary
            return out_dict

        # If views is not a list of different images but just a single image
        else:
            # Compute the output features of the encoder
            feats = self.encoder(views)
            # Compute the output of the heads
            out = self.forward_heads(feats)
            # Add the encoder features to the output of the heads
            out["feats"] = feats
            # Return the concatenated dictionary
            return out
