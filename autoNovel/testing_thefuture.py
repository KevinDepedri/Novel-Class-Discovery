import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
class resnet_sim(nn.Module):
    def __init__(self,num_labeled_classes=5, num_unlabeled_classes=5):
        super(resnet_sim,self).__init__()
        self.encoder = models.__dict__['resnet18']()#intializingresnet18 by pytorcch
        self.encoder.fc = nn.Identity()# replace the fullneceted by an identity
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()# I am removign the max pool layer
        self.head1 = nn.Linear(512, num_labeled_classes)  # First head: to classify known classes
        self.head2 = nn.Linear(512, num_unlabeled_classes)  # Second head: to classify unknown classes

    def forward(self, x):
        out = self.encoder(x)
        out1 = self.head1(out)
        out2 = self.head2(out)
        return out1,out2,out
# ckpt_path= 'trained_models/cifar10/barlow_twins/barlow-cifar10-otu5cw89-ep=999.ckpt' # classifier + projector 0 to 6
# ckpt_path= 'trained_models/cifar10/byol/byol-cifar10-32brzx9a-ep=999.ckpt'# you have 2 resnets and 2 projection heads.
# ckpt_path= 'trained_models/cifar10/deepclusterv2/deepclusterv2-cifar10-1ikqjkrr-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/dino/dino-cifar10-13wu9ixc-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/mocov2plus/mocov2plus-cifar10-1nhrg2pm-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/mocov3/mocov3-cifar10-3gpr99oc-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/nnclr/nnclr-cifar10-2655kevu-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/ressl/ressl-cifar10-3ns2ryc6-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/simclr/simclr-cifar10-b30xch14-ep=999.ckpt'
ckpt_path= 'trained_models/cifar10/simsiam/simsiam-cifar10-252e1tvw-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/supcon/supcon-cifar10-1w8chdt4-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/swav/swav-2rwotcpy-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/vibcreg/vibcreg-cifar10-3ehq2v3f-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/vicreg/vicreg-cifar10-qx5zahvt-ep=999.ckpt'
# ckpt_path= 'trained_models/cifar10/wmse/wmse-cifar10-6z3m2p9o-ep=999.ckpt'
if __name__ == '__main__':
    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    num_labeled_classes = 5
    num_unlabeled_classes = 5
    model= resnet_sim( num_labeled_classes, num_unlabeled_classes).cuda()
    #################################################
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    encoder = models.__dict__['resnet18']()#intializingresnet18 by pytorcch
    feat_dim = encoder.fc.weight.shape[1]#you have a full connected of size of 10000,512
    encoder.fc = nn.Identity()# replace the fullneceted by an identity
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.Identity()# I am removign the max pool layer
    encoder.load_state_dict(state, strict=True)
    f= open("information1.txt","w+")
    for name, param in encoder.named_parameters():
        print(name, param.data)
        f.write("The name is {0}\t the data is {1}  \n".format(str(name),str(param.data)))## edited
    ##############################################################################
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
        if "backbone" in k:
            state['encoder.'+k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state, strict=False)
    f= open("information2.txt","w+")
    for name, param in model.named_parameters():
        print(name, param.data)
        f.write("The name is {0}\t the data is {1}  \n".format(str(name),str(param.data)))## edited

    #################################################
    # print(state)
    # print(model)
    # print(summary(model, (3, 32, 32), batch_size=256))
    
