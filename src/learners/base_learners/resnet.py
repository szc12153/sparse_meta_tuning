import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from torchvision.models import ResNet18_Weights



class ResNetModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Linear(resnet.fc.weight.size(1), args.base_learner.output_dim, bias=args.base_learner.bias)
        for p in resnet.fc.parameters():
            nn.init.zeros_(p)
        self.net = resnet
        if not args.base_learner.keep_bn_stats:
            for m in self.modules():
                if isinstance(m,nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean=None
                    m.running_var=None
        for n, p in self.inner_meta_params():
            print(n)

    def forward(self,x):
        return self.net(x)

    @property
    def input_size(self):
        return 84

    def outer_meta_params(self):
        for n, p in self.named_parameters():
            yield (n, p)

    def inner_meta_params(self):
        for n,p in self.named_parameters():
            if "bn" in n or "downsample.1" in n:
                continue
            yield (n,p)



def get_resnet_model(args):
    return ResNetModel(args)