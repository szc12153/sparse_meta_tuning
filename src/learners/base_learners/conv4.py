import torch
import torch.nn as nn
from collections import OrderedDict

def get_cov4_model(args):
    def conv_block(in_channels, out_channels, non_lin, **kwargs):
        return nn.Sequential(OrderedDict([
            ("conv",nn.Conv2d(in_channels,out_channels,**kwargs)),
            ("norm",nn.BatchNorm2d(out_channels,track_running_stats=False)),
            ("non_linearity",non_lin),
            ("pool",nn.MaxPool2d(2))
        ]))

    class CONV4(nn.Module):
        def __init__(self, n_filters = 64, bias=False):
            super().__init__()
            self.conv1 = conv_block(3, n_filters, nn.ReLU(),
                            kernel_size=3, stride=1,padding=1, bias=bias)
            self.conv2 = conv_block(n_filters, n_filters, nn.ReLU(),
                                    kernel_size=3, stride=1, padding=1, bias=bias)
            self.conv3 = conv_block(n_filters, n_filters, nn.ReLU(),
                                    kernel_size=3, stride=1, padding=1, bias=bias)
            self.conv4 = conv_block(n_filters, n_filters, nn.ReLU(),
                                    kernel_size=3, stride=1, padding=1, bias=bias)
            self.linear=nn.Linear(1600,args.base_learner.output_dim,bias=bias)
            for n, p in self.inner_meta_params():
                print(n)

        def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view((x.size(0), -1))
            x = self.linear(x)
            return x

        @property
        def input_size(self):
            return 84

        def outer_meta_params(self):
            for n, p in self.named_parameters():
                yield (n, p)

        def inner_meta_params(self):
            for n, p in self.named_parameters():
                if "norm" not in n:
                    yield (n, p)

    return CONV4(bias = args.base_learner.bias)