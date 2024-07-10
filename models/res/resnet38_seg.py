import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet38d


class Net(resnet38d.Net):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.fc8_seg_conv1 = nn.Conv2d(4096, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)

        self.fc8_seg_conv2 = nn.Conv2d(512, num_classes, (3, 3), stride=1, padding=12, dilation=12, bias=True)
        torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)
        self.from_scratch_layers = [self.fc8_seg_conv1, self.fc8_seg_conv2]
        # if train_cls:
        #     if isgap:
        #         self.pooling = F.adaptive_avg_pool2d
        #     else:
        #         self.pooling = F.adaptive_max_pool2d
        #     self.fc8_classifier = nn.Conv2d(in_channels=4096, out_channels=num_classes-1, kernel_size=1, bias=False,)
        #     self.from_scratch_layers = [self.fc8_classifier,self.fc8_seg_conv1, self.fc8_seg_conv2]



    def forward(self, x, get_cls=False):
        x,_ = super().forward(x)
        x_seg = F.relu(self.fc8_seg_conv1(x))
        x_seg = self.fc8_seg_conv2(x_seg)
        # if get_cls:
        #     x_cls = self.pooling(xb7, (1, 1))
        #     x_cls = self.fc8_classifier(x_cls)
        #     x_cls = x_cls.view(-1, self.num_classes-1)
        #     return x_seg, x_cls
        return x_seg


    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
