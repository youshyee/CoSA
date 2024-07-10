from typing import Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .modules.convs import DepthwiseSeparableConv
from .modules.backbones.resnet import get_convnet
from .modules.decoders.sep_aspp import  SepASPP


def init_weight(module: nn.Module, a=0, mode='fan_in', nonlinearity='relu'):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decode
    Deeplabv3plus implememts
    This module has five components:

    self.backbone
    self.aspp
    self.projector: an 1x1 conv for lowlevel feature projection
    self.preclassifier: an 3x3 conv for feature mixing, before final classification
    self.classifier: last 1x1 conv for output classification results

    Args:
        backbone: Dict, configs for backbone
        decoder: Dict, configs for decoder

    NOTE: The bottleneck has only one 3x3 conv by default, some implements stack
        two 3x3 convs
    """
    def __init__(self, backbone: Dict, decoder) -> None:
        super(DeepLabV3Plus, self).__init__()

        self.align_corners = decoder['align_corners']
        BN_op = getattr(nn, decoder['norm_layer'])
        channels = decoder['channels']
        #self.backbone = modules.backbones.get_backbone(backbone)
        self.backboneres = get_convnet(**backbone)
        self.freeze_bn = False

        #self.aspp = modules.decoders.get_decoder(decoder)
        decoder.pop('type')
        self.aspp = SepASPP(**decoder)
        self.projector = nn.Sequential(
            nn.Conv2d(
                decoder['lowlevel_in_channels'],
                decoder['lowlevel_channels'],
                kernel_size=1, bias=False),
            BN_op(decoder['lowlevel_channels']),
            nn.ReLU(inplace=True),
        )
        self.pre_classifier = DepthwiseSeparableConv(
            decoder['norm_layer'],
            channels + decoder['lowlevel_channels'],
            channels, 3, padding=1
        )

        self.classifier = nn.Conv2d(channels, decoder['num_classes'], 1, 1)

        init_weight(self.projector)
        init_weight(self.pre_classifier)
        init_weight(self.classifier)

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if self.freeze_bn:
                if 'bn' in name:
                    param.requires_grad = False
                    print('0x',name)
                    continue
            if 'backboneres' not in name:
                print('10x',name)
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if self.freeze_bn:
                if 'bn' in name:
                    param.requires_grad = False
                    print('0x',name)
                    continue
            if 'backboneres' in name:
                print('1x',name)
                yield param

    def forward(self, x: Tensor) -> Tensor:
        size = (x.shape[2], x.shape[3])
        out = self.backboneres(x)
        lowlevel_feature = self.projector(out[1])
        output = self.aspp(out[4])
        output = F.interpolate(
            output,
            size=lowlevel_feature.shape[-2:],
            mode='bilinear', align_corners=self.align_corners
        )
        output = torch.cat([lowlevel_feature, output], dim=1)
        output = self.pre_classifier(output)
        out['embeddings'] = output
        output = self.classifier(output)
        return output
        out['pre_logits'] = output
        out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)

        return out
