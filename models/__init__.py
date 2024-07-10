import torch
from . import vit as vitencoder
from . import decoder as decoder_module
# from . import mmsegmodel
# from . import deeplabs
import torch.nn as nn
import torch.nn.functional as F
# from . import res
# import clip
import sys


def build_model(args):
    try:
        model = eval(args.model)()
    except:
        if hasattr(args,'decoder'):
            pass
        else:
            args.__setattr__('decoder', 'LargeFOV')
        assert args.decoder in ['LargeFOV', 'Maskformer', 'Original','Segformer'], "decoder should be one of ['LargeFOV', 'Maskformer', 'Original','segformer']"
        if args.model == 'vit':
            model = VITNetwork(args.backbone, args.num_classes, args.pretrained,
                               args.aux_layer, args.isgap, decoder=args.decoder)
        # elif args.model == 'res38':
        #     if args.backbone == 'res38':
        #         model = res.WRN38(num_classes=args.num_classes)
        #         if args.pretrained:
        #             weights_dict = torch.load('./ckpts/res38_cls.pth', map_location='cpu')
        #             model.load_state_dict(weights_dict, strict=False)
        #     elif args.backbone=='beco_r101':
        #         model = res.DeepLabV3Plus(
        #             backbone={
        #                 "pretrain": "./ckpts/resnetv1d101_mmcv.pth",
        #                 "variety": "resnet-D",
        #                 "depth": 101,
        #                 "out_indices": [1, 4],
        #                 "output_stride": 16,
        #                 "contract_dilation": False,
        #                 "multi_grid": True,
        #                 "norm_layer": "BatchNorm2d"
        #             },
        #             decoder={
        #                 "type": "SepASPP",
        #                 "in_channels": 2048,
        #                 "channels": 256,
        #                 "lowlevel_in_channels": 256,
        #                 "lowlevel_channels": 48,
        #                 "atrous_rates": [6, 12, 18],
        #                 "dropout_ratio": 0.1,
        #                 "num_classes": args.num_classes,
        #                 "norm_layer": "BatchNorm2d",
        #                 "align_corners": False
        #             }
        #         )
        #     elif args.backbone=='l2g_r101':
        #         model = deeplabs.DeepLabV1_ResNet101_FOV(n_classes=args.num_classes)
        #         weights_dict = torch.load('./ckpts/deeplabv1_resnet101-imagenet.pth', map_location='cpu')
        #         if args.pretrained=='coco':
        #             print('use coco pretrained')
        #             weights_dict = torch.load('./ckpts/deeplabv1_resnet101-coco.pth', map_location='cpu')
        #         msg=model.load_state_dict(weights_dict, strict=False)
        #         print('loading l2g_r101 with pretrained weights', msg)
        #     elif args.backbone=='l2g_r101_v2':
        #         model = deeplabs.DeepLabV2_ResNet101(n_classes=args.num_classes)
        #         weights_dict = torch.load('./ckpts/deeplabv2_cocostaff.pth', map_location='cpu')
        #         msg=model.load_state_dict(weights_dict, strict=False)
        #         print('loading l2g_r101_v2 with pretrained weights', msg)
        #     else:
        #         raise NotImplementedError
        # elif args.model == 'mmseg':
        #     model=mmsegmodel.MMSegModel(args.backbone, args.num_classes)
        # elif args.model == 'swinend2end':
        #     # assert args.backbone in ['uper-swin-b','uper-swin-s','uper-swin-t'], "backbone should be one of ['uper-swin-b','uper-swin-s','uper-swin-t']"
        #     model=mmsegmodel.MMSWIN(args.backbone,args.num_classes,args.aux_layer, args.isgap)

        else:
            raise NotImplementedError
    return model


class VITNetwork(torch.nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True, aux_layer=-3, isgap=False, decoder='LargeFOV'):
        super().__init__()
        assert decoder in ['LargeFOV','Maskformer'], "decoder should be one of LargeFOV, Maskformer"
        self.num_classes = num_classes
        self.encoder = getattr(vitencoder, backbone)(
            pretrained=pretrained, aux_layer=aux_layer)

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(
            self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4

        if isgap:
            self.pooling = F.adaptive_avg_pool2d
        else:
            self.pooling = F.adaptive_max_pool2d

        if decoder == 'LargeFOV':
            self.decoder = decoder_module.LargeFOV(
                in_planes=self.in_channels[-1], out_planes=self.num_classes,)
            self.isdecoder_trans = False
            print('decoder is LargeFOV')
        elif decoder == 'Maskformer':
            edim = self.encoder.embed_dim
            self.decoder = decoder_module.MaskTransformer(
                n_cls=self.num_classes,
                patch_size=self.encoder.patch_size,
                d_encoder=edim,
                n_layers=2,
                n_heads=edim//64,
                d_model=edim,
                d_ff=4*edim,
                drop_path_rate=0.0,
                dropout=0.1,
            )
            self.isdecoder_trans = True
            print('decoder is Maskformer')
        else:
            raise NotImplementedError

        self.classifier = nn.Conv2d(
            in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(
            in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)

    def get_param_groups(self):

        # backbone; backbone_norm; cls_head; seg_head;
        param_groups = [[], [], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'encoder' not in name:
                print('10x',name)
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'encoder' in name:
                print('1x',name)
                yield param

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x, cam_only=False,seg_only=False, detach='none'):
        H, W = x.size(2), x.size(3)
        cls_token, _x, x_aux = self.encoder.forward_features(x)

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w)
        _x_aux = self.to_2D(x_aux, h, w)

        if not self.isdecoder_trans:
            seg = self.decoder(_x4)
        else:
            seg = self.decoder(_x, (H, W))

        if seg_only:
            return seg

        assert detach in ['all', 'feat', 'none', 'cls']
        if detach == 'all':
            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()
        elif detach == 'feat':
            cam = F.conv2d(_x4.detach(), self.classifier.weight)
            cam_aux = F.conv2d(_x_aux.detach(), self.aux_classifier.weight)
        elif detach == 'cls':
            cam = F.conv2d(_x4, self.classifier.weight.detach())
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight.detach())
        elif detach == 'none':
            cam = F.conv2d(_x4, self.classifier.weight)
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight)

        if cam_only:
            return cam, cam_aux

        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)

        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        cls_aux = cls_aux.view(-1, self.num_classes-1)

        return cls_x4, cls_aux, _x4, seg, cam, cam_aux


