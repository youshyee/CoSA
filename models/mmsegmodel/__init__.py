import warnings
from mmengine import build_model_from_cfg
from mmseg.registry import MODELS
import torch
from mmengine.config import Config
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.backbones import SwinTransformer
from mmseg.models.backbones.swin import SwinBlockSequence
from mmseg.models.utils.embed import PatchMerging,PatchEmbed
from mmengine.model import ModuleList
from mmengine.utils import to_2tuple
from mmcv.cnn import build_norm_layer
from .. import decoder as decoder_module

Model_cfg = {
    'uper-swin-b':'./models/mmsegmodel/cfg_swin_b.py',
    'uper-swin-s':'./models/mmsegmodel/cfg_swin_s.py',
    'uper-swin-t':'./models/mmsegmodel/cfg_swin_t.py',
    'deeplab3':'./models/mmsegmodel/deeplab3.py',
    'deeplab3p':'./models/mmsegmodel/deeplab3p.py',
             }
backbone_cfg = {
    'swin-b':'./models/mmsegmodel/swin_b.py',
    'swin-s':'./models/mmsegmodel/swin_s.py',
    'swin-t':'./models/mmsegmodel/swin_t.py',
             }

class MMSegModel(torch.nn.Module):
    def __init__(self, backbone,num_classes):
        super().__init__()
        model_cfg = Config.fromfile(Model_cfg[backbone])
        # assert model_cfg.model.decode_head.num_classes == args.num_classes
        # assert model_cfg.model.auxil == args.in_channels
        model_cfg.model.decode_head.num_classes=num_classes
        self.model = build_model_from_cfg(model_cfg.model, MODELS)
        self.model.init_weights()
        # msg=model.load_state_dict(torch.load('')['state_dict'], strict=False)
        if 'swin' in backbone:
            self.wt_keys=['absolute_pos_embed','relative_position_bias_table','norm']
        else:
            self.wt_keys = []
    def forward(self, x):
        return self.model(x)

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if self.anyin(self.wt_keys,name):
                continue
            if 'backbone' not in name:
                print('10x',name)
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if self.anyin(self.wt_keys,name):
                continue
            if 'backbone' in name:
                print('1x',name)
                yield param

    def get_dec_wt_params(self):
        for name, param in self.named_parameters():
            if self.anyin(self.wt_keys,name):
                print('wt_dec_mult:',name)
                yield param
    def anyin(self,keys,name):
        for key in keys:
            if key in name:
                return True
        return False

class MMSWIN(torch.nn.Module):
    def __init__(self, backbone, num_classes, aux_layer=-3,isgap=False, decoder='LargeFOV',**kwargs):
        super().__init__()
        # assert aux_layer in [-1, -2, -3, -4]
        model_cfg = Config.fromfile(backbone_cfg[backbone])
        model_cfg.backbone.type='SwinTransformer_'
        self.num_classes=num_classes
        self.aux_layer=aux_layer

        self.backbone = build_model_from_cfg(model_cfg.backbone, MODELS)
        self.backbone.init_weights()
        self.wt_keys=['absolute_pos_embed','relative_position_bias_table','norm']

        self.in_channels=[]
        for x,y in zip(model_cfg.backbone.depths, [128, 256, 512, 1024]):
            self.in_channels+=[y]*x

        self.classifier = nn.Conv2d(
            in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(
            in_channels=self.in_channels[aux_layer], out_channels=self.num_classes-1, kernel_size=1, bias=False,)

        if isgap:
            self.pooling = F.adaptive_avg_pool2d
        else:
            self.pooling = F.adaptive_max_pool2d

        if decoder == 'LargeFOV':
            self.decoder = decoder_module.LargeFOV(
                in_planes=self.in_channels[-1], out_planes=self.num_classes,)
            print('decoder is LargeFOV')

    def forward(self, x, cam_only=False,seg_only=False, detach='none'):
        x, xblocks = self.backbone(x)
        _x=x[-1]
        _x_aux=xblocks[self.aux_layer]

        seg=self.decoder(_x)
        if seg_only:
            return seg

        cam = F.conv2d(_x, self.classifier.weight)
        cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight)

        if cam_only:
            return cam, cam_aux


        cls_x = self.pooling(_x, (1, 1))
        cls_x = self.classifier(cls_x)
        cls_x = cls_x.view(-1, self.num_classes-1)

        cls_aux = self.pooling(_x_aux, (1, 1))
        cls_aux = self.aux_classifier(cls_aux)
        cls_aux = cls_aux.view(-1, self.num_classes-1)

        return cls_x, cls_aux, None, seg, cam, cam_aux

    def get_param_groups(self):

        # backbone; backbone_norm; cls_head; seg_head;
        param_groups = [[], [], [], []]

        for name, param in list(self.backbone.named_parameters()):

            if self.anyin(self.wt_keys,name):
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
            if self.anyin(self.wt_keys,name):
                continue
            if 'backbone' not in name:
                print('10x',name)
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if self.anyin(self.wt_keys,name):
                continue
            if 'backbone' in name:
                print('1x',name)
                yield param

    def get_dec_wt_params(self):
        for name, param in self.named_parameters():
            if self.anyin(self.wt_keys,name):
                print('wt_dec_mult:',name)
                yield param
    def anyin(self,keys,name):
        for key in keys:
            if key in name:
                return True
        return False

@MODELS.register_module()
class SwinTransformer_(SwinTransformer):
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=None):
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence_(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        out_all_blocks=[]
        for i, stage in enumerate(self.stages):
            x, hw_shape, out_blocks, out_hw_shape = stage(x, hw_shape)
            out= out_blocks[-1]
            out_all_blocks+=[ x.view(-1, *out_hw_shape,self.num_features[i]).permute(0, 3, 1,2).contiguous() for x in out_blocks]
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs, out_all_blocks

class SwinBlockSequence_(SwinBlockSequence):
    def forward(self, x, hw_shape):
        outs=[]
        for block in self.blocks:
            x = block(x, hw_shape)
            outs.append(x)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, outs, hw_shape
        else:
            return x, hw_shape, outs, hw_shape

if __name__ =="__main__":
    model=MMSWIN('uper-swin-b',21,aux_layer=-2)
    c=model.get_param_groups()
    print(model)
    # model.eval()
    x=torch.randn(2,3,448,448)
    cls_x,cls_aux,_,seg,cam,cam_aux=model(x)
    print('cls_x',cls_x.shape)
    print('cls_aux',cls_aux.shape)
    print('seg',seg.shape)
    print('cam',cam.shape)
    print('cam_aux',cam_aux.shape)




