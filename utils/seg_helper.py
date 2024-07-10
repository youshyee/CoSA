from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable
from bilateralfilter import bilateralfilter, bilateralfilter_batch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import sklearn.mixture as skm

PALETTE = [0, 0, 0,
           128, 0, 0,
           0, 128, 0,
           128, 128, 0,
           0, 0, 128,
           128, 0, 128,
           0, 128, 128,
           128, 128, 128,
           64, 0, 0,
           192, 0, 0,
           64, 128, 0,
           192, 128, 0,
           64, 0, 128,
           192, 0, 128,
           64, 128, 128,
           192, 128, 128,
           0, 64, 0,
           128, 64, 0,
           0, 192, 0,
           128, 192, 0,
           0, 64, 128,
           128, 64, 128,
           0, 192, 128,
           128, 192, 128,
           64, 64, 0,
           192, 64, 0,
           64, 192, 0,
           192, 192, 0]

PALETTE81=[0,  0,   0,
           158,   1,  66,
           164,   8,  68,
           171,  15,  69,
           180,  25,  71,
           186, 32,  73,
           193,  39,  74,
           199,  46,  76,
           205,  54,  77,
           214,  63, 79,
           217,  68,  77,
           221,  74,  76,
           225,  80,  75,
           228,  85,  73,
           232,  91,  72,
           237,  98,  70,
           240, 103,  68,
           244, 109,  67,
           245, 117,  71,
           246, 124,  74,
           248, 134,  79,
           249, 142,  82,
           250, 150, 86,
           251, 157,  89,
           252, 165,  93,
           253, 173,  96,
           253, 181, 103,
           253, 187, 108,
           253, 193, 113,
           253, 199, 118,
           254, 204, 123,
           254, 212, 129,
           254, 218, 134,
           254, 224, 139,
           254, 228, 145,
           254, 231, 151,
           254, 236, 159,
           255, 240, 166,
           255, 243, 172,
           255, 247, 178,
           255, 251, 184,
           255, 255, 190,
           252, 254, 186,
           249, 252, 181,
           246, 251, 176,
           243, 250, 172,
           240, 249, 167,
           236, 247, 161,
           233, 246, 157,
           230, 245, 152,
           223, 242, 153,
           216, 239, 155,
           207, 236, 157,
           200, 233, 158,
           193, 230, 160,
           186, 227, 161,
           179, 224, 162,
           172, 221, 164,
           162, 217, 164,
           153, 214, 164,
           145, 211, 164,
           137, 208, 164,
           129, 205, 165,
           118, 200, 165,
           110, 197, 165,
           102, 194, 165,
           96, 187, 168,
           90, 180, 171,
           82, 171, 174,
           75, 164, 177,
           69, 158, 180,
           63, 151, 183,
           57, 144, 186,
           51, 135, 188,
           56, 128, 185,
           61, 121, 182,
           66, 115, 179,
           72, 108, 176,
           77, 101, 173,
           84,  92, 168,
           89,  86, 165,
           94,  79, 162]

def mask_to_onehot(mask, num_classes):
    """
    Convert a mask to one-hot code mask.

    :param mask: A torch tensor with shape Bx1xHxW representing the mask.
    :param num_classes: The number of classes in the one-hot code mask.
    :return: A torch tensor with shape BxCxHxW representing the one-hot code mask.
    """
    assert num_classes > 0, "num_classes must be positive"
    assert len(mask.shape) == 4, "mask must have shape Bx1xHxW"
    assert mask.shape[1] == 1, "mask must have only one channel"

    batch_size, height, width = mask.shape[0], mask.shape[2], mask.shape[3]
    mask_onehot = torch.FloatTensor(batch_size, num_classes, height, width).zero_()
    mask_onehot.scatter_(1, mask.long(), 1)

    return mask_onehot

def save_seg(seg,filepath,classnum):
    # save seg mask after the argmax np HxW
    out = seg.astype(np.uint8)
    out = Image.fromarray(out, mode='P')
    if classnum>28:
        out.putpalette(PALETTE81)
    out.putpalette(PALETTE)
    out.save(filepath)

def save_cam_on_image(img, mask, save_path):
    '''
    img: np array. HxWx3 RGB 0-255
    mask: np array. HxW
    '''
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cam[:, :, ::-1]
    Image.fromarray(cam).save(save_path)
    # cv2.imwrite(save_path, cam)

# merged save cam and seg and original image and gt
def save_merge(img_org, mask, gt, seg, save_path):
    '''
    img: np array. HxWx3 RGB 0-255
    mask: np array. HxW
    gt: np array. HxW
    seg: np array. HxW
    '''
    img = np.float32(img_org) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cam[:, :, ::-1]
    gt=Image.fromarray(gt.astype(np.uint8),mode='P')
    gt.putpalette([0,0,0,10,186,181])
    gt=np.array(gt.convert('RGB'))
    seg=Image.fromarray(seg.astype(np.uint8),mode='P')
    seg.putpalette([0,0,0,10,186,181])
    seg=np.array(seg.convert('RGB'))
    merged=np.concatenate((cam,seg,gt,img_org),axis=1)
    Image.fromarray(merged).save(save_path)
    # cv2.imwrite(save_path, merged)

class DenseEnergyLoss(torch.nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True)
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)

def get_energy_loss(img,
                    logit,
                    label,
                    img_box,
                    loss_layer,
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]):

    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:,0,...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )
    return loss.cuda()

def multi_scale_camseg(model, imgs, scales):
    # teacher model forward get pseudo labels for CAM and Segs
    # using multi-scale inputs an its flip version
    # input image shape: BxCxHxW
    # model for getting CAM and Segs
    # scales: list of scales
    b, c, h, w = imgs.shape
    cam_list = []
    cam_aux_list = []
    seg_list = []
    assert 1.0 in scales, 'scale 1.0 must be in scales'

    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            _,_,_, _seg,_cam, _cam_aux = model(imgs_cat, cam_only=False)
            _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
            _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
            _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

            cam_list.append(F.relu(_cam))
            cam_aux_list = [F.relu(_cam_aux)]

            _seg = F.interpolate(_seg, size=(h,w), mode='bilinear', align_corners=False)
            seg = torch.sum(torch.stack([_seg[:b,...], _seg[b:,...].flip(-1)],dim=0), dim=0)
            seg_list.append(seg)

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

        # seg
        seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)

    return cam, cam_aux, seg

def multi_scale_camsegv4(model, imgs, scales, cls_label):
    # use global max instead of channel max
    # teacher model forward get pseudo labels for CAM and Segs
    # using multi-scale inputs an its flip version
    # input image shape: BxCxHxW
    # model for getting CAM and Segs
    # scales: list of scales
    b, c, h, w = imgs.shape
    cam_list = []
    cam_aux_list = []
    seg_list = []
    assert 1.0 in scales, 'scale 1.0 must be in scales'

    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            _,_,_, _seg,_cam, _cam_aux = model(imgs_cat, cam_only=False)
            _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
            _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
            _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

            cam_list.append(F.relu(_cam))
            cam_aux_list = [F.relu(_cam_aux)]

            _seg = F.interpolate(_seg, size=(h,w), mode='bilinear', align_corners=False)
            seg = torch.sum(torch.stack([_seg[:b,...], _seg[b:,...].flip(-1)],dim=0), dim=0)
            seg_list.append(seg)

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam=cam_validation(cam, cls_label)
        cam = cam - cam.min()
        cam /= cam.max() + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux=cam_validation(cam_aux, cls_label)
        cam_aux = cam_aux -cam_aux.min()
        cam_aux /= cam_aux.max() + 1e-5

        # seg
        seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)

    return cam, cam_aux, seg

def multi_scale_camsegv2(model,
                         imgs,
                         scales,
                         cam_fuse=['max','sum'],# the first one for flip fuse, the second one for multi-scale fuse
                         seg_fuse=['max','sum']
                         ):
    # teacher model forward get pseudo labels for CAM and Segs
    # using multi-scale inputs an its flip version
    # input image shape: BxCxHxW
    # model for getting CAM and Segs
    # scales: list of scales
    b, c, h, w = imgs.shape
    cam_list = []
    cam_aux_list = []
    seg_list = []
    assert 1.0 in scales, 'scale 1.0 must be in scales'

    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            _,_,_, _seg,_cam, _cam_aux = model(imgs_cat, cam_only=False)

            _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
            _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)

            if cam_fuse[0]=='max':
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
            elif cam_fuse[0]=='sum':
                _cam = torch.sum(torch.stack([_cam[:b,...], _cam[b:,...].flip(-1)],dim=0), dim=0)
                _cam_aux = torch.sum(torch.stack([_cam_aux[:b,...], _cam_aux[b:,...].flip(-1)],dim=0), dim=0)
            else:
                raise NotImplementedError

            cam_list.append(F.relu(_cam))
            cam_aux_list = [F.relu(_cam_aux)]

            _seg = F.interpolate(_seg, size=(h,w), mode='bilinear', align_corners=False)
            if seg_fuse[0]=='max':
                seg = torch.max(_seg[:b,...], _seg[b:,...].flip(-1))
            elif seg_fuse[0]=='sum':
                seg = torch.sum(torch.stack([_seg[:b,...], _seg[b:,...].flip(-1)],dim=0), dim=0)
            else:
                raise NotImplementedError
            seg_list.append(seg)

        if cam_fuse[1]=='sum':
            cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        elif cam_fuse[1]=='max':
            cam = torch.max(torch.stack(cam_list, dim=0), dim=0)[0]
        else:
            raise NotImplementedError

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

        # seg
        if seg_fuse[1]=='sum':
            seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)
        elif seg_fuse[1]=='max':
            seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
        else:
            raise NotImplementedError

    return cam, cam_aux, seg

def multi_scale_camsegv3(model, imgs, scales,getcls=False):
    # teacher model forward get pseudo labels for CAM and Segs
    # using multi-scale inputs an its flip version
    # input image shape: BxCxHxW
    # model for getting CAM and Segs
    # scales: list of scales
    b, c, h, w = imgs.shape
    cam_list = []
    cam_aux_list = []
    seg_list = []
    if getcls:
        cls_f_ = 0
        cls_a_ = 0
    assert 1.0 in scales, 'scale 1.0 must be in scales'

    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            cls_f,cls_a,_, _seg,_cam, _cam_aux = model(imgs_cat, cam_only=False)
            _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
            _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
            _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

            cam_list.append(F.relu(_cam))
            cam_aux_list = [F.relu(_cam_aux)]

            _seg = F.interpolate(_seg, size=(h,w), mode='bilinear', align_corners=False)
            seg = torch.sum(torch.stack([_seg[:b,...], _seg[b:,...].flip(-1)],dim=0), dim=0)
            seg_list.append(seg)
            if getcls:
                cls_f_ += torch.sum(cls_f,dim=0,keepdim=True)
                cls_a_ += torch.sum(cls_a,dim=0,keepdim=True)

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

        # seg
        seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)
        if getcls:
            return cam, cam_aux, seg, cls_f_, cls_a_

    return cam, cam_aux, seg

def multi_scale_seg(model, imgs, scales):
    b, c, h, w = imgs.shape
    seg_list = []
    assert 1.0 in scales, 'scale 1.0 must be in scales'

    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            _seg_pred = model(x=imgs_cat)

            _seg_pred = F.interpolate(_seg_pred, size=(h,w), mode='bilinear', align_corners=False)
            seg = torch.sum(torch.stack([_seg_pred[:b,...], _seg_pred[b:,...].flip(-1)],dim=0), dim=0)
            seg_list.append(seg)

        # seg
        seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)
    return seg

def multi_scale_seg_(model, imgs, scales):
    b, c, h, w = imgs.shape
    seg_list = []
    assert 1.0 in scales, 'scale 1.0 must be in scales'

    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            _seg_pred = model(x=imgs_cat,seg_only=True)

            _seg_pred = F.interpolate(_seg_pred, size=(h,w), mode='bilinear', align_corners=False)
            seg = torch.sum(torch.stack([_seg_pred[:b,...], _seg_pred[b:,...].flip(-1)],dim=0), dim=0)
            seg_list.append(seg)

        # seg
        seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)
    return seg

def multi_scale_cls(exmodel, imgs, scales):
    b, c, h, w = imgs.shape
    cls_list = []
    assert 1.0 in scales, 'scale 1.0 must be in scales'
    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                imgs_ = F.interpolate(imgs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            else:
                imgs_=imgs
            imgs_cat = torch.cat([imgs_, imgs_.flip(-1)], dim=0)
            ex_outputs = exmodel(x=imgs_cat)
            _cls_logits=ex_outputs[0]
            cls_logits = torch.sum(torch.stack([_cls_logits[:b,...], _cls_logits[b:,...]],dim=0), dim=0)
            cls_list.append(cls_logits)
        # seg
        cls = torch.sum(torch.stack(cls_list, dim=0), dim=0)
    return cls

def cam_to_label(cam,
                 cls_label,
                 img_box=None,
                 bkg_thre=None,
                 high_thre=None,
                 low_thre=None,
                 ignore_mid=False,
                 ignore_index=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    if cls_label is not None:
        cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
        valid_cam = cls_label_rep * cam
    else:
        valid_cam = cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def cam_validation(cam, cls_label):
    b, c, h, w = cam.shape
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    return valid_cam

def seg_refine_by_label(seg,cls_label,softmaxtemp,after_softmax=False):
    # input seg B C+1 H W without softmax
    b, c, h, w = seg.shape
    cls_label_bk=torch.cat([torch.ones(b,1).long().to(cls_label.device),
                            cls_label],dim=1)
    if after_softmax:
        cls_label_rep = cls_label_bk.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
        seg=F.softmax(seg/softmaxtemp,dim=1)
        valid_seg = cls_label_rep * seg
    else:
        # assign large neg value first and do softmax
        valid_seg = seg.clone()
        valid_seg[cls_label_bk==0]=-1e5
        valid_seg=F.softmax(valid_seg/softmaxtemp,dim=1)

    return valid_seg

def seg_get_pseudo(seg,greater=1.5):
    # seg: valid seg after seg validation B C H W --- C:21
    # greater confidence than the second
    seg_prob=seg.clone().softmax(dim=1)
    seg_top2, seg_top2_idx = seg_prob.topk(2,dim=1)
    seg_pseudo_label=seg_top2_idx[:,0,...]
    uncertain_mask=seg_top2[:,0,...]< greater * seg_top2[:,1,...]
    seg_pseudo_label[uncertain_mask]=255
    return seg_pseudo_label


def seg_validation(seg, cls_label):
    # assign large neg value for valid seg
    b, c, h, w = seg.shape
    if cls_label is not None:
        cls_label_bk=torch.cat([torch.ones(b,1).long().to(cls_label.device),
                                cls_label],dim=1)
        valid_seg = seg.clone()
        valid_seg[cls_label_bk==0]=-1e5
        return valid_seg
    else:
        return seg

def cam_loss(cam, seg_ps,is_relu=True):
    B,C,H,W=cam.shape
    seg_ps_fg=seg_ps[:,1:,...]
    seg_ps_fg=F.interpolate(seg_ps_fg,size=[H,W],mode='bilinear',align_corners=False)
    seg_ps_fg_flat=seg_ps_fg.permute(0,2,3,1).contiguous().reshape(B*H*W,C)
    if is_relu:
        cam=F.relu(cam)
    cam_flat=cam.permute(0,2,3,1).contiguous().reshape(B*H*W,C)
    cam_loss=F.multilabel_soft_margin_loss(cam_flat,seg_ps_fg_flat)
    return cam_loss

def cam_lossv2(cam, seg_ps,detach=False):
    # add cam norm compare to the first one

    B,C,H,W=cam.shape
    cam=F.relu(cam)
    if detach:
        d1=F.adaptive_max_pool2d(-cam, (1, 1)).detach()
        d2=F.adaptive_max_pool2d(cam, (1, 1)).detach() +1e-4
    else:
        d1=F.adaptive_max_pool2d(-cam, (1, 1))
        d2=F.adaptive_max_pool2d(cam, (1, 1)) +1e-4
    cam = cam + d1
    cam = cam / d2

    seg_ps_fg=seg_ps[:,1:,...]
    seg_ps_fg=F.interpolate(seg_ps_fg,size=[H,W],mode='bilinear',align_corners=False)
    seg_ps_fg_flat=seg_ps_fg.permute(0,2,3,1).contiguous().reshape(B*H*W,C)

    cam_flat=cam.permute(0,2,3,1).contiguous().reshape(B*H*W,C)
    cam_loss=F.multilabel_soft_margin_loss(cam_flat,seg_ps_fg_flat)
    return cam_loss

def cam_lossv3(cam, seg_label,detach=False,cambgmax=True):
    # add cam norm compare to the first one

    B,H,W= seg_label.shape
    cam=F.relu(cam)

    if detach:
        d1=F.adaptive_max_pool2d(-cam, (1, 1)).detach()
        d2=F.adaptive_max_pool2d(cam, (1, 1)).detach() +1e-4
    else:
        d1=F.adaptive_max_pool2d(-cam, (1, 1))
        d2=F.adaptive_max_pool2d(cam, (1, 1)) +1e-4
    cam = cam + d1
    cam = cam / d2

    cambg=1-torch.mean(cam,dim=1,keepdim=True) if not cambgmax else 1-torch.max(cam,dim=1,keepdim=True)[0]
    cam_mix = torch.cat([cambg,cam],dim=1)
    cam_mix = F.interpolate(cam_mix,size=[H,W],mode='bilinear',align_corners=False)
    # cam shape B C+1 H W
    # seg seg_label shape B H W
    cam_loss =seg_loss(cam_mix,seg_label)

    return cam_loss
def cam_lossv3_wrap(cam,seg_ps,seg_confident_thre=0.25):
    # change seg_ps to seg_label and apply cam_lossv3
    seg_label_value, seg_label=torch.max(seg_ps,dim=1)
    seg_label[seg_label_value<=seg_confident_thre]=255
    return cam_lossv3(cam,seg_label)

def refine_camlabel_(refine_model,
                    images,
                    img_boxes,
                    cams,
                    cls_labels,
                    threshold_high,
                    threshold_low,
                    ignore_index=255,
                    downscale=2,
                    usepar=True
                    ):
    # 1. downscale image by 2
    # 2. create bk_high and bk_low threshold map and cat to cams
    # 3. use refine_model to process images by images for lowcams and highcams
    # 4. merge lowcams and highcams
    #  output is a label mask map of images size
    b,_,h,w = images.shape
    if downscale:
        _images = F.interpolate(images, size=[h//downscale, w//downscale], mode="bilinear", align_corners=False)
    else:
        _images = images
    threshold_temp =torch.ones((b,1,h,w)).to(cams.device)
    cams_high_threshold = torch.cat([
        threshold_temp.clone() * threshold_high,
        cams
    ],dim=1)
    cams_low_threshold = torch.cat([
        threshold_temp.clone() * threshold_low,
        cams
    ],dim=1)
    if downscale:
        _cams_high_threshold = F.interpolate(cams_high_threshold, size=[h//downscale,w//downscale], mode="bilinear", align_corners=False)
        _cams_low_threshold = F.interpolate(cams_low_threshold, size=[h//downscale,w//downscale], mode="bilinear", align_corners=False)
    else:
        _cams_high_threshold = cams_high_threshold
        _cams_low_threshold = cams_low_threshold
    cls_labels_with_bkg = torch.cat([
        torch.ones((b,1)).to(cls_labels.device),
        cls_labels,
    ], dim=1)
    refined_label_mask = threshold_temp.squeeze(1) * ignore_index
    refined_label_mask_high =refined_label_mask.clone()
    refined_label_mask_low = refined_label_mask.clone()
    # for each image
    for b_idx, coord in enumerate(img_boxes):
        current_labels = torch.nonzero(cls_labels_with_bkg[b_idx,...])[:,0]
        active_cams_high = _cams_high_threshold[b_idx, current_labels,...].unsqueeze(0).softmax(dim=1)
        active_cams_low = _cams_low_threshold[b_idx, current_labels,...].unsqueeze(0).softmax(dim=1)

        # use refine model to process this both
        # _refined_label_mask_high= refine_model(_images[[b_idx], ...], active_cams_high)
        _refined_label_mask_high = _refine_cams(refine_model, _images[[b_idx], ...], active_cams_high, current_labels, orig_size=(h,w),usepar=usepar )
        # _refined_label_mask_low = refine_model(_images[[b_idx], ...], active_cams_low)
        _refined_label_mask_low = _refine_cams(refine_model, _images[[b_idx], ...], active_cams_low, current_labels, orig_size=(h,w),usepar=usepar )
        # only consider within the box
        refined_label_mask_high[b_idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_mask_high[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_mask_low[b_idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_mask_low[0, coord[0]:coord[1], coord[2]:coord[3]]

    # merge high and low
    refined_label_mask=refined_label_mask_high.clone() # high fg are fg
    refined_label_mask[refined_label_mask_high==0] = ignore_index # high think 0 are ignore
    refined_label_mask[( refined_label_mask_high + refined_label_mask_low) ==0 ] = 0 # both think 0 are 0

    return refined_label_mask


def cam2mask(
                    images,
                    img_boxes,
                    cams,
                    cls_labels,
                    threshold_high,
                    threshold_low,
                    refine_model=None,
                    ignore_index=255,
                    downscale=2,
                    ):
    # 1. downscale image by 2
    # 2. create bk_high and bk_low threshold map and cat to cams
    # 3. use refine_model to process images by images for lowcams and highcams
    # 4. merge lowcams and highcams
    #  output is a label mask map of images size
    b,_,h,w = images.shape
    if downscale:
        _images = F.interpolate(images, size=[h//downscale, w//downscale], mode="bilinear", align_corners=False)
    else:
        _images = images
    threshold_temp =torch.ones((b,1,h,w)).to(cams.device)
    cams_high_threshold = torch.cat([
        threshold_temp.clone() * threshold_high,
        cams
    ],dim=1)
    cams_low_threshold = torch.cat([
        threshold_temp.clone() * threshold_low,
        cams
    ],dim=1)
    if downscale:
        _cams_high_threshold = F.interpolate(cams_high_threshold, size=[h//downscale,w//downscale], mode="bilinear", align_corners=False)
        _cams_low_threshold = F.interpolate(cams_low_threshold, size=[h//downscale,w//downscale], mode="bilinear", align_corners=False)
    else:
        _cams_high_threshold = cams_high_threshold
        _cams_low_threshold = cams_low_threshold

    cls_labels_with_bkg = torch.cat([
        torch.ones((b,1)).to(cls_labels.device),
        cls_labels,
    ], dim=1)
    refined_label_mask = threshold_temp.squeeze(1) * ignore_index
    refined_label_mask_high =refined_label_mask.clone()
    refined_label_mask_low = refined_label_mask.clone()
    # for each image
    for b_idx, coord in enumerate(img_boxes):
        current_labels = torch.nonzero(cls_labels_with_bkg[b_idx,...])[:,0]
        active_cams_high = _cams_high_threshold[b_idx, current_labels,...].unsqueeze(0).softmax(dim=1)
        active_cams_low = _cams_low_threshold[b_idx, current_labels,...].unsqueeze(0).softmax(dim=1)

        # use refine model to process this both
        # _refined_label_mask_high= refine_model(_images[[b_idx], ...], active_cams_high)
        _refined_label_mask_high = _refine_cams(refine_model, _images[[b_idx], ...], active_cams_high, current_labels, orig_size=(h,w))
        # _refined_label_mask_low = refine_model(_images[[b_idx], ...], active_cams_low)
        _refined_label_mask_low = _refine_cams(refine_model, _images[[b_idx], ...], active_cams_low, current_labels, orig_size=(h,w))
        # only consider within the box
        refined_label_mask_high[b_idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_mask_high[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_mask_low[b_idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_mask_low[0, coord[0]:coord[1], coord[2]:coord[3]]

    # merge high and low
    refined_label_mask=refined_label_mask_high.clone() # high fg are fg
    refined_label_mask[refined_label_mask_high==0] = ignore_index # high think 0 are ignore
    refined_label_mask[( refined_label_mask_high + refined_label_mask_low) ==0 ] = 0 # both think 0 are 0

    return refined_label_mask

def _refine_cams(refine_model, images, cams, valid_key, orig_size):

    if refine_model:
        refined_cams = refine_model(images, cams)
    else:
        refined_cams = cams
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def seg_loss(seg_pred,mask_label,fg_alpha=0.5,ignore_index=255):
    # seg_pred: B C+1 H W
    # label: B H W
    assert fg_alpha>=0 and fg_alpha<=1, "fg_alpha should be in [0,1]"
    bg_label = mask_label.clone()
    bg_label[mask_label!=0] = ignore_index
    bg_loss = F.cross_entropy(seg_pred, bg_label.long(), ignore_index=ignore_index,reduction='sum')/(( bg_label !=ignore_index ).sum()+1e-6)

    fg_label = mask_label.clone()
    fg_label[mask_label==0] = ignore_index
    # fg_loss = F.cross_entropy(seg_pred, fg_label.long(), ignore_index=ignore_index)
    fg_loss = F.cross_entropy(seg_pred, fg_label.long(), ignore_index=ignore_index,reduction='sum')/( (fg_label !=ignore_index).sum()+1e-6)

    return (1-fg_alpha)*bg_loss + fg_alpha*fg_loss

def seg_lossv2(seg_pred,mask_label,ignore_index=255):
    # not distiguish bg and fg
    # seg_pred: B C+1 H W
    # label: B H W
    loss = F.cross_entropy(seg_pred, mask_label.long(), ignore_index=ignore_index,reduction='sum')/(( mask_label !=ignore_index ).sum()+1e-6)

    return loss

def seg_weightloss(seg_pred, mask_label,mask_weights, fg_alpha=0.5,ignore_index=255):
    assert fg_alpha>=0 and fg_alpha<=1, "fg_alpha should be in [0,1]"
    bg_label = mask_label.clone()
    bg_label[mask_label!=0] = ignore_index
    bg_loss_raw = F.cross_entropy(seg_pred, bg_label.long(), ignore_index=ignore_index,reduction='none')
    bg_loss = (bg_loss_raw*mask_weights).sum()/((bg_label!=ignore_index).sum() + 1e-6)

    fg_label = mask_label.clone()
    fg_label[mask_label==0] = ignore_index
    fg_loss_raw = F.cross_entropy(seg_pred, fg_label.long(), ignore_index=ignore_index,reduction='none')
    fg_loss = (fg_loss_raw*mask_weights).sum()/((fg_label!=ignore_index).sum() + 1e-6)

    return (1-fg_alpha)*bg_loss + fg_alpha*fg_loss

def seg_softloss(seg_pred,softprobs,fg_alpha=0.5):
    # seg_pred: B C+1 H W
    # softprobs: B C+1 H W
    labels=softprobs.argmax(dim=1)

    seg_pred_bg=seg_pred.permute(0,2,3,1)[labels==0] # N C+1
    softprobs_bg=softprobs.permute(0,2,3,1)[labels==0] # N C+1
    bgloss=seg_softlossv2(seg_pred_bg,softprobs_bg)

    seg_pred_fg=seg_pred.permute(0,2,3,1)[labels!=0] # N C+1
    softprobs_fg=softprobs.permute(0,2,3,1)[labels!=0] # N C+1
    fgloss=seg_softlossv2(seg_pred_fg,softprobs_fg)

    loss = (1-fg_alpha)*bgloss + fg_alpha*fgloss

    return loss

def seg_softlossv2(seg_pred,softprobs):
    # not distiguish bg and fg
    # seg_pred: B C+1 H W
    # softprobs: B C+1 H W
    loss = - F.log_softmax(seg_pred,dim=1) * softprobs
    loss = loss.sum(dim=1).mean()

    return loss


class DenseEnergyLossFunction(Function):

    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)

        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None

def crf_inference_inf(img, probs, t=10, scale_factor=1, labels=21):

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=4/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83/scale_factor, srgb=5, rgbim=np.copy(img_c), compat=3)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def rungmm(queue,modal,filter_thre=0.05):
    assert modal in [2,3]
    queue=queue.flatten()
    queue=queue[queue>filter_thre]
    queue=queue.reshape(-1,1)

    if modal==3:
        means_init = [[np.min(queue)],[np.median(queue)],[np.max(queue)]]
        weights_init = [1 / 3, 1/3,  1 / 3]
        precisions_init = [[[1.0]],[[1.0]], [[1.0]]]
    elif modal==2:
        means_init = [[np.min(queue)],[np.max(queue)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]],[[1.0]]]
    gmm = skm.GaussianMixture(modal ,weights_init=weights_init,means_init=means_init,precisions_init=precisions_init)
    prediction=gmm.fit_predict(queue)
    if modal==2:
        return max(queue[prediction==0]).item()
    else:
        return max(queue[prediction==0]).item(),min(queue[prediction==2]).item()


class DynamicQueue(object):
    def __init__(self, max_size, dim, batch_size):
        self.max_size = max_size
        self.queue = np.random.random((max_size,dim))
        self.ptr=0
        self.batch_size=batch_size

    def update(self,income):
        # income -> batchsize,dim
        self.queue[self.ptr:self.ptr+self.batch_size,:]=income
        self.ptr=(self.ptr+self.batch_size)%self.max_size

    def getqueue(self):
        return self.queue

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U =unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q
crf_inference_infv2= DenseCRF(
    iter_max=1,
    pos_xy_std=1,   # 3
    pos_w=1,        # 3
    bi_xy_std=121,  # 121, 140
    bi_rgb_std=5,   # 5, 5
    bi_w=4,         # 4, 5
    )
