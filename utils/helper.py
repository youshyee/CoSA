import numpy as np
import torch
from . import misc as utils
import random
import torch.nn.functional as F
from PIL import Image

def set_seed(args):
    if args.random_seed:
        args.seed = np.random.randint(0, 1000000)
    if args.resume:
        resume_args = torch.load(args.resume, map_location='cpu')['args']
        args.seed = resume_args.seed

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")

def init_model(args,model,output_dir,isteacher=False):
    print('Initialized from the pre-training model')
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.pretrained_model_index:
            checkpoint = checkpoint[args.pretrained_model_index]
        msg=model.load_state_dict(checkpoint, strict=False)
        print(msg)
    isresume = False
    # checkpoint format
    # dict: ['model'], ['optimizer'], ['lr_scheduler'], ['args'],['epoch'], ['eval_result'']
    if args.resume:
        isresume = True
        resume = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
    elif (output_dir/'checkpoint.pth').exists():  # auto resume
        isresume = True
        resume = str(output_dir/'checkpoint.pth')
        checkpoint = torch.load(resume, map_location='cpu')

    resume_dict=None
    epoch=0
    if isresume:
        epoch=checkpoint['epoch']
        resume_dict = torch.load(resume, map_location='cpu')
        name='teacher' if isteacher else 'student'
        model_state_dict = resume_dict.pop(name)
        outload = model.load_state_dict(model_state_dict, strict=True)
        print(f'Resume from {resume}', outload)

    return model, isresume, resume_dict, epoch

def save_ckpt(output_dir,
              model,
              optimizer,
              lr_scheduler,
              isbest,
              finish_epoch,
              eval_result,
              args):
        print('Saving checkpoint to ', output_dir)
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if isbest:
            checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'student': model[0].module.state_dict(),
                'teacher': model[1].state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                'epoch': finish_epoch,
                'args': args,
                'eval_result': eval_result,
            }, checkpoint_path)

def _crf_with_alpha(ori_img,cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(ori_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = np.zeros([21, bg_score.shape[1], bg_score.shape[2]])
    n_crf_al[0, :, :] = crf_score[0, :, :]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]
    return n_crf_al

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def cam2seglabel(cam,label,ori_images):
    b,h,w,c=ori_images.shape
    cam_up = compute_cam_up(cam, label, w, h, b)
    seg_label = np.zeros((b, w, h))
    for i in range(b):
        cam_up_single = cam_up[i]
        cam_label = label[i].cpu().numpy()
        ori_img = ori_images[i].astype(np.uint8)
        norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)
        seg_label[i] = compute_seg_label(ori_img, cam_label, norm_cam)
    return seg_label

def compute_cam_up(cam, label, w, h, b):
    cam_up = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    cam_up = cam_up * label.clone().view(b, 20, 1, 1)
    cam_up = cam_up.cpu().data.numpy()
    return cam_up

def compute_seg_label(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8)

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]

    bg_score = np.power(1 - np.max(cam_np, 0), 32)
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))
    _, bg_w, bg_h = bg_score.shape

    cam_img = np.argmax(cam_all, 0)

    crf_la = _crf_with_alpha(ori_img, cam_dict, 4)
    crf_ha = _crf_with_alpha(ori_img, cam_dict, 32)
    crf_la_label = np.argmax(crf_la, 0)
    crf_ha_label = np.argmax(crf_ha, 0)
    crf_label = crf_la_label.copy()
    crf_label[crf_la_label == 0] = 255

    single_img_classes = np.unique(crf_la_label)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)
    for class_i in single_img_classes:
        if class_i != 0:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            cam_class_order = cam_class[cam_class > 0.1]
            cam_class_order = np.sort(cam_class_order)
            confidence_pos = int(cam_class_order.shape[0] * 0.6)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        else:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            class_sure_region = (cam_class > 0.8)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region

    crf_label[crf_ha_label == 0] = 0
    crf_label_np = np.concatenate([np.expand_dims(crf_ha[0, :, :], axis=0), crf_la[1:, :, :]])
    crf_not_sure_region = np.max(crf_label_np, 0) < 0.8
    not_sure_region = np.logical_or(crf_not_sure_region, cam_not_sure_region)

    crf_label[not_sure_region] = 255

    return crf_label

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):
        if self.global_step < (0.5*self.max_step):
            lr_mult = (1 - self.global_step / (0.5*self.max_step)) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        elif self.global_step < self.max_step:
             lr_mult = (1 - (self.global_step-(0.5*self.max_step)) / (self.max_step-(0.5*self.max_step))) ** self.momentum

             for i in range(len(self.param_groups)):
                 self.param_groups[i]['lr'] = 0.0007 * lr_mult

        super().step(closure)

        self.global_step += 1

def save_seg(seg,filepath):
    # save seg mask after the argmax np HxW
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]
    out = seg.astype(np.uint8)
    out = Image.fromarray(out, mode='P')
    out.putpalette(palette)
    out.save(filepath)

