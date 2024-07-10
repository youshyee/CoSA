import numpy as np
# import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
from . import randaug
import torchvision
from PIL import Image
from torchvision import transforms as T
# import random

class_list = ['_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):

    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

class COCODataset(Dataset):
    def __init__(
        self,
        root_dir=None, # coco dir
        name_list_dir=None, # dataloader/coco
        split='train',
        stage='train',
    ):
        super().__init__()

        splitclean=split
        if split[:3]=='val':
            splitclean='val'

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, splitclean+'2014')
        self.label_dir = os.path.join(root_dir, f'SegmentationClass/{splitclean}2014')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(Image.open(img_name).convert('RGB'))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label



class COCOClsDatasetNew(COCODataset):
    def __init__(self,
                 root_dir,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 rescale_range=[0.5, 2.0],
                 crop_size=448,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=81,
                 aug=True,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        assert aug
        self.split=split
        self.ignore_index = ignore_index
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes

        self.gaussian_blur = transforms.GaussianBlur(p=0.5)

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.strong_transfrom = randaug.OneOf(transforms=[
            randaug.Identity(),
            randaug.AutoContrast(),
            randaug.RandEqualize(),
            randaug.RandSolarize(),
            randaug.RandColor(),
            randaug.RandContrast(),
            randaug.RandBrightness(),
            randaug.RandSharpness(),
            randaug.RandPosterize()])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        # basic transform
        image = transforms.random_scaling(image, scale_range=self.rescale_range)
        image = transforms.random_fliplr(image)
        image, img_box = transforms.random_crop(image, crop_size=self.crop_size, mean_rgb=[0,0,0], ignore_index=self.ignore_index)
        imagepil=self.gaussian_blur(Image.fromarray(image))

        weak_img = self.normalize(imagepil)
        strong_pil=self.strong_transfrom(imagepil)
        strong_img = self.normalize(strong_pil)

        return weak_img, strong_img, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, _ = super().__getitem__(idx)


        wimage, simage, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]


        return img_name, wimage, simage, cls_label, img_box

class COCOSegDataset(COCODataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.split = split
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        # self.name_list = self.name_list[:100]

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(image, label, crop_size=self.crop_size, mean_rgb=[123.675, 116.28, 103.53], ignore_index=self.ignore_index)

        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        if self.stage == "test":
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label

    def __str__(self):
        return 'dataset name: {}\n'.format(self.__class__.__name__)  +  f'length: {len(self)} \n' +'split: {}\n'.format(self.split)

# class VOC12SegDatasetNew(VOC12Dataset):
#     def __init__(self,
#                  root_dir,
#                  name_list_dir=None,
#                  split='train_aug',
#                  stage='train',
#                  rescale_range=[0.5, 2.0],
#                  crop_size=448,
#                  img_fliplr=True,
#                  ignore_index=255,
#                  num_classes=21,
#                  aug=True,
#                  **kwargs):
#
#         super().__init__(root_dir, name_list_dir, split, stage)
#
#         self.aug = aug
#         assert aug
#         self.split=split
#         self.ignore_index = ignore_index
#         self.rescale_range = rescale_range
#         self.crop_size = crop_size
#         self.img_fliplr = img_fliplr
#         self.num_classes = num_classes
#
#         self.gaussian_blur = transforms.GaussianBlur(p=0.5)
#
#         self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
#
#         self.normalize = T.Compose([
#             T.ToTensor(),
#             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
#
#         self.strong_transfrom = randaug.OneOf(transforms=[
#             randaug.Identity(),
#             randaug.AutoContrast(),
#             randaug.RandEqualize(),
#             randaug.RandSolarize(),
#             randaug.RandColor(),
#             randaug.RandContrast(),
#             randaug.RandBrightness(),
#             randaug.RandSharpness(),
#             randaug.RandPosterize()])
#
#     def __len__(self):
#         return len(self.name_list)
#
#     def __transforms(self, image,label):
#         img_box = None
#         # basic transform
#         image, label = transforms.random_scaling(image,label=label, scale_range=self.rescale_range)
#         image, label = transforms.random_fliplr(image,label=label)
#         image, label, img_box = transforms.random_crop(image,label=label, crop_size=self.crop_size, mean_rgb=[0,0,0], ignore_index=self.ignore_index)
#         imagepil=self.gaussian_blur(Image.fromarray(image))
#
#         weak_img = self.normalize(imagepil)
#         strong_pil=self.strong_transfrom(imagepil)
#         strong_img = self.normalize(strong_pil)
#
#         return weak_img, strong_img, img_box, label
#
#
#     def __getitem__(self, idx):
#
#         img_name, image, label = super().__getitem__(idx)
#
#
#         wimage, simage, img_box, label = self.__transforms(image=image,label=label)
#
#         cls_label = self.label_list[img_name]
#
#
#         return img_name, wimage, simage, cls_label, img_box, label
