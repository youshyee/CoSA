import numpy as np
# import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
from . import randaug
from PIL import Image
from torchvision import transforms as T
# import random

class_list = ["_background_",
              'aeroplane',
              'bicycle',
              'bird',
              'boat',
              'bottle',
              'bus',
              'car',
              'cat',
              'chair',
              'cow',
              'diningtable',
              'dog',
              'horse',
              'motorbike',
              'person',
              'pottedplant',
              'sheep',
              'sofa',
              'train',
              'tvmonitor']

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):

    return np.load(os.path.join(name_list_dir,'cls_labels_onehot.npy'), allow_pickle=True).item()

class VOC12Dataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir,  'JPEGImages_test' if split=='test' else 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(imageio.imread(img_name))

        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name+'.png')
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":
            label = image[:,:,0]

        return _img_name, image, label


class VOC12ClsDataset(VOC12Dataset):
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
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.split=split
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        # self.color_jittor = transforms.PhotoMetricDistortion()

        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=1.0),
            # self.normalize,
        ])
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.1),
            self.solarization,
            self.normalize,
        ])
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        local_image = None
        if self.aug:

            if self.rescale_range:
                image = transforms.random_scaling(image, scale_range=self.rescale_range)
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(image, crop_size=self.crop_size, mean_rgb=[0,0,0], ignore_index=self.ignore_index)

            local_image = self.local_view(Image.fromarray(image))
            image = self.global_view1(Image.fromarray(image))

        image = self.normalize(image)

        return image, local_image, img_box

    def __str__(self):
        output_str = 'dataset name: {}\n'.format(self.__class__.__name__)  +  f'length: {len(self)} \n' +'split: {}\n'.format(self.split) + 'stage: {}\n'.format(self.stage) + 'num images: {}\n'.format(len(self.name_list)) + \
            'Transforms used: ' + f'( Isaug: {self.aug} ) \n' +'global_view1: \n' + 'random_scaling' if self.rescale_range else '' + 'random_fliplr' if self.img_fliplr else '' + 'random_crop' if self.crop_size else '' + \
            'global_view2: \n' + 'random_resized_crop' if self.crop_size else '' + 'random_fliplr' if self.img_fliplr else '' + 'random_blur' + 'random_solarization' + 'normalize' + \
            'local_view: \n' + 'random_resized_crop' if self.local_crop_size else '' + 'random_fliplr' if self.img_fliplr else '' + 'random_blur' + 'normalize' + \
            'label: \n' + 'onehot' + 'num_classes: {}'.format(self.num_classes) + 'ignore_index: {}'.format(self.ignore_index)
        return output_str

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

        pil_image = Image.fromarray(image)

        image, local_image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[img_name]

        if self.aug:

            crops = []
            crops.append(image)
            crops.append(self.global_view2(pil_image))
            crops.append(local_image)
            # for _ in range(8):
            #     crops.append(self.local_view(pil_image))

            return img_name, image, cls_label, img_box, crops
        else:
            return img_name, image, cls_label


class VOC12ClsDatasetNew(VOC12Dataset):
    def __init__(self,
                 root_dir,
                 name_list_dir=None,
                 split='train_aug',
                 stage='train',
                 rescale_range=[0.5, 2.0],
                 crop_size=448,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
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

class VOC12SegDataset(VOC12Dataset):
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
                image, label,_ = transforms.random_crop(image, label, crop_size=self.crop_size, mean_rgb=[123.675, 116.28, 103.53], ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
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

class VOC12SegDatasetNew(VOC12Dataset):
    def __init__(self,
                 root_dir,
                 name_list_dir=None,
                 split='train_aug',
                 stage='train',
                 rescale_range=[0.5, 2.0],
                 crop_size=448,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
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

    def __transforms(self, image,label):
        img_box = None
        # basic transform
        image, label = transforms.random_scaling(image,label=label, scale_range=self.rescale_range)
        image, label = transforms.random_fliplr(image,label=label)
        image, label, img_box = transforms.random_crop(image,label=label, crop_size=self.crop_size, mean_rgb=[0,0,0], ignore_index=self.ignore_index)
        imagepil=self.gaussian_blur(Image.fromarray(image))

        weak_img = self.normalize(imagepil)
        strong_pil=self.strong_transfrom(imagepil)
        strong_img = self.normalize(strong_pil)

        return weak_img, strong_img, img_box, label


    def __getitem__(self, idx):

        img_name, image, label = super().__getitem__(idx)


        wimage, simage, img_box, label = self.__transforms(image=image,label=label)

        cls_label = self.label_list[img_name]


        return img_name, wimage, simage, cls_label, img_box, label
