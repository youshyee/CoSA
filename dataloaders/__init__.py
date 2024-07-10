from . import voc
from . import coco
# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch




def build_val_dataset(args):
    if args.dataset =='VOC12':
        val_dataset = voc.VOC12SegDataset(
            root_dir=args.voc12_root,
            name_list_dir='./dataloaders/voc/',
            split='val',
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    elif args.dataset =='COCO':
        val_dataset = coco.COCOSegDataset(
            root_dir=args.coco_root,
            name_list_dir='./dataloaders/coco/',
            split='val_part' if not args.valfull else 'val',
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    else:
        raise NotImplementedError
    return val_dataset

def build_test_dataset(args):
    if args.dataset =='VOC12':
        val_dataset = voc.VOC12SegDataset(
            root_dir=args.voc12_root,
            name_list_dir='./dataloaders/voc/',
            split='val',
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    elif args.dataset =='COCO':
        val_dataset = coco.COCOSegDataset(
            root_dir=args.coco_root,
            name_list_dir='./dataloaders/coco/',
            split='val',
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    else:
        raise NotImplementedError
    return val_dataset

def build_train_datasetv2(args):
    if args.dataset =='VOC12':
        train_dataset = voc.VOC12ClsDatasetNew(
            root_dir=args.voc12_root,
            name_list_dir='./dataloaders/voc/',
            split='train_aug',
            stage='train',
            aug=True,
            rescale_range=args.scales,
            crop_size=args.crop_size,
            img_fliplr=True,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    elif args.dataset == 'COCO':
        train_dataset = coco.COCOClsDatasetNew(
            root_dir=args.coco_root,
            name_list_dir='./dataloaders/coco/',
            split='train',
            stage='train',
            aug=True,
            rescale_range=args.scales,
            crop_size=args.crop_size,
            img_fliplr=True,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
        )
    else:
        raise NotImplementedError
    return train_dataset



def build_dataloader(args, is_train=True):
    if is_train:
        train_dataset=build_train_datasetv2(args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   sampler=train_sampler,
                                                   )
        val_dataset=build_val_dataset(args)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 pin_memory=False,
                                                 sampler=val_sampler,
                                                 drop_last=False)
        return train_loader, val_loader
    else:
        test_dataset=build_test_dataset(args)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 pin_memory=False,
                                                  sampler=test_sampler,
                                                 drop_last=False)
        return test_loader

