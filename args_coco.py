import argparse

default_args = dict(
    ###########
    #  model  #
    ###########
    model='vit',
    backbone='vit_base_patch16_224',
    decoder='LargeFOV',
    pretrained=True,
    freeze_norm=False,
    aux_layer=-3,
    isgap=False,
    ##########
    #  misc  #
    ##########
    finalval=True,
    seed=0,
    work_dir = '', #'/path/to/work_dir',
    device='cuda',
    save_per_eval=10,
    eval_iters=6000,
    turnon_rawcam=False,
    fasteval=False,
    valfull=False,
    ##########
    #  data  #
    ##########
    dataset='COCO',
    crop_size=448,
    scales=(0.5,2),
    ignore_index=255,
    num_classes=81,
    coco_root='', #'/path/to/coco',
    batch_size=4,
    num_workers=4,
    ###########
    #  train  #
    ###########
    max_iters=60000,
    warmup_iters=10000,
    lr=6e-5,
    min_mult=0.,
    wt_dec=1e-2,
    wt_dec_mult=1.,
    momentum=0.9994,
    seg_weight=0.1,
    segfg_alpha=0.5,
    cam_weight=0.05,
    camloss_version='v1',
    segconf_thre=0.25,
    seg_softmaxtemp=0.01,
    reg_weight=0.05,
    pseudo_scales=[1.0,0.5,1.5],
    high_thre=0.65,
    high_thre_aux=0.7,
    bkg_thre=0.5,
    low_thre=0.25,
    low_thre_aux=0.25,
    usegmm=False,
    usegmmaux=False,
    gmmscale=16,
    gmmfilter_thre=0.05,
    gmmemadecay=0.99,
    queue_update_ratio=100, # 1 out of 100 in the queue are updated
    camweight_beta=1.0,
    par_downscale=2, # 0 for no downscale
    usepar=False, # usepar by default
    aux_cam2seg=True,
    aux_cam2seg_traditional=True,
    aux_cam2seg_alpha=0.5,
    aux_seg2cam=False,
    aux_seg2cam_alpha=0.5,
    after_softmax=False,
    detach='none', #['all', 'feat', 'none','cls']
    use_cammix=False,
    oracle_camloss_version='v1',
    oracle_camloss_detach=False,
    oracle_camloss_bgmax=True,
)


def get_parser():
    parser = argparse.ArgumentParser(
        'End to end weakly supervised segmentation model', add_help=False)
    parser.add_argument('name', type=str)
    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--decoder', type=str)
    parser.add_argument('--pretrained', type=str2bool)
    parser.add_argument('--freeze_norm', default=False, action='store_true')
    parser.add_argument('--aux_layer', type=int)
    parser.add_argument('--isgap', type=str2bool)
    # misc
    parser.add_argument('--finalval', type=str2bool)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--random_seed', action='store_true')
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--save_per_eval', type=int)
    parser.add_argument('--eval_iters', type=int)
    parser.add_argument('--turnon_rawcam', action='store_true')
    parser.add_argument('--fasteval', action='store_true')
    parser.add_argument('--valfull', action='store_true')
    parser.add_argument('--eval_threshold_filters', type=float, metavar='N', nargs='+',default=None)
    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--coco_root', type=str)
    parser.add_argument('--crop_size', type=str)
    parser.add_argument('--scales', type=tuple)
    parser.add_argument('--ignore_index', type=int)
    parser.add_argument('--num_classes', type=int)
    # parser.add_argument('--train_list', type=str)
    # parser.add_argument('--val_list', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    # train
    parser.add_argument('--max_iters', type=int)
    parser.add_argument('--warmup_iters', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lrscale', default=10., type=float)
    parser.add_argument('--min_mult', type=float)
    parser.add_argument("--wt_dec", type=float)
    parser.add_argument("--wt_dec_mult", type=float)
    parser.add_argument("--cam_weight", type=float)
    parser.add_argument("--camloss_version", type=str)
    parser.add_argument("--seg_weight", type=float)
    parser.add_argument("--segfg_alpha", type=float)
    parser.add_argument("--reg_weight", type=float)
    # parser.add_argument('--grad_norm', type=int)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--pseudo_scales', type=float,metavar='N', nargs='+')
    parser.add_argument('--high_thre', type=float)
    parser.add_argument('--high_thre_aux', type=float)
    parser.add_argument('--bkg_thre', type=float)
    parser.add_argument('--low_thre', type=float)
    parser.add_argument('--low_thre_aux', type=float)
    parser.add_argument('--usegmm', type=str2bool)
    parser.add_argument('--usegmmaux', type=str2bool)
    parser.add_argument('--gmmscale', type=int)
    parser.add_argument('--gmmfilter_thre', type=float)
    parser.add_argument('--gmmemadecay', type=float)
    parser.add_argument('--queue_update_ratio', type=int)
    parser.add_argument('--camweight_beta', type=float)
    parser.add_argument('--par_downscale', type=int)
    parser.add_argument('--usepar', type=str2bool)
    parser.add_argument('--aux_cam2seg', type=str2bool)
    parser.add_argument('--aux_cam2seg_traditional', type=str2bool)
    parser.add_argument('--aux_cam2seg_alpha', type=float)
    parser.add_argument('--aux_seg2cam', type=str2bool)
    parser.add_argument('--aux_seg2cam_alpha', type=float)
    parser.add_argument('--seg_softmaxtemp', type=float)
    parser.add_argument('--segconf_thre', type=float)
    parser.add_argument('--after_softmax', type=str2bool)
    parser.add_argument('--detach', type=str)
    parser.add_argument('--use_cammix', type=str2bool)
    parser.add_argument('--oracle_camloss_version', type=str)
    parser.add_argument('--oracle_camloss_detach', type=str2bool)
    parser.add_argument('--oracle_camloss_bgmax', type=str2bool)
    parser.add_argument('--find_unused', type=str2bool,default=True)
    # parser.add_argument('--seg_eval_use_crf', action='store_true')
    # parser.add_argument('--eval_save_seg', action='store_const', const=True)
    return parser


def handle_defaults(args, default_args=default_args):
    changed = {}
    runtime_args = vars(args)
    for k, v in default_args.items():
        args_v = runtime_args[k]
        if v is not None:
            assert type(args_v) == type(
                v) or args_v is None, f'{k} is {type(v)} not of type {type(args_v)}'
        if args_v is None:
            setattr(args, k, v)
        else:
            changed[k] = args_v
    return args, changed

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
