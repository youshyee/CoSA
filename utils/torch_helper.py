import torch
import numpy as np
import random
import os
import subprocess
from sklearn.metrics import average_precision_score
import datetime
import time
from . import misc as utils
from texttable import Texttable

def format_tabs(scores, name_list, cat_list=None,getmIoU_list=True):

    _keys = list(scores[0]['iou'].keys())
    _values = []

    for i in range(len(name_list)):
        _values.append(list(scores[i]['iou'].values()))

    _values = np.round(np.array(_values) * 100,2)

    t = Texttable(max_width=150)
    t.header(["Class"] + name_list)

    for i in range(len(_keys)):
        t.add_row([cat_list[i]] + list(_values[:, i]))

    t.add_row(["mIoU"] + list(_values.mean(1)))

    return t.draw(), _values.mean(1)[-1], list(_values.mean(1))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def cal_itertime(prevtime,iters):
    time_now = time.time()
    diff=time_now-prevtime
    return round(diff/iters,2), time_now

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

class EMAtracker:
    def __init__(self, initial_value=0, decay=0.9):
        self.X = initial_value
        self.decay = decay

    def update(self, newvlaue):
        self.X = self.X * self.decay + newvlaue * (1-self.decay)

    def get(self):
        return self.X

def save_best(output_dir,
              model,
              finish_epoch,
              result,
              args,
              s_or_t,
              comment='',
              ):
        print('Saving checkpoint to ', output_dir)
        checkpoint_path=output_dir / f'best_{comment}.pth'
        utils.save_on_master({
            's_or_t': s_or_t,
            'model': model.state_dict(),
            'epoch': finish_epoch,
            'args': args,
            'result': result,
        }, checkpoint_path)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # input: output: NxC torch tensor of logit
    # target: N torch tensor of long
    # output: list of topk predictions
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
    return AP


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        # print('I am here ',args.gpu)
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        raise NotImplementedError
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0, workdir=str(args.output_dir))

def setup_for_distributed(is_master, workdir):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
            # for logging purposes
            with open(os.path.join(workdir, 'print.out'), 'a') as f:
                builtin_print(*args, file=f, **kwargs)

    __builtin__.print = print
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

class CosWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8,)

        self.global_step = 0
        self.warmup_iter = np.float(warmup_iter)
        self.warmup_ratio = warmup_ratio
        self.max_iter = np.float(max_iter)
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = self.global_step / self.warmup_iter
            lr_add = (1 - self.global_step / self.warmup_iter) * self.warmup_ratio
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult + lr_add

        elif self.global_step < self.max_iter:

            lr_mult = np.cos((self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter) * np.pi) * 0.5 + 0.5
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyWarmupAdamW(torch.optim.AdamW):

    def __init__(self, params, lr, weight_decay, betas, warmup_iter, max_iter, warmup_ratio, power, min_mult=0, **kwargs):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8,)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power
        self.min_mult = min_mult

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        elif self.global_step < self.max_iter:

            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
            lr_mult = max(lr_mult, self.min_mult)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyWarmupSGD(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, warmup_iter=None, max_iter=None, warmup_ratio=None, power=None, **kwargs):
        super().__init__(params, lr=lr, momentum=0.9, weight_decay=weight_decay,)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_lr = warmup_ratio
        self.max_iter = max_iter
        self.power = power

        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        ## adjust lr
        if self.global_step < self.warmup_iter:

            lr_mult = (1 - self.global_step / self.warmup_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult * 10

        elif self.global_step < self.max_iter:

            lr_mult = (1 - (self.global_step - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # step
        super().step(closure)

        self.global_step += 1

class PolyOptimizer_cls(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                if i == 4:
                    self.param_groups[i]['lr'] = self.__initial_lr[i]
                else:
                    self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def denormalize_img_(imgs, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img(imgs):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img_(imgs)

    return imgs / 255.0

