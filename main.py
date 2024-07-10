import argparse
import datetime
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import utils.misc as utils
from args import get_parser as get_parser_voc
from args import handle_defaults as handle_defaults_voc
from args_coco import get_parser as get_parser_coco
from args_coco import handle_defaults as handle_defaults_coco
from dataloaders import build_dataloader
from evaluation_engine import evaluate
from models import build_model
from utils import seg_helper, torch_helper


def main(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.work_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    if args.random_seed:
        args.seed=random.randint(1,10000)
    torch_helper.setup_seed(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("{}".format(args).replace(', ', ',\n'))

    train_loader, val_loader = build_dataloader(args)
    dataset_train = train_loader.dataset
    dataset_val = val_loader.dataset
    print(dataset_train)
    print(dataset_val)

    model_ON = build_model(args).to(device)
    model_AN = build_model(args).to(device)
    param_groups = model_ON.get_param_groups()

    model_ON = torch.nn.parallel.DistributedDataParallel(
        model_ON, device_ids=[args.gpu],find_unused_parameters=args.find_unused)

    n_parameters = sum(p.numel()
                       for p in model_ON.parameters() if p.requires_grad)
    print('Number of trainable params for Network: {}M'.format(n_parameters//1000000))


    optimizer = torch_helper.PolyWarmupAdamW(
        params=[
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': args.lr if not args.freeze_norm else 0, 'weight_decay': args.wt_dec * args.wt_dec_mult if not args.freeze_norm else 0},
        {'params': param_groups[2], 'lr': args.lrscale*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': args.lrscale*args.lr, 'weight_decay': args.wt_dec},
    ],
        lr=args.lr,
        weight_decay=args.wt_dec,
        betas=(0.9, 0.999),
        warmup_iter=1500,
        max_iter=args.max_iters,
        warmup_ratio=1e-6,
        power=0.9,
        min_mult=args.min_mult,
    )

    train_loader.sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = torch_helper.AverageMeter()
    reguliser_layer = seg_helper.DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    print("Start training")
    start_time = time.time()
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    df=None
    loss_df={'overall_loss':[],'cls_loss':[],'cls_acc':[],'cls_aux_loss':[],'cls_aux_acc':[],'seg_loss':[],'cam_loss':[],'reg_loss':[],'iters':[]}

    best_seg=-1
    best_cam=-1

    hist_lowthre=[]
    hist_highthre=[]
    hist_auxlowthre=[]
    hist_auxhighthre=[]

    if args.usegmm:
        queuedim=(args.crop_size // args.gmmscale) * (args.crop_size // args.gmmscale) * 1
        cam_queue = seg_helper.DynamicQueue(args.batch_size*args.queue_update_ratio,dim=queuedim,batch_size=args.batch_size)
        ema_lowthre=torch_helper.EMAtracker(args.low_thre,decay=args.gmmemadecay)
        ema_highthre=torch_helper.EMAtracker(args.high_thre,decay=args.gmmemadecay)

        queuedim=(args.crop_size // args.gmmscale) * (args.crop_size // args.gmmscale) * 1
        camaux_queue = seg_helper.DynamicQueue(args.batch_size*args.queue_update_ratio,dim=queuedim,batch_size=args.batch_size)
        ema_auxlowthre=torch_helper.EMAtracker(args.low_thre_aux,decay=args.gmmemadecay)
        ema_auxhighthre=torch_helper.EMAtracker(args.high_thre_aux,decay=args.gmmemadecay)

    currenttime=time.time()
    for n_iter in range(args.max_iters):

        try:
            img_name, wimg, simg, cls_label, img_box = next(train_loader_iter)
        except:
            train_loader.sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, wimg, simg, cls_label, img_box = next(train_loader_iter)
        wimg = wimg.cuda()
        simg = simg.cuda()

        img_denorm=torch_helper.denormalize_img(simg)
        cls_label = cls_label.cuda()

        # pseudo label forward with teacher model
        cam_ps,cam_aux_ps,seg_ps=seg_helper.multi_scale_camseg(model_AN, wimg, args.pseudo_scales)

        # learning model forward
        cls_final, cls_aux, feature_map, seg_pred, cam_pred, cam_aux_pred= model_ON(simg, cam_only=False, detach=args.detach)

        # cls loss and aux loss
        cls_loss = F.multilabel_soft_margin_loss(cls_final, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        #######################
        #  Seg Loss from CAM  #
        #######################
         # refine cam as pseudo label for seg
        with torch.no_grad():
            if args.use_cammix:
                cam_ps = (cam_ps + cam_aux_ps)/2
            valid_cam_ps=seg_helper.cam_validation(cam_ps,cls_label)
            if args.usegmm:
                valid_cam_ps_reduced=F.interpolate(valid_cam_ps.clone(),
                                                   size=(args.crop_size // args.gmmscale, args.crop_size // args.gmmscale),
                                                   mode='bilinear', align_corners=False)
                valid_cam_max,_=torch.max(valid_cam_ps_reduced,dim=1,keepdim=False)
                valid_cam_max = valid_cam_max.cpu().reshape(args.batch_size,-1).numpy()
                # print(valid_cam_max.shape)
                cam_queue.update(valid_cam_max)
                # if n_iter>args.warmup_iters:
                if n_iter%1==0:
                    threlow_,threhigh_=seg_helper.rungmm(cam_queue.getqueue(),modal=3,filter_thre=args.gmmfilter_thre)
                    ema_lowthre.update(threlow_)
                    ema_highthre.update(threhigh_)
                threlow,threhigh=ema_lowthre.get(),ema_highthre.get()
            else:
                threlow, threhigh = args.low_thre, args.high_thre

            hist_lowthre.append(threlow)
            hist_highthre.append(threhigh)

            refine_mask_label = seg_helper.cam2mask(
                images=img_denorm,
                img_boxes=img_box,
                cams=valid_cam_ps,
                cls_labels=cls_label,
                threshold_high=threhigh,
                threshold_low=threlow,
                downscale=args.par_downscale,
            )
        seg_pred = F.interpolate(seg_pred, size=refine_mask_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = seg_helper.seg_loss(seg_pred,
                                       refine_mask_label,
                                       fg_alpha=args.segfg_alpha)
        if args.aux_cam2seg:
            with torch.no_grad():
                valid_cam_aux_ps=seg_helper.cam_validation(cam_aux_ps, cls_label)
                if args.usegmm:
                    valid_cam_aux_ps_reduced=F.interpolate(valid_cam_aux_ps.clone(), size=(args.crop_size // args.gmmscale, args.crop_size // args.gmmscale), mode='bilinear', align_corners=False)
                    valid_cam_aux_ps_reduced,_=torch.max(valid_cam_aux_ps_reduced,dim=1,keepdim=False)
                    valid_cam_aux_ps_reduced = valid_cam_aux_ps_reduced.cpu().reshape(args.batch_size,-1).numpy()
                    camaux_queue.update(valid_cam_aux_ps_reduced)
                    # if n_iter>args.warmup_iters:
                    if n_iter%1==0:
                        auxthrelow_,auxthrehigh_=seg_helper.rungmm(camaux_queue.getqueue(),modal=3)
                        ema_auxlowthre.update(auxthrelow_)
                        ema_auxhighthre.update(auxthrehigh_)
                    auxthrelow,auxthrehigh=ema_auxlowthre.get(),ema_auxhighthre.get()
                else:
                    auxthrelow,auxthrehigh=args.low_thre_aux,args.high_thre_aux

                hist_auxlowthre.append(auxthrelow)
                hist_auxhighthre.append(auxthrehigh)

                refine_mask_label_aux = seg_helper.cam2mask(
                    images=img_denorm,
                    img_boxes=img_box,
                    cams=valid_cam_aux_ps,
                    cls_labels=cls_label,
                    threshold_high=auxthrehigh,
                    threshold_low=auxthrelow,
                    downscale=args.par_downscale,
                ) # bhw uint8 hw same as img_denorm
            seg_loss_aux = seg_helper.seg_loss(seg_pred,
                                           refine_mask_label_aux,
                                           fg_alpha=args.segfg_alpha)
            seg_loss = (1 - args.aux_cam2seg_alpha)* seg_loss + args.aux_cam2seg_alpha* seg_loss_aux

        ##############
        #  Reg Loss  #
        ##############
        reg_loss = seg_helper.get_energy_loss(img=simg,
                                              logit=seg_pred,
                                              label=refine_mask_label,
                                              img_box=img_box,
                                              loss_layer=reguliser_layer)
        #######################
        #  CAM Loss from seg  #
        #######################
        if args.camloss_version == 'v1':
            camlossfunc=seg_helper.cam_loss
        elif args.camloss_version == 'v2':
            camlossfunc=seg_helper.cam_lossv2
        elif args.camloss_version == 'v3':
            camlossfunc=seg_helper.cam_lossv3_wrap
            camlossfunc=partial(camlossfunc,seg_confident_thre=args.segconf_thre)
        else:
            raise NotImplementedError

        with torch.no_grad():
            valid_seg_ps = seg_helper.seg_refine_by_label(seg_ps, cls_label,softmaxtemp=args.seg_softmaxtemp,after_softmax=args.after_softmax)
        cam_loss=camlossfunc(cam_pred,valid_seg_ps)

        if args.aux_seg2cam:
            cam_aux_loss=camlossfunc(cam_aux_pred,valid_seg_ps)
            aux_seg2cam_alpha=args.aux_seg2cam_alpha
            cam_loss = (1-aux_seg2cam_alpha)* cam_loss + aux_seg2cam_alpha* cam_aux_loss

        ##############
        #  Training  #
        ##############

        # warmup
        if n_iter <= args.warmup_iters:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux +  0.0 * seg_loss + 0.0 * cam_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.seg_weight * seg_loss + args.cam_weight * cam_loss +  args.reg_weight * reg_loss

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # update assignment network
        with torch.no_grad():
            for param_o, param_a in zip(model_ON.module.parameters(), model_AN.parameters()):
                param_a.data.mul_(args.momentum).add_((1 - args.momentum) * param_o.detach().data)

        ##########
        #  Logs  #
        ##########
        cls_acc=np.mean(torch_helper.compute_mAP(cls_label,torch.sigmoid(cls_final.detach())))
        cls_aux_acc=np.mean(torch_helper.compute_mAP(cls_label,torch.sigmoid(cls_aux.detach())))
        avg_meter.add({
            'overall_loss': loss.item(),
            'cls_loss': cls_loss.item(),
            'cls_acc': cls_acc,
            'cls_aux_loss': cls_loss_aux.item(),
            'cls_aux_acc': cls_aux_acc,
            'seg_loss': seg_loss.item(),
            'cam_loss': cam_loss.item(),
            'reg_loss': reg_loss.item(),
        })
        log_iters=20
        if (n_iter + 1) % log_iters == 0:
            delta, eta = torch_helper.cal_eta(time0, n_iter + 1, args.max_iters)
            itertime, currenttime = torch_helper.cal_itertime(currenttime,log_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            if args.rank == 0:
                current_overall_loss=avg_meter.pop('overall_loss')
                current_cls_loss=avg_meter.pop('cls_loss')
                current_cls_acc=avg_meter.pop('cls_acc')
                current_cls_aux_loss=avg_meter.pop('cls_aux_loss')
                current_cls_aux_acc=avg_meter.pop('cls_aux_acc')
                current_seg_loss=avg_meter.pop('seg_loss')
                current_cam_loss=avg_meter.pop('cam_loss')
                current_reg_loss=avg_meter.pop('reg_loss')
                loss_df['overall_loss'].append(current_overall_loss)
                loss_df['cls_loss'].append(current_cls_loss)
                loss_df['cls_acc'].append(current_cls_acc)
                loss_df['cls_aux_loss'].append(current_cls_aux_loss)
                loss_df['cls_aux_acc'].append(current_cls_aux_acc)
                loss_df['seg_loss'].append(current_seg_loss)
                loss_df['cam_loss'].append(current_cam_loss)
                loss_df['reg_loss'].append(current_reg_loss)
                loss_df['iters'].append(n_iter+1)
                print("Iter: %d; Elasped: %s; ETA: %s; Itertime: %.2f; LR: %.3e; \n overall_loss: %.4f, cls_loss: %.4f, cls_acc: %.3f,  cls_aux_loss: %.4f, cls_aux_acc: %.3f, seg_loss: %.4f, cam_loss: %.4f, reg_loss: %.4f ..." %
                      (n_iter + 1,
                       delta,
                       eta,
                       itertime,
                       cur_lr,
                       current_overall_loss,
                       current_cls_loss,
                       current_cls_acc,
                       current_cls_aux_loss,
                       current_cls_aux_acc,
                       current_seg_loss,
                       current_cam_loss,
                       current_reg_loss,
                       )
                      )

        ################
        #  Validation  #
        ################
        eval_iters=args.eval_iters
        if (n_iter + 1 ) % eval_iters == 0:
            tab_results, segvd_miou, cam_iou, df, cls_aps_o= evaluate(
                model_ON.module,
                val_loader,
                args,
                df=df,
                save_result=False,
                epoch=n_iter+1,
                getcrf=False,
                threshold_filters=args.eval_threshold_filters,
                s_or_t='s',
                get_camiou=True,
            )
            if args.rank == 0:
                print(f'ON Model Classification: cls:{cls_aps_o[0]}, clsaux: {cls_aps_o[1]}')
                print(tab_results)

            tab_results_a,segvd_miou_a,cam_iou_a, df,cls_aps_a = evaluate(
                model_AN,
                val_loader,
                args,
                df=df,
                save_result=False,
                epoch=n_iter+1,
                save_rawcam= args.turnon_rawcam,
                getcrf=False,
                threshold_filters=args.eval_threshold_filters,
                s_or_t='t',
                get_camiou=True,
            )
            if args.rank == 0:
                print(f'AN: cls:{cls_aps_a[0]}, clsaux: {cls_aps_a[1]}')
                print(tab_results_a)

                # save model for seg
                seg_cmp_list=[round(segvd_miou,2),round(segvd_miou_a,2),best_seg]
                best_seg_index=max(range(len(seg_cmp_list)), key=seg_cmp_list.__getitem__)
                best_seg= max(seg_cmp_list)
                if best_seg_index!=2:
                    torch_helper.save_best(output_dir,
                                           model_ON.module if best_seg_index==0 else model_AN,
                                           finish_epoch=n_iter + 1,
                                           result=best_seg,
                                           args=args,
                                           s_or_t='s' if best_seg_index==0 else 't',
                                           comment='seg',
                                           )

                # save model for cam
                cam_cmp_list=[round(cam_iou,2),round(cam_iou_a,2),best_cam]
                best_cam_index=max(range(len(cam_cmp_list)), key=cam_cmp_list.__getitem__)
                best_cam= max(cam_cmp_list)
                if best_cam_index!=2:
                    torch_helper.save_best(output_dir,
                                           model_ON.module if best_cam_index==0 else model_AN,
                                           finish_epoch=n_iter + 1,
                                           result=best_cam,
                                           args=args,
                                           s_or_t='s' if best_cam_index==0 else 't',
                                           comment='cam',
                                           )

                # log results for each validation
                with (output_dir / "log_val.txt").open("a") as f:
                    f.write(f'iters:{n_iter}' + "\n")
                    if tab_results:
                        f.write(f'ON model: cls:{cls_aps_o[0]}, clsaux: {cls_aps_o[1]}'+ "\n")
                        f.write(tab_results + "\n")
                    f.write(f'AN model: cls:{cls_aps_a[0]}, clsaux: {cls_aps_a[1]}'+ "\n")
                    f.write(tab_results_a + "\n")

        torch.distributed.barrier()

    # after training
    if args.rank==0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str), 'Best val Seg mIoU: %.2f' % best_seg, 'Best val CAM mIoU: %.2f' % best_cam)

        loss_df=pd.DataFrame(loss_df)
        torch.save(loss_df,output_dir / 'loss_dataframe.pt')

    if args.finalval:
        args.bestseg_path = output_dir / 'best_seg.pth'
        print('Perform final validation on best model')
        finaleval(args)

@torch.no_grad()
def finaleval(args):
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.work_dir) / args.name
    args.output_dir = output_dir
    device = torch.device(args.device)
    model = build_model(args)
    ckpt_dict = torch.load(args.bestseg_path, map_location="cpu")
    state_dict = ckpt_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model=model.to(device)
    test_loader = build_dataloader(args, is_train=False)
    tab_results, _, _, _ = evaluate(
        model,
        test_loader,
        args,
        df=None,
        save_result=False,
        epoch='best1',
        save_rawcam= args.turnon_rawcam,
        getcrf=True,
        isfinal=True,
    )
    if args.rank==0:
        print('Final Model Result:')
        print(tab_results)
        with (output_dir / "log_val.txt").open("a") as f:
            f.write('------------'*3 + "\n")
            f.write(f'Final Model Result:' + "\n")
            f.write('------------'*3 + "\n")
            f.write(tab_results + "\n")

if __name__ =="__main__":
    parser = argparse.ArgumentParser(
        'Weakly Supervised end-to-end for Semantic Segmentation',
        parents=[get_parser_voc()])
    args = parser.parse_args()
    args, changed = handle_defaults_voc(args)
    if args.dataset =='VOC12':
        pass
    elif args.dataset=='COCO':
        parser = argparse.ArgumentParser(
            'Weakly Supervised end-to-end for Semantic Segmentation',
            parents=[get_parser_coco()])
        args = parser.parse_args()
        args, changed = handle_defaults_coco(args)
    else:
        raise NotImplementedError
    print('runnning on {args.dataset}')
    print("Changed arguments:")
    print(changed)
    main(args)
