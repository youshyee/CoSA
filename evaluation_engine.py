import torch
import torch.nn.functional as F
import numpy as np
from utils import seg_helper, torch_helper, evaluation
from tqdm import tqdm
from dataloaders import voc,coco
import os
import torch.distributed as dist


def evaluate(model,
             data_loader,
             args,
             df=None,
             save_result=False,
             save_rawcam=False,
             epoch=None,
             threshold_filters=None, # a list of threshold [0.11, 0.25]
             getcrf=False,
             s_or_t='t',
             get_camiou=False,
             isfinal=False,
             ):
    if isfinal:
        get_crf=True
    dataset=voc if args.dataset=='VOC12' else coco
    print('model validating...')
    assert s_or_t in ['s','t']
    avg_meter = torch_helper.AverageMeter()
    assert epoch if save_result else True
    current_rank = dist.get_rank()
    print('current_rank',current_rank)
    store={
        'preds_ps': [],
        'preds_vd': [],
        'gts': [],
        'cams': [],
        'cams_aux': [],
    }
    if getcrf:
        store['vd_crfs']=[]

    if threshold_filters:
        cams_threshed={}
        cams_aux_threshed={}
        for thre in threshold_filters:
            cams_threshed[thre]=[]
            cams_aux_threshed[thre]=[]
        store['cams_threshed']=cams_threshed
        store['cams_aux_threshed']=cams_aux_threshed

    model.eval()
    epoch_int = epoch
    epoch = str(epoch).zfill(5) # 20000

    if save_result:
        seg_dir=args.output_dir/epoch/"seg"
        seg_dir.mkdir(exist_ok=True,parents=True)
        cam_dir=args.output_dir/epoch/"cam"
        cam_dir.mkdir(exist_ok=True,parents=True)
        cam_aux_dir=args.output_dir/epoch/"camaux"
        cam_aux_dir.mkdir(exist_ok=True,parents=True)

        # merged dir only save cam and seg and original image and gt
        merged_dir=args.output_dir/epoch/"merged"
        merged_dir.mkdir(exist_ok=True,parents=True)
        cmp_dict ={}
        cmp_dict_path=args.output_dir/epoch/"iou_dic.pth"

    if save_rawcam:
        camraw_dir=args.output_dir/epoch/'camraw_dir'
        camraw_dir.mkdir(exist_ok=True,parents=True)

    with torch.no_grad():
        for data in tqdm(data_loader):
            name, img_org, labels, cls_label = data

            labels = labels.cuda()
            cls_label = cls_label.cuda()
            img_denorm=torch_helper.denormalize_img(img_org.cuda())
            inputs  = F.interpolate(img_org.cuda(), size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            _cams, _cams_aux, seg_ps, cls_final, cls_aux=seg_helper.multi_scale_camsegv3(model,
                                                                                         inputs.clone(),
                                                                                         [1.0,0.5,1.5,0.75,1.25],
                                                                                         getcls=True)

            cls_acc=np.mean(torch_helper.compute_mAP(cls_label,torch.sigmoid(cls_final.detach())))
            cls_aux_acc=np.mean(torch_helper.compute_mAP(cls_label,torch.sigmoid(cls_aux.detach())))
            avg_meter.add({
                'cls_acc': cls_acc,
                'cls_aux_acc': cls_aux_acc,
            })

            # for cam
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = seg_helper.cam_to_label(resized_cam.clone(),
                                                cls_label,
                                                bkg_thre=args.bkg_thre,
                                                high_thre=args.high_thre,
                                                low_thre=args.low_thre,
                                                ignore_index=args.ignore_index)

            # for cam aux
            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = seg_helper.cam_to_label(resized_cam_aux.clone(),
                                                    cls_label,
                                                    bkg_thre=args.bkg_thre,
                                                    high_thre=args.high_thre,
                                                    low_thre=args.low_thre,
                                                    ignore_index=args.ignore_index)

            # mix_cam_avg=(resized_cam+resized_cam_aux)/2
            # # for cam mix
            # cam_label_mix_avg = seg_helper.cam_to_label(mix_cam_avg.clone(),
            #                                             cls_label,
            #                                             bkg_thre=args.bkg_thre,
            #                                             high_thre=args.high_thre,
            #                                             low_thre=args.low_thre,
            #                                             ignore_index=args.ignore_index)

            resized_seg_ps = F.interpolate(seg_ps,
                                           size=labels.shape[1:],
                                           mode='bilinear',
                                           align_corners=False)
            valid_seg_ps = seg_helper.seg_validation(resized_seg_ps,cls_label)

            # for cam and camaux under threshold
            valid_cam=seg_helper.cam_validation(resized_cam.clone(),cls_label)
            valid_cam_aux=seg_helper.cam_validation(resized_cam_aux.clone(),cls_label)

            if threshold_filters:
                for thre in threshold_filters:
                    cam_label_thred = seg_helper.cam2mask(
                        images=img_denorm.clone(),
                        img_boxes=[ [0,-1,0,-1] ],
                        cams=valid_cam.clone(),
                        cls_labels=cls_label,
                        threshold_high=1-thre,
                        threshold_low=thre,
                    )
                    # for cam aux par
                    cam_aux_label_threshed= seg_helper.cam2mask(
                        images=img_denorm.clone(),
                        img_boxes=[ [0,-1,0,-1] ],
                        cams=valid_cam_aux.clone(),
                        cls_labels=cls_label,
                        threshold_high=1-thre,
                        threshold_low=thre,
                    )

                    store['cams_threshed'][thre] += list(cam_label_thred.cpu().numpy().astype(np.uint8))
                    # cams_threshed[thre] += list(cam_label_thred.cpu().numpy().astype(np.uint8))
                    # cams_aux_threshed[thre] += list(cam_aux_label_threshed.cpu().numpy().astype(np.uint8))
                    store['cams_aux_threshed'][thre] += list(cam_aux_label_threshed.cpu().numpy().astype(np.uint8))

            if save_rawcam:
                save_cam_npv2(resized_cam, name, cls_label, str(camraw_dir))

            if save_result:
                current_name=name[0]
                ori_img=torch_helper.denormalize_img_(img_org).permute(0,2,3,1).cpu().numpy()[0]

                # save seg
                segmask=torch.argmax(valid_seg_ps, dim=1).cpu()[0]
                gt=labels.cpu()[0]
                cmp_iou=assist_seg(segmask,gt,cls_label.cpu()[0])
                cmp_dict[current_name]=cmp_iou

                toshowseg=segmask.numpy().astype(np.uint8)
                segpath=seg_dir/(current_name+'.png')
                seg_helper.save_seg(toshowseg,segpath,classnum=args.num_classes)

                # save cam
                toshwocams=[
                    resized_cam.cpu().numpy()[0],
                    resized_cam_aux.cpu().numpy()[0],
                ]

                ori_cls_label=cls_label.cpu().numpy()[0]
                for j, cls_ in enumerate(ori_cls_label):
                    if cls_>0:
                        savenames=[
                            cam_dir/(current_name+'_'+dataset.class_list[1:][j]+'.png'),
                            cam_aux_dir/(current_name+'_'+dataset.class_list[1:][j]+'.png'),
                        ]
                        assert len(savenames)==len(toshwocams)
                        for savename,toshowcam in zip(savenames,toshwocams):
                            seg_helper.save_cam_on_image(ori_img,toshowcam[j],str(savename))
                        # save merge
                        current_seg_area=(toshowseg == (j + 1) ) # h x w
                        current_gt_area = (labels[0] == (j + 1) ) # h x w
                        seg_helper.save_merge(ori_img,
                                              toshwocams[0][j],
                                              current_gt_area.cpu().numpy(),
                                              current_seg_area,
                                              str(merged_dir/(current_name+'_'+dataset.class_list[1:][j]+'.png')),
                                              )

            store['gts'] += list(labels.cpu().numpy().astype(np.uint8))
            store['preds_ps'] += list(torch.argmax(resized_seg_ps, dim=1).cpu().numpy().astype(np.uint8))
            store['preds_vd'] += list(torch.argmax(valid_seg_ps, dim=1).cpu().numpy().astype(np.uint8))

            # for crf seg
            if getcrf:
                vd_crf=valid_seg_ps.softmax(dim=1)[0].cpu().numpy()
                ori_img=torch_helper.denormalize_img_(img_org).permute(0,2,3,1).cpu().numpy()[0]
                vd_crfv2_score=seg_helper.crf_inference_infv2(ori_img,vd_crf)
                vd_crfv2_score=np.argmax(vd_crfv2_score,axis=0)
                # vd_crfs += [vd_crfv2_score.astype(np.uint8)]
                store['vd_crfs'] += [vd_crfv2_score.astype(np.uint8)]

            store['cams'] += list(cam_label.cpu().numpy().astype(np.uint8))
            store['cams_aux'] += list(cam_label_aux.cpu().numpy().astype(np.uint8))

    # save store
    print('store saving with size:',len(store['gts']), 'at rank:', current_rank)
    if current_rank!=0:
        store_path=args.output_dir / f'_temp_store_{current_rank}.pth'
        torch.save(store,store_path)
        del store
    dist.barrier()

    # load store
    if current_rank==0:
        for r in range(1,dist.get_world_size()):
            store_other=torch.load(args.output_dir / f'_temp_store_{r}.pth')
            for item in store:
                store[item]+=store_other[item]
            os.remove(args.output_dir / f'_temp_store_{r}.pth')
        print('Store loaded with size:',len(store['gts']))

        seg_ps_score = evaluation.scores(store['gts'],store['preds_ps'], args.num_classes)
        seg_vd_score = evaluation.scores(store['gts'],store['preds_vd'], args.num_classes)

        cam_score = evaluation.scores(store['gts'],store['cams'], args.num_classes)
        cam_aux_score = evaluation.scores(store['gts'],store['cams_aux'], args.num_classes)

        # evaluation_metrics=[cam_score, cam_aux_score, seg_ps_score, seg_vd_score]
        evaluation_metrics=[cam_score, cam_aux_score, seg_vd_score]
        # evaluation_metrics_name=["CAM","aux_CAM", "Seg_ps", "Seg_vd"]
        evaluation_metrics_name=["CAM","aux_CAM", "Seg_vd"]
        if isfinal:
            evaluation_metrics_name=["Seg_vd"]
            evaluation_metrics=[seg_vd_score]

        if getcrf:
            vd_crfv2_score = evaluation.scores(store['gts'],store['vd_crfs'], args.num_classes)
            evaluation_metrics+=[vd_crfv2_score]
            evaluation_metrics_name+=["Seg_crf"]

        if threshold_filters:
            thre_cam_scores={}
            thre_aux_scores={}
            for thre in threshold_filters:
                # thre_cam_scores[f'cam_{thre}']=evaluation.pseudo_scores(gts, cams_threshed[thre], args.num_classes)
                thre_cam_scores[f'cam_{thre}']=evaluation.pseudo_scores(store['gts'], store['cams_threshed'][thre], args.num_classes)
                # thre_aux_scores[f'camaux_{thre}']=evaluation.pseudo_scores(gts, cams_aux_threshed[thre], args.num_classes)
                thre_aux_scores[f'camaux_{thre}']=evaluation.pseudo_scores(store['gts'], store['cams_aux_threshed'][thre], args.num_classes)
            evaluation_metrics= evaluation_metrics[:3] + list(thre_cam_scores.values()) + list(thre_aux_scores.values()) + evaluation_metrics[3:]
            evaluation_metrics_name= evaluation_metrics_name[:3] + list(thre_cam_scores.keys()) + list(thre_aux_scores.keys()) + evaluation_metrics_name[3:]

        cls_aps=[avg_meter.pop('cls_acc'),avg_meter.pop('cls_aux_acc')]

        tab_results, _, mioulist = torch_helper.format_tabs(scores=evaluation_metrics,
                                                                   name_list=evaluation_metrics_name,
                                                                   cat_list=dataset.class_list)
        if not df:
            df = {'Iterations': [],
                    'mIoU': [],
                    'Metrics': [],
                    'ST': []}
        assert len(mioulist)==len(evaluation_metrics_name)
        timestep=[epoch_int]*len(evaluation_metrics_name)
        st_type=[s_or_t]*len(evaluation_metrics_name)

        df['Iterations'].extend(timestep)
        df['mIoU'].extend(mioulist)
        df['Metrics'].extend(evaluation_metrics_name)
        df['ST'].extend(st_type)

        if save_result:
            torch.save(cmp_dict, cmp_dict_path)

    dist.barrier()
    model.train()

    if current_rank==0:
        seg_vd_miou = mioulist[-1] if not getcrf else mioulist[-2]
        cam_miou = mioulist[0]
        if get_camiou:
            return tab_results, seg_vd_miou, cam_miou, df, cls_aps
        return tab_results, seg_vd_miou, df, cls_aps
    else:
        if get_camiou:
            return None, None, None, None, None
        return None, None, None, None

def save_cam_npv2(cam,img_names,label,cam_np_dir):
    batch_size, class_num, H, W = cam.shape
    for b in range(batch_size):
        img_name = img_names[b]
        if (label[b].sum()) > 0:
            cam_dict = {}
            for c in range(class_num):
                if label[b,c]>0:
                    cam_img = cam[b,c,:]
                    cam_dict[c] = cam_img.cpu().numpy()
            np.save(os.path.join(cam_np_dir, img_name + '.npy'), cam_dict)

def assist_seg(seg,gt,cls_label):
    ious={}
    assert seg.shape==gt.shape, seg.shape
    assert len(cls_label)>=20, cls_label

    for c,j in enumerate(cls_label):
        assert j in [0,1], j
        if j>0:
            seg_num=c+1
            seg_area=(seg==seg_num)
            gt_area=(gt==seg_num)
            iou=(seg_area*gt_area).sum()/(seg_area+gt_area).sum()
            gt_ratio = gt_area.sum()/gt.numel()
            ious[seg_num]=(iou.item(), gt_ratio.item())
    mean_iou=np.mean([ious[i][0] for i in ious])
    weight_mean_iou = np.sum([ious[i][0]*ious[i][1] for i in ious])/np.sum([ious[i][1] for i in ious])
    ious['miou']=mean_iou
    ious['wmiou']=weight_mean_iou
    return ious


