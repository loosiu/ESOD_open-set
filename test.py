# # Copyright (c) Alibaba, Inc. and its affiliates.
# ############ EDL ############
import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader, norm_imgs
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
    target2mask, check_mask
from utils.metrics import ap_per_class, ConfusionMatrix, cluster_recall, sparse_recall, mask_pr, hm_verbose, \
    heatmap_quality_update, heatmap_quality_summary, \
    patch_quality_update, patch_quality_summary
from utils.plots import plot_images, output_to_target, plot_study_txt, LatencyBucket
from utils.torch_utils import select_device, time_synchronized, model_info
from utils.loss import ComputeLoss

lbucket = LatencyBucket()

@torch.no_grad()
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         use_gt=True,
         sparse_head=False,
         hm_metric=False,
         opt=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        if len(weights) == 1 and weights[0].endswith('.yaml'):
            from models.yolo import Model
            model = Model(weights[0], ch=3, nc=10).to(device).float().fuse()
        else:
            model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size TODO: 32 or 64
        if opt.compute_loss:
            compute_loss = ComputeLoss(model)

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    if sparse_head:
        model.model[-1].set_sparse()
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu' and False:
            model(torch.zeros(1, 3, imgsz//2, imgsz//2).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0., rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'BPR', 'Occupy')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(6, device=device)
    statistic_items = torch.zeros(3, device=device, dtype=torch.float32)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    sp_r, m_p, m_r, attr = [], [], [], []
    gflops, infer_times = [], []
    hm_stats    = dict(tp=0, fp=0, tn=0, fn=0, obj_hit=0, obj_total=0)
    patch_stats = dict(obj_recalled=0, obj_total=0, gt_coverage_sum=0.0,
                       bg_patches=0, total_patches=0)

    # ── EDL routing 통계 monkey-patch ────────────────────────────────────────
    _routing_stats = dict(n_images=0, n_edl_active=0, n_cap_only=0, n_no_cap=0,
                          sure_total=0, explore_total=0,
                          clusters_before=0, clusters_after=0)
    _hmp = None
    for _mod in model.modules():
        if type(_mod).__name__ == 'HeatMapParser':
            _hmp = _mod
            break
    if _hmp is not None:
        import types as _types
        _rs = _routing_stats

        def _patched_forward(self, x):
            x_feat, heatmaps = x
            bs, c, ny, nx = x_feat.shape
            device_ = x_feat.device
            if len(heatmaps) >= 2 and heatmaps[0].shape[1] == 1 and heatmaps[1].shape[1] == 1:
                mask_raw = torch.cat([heatmaps[0], heatmaps[1]], dim=1).detach()
            else:
                mask_raw = heatmaps[0].detach()
            vacuity = None
            if mask_raw.shape[1] == 1:
                mask_pred = mask_raw[:, 0]
                if torch.max(mask_pred) > 1.0 or torch.min(mask_pred) < 0.0:
                    mask_pred = mask_pred.sigmoid()
            else:
                ev = F.softplus(mask_raw)
                al = ev + 1.0
                S  = al.sum(dim=1)
                mask_pred = (al[:, 1] / S).detach()
                vacuity   = (2.0 / S).detach()
            if self.training:
                return self.uni_slicer(x_feat, mask_pred, self.ratio, self.threshold, device=device_)
            total_clusters = self.ada_slicer_fast(mask_pred, self.ratio, self.threshold)
            Kcap = int(self.max_patches or 0)
            rho  = float(self.explore_ratio or 0.0)
            lam  = float(self.explore_lambda or 0.0)
            _rs['n_images'] += bs
            patches, offsets = [], []
            for bi, clusters in enumerate(total_clusters):
                if clusters.numel() == 0:
                    _rs['n_no_cap'] += 1
                    continue
                _rs['clusters_before'] += clusters.shape[0]
                if Kcap > 0 and clusters.shape[0] > Kcap:
                    x1c, y1c, x2c, y2c = clusters[:,0], clusters[:,1], clusters[:,2], clusters[:,3]
                    cx = ((x1c + x2c) // 2).clamp_(0, nx - 1)
                    cy = ((y1c + y2c) // 2).clamp_(0, ny - 1)
                    p_ = mask_pred[bi, cy, cx]
                    if vacuity is not None:
                        v  = vacuity[bi, cy, cx]
                        sure_score = p_ * (1.0 - v)
                        exp_score  = p_ + lam * v
                        K1 = max(0, min(int(round(Kcap * rho)), Kcap))
                        K0 = Kcap - K1
                        k0 = min(K0, clusters.shape[0])
                        top0 = torch.topk(sure_score, k=k0, largest=True).indices
                        if K1 > 0 and clusters.shape[0] > k0:
                            exp2 = exp_score.clone(); exp2[top0] = -1e9
                            k1 = min(K1, clusters.shape[0] - k0)
                            top1 = torch.topk(exp2, k=k1, largest=True).indices
                            keep = torch.cat([top0, top1], dim=0)
                        else:
                            keep = top0; k1 = 0
                        clusters = clusters[keep]
                        _rs['n_edl_active']    += 1
                        _rs['sure_total']      += k0
                        _rs['explore_total']   += (k1 if K1 > 0 and clusters.shape[0] > k0 else 0)
                    else:
                        keep = torch.topk(p_, k=Kcap, largest=True).indices
                        clusters = clusters[keep]
                        _rs['n_cap_only'] += 1
                else:
                    _rs['n_no_cap'] += 1
                _rs['clusters_after'] += clusters.shape[0]
                for _x1, _y1, _x2, _y2 in clusters:
                    patches.append(x_feat[bi, :, _y1:_y2, _x1:_x2])
                    offsets.append(torch.tensor([bi, _x1, _y1, _x2, _y2], device=device_))
            if len(patches):
                return torch.stack(patches), torch.stack(offsets)
            return torch.zeros((0, c, ny, nx), device=device_), torch.zeros((0, 5), device=device_)

        _hmp._orig_forward = _hmp.forward
        _hmp.forward = _types.MethodType(_patched_forward, _hmp)
    # ─────────────────────────────────────────────────────────────────────────
    for batch_i, (img, targets, masks, m_weights, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = norm_imgs(img, model)
        masks = masks.to(device, non_blocking=True)
        masks = masks.half() if half else masks.float()
        m_weights = m_weights.to(device, non_blocking=True)
        m_weights = m_weights.half() if half else m_weights.float()
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        if not training and opt.task == 'measure':
            assert img.shape[0] == 1  # bs=1

            try:
                from fvcore.nn import FlopCountAnalysis
                fca = FlopCountAnalysis(model, inputs=(img,))
                fca.unsupported_ops_warnings(False)
                gflop = fca.total() / 1E9 * 2
            except:
                gflop = model_info(model, inputs=(img,))

            gflops.append(gflop)

        t = time_synchronized()
        # out, train_out = model((img, masks), augment=augment)  # inference and training outputs
        (out, p_det), p_seg = model((img, [masks]) if use_gt else img, augment=augment)  # inference and training outputs
        infer_times.append(time_synchronized() - t)
        t0 += infer_times[-1]
        train_out = (p_det, p_seg)
        if hm_metric:
            # EDL (C=2) → 1채널 logit으로 변환 (mask_pr/sparse_recall은 sigmoid 후 threshold)
            hm_raw = p_seg[0]
            if hm_raw.shape[1] == 2:
                import torch.nn.functional as _F
                _ev = _F.softplus(hm_raw)
                _al = _ev + 1.0
                _p  = (_al[:, 1:2] / _al.sum(dim=1, keepdim=True)).clamp(1e-6, 1 - 1e-6)
                hm_1ch = torch.log(_p / (1.0 - _p))  # logit: sigmoid(hm_1ch) == p_obj
            else:
                hm_1ch = hm_raw
            mask_pr(masks, hm_1ch, targets, m_p, m_r)
            sparse_recall(hm_1ch, targets, sp_r)
            attr.append(targets[:, [1,4,5]])  # class, width, height
            heatmap_quality_update(p_seg[0], masks, targets, hm_stats)

        # Compute loss
        if compute_loss:
            # loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            loss += compute_loss(train_out, targets, imgsz=img.shape, masks=masks, m_weights=m_weights)[1][:6]  # box, obj, cls, pixl, area, dist

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # targets = targets[targets[:, 4] * targets[:, 5] < 32 * 32]  # TODO: small objects
        if isinstance(p_det, tuple):
            clusters = p_det[1][0] if p_det is not None else torch.zeros((0, 5), device=device)
            clusters = [clusters[clusters[:, 0] == bi, 1:] for bi in range(nb)]
            statistic_items += cluster_recall(clusters, targets, imgsz=(width, height), mode='bbox')
            patch_quality_update(clusters, targets, patch_stats)
            if not training and opt.task == 'measure':
                lbucket.add(len(clusters[0]), gflops[-1], infer_times[-1])
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        if out is not None:
            if isinstance(out, torch.Tensor):
                t = time_synchronized()
                out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
                t1 += time_synchronized() - t
        else:
            out = [torch.zeros((0, 6), device=device)] * nb

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    filename = (path.stem + '.txt') if 'uavdt' not in opt.data else \
                        (path.parent.stem + '_' + path.stem + '.txt')
                    with open(save_dir / 'labels' / filename, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    bpr, occupy = statistic_items[0] / nt.sum(), (statistic_items[1] + 1e-6) / (statistic_items[2] + 1e-6)
    if hm_metric:
        sp_r, m_p, m_r = torch.stack(sp_r), torch.stack(m_p), torch.stack(m_r)
        mp, mr, bpr = m_p.mean().item(), m_r.mean().item(), sp_r.mean().item()
        print('\n'.join([str(res) for res in hm_verbose(sp_r, torch.cat(attr))]))

        hq = heatmap_quality_summary(hm_stats)
        print('\n[Heatmap Quality]')
        print('  %-20s %.4f' % ('Pixel Precision',  hq['pixel_precision']))
        print('  %-20s %.4f' % ('Pixel Recall',     hq['pixel_recall']))
        print('  %-20s %.4f' % ('Pixel F1',         hq['pixel_f1']))
        print('  %-20s %.4f' % ('Pixel IoU',        hq['pixel_iou']))
        print('  %-20s %.4f' % ('Object Recall',    hq['obj_recall']))
        print('  %-20s %.4f' % ('Background FPR',   hq['bg_fpr']))

    if patch_stats['total_patches'] > 0:
        pq = patch_quality_summary(patch_stats)
        avg_patches = patch_stats['total_patches'] / max(seen, 1)
        avg_bg = patch_stats['bg_patches'] / max(seen, 1)
        avg_fg = avg_patches - avg_bg
        print('\n[Patch Quality]')
        print('  %-20s %.4f' % ('Patch Recall',     pq['patch_recall']))
        print('  %-20s %.4f' % ('GT Coverage',      pq['gt_coverage']))
        print('  %-20s %.4f' % ('BG Patch Ratio',   pq['bg_patch_ratio']))
        print('  %-20s %.1f' % ('Avg Patches/Image', avg_patches))
        print('  %-20s %.1f (FG) + %.1f (BG)' % ('  Breakdown', avg_fg, avg_bg))


    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, bpr, occupy))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], bpr, occupy))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        fps_infer = 1000. / t[0] if t[0] > 0 else 0
        fps_total = 1000. / t[2] if t[2] > 0 else 0
        if opt.task == 'measure':
            print('GFLOPs: %.1f. ' % np.mean(gflops), end='')
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
        print('FPS (inference only): %.1f' % fps_infer)
        print('FPS (total):          %.1f' % fps_total)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        if 'visdrone' in opt.data.lower():
            anno_json = './datasets/VisDrone/annotations/val.json'  # annotations json
        elif 'tinyperson' in opt.data.lower():
            anno_json = './datasets/TinyPerson/test.json'  # annotations json
            format_tinyperson(jdict)
        else:
            raise NotImplementedError(f'Ground-Truth file for {os.path.basename(opt.data)} not found.')

        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        
            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: \n{e}')

    # Restore monkey-patched forward to avoid pickle issues
    if _hmp is not None and hasattr(_hmp, '_orig_forward'):
        _hmp.forward = _hmp._orig_forward
        del _hmp._orig_forward

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
        lbucket.save(f'{save_dir}/buckets.json')
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, bpr, occupy, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def format_tinyperson(jdict):
    with open('datasets/TinyPerson/mini_annotations/tiny_set_test_all.json', 'r') as f:
        anno = json.load(f)
    image_id_dict = {item['file_name'].split('/')[1][:-4]: item['id'] for item in anno['images']}

    for item in jdict:
        item['image_id'] = image_id_dict[item['image_id']]
        item['category_id'] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--use-gt', action='store_true', help='use gt masks')
    parser.add_argument('--compute-loss', action='store_true', help='compute loss')
    parser.add_argument('--sparse-head', action='store_true', help='use sparse detection head')
    parser.add_argument('--hm-metric', action='store_true', help='use heatmap-related evaluation metrics')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.task in ('train', 'val', 'test', 'measure'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             half_precision=opt.half,
             use_gt=opt.use_gt,
             sparse_head=opt.sparse_head,
             hm_metric=opt.hm_metric,
             opt=opt
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        # x = list(range(832, 1920 + 128, 128))  # x axis (image sizes)
        x = list(range(1024, 2560 + 128, 128))  # x axis (image sizes)
        opt.task = 'measure'
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               sparse_head=opt.sparse_head, use_gt=opt.use_gt, half_precision=opt.half, plots=False, opt=opt)
                y.append((r + t).cpu().numpy())  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot

### graph ###
# import argparse
# import json
# import os
# from pathlib import Path
# from threading import Thread

# import numpy as np
# import torch
# import torch.nn.functional as F
# import yaml
# from tqdm import tqdm

# from models.experimental import attempt_load
# from utils.datasets import create_dataloader, norm_imgs
# from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
#     box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
#     target2mask, check_mask
# from utils.metrics import ap_per_class, ConfusionMatrix, cluster_recall, sparse_recall, mask_pr, hm_verbose
# from utils.plots import plot_images, output_to_target, plot_study_txt, LatencyBucket
# from utils.torch_utils import select_device, time_synchronized, model_info
# from utils.loss import ComputeLoss

# lbucket = LatencyBucket()

# @torch.no_grad()
# def test(data,
#          weights=None,
#          batch_size=32,
#          imgsz=640,
#          conf_thres=0.001,
#          iou_thres=0.6,  # for NMS
#          save_json=False,
#          single_cls=False,
#          augment=False,
#          verbose=False,
#          model=None,
#          dataloader=None,
#          save_dir=Path(''),  # for saving images
#          save_txt=False,  # for auto-labelling
#          save_hybrid=False,  # for hybrid auto-labelling
#          save_conf=False,  # save auto-label confidences
#          plots=True,
#          wandb_logger=None,
#          compute_loss=None,
#          half_precision=True,
#          is_coco=False,
#          use_gt=True,
#          sparse_head=False,
#          hm_metric=False,
#          opt=None,
#          return_confusion=False):
#     # Initialize/load model and set device
#     training = model is not None
#     if training:  # called by train.py
#         device = next(model.parameters()).device  # get model device

#     else:  # called directly
#         set_logging()
#         device = select_device(opt.device, batch_size=batch_size)

#         # Directories
#         save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
#         (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#         # Load model
#         if len(weights) == 1 and weights[0].endswith('.yaml'):
#             from models.yolo import Model
#             model = Model(weights[0], ch=3, nc=10).to(device).float().fuse()
#         else:
#             model = attempt_load(weights, map_location=device)  # load FP32 model
#         gs = max(int(model.stride.max()), 32)  # grid size (max stride)
#         imgsz = check_img_size(imgsz, s=gs)  # check img_size TODO: 32 or 64
#         if opt.compute_loss:
#             compute_loss = ComputeLoss(model)

#         # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
#         # if device.type != 'cpu' and torch.cuda.device_count() > 1:
#         #     model = nn.DataParallel(model)

#     # Half
#     half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
#     if half:
#         model.half()

#     # Configure
#     if sparse_head:
#         model.model[-1].set_sparse()
#     model.eval()
#     if isinstance(data, str):
#         is_coco = data.endswith('coco.yaml')
#         with open(data) as f:
#             data = yaml.safe_load(f)
#     check_dataset(data)  # check
#     nc = 1 if single_cls else int(data['nc'])  # number of classes
#     iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
#     niou = iouv.numel()

#     # Logging
#     log_imgs = 0
#     if wandb_logger and wandb_logger.wandb:
#         log_imgs = min(wandb_logger.log_imgs, 100)
#     # Dataloader
#     if not training:
#         if device.type != 'cpu' and False:
#             model(torch.zeros(1, 3, imgsz//2, imgsz//2).to(device).type_as(next(model.parameters())))  # run once
#         task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
#         dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0., rect=True,
#                                        prefix=colorstr(f'{task}: '))[0]

#     seen = 0
#     confusion_matrix = ConfusionMatrix(nc=nc)
#     names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
#     coco91class = coco80_to_coco91_class()
#     s = ('%20s' + '%11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'BPR', 'Occupy')
#     p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
#     loss = torch.zeros(6, device=device)
#     statistic_items = torch.zeros(3, device=device, dtype=torch.float32)
#     jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
#     sp_r, m_p, m_r, attr = [], [], [], []
#     gflops, infer_times = [], []
#     for batch_i, (img, targets, masks, m_weights, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
#     # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
#         img = img.to(device, non_blocking=True)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         # img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         img = norm_imgs(img, model)
#         masks = masks.to(device, non_blocking=True)
#         masks = masks.half() if half else masks.float()
#         m_weights = m_weights.to(device, non_blocking=True)
#         m_weights = m_weights.half() if half else m_weights.float()
#         targets = targets.to(device)
#         nb, _, height, width = img.shape  # batch size, channels, height, width

#         # Run model
#         if not training and opt.task == 'measure':
#             assert img.shape[0] == 1  # bs=1

#             try:
#                 from fvcore.nn import FlopCountAnalysis
#                 fca = FlopCountAnalysis(model, inputs=(img,))
#                 fca.unsupported_ops_warnings(False)
#                 gflop = fca.total() / 1E9 * 2
#             except:
#                 gflop = model_info(model, inputs=(img,))

#             gflops.append(gflop)

#         t = time_synchronized()
#         # out, train_out = model((img, masks), augment=augment)  # inference and training outputs
#         (out, p_det), p_seg = model((img, [masks]) if use_gt else img, augment=augment)  # inference and training outputs
#         infer_times.append(time_synchronized() - t)
#         t0 += infer_times[-1]
#         train_out = (p_det, p_seg)
#         if hm_metric:
#             mask_pr(masks, p_seg[0], targets, m_p, m_r)
#             sparse_recall(p_seg[0], targets, sp_r)
#             attr.append(targets[:, [1,4,5]])  # class, width, height

#         # Compute loss
#         if compute_loss:
#             # loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
#             loss += compute_loss(train_out, targets, imgsz=img.shape, masks=masks, m_weights=m_weights)[1][:6]  # box, obj, cls, pixl, area, dist

#         # Run NMS
#         targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
#         # targets = targets[targets[:, 4] * targets[:, 5] < 32 * 32]  # TODO: small objects
#         if isinstance(p_det, tuple):
#             clusters = p_det[1][0] if p_det is not None else torch.zeros((0, 5), device=device)
#             clusters = [clusters[clusters[:, 0] == bi, 1:] for bi in range(nb)]
#             statistic_items += cluster_recall(clusters, targets, imgsz=(width, height), mode='bbox')
#             if not training and opt.task == 'measure':
#                 lbucket.add(len(clusters[0]), gflops[-1], infer_times[-1])
#         lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
#         if out is not None:
#             if isinstance(out, torch.Tensor):
#                 t = time_synchronized()
#                 out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
#                 t1 += time_synchronized() - t
#         else:
#             out = [torch.zeros((0, 6), device=device)] * nb

#         # Statistics per image
#         for si, pred in enumerate(out):
#             labels = targets[targets[:, 0] == si, 1:]
#             nl = len(labels)
#             tcls = labels[:, 0].tolist() if nl else []  # target class
#             path = Path(paths[si])
#             seen += 1

#             if len(pred) == 0:
#                 if nl:
#                     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
#                 continue

#             # Predictions
#             if single_cls:
#                 pred[:, 5] = 0
#             predn = pred.clone()
#             scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

#             # Append to text file
#             if save_txt:
#                 gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
#                 for *xyxy, conf, cls in predn.tolist():
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                     filename = (path.stem + '.txt') if 'uavdt' not in opt.data else \
#                         (path.parent.stem + '_' + path.stem + '.txt')
#                     with open(save_dir / 'labels' / filename, 'a') as f:
#                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

#             # W&B logging - Media Panel Plots
#             if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
#                 if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
#                     box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
#                                  "class_id": int(cls),
#                                  "box_caption": "%s %.3f" % (names[cls], conf),
#                                  "scores": {"class_score": conf},
#                                  "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
#                     boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
#                     wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
#             wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

#             # Append to pycocotools JSON dictionary
#             if save_json:
#                 # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
#                 image_id = int(path.stem) if path.stem.isnumeric() else path.stem
#                 box = xyxy2xywh(predn[:, :4])  # xywh
#                 box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
#                 for p, b in zip(pred.tolist(), box.tolist()):
#                     jdict.append({'image_id': image_id,
#                                   'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
#                                   'bbox': [round(x, 3) for x in b],
#                                   'score': round(p[4], 5)})

#             # Assign all predictions as incorrect
#             correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
#             if nl:
#                 detected = []  # target indices
#                 tcls_tensor = labels[:, 0]

#                 # target boxes
#                 tbox = xywh2xyxy(labels[:, 1:5])
#                 scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
#                 if plots:
#                     confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

#                 # Per target class
#                 for cls in torch.unique(tcls_tensor):
#                     ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
#                     pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

#                     # Search for detections
#                     if pi.shape[0]:
#                         # Prediction to target ious
#                         ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

#                         # Append detections
#                         detected_set = set()
#                         for j in (ious > iouv[0]).nonzero(as_tuple=False):
#                             d = ti[i[j]]  # detected target
#                             if d.item() not in detected_set:
#                                 detected_set.add(d.item())
#                                 detected.append(d)
#                                 correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
#                                 if len(detected) == nl:  # all targets already located in image
#                                     break

#             # Append statistics (correct, conf, pcls, tcls)
#             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

#         # Plot images
#         if plots and batch_i < 3:
#             f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
#             Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
#             f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
#             Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

#     # Compute statistics
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
#     if len(stats) and stats[0].any():
#         p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
#         ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
#         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
#     else:
#         nt = torch.zeros(1)
#     bpr, occupy = statistic_items[0] / nt.sum(), (statistic_items[1] + 1e-6) / (statistic_items[2] + 1e-6)
#     if hm_metric:
#         sp_r, m_p, m_r = torch.stack(sp_r), torch.stack(m_p), torch.stack(m_r)
#         mp, mr, bpr = m_p.mean().item(), m_r.mean().item(), sp_r.mean().item()
#         print('\n'.join([str(res) for res in hm_verbose(sp_r, torch.cat(attr))]))


#     # Print results
#     pf = '%20s' + '%11i' * 2 + '%11.3g' * 6  # print format
#     print(pf % ('all', seen, nt.sum(), mp, mr, map50, map, bpr, occupy))

#     # Print results per class
#     if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats) and False:
#         for i, c in enumerate(ap_class):
#             print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], bpr, occupy))

#     # Print speeds
#     t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
#     if not training:
#         if opt.task == 'measure':
#             print('GFLOPs: %.1f. FPS: %.1f ' % (np.mean(gflops), 1000. / t[0]), end='')
#         print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

#     # Plots
#     if plots:
#         confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
#         if wandb_logger and wandb_logger.wandb:
#             val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
#             wandb_logger.log({"Validation": val_batches})
#     if wandb_images:
#         wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

#     # Save JSON
#     if save_json and len(jdict):
#         w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
#         if 'visdrone' in opt.data.lower():
#             anno_json = './datasets/VisDrone/annotations/val.json'  # annotations json
#         elif 'tinyperson' in opt.data.lower():
#             anno_json = './datasets/TinyPerson/test.json'  # annotations json
#             format_tinyperson(jdict)
#         else:
#             raise NotImplementedError(f'Ground-Truth file for {os.path.basename(opt.data)} not found.')

#         pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
#         print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
#         with open(pred_json, 'w') as f:
#             json.dump(jdict, f)

#         try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#             from pycocotools.coco import COCO
#             from pycocotools.cocoeval import COCOeval
        
#             anno = COCO(anno_json)  # init annotations api
#             pred = anno.loadRes(pred_json)  # init predictions api
#             eval = COCOeval(anno, pred, 'bbox')
#             if is_coco:
#                 eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
#             eval.evaluate()
#             eval.accumulate()
#             eval.summarize()
#             # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
#         except Exception as e:
#             print(f'pycocotools unable to run: \n{e}')

#     # Return results
#     model.float()  # for training
#     if not training:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         print(f"Results saved to {save_dir}{s}")
#         lbucket.save(f'{save_dir}/buckets.json')
#     maps = np.zeros(nc) + map
#     for i, c in enumerate(ap_class):
#         maps[c] = ap[i]
#     conf_out = None
#     if return_confusion:
#         # ConfusionMatrix.matrix is (nc+1)x(nc+1) with background last
#         try:
#             cm = confusion_matrix.matrix
#             if isinstance(cm, np.ndarray):
#                 cm = cm.astype(np.float32)
#             conf_out = cm
#         except Exception:
#             conf_out = None

#     if return_confusion:
#         return (mp, mr, map50, map, bpr, occupy, *(loss.cpu() / len(dataloader)).tolist()), maps, t, conf_out
#     return (mp, mr, map50, map, bpr, occupy, *(loss.cpu() / len(dataloader)).tolist()), maps, t



# def format_tinyperson(jdict):
#     with open('datasets/TinyPerson/mini_annotations/tiny_set_test_all.json', 'r') as f:
#         anno = json.load(f)
#     image_id_dict = {item['file_name'].split('/')[1][:-4]: item['id'] for item in anno['images']}

#     for item in jdict:
#         item['image_id'] = image_id_dict[item['image_id']]
#         item['category_id'] += 1


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(prog='test.py')
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
#     parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
#     parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
#     parser.add_argument('--task', default='val', help='train, val, test, speed or study')
#     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--verbose', action='store_true', help='report mAP by class')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
#     parser.add_argument('--project', default='runs/test', help='save to project/name')
#     parser.add_argument('--name', default='exp', help='save to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--use-gt', action='store_true', help='use gt masks')
#     parser.add_argument('--compute-loss', action='store_true', help='compute loss')
#     parser.add_argument('--sparse-head', action='store_true', help='use sparse detection head')
#     parser.add_argument('--hm-metric', action='store_true', help='use heatmap-related evaluation metrics')
#     opt = parser.parse_args()
#     opt.save_json |= opt.data.endswith('coco.yaml')
#     opt.data = check_file(opt.data)  # check file
#     print(opt)
#     check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

#     if opt.task in ('train', 'val', 'test', 'measure'):  # run normally
#         test(opt.data,
#              opt.weights,
#              opt.batch_size,
#              opt.img_size,
#              opt.conf_thres,
#              opt.iou_thres,
#              opt.save_json,
#              opt.single_cls,
#              opt.augment,
#              opt.verbose,
#              save_txt=opt.save_txt | opt.save_hybrid,
#              save_hybrid=opt.save_hybrid,
#              save_conf=opt.save_conf,
#              half_precision=opt.half,
#              use_gt=opt.use_gt,
#              sparse_head=opt.sparse_head,
#              hm_metric=opt.hm_metric,
#              opt=opt
#              )

#     elif opt.task == 'study':  # run over a range of settings and save/plot
#         # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
#         # x = list(range(832, 1920 + 128, 128))  # x axis (image sizes)
#         x = list(range(1024, 2560 + 128, 128))  # x axis (image sizes)
#         opt.task = 'measure'
#         for w in opt.weights:
#             f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
#             y = []  # y axis
#             for i in x:  # img-size
#                 print(f'\nRunning {f} point {i}...')
#                 r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
#                                sparse_head=opt.sparse_head, use_gt=opt.use_gt, half_precision=opt.half, plots=False, opt=opt)
#                 y.append((r + t).cpu().numpy())  # results and times
#             np.savetxt(f, y, fmt='%10.4g')  # save
#         os.system('zip -r study.zip study_*.txt')
#         plot_study_txt(x=x)  # plot