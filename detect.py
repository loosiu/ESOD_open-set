# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import time
from pathlib import Path
import os
import os.path as osp
from os.path import join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, norm_imgs
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, target2mask
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and (not source.endswith('.txt') or True)  # save inference images
    webcam = source.isnumeric() or (source.endswith('.txt') and False) or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ── EDL routing 통계 수집용 monkey-patch ──────────────────────────────
    _routing_stats = dict(
        n_images=0,
        n_edl_active=0,    # clusters > Kcap 이고 vacuity 사용된 횟수
        n_cap_only=0,      # clusters > Kcap 이지만 vacuity 없는 경우
        n_no_cap=0,        # clusters <= Kcap (cap 자체 불필요)
        sure_total=0,      # K0 sure 패치 누적
        explore_total=0,   # K1 explore 패치 누적
        clusters_before=0, # cap 전 cluster 수 누적
        clusters_after=0,  # cap 후 cluster 수 누적
    )
    _hmp = None
    for _mod in model.modules():
        if type(_mod).__name__ == 'HeatMapParser':
            _hmp = _mod
            break
    if _hmp is not None:
        _orig_forward = _hmp.forward.__func__  # unbound

        def _patched_forward(self, x):
            x_feat, heatmaps = x
            bs, c, ny, nx = x_feat.shape
            device = x_feat.device

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
                # EDL: vacuity 기반 객체 탐지
                evidence = F.softplus(mask_raw)
                alpha = evidence + 1.0
                S = alpha.sum(dim=1)
                vacuity = (2.0 / S).detach()
                mask_pred = vacuity  # prob 대신 vacuity 사용

            if self.training:
                return self.uni_slicer(x_feat, mask_pred, self.ratio, self.threshold, device=device)

            total_clusters = self.ada_slicer_fast(mask_pred, self.ratio, self.threshold)
            Kcap = int(self.max_patches or 0)
            rho  = float(self.explore_ratio or 0.0)
            lam  = float(self.explore_lambda or 0.0)

            _routing_stats['n_images'] += bs

            patches, offsets = [], []
            for bi, clusters in enumerate(total_clusters):
                if clusters.numel() == 0:
                    _routing_stats['n_no_cap'] += 1
                    continue

                n_before = clusters.shape[0]
                _routing_stats['clusters_before'] += n_before

                if Kcap > 0 and clusters.shape[0] > Kcap:
                    x1, y1, x2, y2 = clusters[:, 0], clusters[:, 1], clusters[:, 2], clusters[:, 3]
                    cx = ((x1 + x2) // 2).clamp_(0, nx - 1)
                    cy = ((y1 + y2) // 2).clamp_(0, ny - 1)
                    p = mask_pred[bi, cy, cx]

                    if vacuity is not None:
                        v = vacuity[bi, cy, cx]
                        sure_score = p * (1.0 - v)
                        exp_score  = p + lam * v
                        K1 = max(0, min(int(round(Kcap * rho)), Kcap))
                        K0 = Kcap - K1
                        k0 = min(K0, clusters.shape[0])
                        top0 = torch.topk(sure_score, k=k0, largest=True).indices
                        if K1 > 0 and clusters.shape[0] > k0:
                            exp2 = exp_score.clone()
                            exp2[top0] = -1e9
                            k1 = min(K1, clusters.shape[0] - k0)
                            top1 = torch.topk(exp2, k=k1, largest=True).indices
                            keep = torch.cat([top0, top1], dim=0)
                        else:
                            keep = top0
                            k1 = 0
                        clusters = clusters[keep]
                        _routing_stats['n_edl_active'] += 1
                        _routing_stats['sure_total']    += k0
                        _routing_stats['explore_total'] += (k1 if K1 > 0 and n_before > k0 else 0)
                    else:
                        keep = torch.topk(p, k=Kcap, largest=True).indices
                        clusters = clusters[keep]
                        _routing_stats['n_cap_only'] += 1
                else:
                    _routing_stats['n_no_cap'] += 1

                _routing_stats['clusters_after'] += clusters.shape[0]

                for _x1, _y1, _x2, _y2 in clusters:
                    patches.append(x_feat[bi, :, _y1:_y2, _x1:_x2])
                    offsets.append(torch.tensor([bi, _x1, _y1, _x2, _y2], device=device))

            if len(patches):
                return torch.stack(patches), torch.stack(offsets)
            else:
                return torch.zeros((0, c, ny, nx), device=device), torch.zeros((0, 5), device=device)

        import types
        _hmp.forward = types.MethodType(_patched_forward, _hmp)
        print('[EDL-stats] HeatMapParser monkey-patched for routing statistics')
    # ─────────────────────────────────────────────────────────────────────────

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # img /= 255.0  # 0-255 to 0.0-1.0
        img = norm_imgs(img, model)

        # Inference
        t1 = time_synchronized()
        (pred, p_det), masks = model(img, augment=opt.augment)
        _seg = masks[0].float()  # [B, C, H, W]
        if _seg.shape[1] == 2:  # EDL: vacuity map 시각화 (불확실한 곳 = 객체 후보)
            _ev = F.softplus(_seg)
            _al = _ev + 1.0
            _S = _al.sum(dim=1, keepdim=True)
            masks = 2.0 / _S  # vacuity [B,1,H,W]
        else:
            masks = _seg.sigmoid()
        if opt.view_center:
            masks = ((masks == F.max_pool2d(masks, 3, stride=1, padding=1)) & (masks > 0.3)).float()
        clusters = p_det[1][0] if (p_det is not None and p_det[1] is not None) else torch.zeros((0, 5), device=img.device)

        # Apply NMS
        if pred is None:
            pred = [torch.zeros((0, 6), device=img.device)]
        else:
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                       max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            image_name = osp.basename(p).split('.')[0]
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if opt.view_cluster:
                # cv2.imwrite(f'{save_dir}/{image_name}_0_raw.jpg', im0)

                heatmap = (masks[i, 0].cpu().numpy() * 255.).astype(np.uint8)
                heatmap = cv2.resize(heatmap, (im0.shape[1], im0.shape[0]), cv2.INTER_CUBIC)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                image_att = cv2.addWeighted(im0, 0.4, heatmap, 0.5, 0)
                cv2.imwrite(f'{save_dir}/{image_name}_1_attn.jpg', image_att)

                label_path = str(p).replace('images', 'labels').replace('.jpg', '.txt')
                if osp.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.read().splitlines()
                    gt_bboxes = [list(map(float, line.split())) for line in lines]

                    im1 = im0.copy()
                    for ci, xc, yc, w, h in gt_bboxes:
                        c = int(ci)
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]}')
                        xyxy = [(xc - w / 2.) * im0.shape[1], (yc - h / 2.) * im0.shape[0],
                                (xc + w / 2.) * im0.shape[1], (yc + h / 2.) * im0.shape[0]]
                        plot_one_box(xyxy, im1, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                    cv2.imwrite(f'{save_dir}/{image_name}_5_gt.jpg', im1)

                    gt_mask_path = str(p).replace('/images/', '/masks/').replace('_masked.', '.').replace('.jpg', '.npy')
                    if os.path.exists(gt_mask_path):
                        gt_mask = np.load(gt_mask_path)
                        gt_mask = gt_mask[..., :1]

                        heatmap = (gt_mask * 255.).astype(np.uint8)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        image_att = cv2.addWeighted(im0, 0.4, heatmap, 0.5, 0)
                        cv2.imwrite(f'{save_dir}/{image_name}_2_attn_gt.jpg', image_att)

                cluster = clusters[clusters[:, 0] == i, 1:] * 8
                cluster = scale_coords(img.shape[2:], cluster, im0.shape).round()
                im2 = im0.copy()
                for ci, xyxy in enumerate(cluster):
                    x1, y1, x2, y2 = list(map(int, xyxy))
                    plot_one_box((x1, y1, x2, y2), im2, color=(0, 255, 0), line_thickness=opt.line_thickness * 2)
                cv2.imwrite(f'{save_dir}/{image_name}_3_cluster.jpg', im2)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    if opt.view_cluster:
                        cv2.imwrite(f'{save_dir}/{image_name}_4_pred.jpg', im0)
                    else:
                        cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    # ── EDL routing 통계 출력 ─────────────────────────────────────────────
    if _hmp is not None:
        rs = _routing_stats
        n = rs['n_images']
        n_ev = rs['n_edl_active']
        n_cap = rs['n_cap_only']
        n_no = rs['n_no_cap']
        print('\n[EDL Routing Statistics]')
        print('  Total images (batch slots) : %d' % n)
        print('  Kcap=%d  rho=%.2f  lambda=%.2f' % (
            _hmp.max_patches, _hmp.explore_ratio, _hmp.explore_lambda))
        print('  No cap needed  (clusters <= Kcap) : %d  (%.1f%%)' % (n_no,  100.*n_no /max(n,1)))
        print('  Cap w/ EDL     (vacuity used)     : %d  (%.1f%%)' % (n_ev,  100.*n_ev /max(n,1)))
        print('  Cap w/o EDL    (vacuity=None)     : %d  (%.1f%%)' % (n_cap, 100.*n_cap/max(n,1)))
        if n_ev > 0:
            print('  Sure patches   (K0) avg per cap  : %.1f' % (rs['sure_total']    / n_ev))
            print('  Explore patches(K1) avg per cap  : %.1f' % (rs['explore_total'] / n_ev))
        if rs['clusters_before'] > 0:
            print('  Avg clusters before cap : %.1f' % (rs['clusters_before'] / max(n_ev+n_cap, 1)))
            print('  Avg clusters after  cap : %.1f' % (rs['clusters_after']  / max(n_ev+n_cap, 1)))
        if n_ev == 0 and n_cap == 0:
            print('  --> EDL routing never triggered (all images had <= Kcap clusters)')
    # ─────────────────────────────────────────────────────────────────────────

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--view-cluster', action='store_true', help='visualize clusters')
    parser.add_argument('--view-center', action='store_true', help='visualize heatmap centers')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
