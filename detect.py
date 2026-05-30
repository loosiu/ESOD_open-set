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
from utils.plots import colors, plot_one_box, plot_one_box_small, reset_small_label_cache
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
            # Dual calibrated fusion: F=(1-w)·prob_h+w·p_e, w=σ(gate)·(1-u)
            if (len(heatmaps) == 3 and heatmaps[0].shape[1] == 1
                    and heatmaps[1].shape[1] == 2 and heatmaps[2].shape[1] == 1):
                _heat = heatmaps[0].detach()
                _edl  = heatmaps[1].detach()
                _gate = heatmaps[2].detach()
                prob = _heat[:, 0].sigmoid()
                _ev = F.softplus(_edl); _al = _ev + 1.0
                _S = _al.sum(dim=1)
                _u  = 2.0 / _S                                  # vacuity
                _pe = _al[:, 1] / _S                            # EDL 객체확률 (ch1=obj)
                _g  = _gate[:, 0].sigmoid()
                _w  = _g * (1.0 - _u)
                mask_pred = (1.0 - _w) * prob + _w * _pe
            elif len(heatmaps) == 2 and heatmaps[0].shape[1] == 1 and heatmaps[1].shape[1] == 2:
                _heat = heatmaps[0].detach(); _edl = heatmaps[1].detach()
                prob = _heat[:, 0].sigmoid()
                evidence = F.softplus(_edl); alpha = evidence + 1.0
                S = alpha.sum(dim=1)
                vacuity = (2.0 / S).detach()
                mask_pred = torch.max(prob, vacuity)
            elif mask_raw.shape[1] == 1:
                mask_pred = mask_raw[:, 0]
                if torch.max(mask_pred) > 1.0 or torch.min(mask_pred) < 0.0:
                    mask_pred = mask_pred.sigmoid()
            else:
                # Single 2ch EDL Segmenter (Seg-only): u-aware keep score = p_obj + γ·u
                # 기존 mask_pred = vacuity 는 잘못 (불확실 영역만 보게 됨 → 객체 검출 안 됨).
                # 표준 HeatMapParser (yolo.py:117 / common.py:771) 와 동일하게 B-1 공식 사용.
                evidence = F.softplus(mask_raw)
                alpha = evidence + 1.0
                S = alpha.sum(dim=1)                            # [B, H, W]
                p_obj = (alpha[:, 1] / S).detach()              # belief obj (ch1)
                vacuity = (2.0 / S).detach()
                import os as _os_seg
                _gamma_seg = float(_os_seg.environ.get('ESOD_VAC_GAMMA', '0.5'))
                mask_pred = (p_obj + _gamma_seg * vacuity).detach()

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
        # masks: tensor (single) or list [heat,edl] / [heat,edl,gate] (dual)
        if (isinstance(masks, (list, tuple)) and len(masks) == 3 and
            masks[0].shape[1] == 1 and masks[1].shape[1] == 2 and masks[2].shape[1] == 1):
            # Dual calibrated fusion: F=(1-w)·prob_h+w·p_e, w=σ(gate)·(1-u)
            _heat = masks[0].float(); _edl = masks[1].float(); _gate = masks[2].float()
            prob = _heat.sigmoid()
            _ev = F.softplus(_edl); _al = _ev + 1.0
            _S = _al.sum(dim=1, keepdim=True)
            _pe = _al[:, 1:2] / _S                       # EDL 객체확률 (ch1=obj)
            _u  = 2.0 / _S                               # vacuity = 인식 불확실도
            _g  = _gate.sigmoid()
            _w  = _g * (1.0 - _u)                        # 신뢰가중: EDL 확신할 때만 위임
            masks = (1.0 - _w) * prob + _w * _pe         # [0,1] calibrated
        elif (isinstance(masks, (list, tuple)) and len(masks) == 2 and
              masks[0].shape[1] == 1 and masks[1].shape[1] == 2):
            # Dual + max
            _heat = masks[0].float(); _edl = masks[1].float()
            prob = _heat.sigmoid()
            _ev = F.softplus(_edl); _al = _ev + 1.0
            vac = 2.0 / _al.sum(dim=1, keepdim=True)
            masks = torch.max(prob, vac)
        else:
            _seg = masks[0].float()
            if _seg.shape[1] == 2:
                _ev = F.softplus(_seg); _al = _ev + 1.0
                masks = 2.0 / _al.sum(dim=1, keepdim=True)
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
                cv2.imwrite(f'{save_dir}/{image_name}_attn.jpg', image_att)

                label_path = str(p).replace('images', 'labels').replace('.jpg', '.txt')
                if osp.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.read().splitlines()
                    gt_bboxes = [list(map(float, line.split())) for line in lines]

                    # OWOD: unknown class GT 는 magenta + "unknown" 으로 표시 (pred 와 일관)
                    _unk_set = set()
                    if getattr(opt, 'owod_vis', False) and getattr(opt, 'owod_unknown', ''):
                        _unk_set = set(int(x) for x in opt.owod_unknown.split(',') if x.strip())
                    im1 = im0.copy()
                    if getattr(opt, 'small_vis', False):
                        reset_small_label_cache(image_name + '_gt')
                    for ci, xc, yc, w, h in gt_bboxes:
                        c = int(ci)
                        xyxy = [(xc - w / 2.) * im0.shape[1], (yc - h / 2.) * im0.shape[0],
                                (xc + w / 2.) * im0.shape[1], (yc + h / 2.) * im0.shape[0]]
                        if c in _unk_set:
                            label = None if opt.hide_labels else 'unknown'
                            col = (255, 0, 255)
                        else:
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]}')
                            col = colors(c, True)
                        if getattr(opt, 'small_vis', False):
                            plot_one_box_small(xyxy, im1, label=label, color=col,
                                               line_thickness=opt.line_thickness,
                                               font_size=opt.small_font_size,
                                               font_thickness=opt.small_font_thickness,
                                               text_color=(255, 255, 255))
                        else:
                            plot_one_box(xyxy, im1, label=label, color=col,
                                         line_thickness=opt.line_thickness)
                    cv2.imwrite(f'{save_dir}/{image_name}_gt.jpg', im1)

                    gt_mask_path = str(p).replace('/images/', '/masks/').replace('_masked.', '.').replace('.jpg', '.npy')
                    if os.path.exists(gt_mask_path):
                        gt_mask = np.load(gt_mask_path)
                        gt_mask = gt_mask[..., :1]

                        heatmap = (gt_mask * 255.).astype(np.uint8)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        image_att = cv2.addWeighted(im0, 0.4, heatmap, 0.5, 0)
                        cv2.imwrite(f'{save_dir}/{image_name}_attn_gt.jpg', image_att)

                cluster = clusters[clusters[:, 0] == i, 1:] * 8
                cluster = scale_coords(img.shape[2:], cluster, im0.shape).round()
                im2 = im0.copy()
                for ci, xyxy in enumerate(cluster):
                    x1, y1, x2, y2 = list(map(int, xyxy))
                    plot_one_box((x1, y1, x2, y2), im2, color=(0, 255, 0), line_thickness=opt.line_thickness * 2)
                cv2.imwrite(f'{save_dir}/{image_name}_cluster.jpg', im2)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                if getattr(opt, 'small_vis', False):
                    reset_small_label_cache(getattr(p, 'stem', str(p)) + '_pred')

                # Print results — per-class count (+ OWOD unknown count when --owod-vis)
                if getattr(opt, 'owod_vis', False):
                    _unk_mask = det[:, -2] < float(opt.owod_tau)
                    _kn_det = det[~_unk_mask]
                    for c in _kn_det[:, -1].unique():
                        n = (_kn_det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    _n_unk = int(_unk_mask.sum())
                    if _n_unk > 0:
                        s += f"{_n_unk} unknown{'s' * (_n_unk > 1)}, "
                else:
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
                        # OWOD visualization: conf < owod_tau → "unknown"
                        #   color: magenta (BGR 255,0,255) — visdrone palette 와 무충돌
                        #   label: known box 와 동일 규칙 (--hide-labels/--hide-conf 존중)
                        #          plot_one_box 내부에서 글씨는 항상 흰색 (255,255,255), 배경은 box color
                        #   thickness: 2 (얇게 — 작은 객체 가림 최소화)
                        if getattr(opt, 'owod_vis', False) and float(conf) < float(opt.owod_tau):
                            unk_label = None if opt.hide_labels else (
                                'unknown' if opt.hide_conf else f'unknown {conf:.2f}')
                            col_box = (255, 0, 255)
                            if getattr(opt, 'small_vis', False):
                                plot_one_box_small(xyxy, im0, label=unk_label, color=col_box,
                                                   line_thickness=opt.line_thickness,
                                                   font_size=opt.small_font_size,
                                                   font_thickness=opt.small_font_thickness)
                            else:
                                plot_one_box(xyxy, im0, label=unk_label, color=col_box,
                                             line_thickness=opt.line_thickness)
                        else:
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            col_box = colors(c, True)
                            if getattr(opt, 'small_vis', False):
                                plot_one_box_small(xyxy, im0, label=label, color=col_box,
                                                   line_thickness=opt.line_thickness,
                                                   font_size=opt.small_font_size,
                                                   font_thickness=opt.small_font_thickness)
                            else:
                                plot_one_box(xyxy, im0, label=label, color=col_box,
                                             line_thickness=opt.line_thickness)
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
                    # Always save as _pred.jpg for ESOD-style consistent naming (attn/cluster/pred/gt)
                    cv2.imwrite(f'{save_dir}/{image_name}_pred.jpg', im0)
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

    # (EDL routing statistics — silenced for OWOD mode; 활성화하려면 opt.verbose_routing 추가)

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
    parser.add_argument('--owod-vis', action='store_true', help='OWOD visualization: conf<owod_tau detection을 magenta unknown 박스로')
    parser.add_argument('--owod-tau', type=float, default=0.1, help='OWOD unknown 분류 threshold (conf<tau → unknown)')
    parser.add_argument('--owod-unknown', type=str, default='', help='OWOD GT 시각화용 unknown class indices (예: "6,7"). 해당 class GT 도 magenta+"unknown" 표시')
    parser.add_argument('--small-vis', action='store_true', help='작은 객체용 시각화: 얇은 박스선 + 작은 outlined 텍스트 + label collision 회피 배치')
    parser.add_argument('--small-font-size', type=float, default=0.4, help='작은 객체 시각화 font scale (cv2 fontScale)')
    parser.add_argument('--small-font-thickness', type=int, default=1, help='작은 객체 시각화 font thickness')
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
