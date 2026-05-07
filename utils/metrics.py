# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import general


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.9, 0.1]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def mask_pr(true, posi, targets, precision :list, recall: list, threshold=0.5):
    assert true.shape == posi.shape
    bs, _, ny, nx = true.shape
    device = true.device
    posi = posi.sigmoid()
    # threshold = round(true[true > 0.05].min().item(), 1)
    true = true >= threshold
    posi = posi >= threshold

    bboxes = general.xywh2xyxy(targets[:, 2:]).clip(0, 1)
    # gain = torch.tensor([[nx, ny, nx, ny]], device=device)
    # bboxes = torch.cat((targets[:, :1], bboxes * gain), dim=1).long()
    gain = torch.tensor([[nx, ny]], device=device)
    bboxes = torch.cat((targets[:, :1], (bboxes[:, :2] * gain).floor(), (bboxes[:, 2:] * gain).ceil()), dim=1).long()
    
    for bi, x1, y1, x2, y2 in bboxes:
        ins_true = true[bi, 0, y1:y2, x1:x2]
        ins_posi = posi[bi, 0, y1:y2, x1:x2]
        assert ins_true.sum() > 0

        # precision.append(((ins_true & ins_posi).sum() + 1e-9) / (ins_posi.sum() + 1e-9))
        recall.append((ins_true & ins_posi).sum() / ins_true.sum())
    precision.append(((true & posi).sum() + 1e-9) / (posi.sum() + 1e-9))
    # recall.append((true & posi).sum() / (true.sum() + 1e-9))


def cluster_recall(clusters, targets, imgsz=(1024,576), mode='bbox', stride=8):
    assert mode in ['point', 'bbox'], '%s' % mode
    tp, cluster_num = 0, 0
    patch_w, patch_h = 0, 0
    bs = len(clusters)
    for bi, cluster in enumerate(clusters):
        t = targets[targets[:, 0] == bi][:, -4:] / stride  # [xc, yc, w, h]
        cluster_num += len(cluster)
        if len(cluster) == 0 or len(t) == 0:
            continue
        patch_w, patch_h = cluster[0, [-2, -1]] - cluster[0, [-4, -3]]
        # assert len(cluster) <= stride ** 2
        # cluster: shape(m,4) = [x1, y1, x2, y2]; t: shape(n,2) = [xc, yc, w, h]
        if mode == 'point':
            x1, y1, x2, y2 = cluster.T
            xc, yc, w, h = t.T
            tp += ((x1[None, :] <= xc[:, None]) & (xc[:, None] < x2[None, :]) &
                   (y1[None, :] <= yc[:, None]) & (yc[:, None] < y2[None, :])).any(dim=1).sum()
        else:
            # xc, yc, w, h = t.T
            # t = t[w * h > 12. * 12.]  # 4 * 8 = 32, 12 * 8 = 96
            # tp += len(t)
            ios = general.box_ios(general.xywh2xyxy(t), cluster)
            tp += (ios >= 0.5).any(dim=1).sum()
            # tp += ((ios < 0.5).all(dim=1) & (ios >= 0.01).any(dim=1)).sum()  # fn
    # total_patch_num = bs * ratio ** 2
    # total_patch_num = bs * math.ceil(imgsz[0] / patch_w / stride) * math.ceil(imgsz[1] / patch_h / stride)
    # total_patch_num = (imgsz[0] / patch_w / stride) * (imgsz[1] / patch_h / stride)
    cluster_num = (cluster_num * (patch_w * stride) * (patch_h * stride)) / (bs * imgsz[0] * imgsz[1])
    total_patch_num = 1.
    return torch.tensor([tp, cluster_num, total_patch_num], device=targets.device)


def sparse_recall(heatmap, targets, recall :list, threshold=0.3):
    heatmap = heatmap.sigmoid()
    indices_pyrimid = []
    
    # for s in [1, 2, 4]:
    #     heatmap_i = heatmap if s == 1 else F.avg_pool2d(heatmap, s, stride=s, padding=0)
    #     maxima = F.max_pool2d(heatmap_i, 3, stride=1, padding=1) == heatmap_i
    #     response = heatmap_i >= threshold / s
    #     indices = (maxima & response).float()
    #     indices = F.max_pool2d(indices, 3, stride=1, padding=1) # expansion
    #     indices_pyrimid.append(indices)
    
    maxima = F.max_pool2d(heatmap, 3, stride=1, padding=1) == heatmap
    response = heatmap >= threshold
    indices = (maxima & response).float()
    for s in [1, 2, 4]:
        indices_i = indices if s == 1 else F.max_pool2d(indices, s, stride=s, padding=0)
        indices_i = F.max_pool2d(indices_i, 3, stride=1, padding=1) # expansion
        indices_pyrimid.append(indices_i)

    _, _, ny, nx = heatmap.shape
    gain = torch.tensor([[nx, ny, nx, ny]], device=heatmap.device)
    for i, (xc, yc, w, h) in enumerate((gain * targets[:, 2:]).long()):
        bi = targets[i, 0].long()
        # recall.append(indices[bi, 0, yc, xc])
        # li = torch.minimum(w, h).clamp(1).log2().ceil().long().clamp(0, 2).item()
        # res = sum([indices_pyrimid[j][bi, 0, yc//(2**j), xc//(2**j)] for j in range(li + 1)]) > 0
        res = sum([indices_pyrimid[j][bi, 0, yc//(2**j), xc//(2**j)] for j in range(3)]) > 0
        recall.append(res.float())


def hm_verbose(recall, attr):
    assert len(recall) == len(attr)
    c, w, h = attr.T
    s = w * h

    res_cls = []
    for ci in range(c.max().long().item()+1):
        ind = c == ci
        res_cls.append(recall[ind].mean().item())

    s0 = 32 / 1920 * 32 / 1080
    size_sep = [0, s0 / 4, s0, s0 * 4, 1.]
    res_size = []
    for i in range(len(size_sep)-1):
        ind = (size_sep[i] < s) & (s <= size_sep[i+1])
        res_size.append(recall[ind].mean().item())

    return res_cls, res_size


def patch_quality_update(clusters, targets, stats, stride=8, ios_thresh=0.5):
    """
    Accumulate patch quality stats for one batch.

    clusters  : list[Tensor[N,4]]  xyxy in feature-map coords (pixels/stride), one per image
    targets   : Tensor[M,6]        [bi, cls, xc_px, yc_px, w_px, h_px] already scaled to pixels
    stats     : dict with keys obj_recalled, obj_total, gt_coverage_sum, bg_patches, total_patches
    """
    for bi, cluster in enumerate(clusters):
        t = targets[targets[:, 0] == bi][:, 2:]   # [M,4] xc,yc,w,h pixels
        t_fm   = t / stride                         # feature-map space
        t_xyxy = general.xywh2xyxy(t_fm)           # [M,4] xyxy feature-map

        n_patches = cluster.shape[0]
        n_gt      = len(t)
        stats['total_patches'] += n_patches
        stats['obj_total']     += n_gt

        if n_patches == 0 or n_gt == 0:
            stats['bg_patches'] += n_patches
            continue

        ios = general.box_ios(t_xyxy, cluster)      # [M, N]  inter/min(area_gt, area_patch)

        # Patch Recall: fraction of GT objects covered by at least one patch
        stats['obj_recalled']    += int((ios >= ios_thresh).any(dim=1).sum())

        # GT Coverage: average best-patch IoS per GT object
        stats['gt_coverage_sum'] += float(ios.max(dim=1).values.sum())

        # Background patches: patches that don't cover any GT object
        stats['bg_patches'] += int((~(ios >= ios_thresh).any(dim=0)).sum())


def patch_quality_summary(stats):
    """Compute final patch quality metrics from accumulated stats dict."""
    patch_recall   = stats['obj_recalled']    / (stats['obj_total']     + 1e-9)
    gt_coverage    = stats['gt_coverage_sum'] / (stats['obj_total']     + 1e-9)
    bg_patch_ratio = stats['bg_patches']      / (stats['total_patches'] + 1e-9)
    return {
        'patch_recall':   patch_recall,
        'gt_coverage':    gt_coverage,
        'bg_patch_ratio': bg_patch_ratio,
    }


def heatmap_quality_update(p_seg, masks, targets, stats, threshold=0.5):
    """
    Accumulate heatmap quality stats for one batch.

    p_seg  : [B, C, H, W]  C=1 (sigmoid logit) or C=2 (EDL evidence logits)
    masks  : [B, 1, H, W]  binary GT mask (0/1 float)
    targets: [N, 6]        [batch_idx, cls, xc, yc, w, h] normalized
    stats  : dict with int keys 'tp','fp','tn','fn','obj_hit','obj_total'
    """
    device = p_seg.device
    B, C, H, W = p_seg.shape

    # → detection map [B,1,H,W]
    if C == 2:
        # EDL: vacuity 기반 탐지 (불확실한 곳 = 객체 후보)
        evidence = F.softplus(p_seg)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        detection_map = 2.0 / S           # vacuity [B,1,H,W]
    else:
        detection_map = p_seg.sigmoid()

    pred_bin = detection_map >= threshold  # [B,1,H,W]
    gt_bin   = masks >= 0.5               # [B,1,H,W]

    stats['tp'] += int((pred_bin &  gt_bin).sum())
    stats['fp'] += int((pred_bin & ~gt_bin).sum())
    stats['tn'] += int((~pred_bin & ~gt_bin).sum())
    stats['fn'] += int((~pred_bin &  gt_bin).sum())

    # object recall: ≥1 positive pixel inside each GT box
    if len(targets):
        gain = torch.tensor([[W, H, W, H]], device=device, dtype=torch.float32)
        bboxes = general.xywh2xyxy(targets[:, 2:6])   # normalized xyxy
        bboxes = (bboxes * gain).long()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(0, W)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(0, H)
        for i in range(len(targets)):
            bi = int(targets[i, 0].long())
            x1, y1, x2, y2 = bboxes[i].tolist()
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            hit = pred_bin[bi, 0, y1:y2, x1:x2].any()
            stats['obj_hit']   += int(hit)
            stats['obj_total'] += 1


def heatmap_quality_summary(stats):
    """Compute final metrics from accumulated stats dict."""
    tp, fp, tn, fn = stats['tp'], stats['fp'], stats['tn'], stats['fn']
    prec  = tp / (tp + fp + 1e-9)
    rec   = tp / (tp + fn + 1e-9)
    f1    = 2 * prec * rec / (prec + rec + 1e-9)
    iou   = tp / (tp + fp + fn + 1e-9)
    obj_r = stats['obj_hit'] / (stats['obj_total'] + 1e-9)
    bg_fpr = fp / (fp + tn + 1e-9)
    return {
        'pixel_precision': prec,
        'pixel_recall':    rec,
        'pixel_f1':        f1,
        'pixel_iou':       iou,
        'obj_recall':      obj_r,
        'bg_fpr':          bg_fpr,
    }


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
