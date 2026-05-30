# # Loss functions
# # Copyright (c) Alibaba, Inc. and its affiliates.

# ############ EDL ############

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.general import bbox_iou, box_iou, wh_iou, xywh2xyxy
from utils.torch_utils import is_parallel, time_synchronized
from utils import edl_det


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)
            # BCEobj = FocalLoss(BCEobj, g)
        # else:
        #     BCEobj = QFocalLoss(BCEobj, gamma=1.5, alpha=0.5)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.det = det  # ECPR: Detect.graph_out 접근용
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'anchor_grid', 'stride':
            setattr(self, k, getattr(det, k))
        self.neg_anchor_iou_thres = 0.7
        self.pos_anchor_iou_thres = 0.15
        self.pos_anchor_num = 4
        self.lpixl_critreia = None

    def __call__(self, p, targets, imgsz=None, masks=None, m_weights=None):  # predictions, targets, model
        p_det, p_seg = p
        offsets = []
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        ledl = torch.zeros(1, device=device)  # EDL parallel head loss (IoU-aware R-EDL)
        
        if p_det is not None and p_det[0] is not None and p_det[1] is not None:  # stupid
            # ta = time_synchronized()
            if isinstance(p_det, tuple):
                p, offsets = p_det
                tcls, tbox, indices, anchors = self.build_patch_targets(offsets, targets, imgsz)  # targets
            else:
                p = p_det
                tcls, tbox, indices, anchors = self.build_targets(p, targets)
            # print(f'build_targets: {time_synchronized() - ta:.3f}s.')

            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
    
                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    
                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss
    
                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
    
                    # Classification — BCE (default) OR EOD-style Evidential (env ESOD_EVIDENTIAL_CLS=1)
                    if self.nc > 1:
                        import os as _os
                        _ev_cls = _os.environ.get('ESOD_EVIDENTIAL_CLS', '').strip() == '1'
                        if _ev_cls:
                            # EOD AAAI'24: K-class R-EDL Dirichlet MSE on positive anchors
                            # cls_logit [n, K], target = tcls[i] long [n]
                            lcls += edl_det.edl_cls_loss_Kch(
                                ps[:, 5:].clamp(-9.21, 9.21), tcls[i],
                                pos_mask=None,  # positive anchor만 들어옴
                                label_smoothing=self.hyp.get('label_smoothing', 0.0))
                        else:
                            t = torch.full_like(ps[:, 5:], self.cn, device=device)
                            t[range(n), tcls[i]] = self.cp
                            lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    
                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
                # OWOD: Unknown box overlap anchor 의 objectness loss 를 ignore (표준 spec)
                #   self._last_unknown_targets [Nu, 6] (bi, cls=999, x, y, w, h) normalized in image space
                #   → 해당 patch 내 anchor cell 이 unknown box 와 IoU > 0.1 이면 ignore mask True
                _unk_t = getattr(self, '_last_unknown_targets', None)
                _ignore_obj = None
                if _unk_t is not None and len(_unk_t) > 0:
                    _ignore_obj = self._compute_obj_ignore_mask(
                        _unk_t, offsets[i] if (offsets is not None and len(offsets) > i) else None,
                        pi.shape, i, imgsz)

                if edl_det.EDL_DET:
                    # detector-stage IoU-aware R-EDL: objectness를 evidential로 학습
                    pos = torch.zeros_like(tobj, dtype=torch.bool)
                    if n:
                        pos[b, a, gj, gi] = True
                    obji = edl_det.iou_aware_edl_loss(
                        pi[..., 4].clamp(-9.21, 9.21).reshape(-1), tobj.reshape(-1),
                        pos_mask=pos.reshape(-1), pos_weight=self.hyp['obj_pw'])
                else:
                    if _ignore_obj is not None and _ignore_obj.any():
                        # mask-weighted BCEobj (ignore mask 영역 weight 0)
                        _obj_w = (~_ignore_obj).float()
                        _bce_raw = F.binary_cross_entropy_with_logits(
                            pi[..., 4].clamp_(-9.21, 9.21), tobj, reduction='none')
                        obji = (_bce_raw * _obj_w).sum() / (_obj_w.sum().clamp(min=1.0))
                    else:
                        obji = self.BCEobj(pi[..., 4].clamp_(-9.21, 9.21), tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

                # EDL parallel evidence head (m_ev) — IoU-aware R-EDL, conf branch와 별개로 학습
                # conf branch (BCEobj 위)는 그대로 → mAP 영향 0.  evidence head는 OOD/calibration 신호용.
                head_i = self.det.m[i] if self.det is not None else None
                if head_i is not None and getattr(head_i, 'last_ev', None) is not None:
                    ev_i = head_i.last_ev.permute(0, 1, 3, 4, 2).contiguous()  # [B, na, ny, nx, 2]
                    if ev_i.shape[:-1] == tobj.shape:
                        pos_ev = torch.zeros_like(tobj, dtype=torch.bool)
                        if n:
                            pos_ev[b, a, gj, gi] = True
                        l_ev_i = edl_det.iou_aware_edl_loss_2ch(
                            ev_i.reshape(-1, 2), tobj.reshape(-1),
                            pos_mask=pos_ev.reshape(-1), pos_weight=self.hyp['obj_pw'])
                        ledl += l_ev_i * self.balance[i]
        
        # bs = tobj.shape[0]  # batch size
        bs = p_seg[0].shape[0] if (p_seg is not None and len(p_seg) > 0) else tobj.shape[0]
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj'] * 0.5 #(0.5 if (len(offsets) and len(offsets[0]) > bs) else 1.)   # adaoff: 0.178
        lcls *= self.hyp['cls']
        
        if masks is not None and p_seg is not None:
            # Cases:
            #   len(p_seg)==1, [B,1,H,W]                            → BCE only (legacy ESOD)
            #   len(p_seg)==1, [B,2,H,W]                            → EDL only
            #   len(p_seg)==2 [heat(1ch), edl(2ch)]                 → Dual + max fusion (raw vacuity)
            #   len(p_seg)==3 [heat(1ch), edl(2ch), vacuity_cal]    → Dual + max fusion + calibration ⭐
            #   ESOD_ROLE_DUAL=1 (env)  + len==3                    → 3-C role-separated dual ✨
            import os as _os
            _role = _os.environ.get('ESOD_ROLE_DUAL', '').strip() == '1'
            if (len(p_seg) == 3 and p_seg[0].shape[1] == 1
                    and p_seg[1].shape[1] == 2 and p_seg[2].shape[1] == 1):
                if _role:
                    lpixl, larea, ldist = self.compute_loss_seg_role_dual(
                        p_seg[0], p_seg[1], masks, targets, weight=m_weights)
                else:
                    lpixl, larea, ldist = self.compute_loss_seg_dual_cal(
                        p_seg[0], p_seg[1], p_seg[2], masks, targets, weight=m_weights)
            elif len(p_seg) == 2 and p_seg[0].shape[1] == 1 and p_seg[1].shape[1] == 2:
                lpixl, larea, ldist = self.compute_loss_seg_dual(p_seg[0], p_seg[1], masks, targets, weight=m_weights)
            else:
                assert len(p_seg) == 1
                lpixl, larea, ldist = self.compute_loss_seg(p_seg[0], masks, targets, weight=m_weights)
        
        # Segmentation loss weight 0.2 (ESOD 원본과 동일).
        # EDL parallel head loss weight: 작게 (0.1) — conf branch는 그대로(BCEobj), 별개 신호.
        # 환경변수 ESOD_EDL_DET_LAMBDA로 override 가능 (baseline 학습 시 0 으로 끄기).
        import os as _os
        _env_lam = _os.environ.get('ESOD_EDL_DET_LAMBDA', None)
        ledl_w = float(_env_lam) if _env_lam is not None else getattr(ComputeLoss, 'EDL_DET_LAMBDA', 0.1)
        loss = (lbox + lobj + lcls) * 1.0 + (lpixl + larea + ldist) * 0.2 + ledl * ledl_w
        # deepcopy 안전: head.last_ev (학습 grad-tied) 는 loss 끝에서 해제
        if self.det is not None:
            for _hi in range(self.nl):
                _h = self.det.m[_hi]
                if getattr(_h, 'last_ev', None) is not None:
                    _h.last_ev = None
        loss_items = torch.cat((lbox, lobj, lcls, lpixl, larea, ldist, loss)).detach()
        return loss * bs, loss_items

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h), 0~1
        # OWOD ignore: cls>=100 (sentinel 999) target 은 *positive matching* 에서 skip + unknown_targets 보관
        # → ComputeLoss.__call__ 에서 unknown box overlap anchor 의 objectness loss 를 ignore
        self._last_unknown_targets = targets[targets[:, 1] >= 100].detach().clone()
        targets = targets[targets[:, 1] < 100]
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)

    def _compute_obj_ignore_mask(self, unknown_targets, patch_offset, pi_shape, layer_idx, imgsz):
        """OWOD: Unknown box overlap anchor 의 objectness ignore mask. (vectorized — no Python loop / .item)

        unknown_targets: [Nu, 6] (bi, cls=999, x, y, w, h) normalized in image space
        patch_offset:    [Bp, 5] (bi, x1, y1, x2, y2) grid space coords at layer i
        pi_shape:        [Bp(or B), na, ny, nx, no]
        layer_idx:       layer i
        imgsz:           input image shape (bs, ch, H, W)
        반환: ignore_mask [Bp, na, ny, nx] bool — True 면 objectness loss 제외
        """
        if unknown_targets is None or len(unknown_targets) == 0 or patch_offset is None or imgsz is None:
            return None
        Bp, na, ny, nx = pi_shape[:4]
        Nu = unknown_targets.shape[0]
        device = unknown_targets.device
        H_img = float(imgsz[-2]); W_img = float(imgsz[-1])
        stride_i = float(self.stride[layer_idx]) if hasattr(self, 'stride') else 8.0 * (2 ** layer_idx)

        # unknown box pixel ranges [Nu]
        u_bi = unknown_targets[:, 0].long()
        u_cx = unknown_targets[:, 2] * W_img
        u_cy = unknown_targets[:, 3] * H_img
        u_w  = unknown_targets[:, 4] * W_img
        u_h  = unknown_targets[:, 5] * H_img
        u_x1 = u_cx - u_w / 2
        u_x2 = u_cx + u_w / 2
        u_y1 = u_cy - u_h / 2
        u_y2 = u_cy + u_h / 2

        # patch pixel-space origins [Bp]
        p_bi = patch_offset[:, 0].long()
        p_x1 = patch_offset[:, 1].float() * stride_i
        p_y1 = patch_offset[:, 2].float() * stride_i

        # broadcast: [Bp, Nu] grid cell ranges (clamp to [0, nx], [0, ny])
        gxs = ((u_x1.unsqueeze(0) - p_x1.unsqueeze(1)) / stride_i).floor().clamp(0, nx).long()  # [Bp, Nu]
        gxe = ((u_x2.unsqueeze(0) - p_x1.unsqueeze(1)) / stride_i).ceil().clamp(0, nx).long()
        gys = ((u_y1.unsqueeze(0) - p_y1.unsqueeze(1)) / stride_i).floor().clamp(0, ny).long()
        gye = ((u_y2.unsqueeze(0) - p_y1.unsqueeze(1)) / stride_i).ceil().clamp(0, ny).long()

        # validity: same image_idx AND non-empty rectangle
        bi_match = (p_bi.unsqueeze(1) == u_bi.unsqueeze(0))                                 # [Bp, Nu]
        valid = bi_match & (gxe > gxs) & (gye > gys)                                        # [Bp, Nu]

        # cell coordinate grids [ny, nx] → broadcast to [1, 1, ny, nx]
        yy = torch.arange(ny, device=device).view(1, 1, ny, 1)
        xx = torch.arange(nx, device=device).view(1, 1, 1, nx)

        # rectangle bounds expanded to [Bp, Nu, 1, 1]
        gxs_b = gxs.unsqueeze(-1).unsqueeze(-1)
        gxe_b = gxe.unsqueeze(-1).unsqueeze(-1)
        gys_b = gys.unsqueeze(-1).unsqueeze(-1)
        gye_b = gye.unsqueeze(-1).unsqueeze(-1)
        v_b   = valid.unsqueeze(-1).unsqueeze(-1)

        # full [Bp, Nu, ny, nx] memory may be large — chunk over Nu if needed
        # 안전 chunk: max ~ 32 unknown boxes per pass to bound peak memory
        cell_ignore = torch.zeros((Bp, ny, nx), dtype=torch.bool, device=device)
        chunk = min(Nu, 32)
        for s in range(0, Nu, chunk):
            e = min(s + chunk, Nu)
            in_rect = ((yy >= gys_b[:, s:e]) & (yy < gye_b[:, s:e]) &
                       (xx >= gxs_b[:, s:e]) & (xx < gxe_b[:, s:e]) & v_b[:, s:e])  # [Bp, k, ny, nx]
            cell_ignore = cell_ignore | in_rect.any(dim=1)

        # expand to all anchors → [Bp, na, ny, nx]
        return cell_ignore.unsqueeze(1).expand(-1, na, -1, -1).contiguous()

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter shape(nt_,7), [bi, ci, xc, yc, w, h, ai]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
  
    def build_patch_targets(self, patch_offsets, targets, imgsz):  # for fast-mode, fixed patch division
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # OWOD ignore: cls>=100 (sentinel 999) target 은 *positive matching*에서 skip
        #              + unknown box overlap anchor 의 objectness loss 도 ignore (표준 OWOD spec)
        # → unknown_targets 를 attribute 로 보관 → __call__ 에서 objectness ignore mask 생성용
        self._last_unknown_targets = targets[targets[:, 1] >= 100].detach().clone()
        targets = targets[targets[:, 1] < 100]
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        dtype, device = targets.dtype, targets.device
        tcls, tbox, indices, anch = [], [], [], []
        bs, _, height, width = imgsz
        
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)
        bi_ = torch.arange(patch_offsets[0].shape[0], device=device)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=device).float() * g  # offsets

        for i in range(self.nl):
            patch_off = patch_offsets[i]
            anchors = self.anchors[i]
            r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
            gain[2:6] = torch.tensor([width, height, width, height], dtype=dtype) / (8 * r)  # TODO: from 4 to 32
            # grid_w, grid_h = patch_off[0, [3, 4]] - patch_off[0, [1, 2]]
            grid_wh = patch_off[:1, [3, 4]] - patch_off[:1, [1, 2]]

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter, shape(nt_, 7)

                tb, txc, tyc = t[:, [0, 2, 3]].chunk(3, dim=1)  # shape(n,1)
                pb, px1, py1, px2, py2 = (patch_off.T).chunk(5, dim=0)  # shape(1,m)
                contained = (tb == pb) & (txc > px1 - g) & (txc < px2 - g) & (tyc > py1 - g) & (tyc < py2 - g)  # shape(n,m)
                ti, pj = torch.nonzero(contained).T  # i-th target is contained within j-th patch
                t = t[ti]  # shape(n,7)
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = grid_wh - gxy  # inverse
                j, k = ((gxy - gxy.floor() < g) & (gxy > 0.-g)).T
                l, m = ((gxi - gxi.floor() < g) & (gxi > 1.-g)).T
                # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                
                t[:, 0] = bi_[pj]  # converted batch-indices
                t[:, 2:4] -= patch_off[pj, 1:3]  # converted xc, yc (minus px1, py1)

                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # assert ((gj >= 0) & (gj <= grid_wh[0,1] - 1) & (gi >= 0) & (gi <= grid_wh[0,0] - 1)).all()
            # indices.append((b, a, gj.clamp_(0, grid_wh[0,1] - 1), gi.clamp_(0, grid_wh[0,0] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    # def compute_loss_seg(self, p, masks, targets, weight=None):
    #     dtype, device = targets.dtype, targets.device
    #     bs, nc, ny, nx = masks.shape
    #     assert nc == 1
    #     lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), \
    #                           torch.zeros(1, device=device)
        
    #     # weight = None
    #     lpixl += F.binary_cross_entropy_with_logits(p, masks, weight=weight)

    #     nt = targets.shape[0]
    #     if nt:  # number of targets
    #         pass

    #         # larea += self.dice_loss(p, masks)
    #         # ldist += self.sigmoid_focal_loss(p, masks) * 20
            
    #         # larea += self.quality_dice_loss(p, masks, weight=weight)
    #         # ldist += self.sigmoid_quality_focal_loss(p, masks, weight=weight) * 20

    
    #     return lpixl, larea, ldist
    
    # ============================================================
    # Dual-branch (방법 1: Patch-level Uncertainty Selection)
    # = BCE(heat) + EDL(edl)
    #
    # 설계 철학:
    #   - Heat: 객체 위치 학습 (BCE, ESOD 원본과 동일)
    #   - EDL:  픽셀별 evidence 학습 (Dirichlet loss, class-balanced)
    #   - 두 branch는 독립적으로 학습
    #   - Patch-level fusion은 inference에서 HeatMapParser가 수행
    #     (prob으로 patch 생성 → vacuity로 patch 필터링)
    # ============================================================

    LAMBDA_FUSED   = 0.5    # auxiliary MSE(F) weight (gate가 학습되는 경로)
    LAMBDA_ENTROPY = 0.10   # gate entropy regularization (0.05→0.10, collapse 방지 강화)
    LAMBDA_HARD    = 0.0    # OFF: ablation 결과 hard-aware는 mAP 도움 안 됨(0.573<0.577).
                            # base 모델은 calib-gating 단독(0.577). 2.0=ablation 재현용

    def compute_loss_seg_dual_cal(self, p_heat, p_edl, p_gate, masks, targets, weight=None):
        """
        p_heat: [B,1,H,W]  heat branch logits
        p_edl:  [B,2,H,W]  edl  branch logits
        p_gate: [B,1,H,W]  gate branch logits

        Calibrated gating fusion:
          prob = sigmoid(heat)
          p_e  = alpha_obj/S ; u (vacuity) = 2/S
          gate = sigmoid(p_gate)
          w    = gate·(1-u)
          F = (1-w)·prob + w·p_e

        Total = BCE(heat, GT)              ← Heat 학습 (ESOD 동일)
              + EDL(edl, GT)               ← EDL 학습 (class-balanced)
              + λf · BCE(F, GT)            ← gate 학습 경로
              + λe · entropy_reg(gate)     ← gate collapse 방지
        """
        device = targets.device

        # 1) Heat — 표준 BCE
        lpixl_heat = F.binary_cross_entropy_with_logits(p_heat, masks, weight=weight)

        # 2) EDL — Dirichlet MSE + KL (class-balanced) + STEP2 hard-aware 재가중
        #    heat이 틀린 영역에서 EDL loss를 키워 EDL을 heat 실패 모드에 특화시킨다
        #    (boosting 원리). detach → EDL→heat 역방향 피드백 차단, 단일 학습 안정.
        lambda_hard = getattr(ComputeLoss, 'LAMBDA_HARD', 2.0)
        heat_err = (p_heat.sigmoid().detach() - masks).abs()       # [B,1,H,W] ∈[0,1]
        base_w = weight if weight is not None else torch.ones_like(masks)
        w_edl = base_w * (1.0 + lambda_hard * heat_err)
        lpixl_edl, _, _ = self.compute_loss_seg(p_edl, masks, targets, weight=w_edl)

        # 3) Fusion (calibrated): belief는 결합, vacuity는 '신뢰 게이트'로만 사용
        #    prob_h, p_e 둘 다 [0,1] calibrated 객체확률 → 정규화 불필요, F도 calibrated
        #    → threshold 0.5가 ESOD heat과 동일 의미 (공정 비교 복구)
        prob_h = p_heat.sigmoid()                            # [B,1,H,W] heat 객체확률
        evidence = F.softplus(p_edl)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)                   # [B,1,H,W]
        p_e = alpha[:, 1:2] / S                              # EDL 객체확률 (ch1=obj)
        u   = 2.0 / S                                        # vacuity = 인식 불확실도
        gate = p_gate.sigmoid()                              # learnable gate (live: fused MSE + entropy)
        # fused 항은 'gate만' 학습: branch 출력/u detach
        # → heat=순수 BCE, EDL=순수 Dirichlet (단일 branch와 동일) → 손상 0 + ablation 정합
        w   = gate * (1.0 - u.detach())                      # gate만 gradient
        fused = (1.0 - w) * prob_h.detach() + w * p_e.detach()   # [0,1] calibrated, gate-only grad
        eps = 1e-6

        # 확률공간 fusion → MSE (gradient bounded 2(F-y), autocast-safe).
        # log-odds 역변환(BCE-with-logits)은 clamp 경계에서 야코비안 1/(F(1-F))≈1e6
        # 폭발 → NaN 유발하므로 사용 금지. EDL Dirichlet-MSE와도 일관.
        if weight is not None:
            lpixl_fused = (weight * (fused - masks).pow(2)).mean()
        else:
            lpixl_fused = F.mse_loss(fused, masks)

        # 4) Gate entropy regularization (α가 0/1로 collapse 방지)
        ge = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(ge * ge.log() + (1.0 - ge) * (1.0 - ge).log())
        entropy_reg = -gate_entropy.mean()  # entropy 최대화 = -entropy 최소화

        lambda_f = getattr(ComputeLoss, 'LAMBDA_FUSED', 0.5)
        lambda_e = getattr(ComputeLoss, 'LAMBDA_ENTROPY', 0.05)

        lpixl = (lpixl_heat.reshape(1)
                 + lpixl_edl.reshape(1)
                 + lambda_f * lpixl_fused.reshape(1)
                 + lambda_e * entropy_reg.reshape(1))

        # 모니터링 (loss 영향 X)
        with torch.no_grad():
            ComputeLoss.LAST_VAC_RAW_MEAN = float(u.mean().item())
            ComputeLoss.LAST_VAC_RAW_MAX  = float(u.max().item())
            ComputeLoss.LAST_GATE_MEAN    = float(gate.mean().item())
            ComputeLoss.LAST_GATE_MIN     = float(gate.min().item())
            ComputeLoss.LAST_GATE_MAX     = float(gate.max().item())

        larea = torch.zeros(1, device=device)
        ldist = torch.zeros(1, device=device)
        return lpixl, larea, ldist

    # ============================================================
    # 3-C Role-separated dual: heat (GT) + edl (heat-error mask) + noisy-OR fusion
    # ============================================================
    def compute_loss_seg_role_dual(self, p_heat, p_edl, masks, targets, weight=None):
        """Role-separated dual segmenter (3-C).

        Heat branch : GT mask 그대로 학습 (BCE class-balanced) — closed-set easy/큰 객체
        EDL branch  : heat이 약한 GT region 만 학습 (R-EDL Dirichlet-MSE) — hard/missed object

        EDL target (soft, detach):  hard = GT * (1 - sigmoid(heat).detach())
          - heat 강함(≈1) → hard≈0  → EDL은 그 region 학습 안 함
          - heat 약함(≈0) & GT region → hard≈1 → EDL이 그 region 집중 학습

        두 branch가 *다른 분포*를 학습 → ensemble diversity 가능 (fusion-redundancy 회피).
        Inference fusion(별도, noisy-OR): F = 1 - (1-σ(heat)) * (1-p_e_obj)
        """
        device = targets.device
        eps = 1e-6

        # 1) Heat — class-balanced BCE
        bce = F.binary_cross_entropy_with_logits(p_heat, masks, reduction='none')
        if weight is not None:
            bce = bce * weight
        obj_m = masks > 0.5; bg_m = ~obj_m
        l_obj_h = bce[obj_m].mean() if obj_m.any() else bce.new_zeros(())
        l_bg_h = bce[bg_m].mean() if bg_m.any() else bce.new_zeros(())
        pw = getattr(ComputeLoss, 'POS_WEIGHT', 3.0)
        lpixl_heat = (pw * l_obj_h + l_bg_h) / (pw + 1.0)

        # 2) EDL — R-EDL Dirichlet-MSE on hard-region target (soft, detach)
        heat_prob = p_heat.sigmoid().detach()                  # [B,1,H,W]
        hard_target = masks * (1.0 - heat_prob)                # [B,1,H,W] ∈[0,1]

        evidence = F.softplus(p_edl)                            # [B,2,H,W] e_bg, e_obj
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)                      # [B,1,H,W]
        y = hard_target
        one_hot = torch.cat([1.0 - y, y], dim=1)                # [B,2,H,W] target prob
        probs = alpha / S
        err = (one_hot - probs).pow(2)
        var = alpha * (S - alpha) / (S.pow(2) * (S + 1.0))
        edl_mse = (err + var).sum(dim=1, keepdim=True)          # [B,1,H,W]

        # class-balanced: hard region이 매우 sparse → object/background pixel separate
        # hard threshold 0.3 ≈ heat sigmoid<0.7 + GT region
        hard_m = hard_target > 0.3
        bg_m_e = ~hard_m
        l_h = edl_mse[hard_m].mean() if hard_m.any() else edl_mse.new_zeros(())
        l_bg_e = edl_mse[bg_m_e].mean() if bg_m_e.any() else edl_mse.new_zeros(())
        pw_e = getattr(ComputeLoss, 'EDL_HARD_POS_WEIGHT', 5.0)
        lpixl_edl = (pw_e * l_h + l_bg_e) / (pw_e + 1.0)

        # 3) Total — gate output 학습 안 시킴 (zero-init 유지 → fusion에서 unused)
        lpixl = (lpixl_heat.reshape(1) + lpixl_edl.reshape(1))

        # monitoring
        with torch.no_grad():
            u = 2.0 / S
            ComputeLoss.LAST_VAC_RAW_MEAN = float(u.mean().item())
            ComputeLoss.LAST_HARD_MEAN = float(hard_target.mean().item())
            ComputeLoss.LAST_HARD_FRAC = float(hard_m.float().mean().item())

        larea = torch.zeros(1, device=device)
        ldist = torch.zeros(1, device=device)
        return lpixl, larea, ldist

    # ============================================================
    # Dual-branch loss (max fusion, no MoE): BCE + EDL + Decorrelation
    # ============================================================

    def compute_loss_seg_dual(self, p_heat, p_edl, masks, targets, weight=None):
        """
        p_heat: [B,1,H,W]  heat branch logits (sigmoid → prob)
        p_edl:  [B,2,H,W]  edl branch logits (softplus → evidence)
        masks:  [B,1,H,W]  binary GT (0/1)

        Total = BCE(heat) + EDL(edl) + λ * Decorrelation(prob, vacuity)
        """
        device = targets.device
        # 1) Heat branch: 표준 BCE (ESOD 원본과 동일)
        lpixl_heat = F.binary_cross_entropy_with_logits(p_heat, masks, weight=weight)
        # 2) EDL branch: Dirichlet MSE + KL
        lpixl_edl, _, _ = self.compute_loss_seg(p_edl, masks, targets, weight=weight)

        # 3) Decorrelation Loss
        #    두 브랜치가 같은 신호를 학습하지 않도록 강제
        #    prob과 vacuity 사이 양의 상관관계만 페널티 (anti-correlation은 OK)
        prob = p_heat.sigmoid()                          # [B,1,H,W]
        evidence = F.softplus(p_edl)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        vacuity = 2.0 / S                                # [B,1,H,W]

        # Pearson correlation (배치 내 모든 픽셀)
        p_flat = prob.flatten()
        v_flat = vacuity.flatten()
        p_c = p_flat - p_flat.mean()
        v_c = v_flat - v_flat.mean()
        denom = (p_c.std(unbiased=False) * v_c.std(unbiased=False) + 1e-8)
        corr = (p_c * v_c).mean() / denom

        # 양의 상관관계만 페널티 (≤0이면 두 브랜치가 다른 신호 = OK)
        decorr_loss = corr.clamp(min=0.0)

        lambda_decorr = getattr(ComputeLoss, 'LAMBDA_DECORR', 0.2)

        lpixl = (lpixl_heat.reshape(1)
                 + lpixl_edl.reshape(1)
                 + lambda_decorr * decorr_loss.reshape(1))
        # 모니터링: corr 값을 클래스 속성으로 저장 (loss에 영향 X)
        ComputeLoss.LAST_CORR = float(corr.detach().item())
        larea = torch.zeros(1, device=device)
        ldist = torch.zeros(1, device=device)
        return lpixl, larea, ldist

    # EDL 기반 Segmentation Loss (Class-balanced for object/background imbalance)
    def compute_loss_seg(self, p, masks, targets, weight=None):
        """
        p: [B,2,H,W]  (bg,obj evidence logits)
        masks: [B,1,H,W] (0/1 binary GT mask)

        개선:
        - 배경 픽셀이 ~99%, 객체 픽셀이 ~1%인 imbalance 해결
        - 객체/배경 픽셀 loss를 별도로 평균 → 동등 가중
        - 이렇게 해야 객체 영역에서 vacuity가 의미있게 낮아짐
        """
        device = targets.device
        bs, nc_t, ny, nx = masks.shape
        assert nc_t == 1, "GT mask는 1채널(0/1)이어야 함"

        # 1채널 plain heat segmenter (ESOD baseline) — class-balanced BCE
        if p.shape[1] == 1:
            bce = F.binary_cross_entropy_with_logits(p, masks, reduction='none')
            if weight is not None:
                bce = bce * weight
            obj_m = masks > 0.5
            bg_m = ~obj_m
            l_obj = bce[obj_m].mean() if obj_m.any() else bce.new_zeros(())
            l_bg = bce[bg_m].mean() if bg_m.any() else bce.new_zeros(())
            pw = getattr(ComputeLoss, 'POS_WEIGHT', 3.0)
            lpixl = ((pw * l_obj + l_bg) / (pw + 1.0)).reshape(1)
            zero = torch.zeros(1, device=device)
            return lpixl, zero, zero

        assert p.shape[1] == 2, f"EDL 사용 시 pred channels must be 2, got {p.shape}"

        # evidence / alpha
        evidence = F.softplus(p)          # [B,2,H,W] >= 0
        alpha = evidence + 1.0            # Dirichlet params
        S = alpha.sum(dim=1, keepdim=True)  # [B,1,H,W]

        # one-hot GT
        y = masks                         # [B,1,H,W]
        one_hot = torch.cat([1.0 - y, y], dim=1)  # [B,2,H,W]

        # 1) EDL MSE loss (Type-II Maximum Likelihood)
        probs = alpha / S
        err = (one_hot - probs) ** 2
        var = alpha * (S - alpha) / (S ** 2 * (S + 1))
        edl_mse = (err + var).sum(dim=1)  # [B,H,W]

        # 2) KL divergence: 틀린 클래스의 evidence를 0으로 보냄
        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        kl = self._kl_divergence_dirichlet(alpha_tilde)  # [B,H,W]

        # KL annealing: 더 천천히 (epoch/10 → epoch/25)
        # → Evidence 충분히 학습된 후 KL 적용
        # → vacuity가 정상적으로 학습됨
        # KL annealing: λ_max=0.1 (spec / EOD AAAI'24 권장 — dense prediction에서 1.0은 collapse 유발)
        # ESOD_KL_LAMBDA_MAX 환경변수로 override 가능 (default 0.1)
        import os as _os_kl
        _kl_lam_max = float(_os_kl.environ.get('ESOD_KL_LAMBDA_MAX', '0.1'))
        annealing_coef = _kl_lam_max * min(1.0, getattr(ComputeLoss, '_edl_epoch', 1) / 25.0)
        edl_per_pixel = edl_mse + annealing_coef * kl  # [B,H,W]

        if weight is not None:
            edl_per_pixel = edl_per_pixel * weight.squeeze(1)

        # ───────────────────────────────────────────────────
        # Class-balanced averaging (핵심 개선)
        # 배경 ~99%, 객체 ~1% imbalance → 객체/배경 분리 평균
        # ───────────────────────────────────────────────────
        obj_mask_bool = (y.squeeze(1) > 0.5)  # [B,H,W]
        bg_mask_bool  = ~obj_mask_bool

        n_obj = obj_mask_bool.sum().clamp(min=1)
        n_bg  = bg_mask_bool.sum().clamp(min=1)

        loss_obj = (edl_per_pixel * obj_mask_bool.float()).sum() / n_obj
        loss_bg  = (edl_per_pixel * bg_mask_bool.float()).sum() / n_bg

        # 객체에 더 큰 가중 (배경 ~99%, 객체 ~1% 분포 보완)
        # POS_WEIGHT 3.0: 객체:배경 = 3:1로 가중
        # → 객체 영역에서 evidence 더 강하게 학습 → vacuity 낮아짐
        pos_weight = getattr(ComputeLoss, 'POS_WEIGHT', 3.0)
        lpixl = (pos_weight * loss_obj + loss_bg) / (pos_weight + 1.0)
        lpixl = lpixl.reshape(1)

        larea = torch.zeros(1, device=device)
        ldist = torch.zeros(1, device=device)
        return lpixl, larea, ldist

    @staticmethod
    def _kl_divergence_dirichlet(alpha):
        """KL(Dir(alpha) || Dir(1,...,1)) per pixel. alpha: [B, K, H, W]"""
        K = alpha.shape[1]
        S = alpha.sum(dim=1)  # [B, H, W]
        kl = torch.lgamma(S) - torch.lgamma(torch.tensor(float(K), device=alpha.device)) \
             - torch.lgamma(alpha).sum(dim=1) \
             + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
        return kl

    @staticmethod
    def dice_loss(inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    @staticmethod
    def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        if weight is not None:
            weight = weight.flatten(1)
            inputs = inputs * weight
            targets = targets * weight

        numerator = 2 * (inputs - targets).abs().sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
        loss = ce_loss * ((prob - targets).abs() ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

### graph ###
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math

# from utils.general import bbox_iou, box_iou, wh_iou, xywh2xyxy
# from utils.torch_utils import is_parallel, time_synchronized


# def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
#     # return positive, negative label smoothing BCE targets
#     return 1.0 - 0.5 * eps, 0.5 * eps


# class BCEBlurWithLogitsLoss(nn.Module):
#     # BCEwithLogitLoss() with reduced missing label effects.
#     def __init__(self, alpha=0.05):
#         super(BCEBlurWithLogitsLoss, self).__init__()
#         self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
#         self.alpha = alpha

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred = torch.sigmoid(pred)  # prob from logits
#         dx = pred - true  # reduce only missing label effects
#         # dx = (pred - true).abs()  # reduce missing label and false label effects
#         alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
#         loss *= alpha_factor
#         return loss.mean()


# class FocalLoss(nn.Module):
#     # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(FocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply FL to each element

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         # p_t = torch.exp(-loss)
#         # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

#         # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class QFocalLoss(nn.Module):
#     # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(QFocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply FL to each element

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)

#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = torch.abs(true - pred_prob) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class ComputeLoss:
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         super(ComputeLoss, self).__init__()
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls = FocalLoss(BCEcls, g)
#             # BCEobj = FocalLoss(BCEobj, g)
#         # else:
#         #     BCEobj = QFocalLoss(BCEobj, gamma=1.5, alpha=0.5)

#         det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
#         for k in 'na', 'nc', 'nl', 'anchors', 'anchor_grid', 'stride':
#             setattr(self, k, getattr(det, k))
#         self.neg_anchor_iou_thres = 0.7
#         self.pos_anchor_iou_thres = 0.15
#         self.pos_anchor_num = 4
#         self.lpixl_critreia = None

#     def __call__(self, p, targets, imgsz=None, masks=None, m_weights=None):  # predictions, targets, model
#         p_det, p_seg = p
#         offsets = []
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
#         if p_det is not None and p_det[0] is not None and p_det[1] is not None:  # stupid
#             # ta = time_synchronized()
#             if isinstance(p_det, tuple):
#                 p, offsets = p_det
#                 tcls, tbox, indices, anchors = self.build_patch_targets(offsets, targets, imgsz)  # targets
#             else:
#                 p = p_det
#                 tcls, tbox, indices, anchors = self.build_targets(p, targets)
#             # print(f'build_targets: {time_synchronized() - ta:.3f}s.')

#             # Losses
#             for i, pi in enumerate(p):  # layer index, layer predictions
#                 b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#                 tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
    
#                 n = b.shape[0]  # number of targets
#                 if n:
#                     ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    
#                     # Regression
#                     pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                     pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                     pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                     iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                     lbox += (1.0 - iou).mean()  # iou loss
    
#                     # Objectness
#                     tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
    
#                     # Classification
#                     if self.nc > 1:  # cls loss (only if multiple classes)
#                         t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                         t[range(n), tcls[i]] = self.cp
#                         lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    
#                     # Append targets to text file
#                     # with open('targets.txt', 'a') as file:
#                     #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
#                 obji = self.BCEobj(pi[..., 4].clamp_(-9.21, 9.21), tobj)
#                 lobj += obji * self.balance[i]  # obj loss
#                 if self.autobalance:
#                     self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
#         # bs = tobj.shape[0]  # batch size
#         bs = p_seg[0].shape[0] if p_seg is not None else tobj.shape[0]
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj'] * 0.5 #(0.5 if (len(offsets) and len(offsets[0]) > bs) else 1.)   # adaoff: 0.178
#         lcls *= self.hyp['cls']
        
#         if masks is not None and p_seg is not None:
#             assert len(p_seg) == 1
#             lpixl, larea, ldist = self.compute_loss_seg(p_seg[0], masks, targets, weight=m_weights)
        
#         loss = (lbox + lobj + lcls) * 1.0 + (lpixl + larea + ldist) * 0.2
#         loss_items = torch.cat((lbox, lobj, lcls, lpixl, larea, ldist, loss)).detach()
#         return loss * bs, loss_items

#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h), 0~1
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)

#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=targets.device).float() * g  # offsets

#         for i in range(self.nl):
#             anchors = self.anchors[i]
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter shape(nt_,7), [bi, ci, xc, yc, w, h, ai]

#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices

#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch
  
#     def build_patch_targets(self, patch_offsets, targets, imgsz):  # for fast-mode, fixed patch division
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         dtype, device = targets.dtype, targets.device
#         tcls, tbox, indices, anch = [], [], [], []
#         bs, _, height, width = imgsz
        
#         gain = torch.ones(7, device=device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)
#         bi_ = torch.arange(patch_offsets[0].shape[0], device=device)

#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=device).float() * g  # offsets

#         for i in range(self.nl):
#             patch_off = patch_offsets[i]
#             anchors = self.anchors[i]
#             r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
#             gain[2:6] = torch.tensor([width, height, width, height], dtype=dtype) / (8 * r)  # TODO: from 4 to 32
#             # grid_w, grid_h = patch_off[0, [3, 4]] - patch_off[0, [1, 2]]
#             grid_wh = patch_off[:1, [3, 4]] - patch_off[:1, [1, 2]]

#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter, shape(nt_, 7)

#                 tb, txc, tyc = t[:, [0, 2, 3]].chunk(3, dim=1)  # shape(n,1)
#                 pb, px1, py1, px2, py2 = (patch_off.T).chunk(5, dim=0)  # shape(1,m)
#                 contained = (tb == pb) & (txc > px1 - g) & (txc < px2 - g) & (tyc > py1 - g) & (tyc < py2 - g)  # shape(n,m)
#                 ti, pj = torch.nonzero(contained).T  # i-th target is contained within j-th patch
#                 t = t[ti]  # shape(n,7)
                
#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = grid_wh - gxy  # inverse
#                 j, k = ((gxy - gxy.floor() < g) & (gxy > 0.-g)).T
#                 l, m = ((gxi - gxi.floor() < g) & (gxi > 1.-g)).T
#                 # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
                
#                 t[:, 0] = bi_[pj]  # converted batch-indices
#                 t[:, 2:4] -= patch_off[pj, 1:3]  # converted xc, yc (minus px1, py1)

#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices

#             # Append
#             a = t[:, 6].long()  # anchor indices
#             # assert ((gj >= 0) & (gj <= grid_wh[0,1] - 1) & (gi >= 0) & (gi <= grid_wh[0,0] - 1)).all()
#             # indices.append((b, a, gj.clamp_(0, grid_wh[0,1] - 1), gi.clamp_(0, grid_wh[0,0] - 1)))  # image, anchor, grid indices
#             indices.append((b, a, gj, gi))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch

#     # def compute_loss_seg(self, p, masks, targets, weight=None):
#     #     dtype, device = targets.dtype, targets.device
#     #     bs, nc, ny, nx = masks.shape
#     #     assert nc == 1
#     #     lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), \
#     #                           torch.zeros(1, device=device)
        
#     #     # weight = None
#     #     lpixl += F.binary_cross_entropy_with_logits(p, masks, weight=weight)

#     #     nt = targets.shape[0]
#     #     if nt:  # number of targets
#     #         pass

#     #         # larea += self.dice_loss(p, masks)
#     #         # ldist += self.sigmoid_focal_loss(p, masks) * 20
            
#     #         # larea += self.quality_dice_loss(p, masks, weight=weight)
#     #         # ldist += self.sigmoid_quality_focal_loss(p, masks, weight=weight) * 20

    
#     #     return lpixl, larea, ldist
    
#     # EDL 기반 Segmentation Loss
#     def compute_loss_seg(self, p, masks, targets, weight=None):
#         """
#         p: [B,2,H,W]  (bg,obj evidence logits)
#         masks: [B,1,H,W] (0/1 binary GT mask)
#         """
#         device = targets.device
#         bs, nc_t, ny, nx = masks.shape
#         assert nc_t == 1, "GT mask는 1채널(0/1)이어야 함"
#         assert p.shape[1] == 2, f"EDL 사용 시 pred channels must be 2, got {p.shape}"

#         # evidence / alpha
#         evidence = F.softplus(p)          # [B,2,H,W]
#         alpha = evidence + 1.0
#         S = alpha.sum(dim=1, keepdim=True)  # [B,1,H,W]
#         probs = alpha / S                  # expected prob

#         y = masks
#         y2 = torch.cat([1.0 - y, y], dim=1)  # [B,2,H,W]

#         # expected CE
#         ce = -(y2 * (probs.clamp_min(1e-8)).log()).sum(dim=1, keepdim=True)  # [B,1,H,W]
#         lpixl = ce.mean()  # <-- 0-dim 이 됨

#         # evidence regularizer (과확신 억제 + 붕괴 방지)
#         e_bg = evidence[:, 0:1]
#         e_obj = evidence[:, 1:2]
#         reg = (y * e_bg + (1.0 - y) * e_obj).mean()
#         lpixl = lpixl + 0.01 * reg

#         # ===== 핵심: cat에 들어가도록 (1,) shape로 강제 =====
#         lpixl = lpixl.reshape(1)

#         larea = torch.zeros(1, device=device)
#         ldist = torch.zeros(1, device=device)
#         return lpixl, larea, ldist

#     @staticmethod
#     def dice_loss(inputs, targets):
#         """
#         Compute the DICE loss, similar to generalized IOU for masks
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#         """
#         inputs = inputs.sigmoid().flatten(1)
#         targets = targets.flatten(1)
#         numerator = 2 * (inputs * targets).sum(-1)
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = 1 - (numerator + 1) / (denominator + 1)
#         return loss.mean()

#     @staticmethod
#     def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
#         """
#         Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#             alpha: (optional) Weighting factor in range (0,1) to balance
#                     positive vs negative examples. Default = -1 (no weighting).
#             gamma: Exponent of the modulating factor (1 - p_t) to
#                 balance easy vs hard examples.
#         Returns:
#             Loss tensor
#         """
#         prob = inputs.sigmoid()
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         loss = ce_loss * ((1 - p_t) ** gamma)

#         if alpha >= 0:
#             alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#             loss = alpha_t * loss

#         return loss.mean()

#     @staticmethod
#     def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2):
#         """
#         Compute the DICE loss, similar to generalized IOU for masks
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#         """
#         inputs = inputs.sigmoid().flatten(1)
#         targets = targets.flatten(1)
#         if weight is not None:
#             weight = weight.flatten(1)
#             inputs = inputs * weight
#             targets = targets * weight

#         numerator = 2 * (inputs - targets).abs().sum(-1)
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = (numerator + 1) / (denominator + 1)
#         return loss.mean()

#     @staticmethod
#     def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
#         """
#         Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#             alpha: (optional) Weighting factor in range (0,1) to balance
#                     positive vs negative examples. Default = -1 (no weighting).
#             gamma: Exponent of the modulating factor (1 - p_t) to
#                 balance easy vs hard examples.
#         Returns:
#             Loss tensor
#         """
#         prob = inputs.sigmoid()
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
#         loss = ce_loss * ((prob - targets).abs() ** gamma)

#         if alpha >= 0:
#             alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#             loss = alpha_t * loss

#         return loss.mean()