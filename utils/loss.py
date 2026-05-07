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
    
                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    
                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
                obji = self.BCEobj(pi[..., 4].clamp_(-9.21, 9.21), tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # bs = tobj.shape[0]  # batch size
        bs = p_seg[0].shape[0] if p_seg is not None else tobj.shape[0]
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj'] * 0.5 #(0.5 if (len(offsets) and len(offsets[0]) > bs) else 1.)   # adaoff: 0.178
        lcls *= self.hyp['cls']
        
        if masks is not None and p_seg is not None:
            assert len(p_seg) == 1
            lpixl, larea, ldist = self.compute_loss_seg(p_seg[0], masks, targets, weight=m_weights)
        
        loss = (lbox + lobj + lcls) * 1.0 + (lpixl + larea + ldist) * 0.2
        loss_items = torch.cat((lbox, lobj, lcls, lpixl, larea, ldist, loss)).detach()
        return loss * bs, loss_items

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h), 0~1
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)

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
    
    # EDL 기반 Segmentation Loss (vacuity 기반 탐지를 위한 Dirichlet loss)
    def compute_loss_seg(self, p, masks, targets, weight=None):
        """
        p: [B,2,H,W]  (bg,obj evidence logits)
        masks: [B,1,H,W] (0/1 binary GT mask)

        학습 목표: 배경은 evidence를 높여 vacuity↓, 객체도 evidence를 높여 vacuity↓
        → 학습이 부족한 영역(작은 객체 등)은 자연스럽게 vacuity↑ → 추론 시 탐지 후보
        """
        device = targets.device
        bs, nc_t, ny, nx = masks.shape
        assert nc_t == 1, "GT mask는 1채널(0/1)이어야 함"
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
        #    → 근거 없이 아는 척(overconfident) 방지 → vacuity가 의미있게 학습됨
        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        kl = self._kl_divergence_dirichlet(alpha_tilde)  # [B,H,W]

        annealing_coef = min(1.0, getattr(ComputeLoss, '_edl_epoch', 1) / 10.0)
        edl_loss = edl_mse + annealing_coef * kl

        if weight is not None:
            edl_loss = edl_loss * weight.squeeze(1)

        lpixl = edl_loss.mean().reshape(1)
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