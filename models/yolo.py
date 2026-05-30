# """YOLOv5-specific modules

# Usage:
#     $ python path/to/models/yolo.py --cfg yolov5s.yaml
# """
# # Copyright (c) Alibaba, Inc. and its affiliates.

# ############ EDL ############
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import warnings

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.common import *
from models.replknet import *
try:
    from models.gpvit import *
except:
    warnings.warn('Package mmdet is not installed. You can follow https://github.com/ChenhongyiYang/GPViT to install dependencies.')
    SpatialPriorModule = GPViTAdapterSingleStageESOD = None
from models.spconv import SPYOLOv5Head, SPYOLOv6Head
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging, xyxy2xywh
from utils import edl_det
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = get_decoupled_heads(ch, self.nc, self.na)  # decoupled head
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

        self.sparse = False
        self.register_buffer('sparse_gird', torch.zeros(1))

        # ECPR Stage 1: cross-patch graph refinement (det_graph.ENABLE=False → 미생성 = baseline)
        from models import det_graph as _dg
        self.graph = _dg.DetGraphRefine(self.nc) if _dg.ENABLE else None
        self.graph_out = None      # 학습 시 refine된 per-image 검출 (ComputeLoss 가 읽음)
        if self.graph is not None:
            self.inplace = False   # 학습-시 decode를 autograd-safe하게 (in-place 회피)

    def set_sparse(self):
        sp_dict = {nn.Conv2d: SPYOLOv5Head, YOLOv6Head: SPYOLOv6Head}
        sp_head = sp_dict[type(self.m[0])]
        self.m = nn.ModuleList(sp_head(m) for m in self.m)
        self.sparse = True
    
    @torch.no_grad()
    def get_indices(self, offsets, mask, thresh=0.3):
        # mask: tensor (single) or list [heat,edl] / [heat,edl,vacuity_cal] (dual)
        if (isinstance(mask, (list, tuple)) and len(mask) == 3
                and mask[0].shape[1] == 1 and mask[1].shape[1] == 2 and mask[2].shape[1] == 1):
            # Dual fusion (calib-gating or 3-C noisy-OR per ESOD_ROLE_DUAL env)
            mask = edl_det.fuse_dual_seg(mask[0].detach(), mask[1].detach(), mask[2].detach())
        elif isinstance(mask, (list, tuple)) and len(mask) == 2:
            # Dual + max
            _heat, _edl = mask[0].detach(), mask[1].detach()
            prob = _heat.sigmoid()
            _ev = F.softplus(_edl); _alpha = _ev + 1.0
            _S = _alpha.sum(dim=1, keepdim=True)
            vac = 2.0 / _S
            mask = torch.max(prob, vac)
        elif mask.dim() == 4 and mask.shape[1] == 2:
            # Single 2ch EDL: u-aware keep
            #   B-1 (sum, default): mask = p_obj + gamma * u
            #   B-2 (dual OR)     : keep = (p_obj > tau_obj) | (u > tau_u); mask = (p_obj + gamma*u) * keep
            # ESOD_VAC_KEEP_MODE: 'sum' (default) or 'dual'
            evidence = F.softplus(mask.detach())
            alpha = evidence + 1.0
            S = alpha.sum(dim=1, keepdim=True)
            p_obj = alpha[:, 1:2] / S                           # belief object (ch1)
            u = 2.0 / S                                         # vacuity
            import os as _os_uaw
            _gamma = float(_os_uaw.environ.get('ESOD_VAC_GAMMA', '0.5'))
            _keep_mode = _os_uaw.environ.get('ESOD_VAC_KEEP_MODE', 'sum').strip().lower()
            if _keep_mode == 'dual':
                # B-2: OR-rule. Two independent score paths, then max → NMS-friendly.
                # u path is scaled so that u=tau_u → 0.5 (safely above default thresh=0.3).
                _tau_obj = float(_os_uaw.environ.get('ESOD_VAC_TAU_OBJ', '0.1'))
                _tau_u   = float(_os_uaw.environ.get('ESOD_VAC_TAU_U',   '0.13'))
                keep_obj = p_obj > _tau_obj
                keep_u   = u > _tau_u
                score_p = torch.where(keep_obj, p_obj, torch.zeros_like(p_obj))
                _u_scale = 0.5 / max(_tau_u, 1e-6)
                score_u = torch.where(keep_u, u * _u_scale, torch.zeros_like(u))
                mask = torch.maximum(score_p, score_u)
            else:
                mask = p_obj + _gamma * u                       # u-aware patch score (B-1)
        elif torch.max(mask) > 1. or torch.min(mask) < 0.:
            mask = mask.detach().sigmoid()
        device, dtype = mask.device, mask.dtype

        patch_w, patch_h = offsets[0, 3:5] - offsets[0, 1:3]
        if not hasattr(self, 'sparse_gird') or self.sparse_gird is None or self.sparse_gird[0].shape != (1,patch_h,patch_w):
            yv, xv = torch.meshgrid([torch.arange(patch_h), torch.arange(patch_w)])
            yv, xv = yv.to(device), xv.to(device)
            self.sparse_gird = torch.stack((torch.zeros_like(yv), yv, xv)).view(3,1,patch_h,patch_w)  # shape(1,ph,pw)
        gb, gy, gx = self.sparse_gird
        ob1, ox1, oy1 = offsets[:, :3].unsqueeze(-1).chunk(3, dim=1)  # shape(n,1,1)
        ob, ox, oy = (ob1 + gb).view(-1), (ox1 + gx).view(-1), (oy1 + gy).view(-1)
        
        maxima = F.max_pool2d(mask, 3, stride=1, padding=1) == mask
        response = mask >= thresh
        indices = (maxima & response).to(dtype)
        indices = F.max_pool2d(indices, 3, stride=1, padding=1)  # expansion  

        slices = indices[ob, 0, oy, ox].view(offsets.shape[0], 1, patch_h, patch_w)

        indices_per_layer = []
        for i in range(self.nl):
            s = 2 ** i

            if i != 0:
                slices_i = F.max_pool2d(slices, s, stride=s, padding=0)
                slices_i = F.max_pool2d(slices_i, 3, stride=1, padding=1)  # expansion
            else:
                slices_i = slices

            indices_per_layer.append(torch.nonzero(slices_i[:, 0, :, :]))

        ###################
        
        # indices_per_layer = []
        # for i in range(self.nl):
        #     s = 2 ** i
        #     if s > 1:
        #         # TODO: size-adaptive?
        #         mask_i = F.avg_pool2d(mask, s, stride=s, padding=0)
        #         # mask_i = F.max_pool2d(mask, s, stride=s, padding=0)
        #     else:
        #         mask_i = mask
            
        #     maxima = F.max_pool2d(mask_i, 3, stride=1, padding=1) == mask_i
        #     response = mask_i > thresh
        #     indices = (maxima & response).float()
        #     indices = F.max_pool2d(indices, 3, stride=1, padding=1)  # expansion for 3x3 conv

        #     sw, sh = patch_w // s, patch_h // s
        #     ob, ox, oy = (ob1 + gb[:,:sh,:sw]).view(-1), (ox1//s + gx[:,:sh,:sw]).view(-1), (oy1//s + gy[:,:sh,:sw]).view(-1)
        #     slices = indices[ob, 0, oy, ox].view(offsets.shape[0], sh, sw)

        #     indices_per_layer.append(torch.nonzero(slices))

        return indices_per_layer

    def forward(self, x):
        # x = x.copy()  # for profiling
        masks, offsets, indices_per_layer = None, None, None
        self.graph_out = None  # deepcopy 안전: 학습 forward 끝~loss 사이에만 채워짐
        if isinstance(x, tuple):
            if len(x) == 2:
                x, offsets = x  # offsets(bi,x1,y1,x2,y2)
            else:
                x, offsets, masks = x
                # masks: [t] (Segmenter), [heat,edl] (Dual), [heat,edl,gate] (Dual+MoE)
                assert len(masks) in (1, 2, 3) and not isinstance(masks, torch.Tensor)
                if offsets is not None and hasattr(self, 'sparse') and self.sparse:
                    mask_arg = masks if len(masks) >= 2 else masks[0]
                    indices_per_layer = self.get_indices(offsets, mask_arg)
            if offsets is not None:
                img_bs = torch.max(offsets[:, 0]).int().item() + 1
            else:
                img_bs = x[0].shape[0]
        else:
            img_bs = x[0].shape[0]
        
        device = x[0].device
        z = []  # inference output
        patch_offsets = []
        for i in range(self.nl):
            # if len(x) > self.nl:
            #     hid_feat_i = F.max_pool2d(x[self.nl * 2 - 1 - i], kernel_size=8, stride=8, padding=0)  # 这里有一个倒序关系
            #     x[i] = torch.cat((x[i], hid_feat_i), dim=1)
            
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            if offsets is not None:
                r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
                patch_off = torch.cat((offsets[:, :1], offsets[:, 1:] / r), dim=1)  # TODO: from 4 to 32
                patch_off_xy = patch_off[:, 1:3].view(-1, 1, 1, 1, 2)
                patch_offsets.append(patch_off)
            
            if indices_per_layer is not None:
                sp_x = self.m[i](x[i], indices_per_layer[i])  # sparse conv
                # x[i] = sp_x.dense(channels_first=True) deprecated # for training
            else:
                sp_x = None
                x[i] = self.m[i](x[i])  # conv
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if (not self.training) or (self.graph is not None):  # inference / ECPR decode
                if self.grid[i].shape[2:4] != (ny, nx) or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(device)

                # EOD-style evidential cls activation (env ESOD_EVIDENTIAL_CLS=1)
                import os as _os_ev
                _ev_cls_on = _os_ev.environ.get('ESOD_EVIDENTIAL_CLS', '').strip() == '1'

                if sp_x is not None:
                    y = sp_x.features.sigmoid().view(-1, self.na, self.no)
                    if edl_det.EDL_DET:
                        y[..., 4] = edl_det.redl_p_obj(
                            sp_x.features.view(-1, self.na, self.no)[..., 4])
                    if _ev_cls_on and self.nc > 1:
                        _cls_logit = sp_x.features.view(-1, self.na, self.no)[..., 5:].clamp(-9.21, 9.21)
                        _belief, _, _ = edl_det.redl_Kch(_cls_logit)
                        y[..., 5:] = _belief.to(y.dtype)
                    bi, yi, xi = sp_x.indices.long().T
                    assert offsets is not None
                    grid_off = self.grid[i][0, 0, yi, xi].view(-1, 1, 2) + patch_off_xy[bi, ...].view(-1, 1, 2)
                    anch_wh = self.anchor_grid[i].view(1, self.na, 2)
                    batch_ind = offsets[bi, 0]  # [num_patches, 5] --> [num_objects, 5], compatible for box concat
                else:
                    y = x[i].sigmoid()
                    if edl_det.EDL_DET:
                        y[..., 4] = edl_det.redl_p_obj(x[i][..., 4])
                    if _ev_cls_on and self.nc > 1:
                        _cls_logit = x[i][..., 5:].clamp(-9.21, 9.21)
                        _belief, _, _ = edl_det.redl_Kch(_cls_logit)
                        # non-inplace 안전 (autograd graph 보존)
                        y = torch.cat([y[..., :5], _belief.to(y.dtype)], dim=-1)
                    anch_wh = self.anchor_grid[i].view(1, self.na, 1, 1, 2)
                    if offsets is not None:
                        grid_off = self.grid[i] + patch_off_xy
                        batch_ind = offsets[:, 0]
                    else:
                        grid_off = self.grid[i]
                        batch_ind = None

                if self.inplace and self.graph is None:  # ECPR: graph on → autograd-safe 비in-place 경로
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_off) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anch_wh  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + grid_off) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * anch_wh  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # y[..., 4] = 1.0
                
                if offsets is not None:
                    pbox = []
                    for bi in range(img_bs):
                        pbox_bi = y[batch_ind == bi]
                        np = len(pbox_bi)
                        if np:
                            pbox.append(pbox_bi.view(-1, self.no))
                        else:
                            pbox.append(torch.zeros((0, self.no), device=device))
                    max_pnum = max([len(boxes) for boxes in pbox])
                    z.append(torch.stack(
                        [torch.cat((boxes, torch.zeros((max_pnum - len(boxes), self.no), device=device))) for boxes in pbox]
                    ))
                else:
                    z.append(y.view(bs, -1, self.no))
            
        if offsets is not None:
            x = (x, patch_offsets)

        # Phase 2: Detect parallel evidence head → per-image vacuity map (inference 전용)
        # layer 0 (P3, stride 8) 의 anchor-min vacuity 를 patch offsets 로 image-space 에 paste.
        # OOD eval 에서 detection box footprint sample 용.
        if (not self.training) and offsets is not None and len(self.m) > 0 and \
                getattr(self.m[0], 'last_ev', None) is not None:
            stride0 = int(self.stride[0])
            ev0 = self.m[0].last_ev.detach()              # [Bp, na, 2, ny, nx]
            e_bg = F.softplus(ev0[:, :, 0, :, :])
            e_obj = F.softplus(ev0[:, :, 1, :, :])
            vac_cells = 2.0 / (e_bg + e_obj + 2.0)        # [Bp, na, ny, nx]
            vac_p = vac_cells.min(dim=1).values           # [Bp, ny, nx] anchor-min
            # image size 추정: offsets [Bp, 5] (bi, x1, y1, x2, y2) — input pixel space
            img_w = int(offsets[:, 3].max().item())
            img_h = int(offsets[:, 4].max().item())
            Hf = max(1, img_h // stride0); Wf = max(1, img_w // stride0)
            img_vac = torch.ones((img_bs, Hf, Wf), device=vac_p.device, dtype=vac_p.dtype)
            for pidx in range(vac_p.shape[0]):
                bi = int(offsets[pidx, 0])
                x1 = int(offsets[pidx, 1]) // stride0
                y1 = int(offsets[pidx, 2]) // stride0
                _, ny_p, nx_p = vac_p[pidx].shape if vac_p[pidx].dim() == 3 else (None,) + tuple(vac_p[pidx].shape)
                ny_p, nx_p = vac_p[pidx].shape
                x2 = min(x1 + nx_p, Wf); y2 = min(y1 + ny_p, Hf)
                if x2 > x1 and y2 > y1:
                    img_vac[bi, y1:y2, x1:x2] = torch.minimum(
                        img_vac[bi, y1:y2, x1:x2], vac_p[pidx, :y2 - y1, :x2 - x1])
            self.last_det_vac_image = img_vac             # [B_img, Hf, Wf]

        # EOD-style cls vacuity image-paste (env ESOD_EVIDENTIAL_CLS=1) — OWOD unknown ID 용
        # layer 0 cls logit → softplus → α=e+λ → S → vacuity = K·λ/S, anchor-min, image paste
        import os as _os_cv
        if (not self.training) and offsets is not None and len(self.m) > 0 and \
                _os_cv.environ.get('ESOD_EVIDENTIAL_CLS', '').strip() == '1':
            try:
                _x0 = x[0] if isinstance(x, list) else x[0][0]  # x already maybe (x,patch_offsets)
            except Exception:
                _x0 = None
            if _x0 is not None and _x0.dim() == 5 and _x0.shape[-1] >= 5 + self.nc:
                stride0 = int(self.stride[0])
                _cls0 = _x0[..., 5:5 + self.nc].clamp(-9.21, 9.21).detach()  # [Bp, na, ny, nx, K]
                _K = float(self.nc)
                _S = (F.softplus(_cls0) + edl_det.LAM).sum(-1)           # [Bp, na, ny, nx]
                _cls_vac_cells = (_K * edl_det.LAM) / _S                   # [Bp, na, ny, nx]
                _cls_vac_p = _cls_vac_cells.min(dim=1).values              # [Bp, ny, nx] anchor-min
                img_w = int(offsets[:, 3].max().item())
                img_h = int(offsets[:, 4].max().item())
                Hf = max(1, img_h // stride0); Wf = max(1, img_w // stride0)
                _cls_vac_img = torch.ones((img_bs, Hf, Wf),
                                          device=_cls_vac_p.device, dtype=_cls_vac_p.dtype)
                for pidx in range(_cls_vac_p.shape[0]):
                    bi = int(offsets[pidx, 0])
                    x1 = int(offsets[pidx, 1]) // stride0
                    y1 = int(offsets[pidx, 2]) // stride0
                    ny_p, nx_p = _cls_vac_p[pidx].shape
                    x2 = min(x1 + nx_p, Wf); y2 = min(y1 + ny_p, Hf)
                    if x2 > x1 and y2 > y1:
                        _cls_vac_img[bi, y1:y2, x1:x2] = torch.minimum(
                            _cls_vac_img[bi, y1:y2, x1:x2],
                            _cls_vac_p[pidx, :y2 - y1, :x2 - x1])
                self.last_cls_vac_image = _cls_vac_img                    # [B_img, Hf, Wf]

        # ECPR: cross-patch graph refinement on per-image decoded detections
        if self.graph is not None and len(z):
            refined, subset = self._apply_graph(torch.cat(z, 1))
        else:
            refined, subset = None, None
        if self.training:
            self.graph_out = subset
            return x
        return ((refined if refined is not None else torch.cat(z, 1)), x)

    def _apply_graph(self, cat_z, topk=4096):
        # cat_z [B,N,no] decoded dets → graph-refined. 반환 (full[B,N,no], subset[list])
        eps = 1e-6
        gdt = next(self.graph.parameters()).dtype              # graph weight dtype (half/float)
        full, subset = [], []
        for i in range(cat_z.shape[0]):
            p = cat_z[i]
            idx = (p[:, 4] > 0).nonzero(as_tuple=True)[0]      # 유효 검출 (padding 제외)
            if idx.numel() < 2:
                full.append(p); subset.append(p[:0]); continue
            if idx.numel() > topk:                             # objectness top-K (O(N^2) knn 방지)
                idx = idx[p[idx, 4].topk(topk).indices]
            v = p[idx].to(gdt)                                 # graph dtype 로 맞춤 (half/float 혼용 방지)
            o = v[:, 4].clamp(eps, 1 - eps)
            c = v[:, 5:].clamp(eps, 1 - eps)
            r_obj, r_cls, r_box = self.graph(
                v[:, :4], torch.log(o / (1 - o)), torch.log(c / (1 - c)))
            ref = torch.cat([r_box, r_obj.sigmoid().unsqueeze(-1),
                             r_cls.sigmoid()], dim=-1).to(p.dtype)   # 원래 dtype 복귀
            oi = p.clone()
            oi[idx] = ref
            full.append(oi); subset.append(ref)
        return torch.stack(full, 0), subset

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Segmenter(nn.Module):
    def __init__(self, nc=10, ch=()):
        super(Segmenter, self).__init__()
        self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)  # output conv

    def forward(self, x):
        return [self.m[i](x[i]) for i in range(len(x))]


class DualSegmenter(nn.Module):
    """
    Dual-branch segmentation head with Learnable Spatial Gating:
      - heat branch:  1ch logit (sigmoid → prob, ESOD baseline)
      - edl branch:   2ch logit (softplus → evidence → vacuity)
      - gate branch:  1ch logit (sigmoid → spatial gate α)

    Returns [heat, edl, gate]  (all logits; shapes 1ch / 2ch / 1ch)

    Downstream (HeatMapParser/Loss):
      prob = sigmoid(heat)
      evidence = softplus(edl) ; alpha = evidence + 1 ; S = Σα
      p_e = alpha_obj / S      ; u (vacuity) = 2/S
      gate = sigmoid(gate)
      w = gate · (1 - u)
      F = (1 - w) · prob + w · p_e     (calibrated gating fusion)
    """
    def __init__(self, ch=()):
        super(DualSegmenter, self).__init__()
        self.heat = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in ch)
        self.edl  = nn.ModuleList(nn.Conv2d(c, 2, 1) for c in ch)
        self.gate = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in ch)
        # gate bias 0 → sigmoid(0)=0.5 (시작은 H/V 균형, collapse 방지 초기화)
        for g in self.gate:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)

    def forward(self, x):
        # x: list of feature tensors (typically 1 element)
        heat = [self.heat[i](x[i]) for i in range(len(x))]
        edl  = [self.edl[i](x[i])  for i in range(len(x))]
        gate = [self.gate[i](x[i]) for i in range(len(x))]
        return [heat[0], edl[0], gate[0]]


class Center(nn.Module):
    def __init__(self, nc=80, ch=()):  # detection layer
        super(Center, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 4  # number of outputs
        self.nl = 1
        self.na = 0
        self.anchors = None
        self.anchor_grid = None
        self.grid = torch.zeros(1)
        self.b = torch.zeros(0)
        self.c = torch.zeros(0)
        self.stride = torch.tensor([4, 32])  # fake

        assert len(ch) == 1
        ch = ch[0]
        self.m = nn.ModuleList([
            nn.Sequential(  # hm
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, nc, kernel_size=1, stride=1, padding=0, bias=True),
            ),
            nn.Sequential(  # wh
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # reg
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            ),
        ])  # output convs
        self.m[0][-1].bias.data.fill_(-2.19)  # hm, expect 0.01
    
    def forward(self, x):
        assert isinstance(x, tuple)
        x, offsets = x  # offsets(bi,x1,y1,x2,y2)
        assert len(x) == 1
        x = x[0]
        hm, wh, reg = [self.m[i](x) for i in range(3)]
        
        # offsets = torch.cat((offsets[:, :1], offsets[:, 1:] * 2.), dim=1)
            
        if not self.training:  # inference
            nb, nc, ny, nx = hm.shape
            device = hm.device
            
            wh_ = wh.permute(0, 2, 3, 1).contiguous()
            reg_ = reg.permute(0, 2, 3, 1).contiguous()
            if self.grid.shape[1:2] != wh_.shape[1:2]:
                self.grid = self._make_grid(nx, ny).to(x.device)
            # [nb, ny, nx, 4], absolute pixel relative to input size
            if offsets is not None:
                offsets_xy = offsets[:, 1:3].view(-1, 1, 1, 2) * 2
                bbox = torch.cat([self.grid + offsets_xy + reg_ - wh_ / 2.,
                                  self.grid + offsets_xy + reg_ + wh_ / 2.], dim=-1)
            else:
                bbox = torch.cat([self.grid + reg_ - wh_ / 2., self.grid + reg_ + wh_ / 2.], dim=-1)
            # no clamp
            # bbox[..., [0, 2]].clamp_(0, nx)
            # bbox[..., [1, 3]].clamp_(0, ny)
            # [nb, nc, ny, nx, 4]
            # print(self.grid.shape, wh_.shape, hm.shape, offsets_xy.shape, bbox.shape)
            bbox = bbox.view(nb, 1, ny, nx, 4).repeat((1, nc, 1, 1, 1)) * self.stride[0]
            
            if self.c.shape[:-1] != hm.shape:
                # [nb, nc, ny, nx, 1]
                self.c = torch.arange(nc).to(x.device).view(1, nc, 1, 1, 1).repeat((nb, 1, ny, nx, 1))
            self.b = offsets[:, :1].view(nb, 1, 1, 1, 1).repeat((1, nc, ny, nx, 1))
            
            hm_ = hm.sigmoid()
            hmax = F.max_pool2d(hm_, 3, stride=1, padding=1)
            maxima = hmax == hm_
            # hmax_cls = torch.argmax(hm_, dim=1, keepdim=True)
            # maxima_class = hmax_cls == self.c[..., 0]
            # maxima &= maxima_class
            
            # [n, 6]
            preds = torch.cat([bbox[maxima], hm_[maxima].view(-1, 1), self.c[maxima]], dim=1)
            bi = self.b[maxima][:, 0]

            pbox = []
            for i in range(torch.max(bi).int().item() + 1):
                pbox_bi = preds[bi == i]
                # topk_indices = torch.argsort(pbox_bi[:, 4], descending=True)[:500]
                if len(pbox_bi):
                    pbox.append(
                        torch.cat((xyxy2xywh(pbox_bi[:, :4]), pbox_bi[:, 4:5],
                                   F.one_hot(pbox_bi[:, 5].long(), self.nc)), dim=1)
                    )
                else:
                    pbox.append(torch.zeros((0, 5 + self.nc), device=device))
            max_pnum = max([len(boxes) for boxes in pbox])
            predictions = torch.stack(
                [torch.cat((boxes, torch.zeros((max_pnum - len(boxes), 5 + self.nc), device=device))) for boxes in pbox]
            )
        else:
            predictions = None

        x = ((hm, wh, reg), [offsets])  # for consistency when testing
            
        return x if self.training else (predictions, x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()
    

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # TODO
            # m.stride = torch.tensor([ 4., 8., 16., 32.])
            m.stride = torch.tensor([ 8., 16., 32.])
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())
        elif isinstance(m, Center):
            # m.stride = torch.tensor(m.stride)  # no forward
            self.stride = m.stride
        # elif isinstance(m, Detect2):  deprecated
        #     s = 256  # 2x min stride
        #     m.inplace = self.inplace
        #     # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     self.forward(torch.zeros(1, ch, s, s))  # forward
        #     m.stride = torch.tensor(m.stride)  # no forward
        #     # m.anchors /= m.stride.view(-1, 1, 1) 这里不归一化
        #     self.stride = m.stride

        # Init weights, biases
        initialize_weights(self)
        try:
            self.info()
        except:
            logger.info('Failed to capture the model info')
        logger.info('')

    def forward(self, x, augment=False, profile=False, hm_only=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        else:
            return self.forward_once(x, profile, hm_only=hm_only)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, hm_only=False):
        y, dt = [], []  # outputs
        
        masks, pred_masks, offsets = None, None, None
        heatmap = None
        if isinstance(x, tuple):
            x, masks = x  # ground-truth masks
        x0 = x
        B, C, H, W = x.shape
        for mi, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else (x0 if j == -100 else y[j]) for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if isinstance(m, HeatMapParser):
                # HeatMapParser는 항상 Segmenter의 예측 heatmap을 받아야 함
                assert pred_masks is not None, "Segmenter output(pred_masks) must exist before HeatMapParser"
                x = (x[0], pred_masks)
            elif type(m) in [Detect, Center] and offsets is not None:
                x = (x, offsets)
                x = (*x, masks if masks is not None else pred_masks)
            elif isinstance(m, MaskedC3TR):
                x = (x, heatmap)
            elif isinstance(m, Token2Image):
                x = [x, (H, W)]
            
            x = m(x)  # run

            if isinstance(m, (Segmenter, DualSegmenter)):
                pred_masks = x  # Segmenter: [tensor], DualSegmenter: [heat, edl]
                if hm_only:
                    return (None, None), pred_masks
                if masks is None:
                    masks = pred_masks
            elif isinstance(m, HeatMapParser):
                if isinstance(x, torch.Tensor):
                    offsets = x
                    if offsets.size(0) == 0:
                        return (None, None), pred_masks
                elif isinstance(x[1], torch.Tensor):
                    x, offsets = x
                    if len(x) == 0:
                        return (None, None), pred_masks
                else:
                    x, thresh = x
                    # Dual fusion (calib-gating or 3-C noisy-OR per ESOD_ROLE_DUAL env)
                    if (len(pred_masks) == 3 and pred_masks[0].shape[1] == 1
                            and pred_masks[1].shape[1] == 2 and pred_masks[2].shape[1] == 1):
                        from utils import edl_det as _edl_det_mod
                        heatmap = _edl_det_mod.fuse_dual_seg(
                            pred_masks[0].detach(), pred_masks[1].detach(), pred_masks[2].detach())
                    elif (len(pred_masks) == 2 and pred_masks[0].shape[1] == 1
                          and pred_masks[1].shape[1] == 2):
                        # Dual + max (legacy)
                        _heat = pred_masks[0].detach()
                        _edl  = pred_masks[1].detach()
                        prob  = _heat.sigmoid()
                        _ev   = F.softplus(_edl); _alpha = _ev + 1.0
                        _S    = _alpha.sum(dim=1, keepdim=True)
                        vac   = 2.0 / _S
                        heatmap = torch.max(prob, vac)
                    else:
                        _pm = pred_masks[0].detach()
                        if _pm.shape[1] == 2:
                            _ev = F.softplus(_pm)
                            _alpha = _ev + 1.0
                            _S = _alpha.sum(dim=1, keepdim=True)
                            heatmap = 2.0 / _S
                        else:
                            heatmap = _pm.sigmoid()
                    heatmap = heatmap > thresh
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
            
        return x, pred_masks

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        try:
            for mi, s in zip(m.m, m.stride):  # from YOLOv5 head
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        except AttributeError:
            for mi, s in zip(m.m, m.stride):  # from decoupled head
                mi.obj_pred.bias.data.fill_(math.log(8 / (640 / s) ** 2))  # obj (8 objects per 640 image)
                mi.cls_pred.bias.data.fill_(math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum()))

        for m_ in self.model:
            if str(m_.type) == 'models.yolo.Segmenter':  # stupid
                for mi in m_.m:
                    b = mi.bias.view(-1)
                    b.data += math.log(0.6 / (m.nc - 0.99) if cf is None else torch.log(cf / cf.sum()))  # cls
                    mi.bias = torch.nn.Parameter(b, requires_grad=True)
                break

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        try:
            self.info()
        except:
            print('Failed to capture the model info')
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, ASPP, SPPF,
                 DWConv, DCN, RepLKConv, MixConv2d, Focus, Blur, CrossConv,
                 BottleneckCSP, C3, C3TR, MaskedC3TR, C2f, ResBlockLayer, RTMDetCSPLayer,
                 HeatMapParser]:
            c2 = args[0]
            if c2 != no and 'GPViTAdapterSingleStageESOD' not in [_x[2] for _x in d['backbone']]:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            if m is HeatMapParser:
                args = [c2, *args[1:]]
            else:
                args = [ch[f], c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR, MaskedC3TR, C2f, RTMDetCSPLayer]:
                    args.insert(2, n)  # number of repeats
                    n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Add, nn.Identity]:
            pass
        elif m in [Detect, Segmenter, DualSegmenter]:  # Detect2 deprecated
            args.append([ch[x] for x in f])
            if len(args) > 1 and isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Center:
            args.append([ch[x] for x in f])
        elif m is SpatialPriorModule:
            c2 = args[0] * 2
        elif m is GPViTAdapterSingleStageESOD:
            pass
        elif m is Indexer:
            c2 = args[0]
            args = args[1:]
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x not in [-1, -100])  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard

### graph ###
# import argparse
# import logging
# import sys
# from copy import deepcopy
# from pathlib import Path
# import warnings

# sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
# logger = logging.getLogger(__name__)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from models.common import *
# from models.replknet import *
# try:
#     from models.gpvit import *
# except:
#     warnings.warn('Package mmdet is not installed. You can follow https://github.com/ChenhongyiYang/GPViT to install dependencies.')
#     SpatialPriorModule = GPViTAdapterSingleStageESOD = None
# from models.spconv import SPYOLOv5Head, SPYOLOv6Head
# from models.experimental import *
# from utils.autoanchor import check_anchor_order
# from utils.general import make_divisible, check_file, set_logging, xyxy2xywh
# from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
#     select_device, copy_attr

# try:
#     import thop  # for FLOPs computation
# except ImportError:
#     thop = None

# class Detect(nn.Module):
#     stride = None  # strides computed during build
#     onnx_dynamic = False  # ONNX export parameter

#     def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
#         super(Detect, self).__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 5  # number of outputs per anchor
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.grid = [torch.zeros(1)] * self.nl  # init grid
#         a = torch.tensor(anchors).float().view(self.nl, -1, 2)
#         self.register_buffer('anchors', a)  # shape(nl,na,2)
#         self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
#         # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
#         self.m = get_decoupled_heads(ch, self.nc, self.na)  # decoupled head
#         self.inplace = inplace  # use in-place ops (e.g. slice assignment)

#         # Optional: class-graph refiner (set by Model.enable_class_graph)
#         self.class_graph_refiner = None

#         self.sparse = False
#         self.register_buffer('sparse_gird', torch.zeros(1))

#     def set_sparse(self):
#         sp_dict = {nn.Conv2d: SPYOLOv5Head, YOLOv6Head: SPYOLOv6Head}
#         sp_head = sp_dict[type(self.m[0])]
#         self.m = nn.ModuleList(sp_head(m) for m in self.m)
#         self.sparse = True
    
#     @torch.no_grad()
#     def get_indices(self, offsets, mask, thresh=0.3):
#         device, dtype = mask.device, mask.dtype
#         if torch.max(mask) > 1. or torch.min(mask) < 0.:
#             mask = mask.detach().sigmoid()

#         patch_w, patch_h = offsets[0, 3:5] - offsets[0, 1:3]
#         if not hasattr(self, 'sparse_gird') or self.sparse_gird is None or self.sparse_gird[0].shape != (1,patch_h,patch_w):
#             yv, xv = torch.meshgrid([torch.arange(patch_h), torch.arange(patch_w)])
#             yv, xv = yv.to(device), xv.to(device)
#             self.sparse_gird = torch.stack((torch.zeros_like(yv), yv, xv)).view(3,1,patch_h,patch_w)  # shape(1,ph,pw)
#         gb, gy, gx = self.sparse_gird
#         ob1, ox1, oy1 = offsets[:, :3].unsqueeze(-1).chunk(3, dim=1)  # shape(n,1,1)
#         ob, ox, oy = (ob1 + gb).view(-1), (ox1 + gx).view(-1), (oy1 + gy).view(-1)
        
#         maxima = F.max_pool2d(mask, 3, stride=1, padding=1) == mask
#         response = mask >= thresh
#         indices = (maxima & response).to(dtype)
#         indices = F.max_pool2d(indices, 3, stride=1, padding=1)  # expansion  

#         slices = indices[ob, 0, oy, ox].view(offsets.shape[0], 1, patch_h, patch_w)

#         indices_per_layer = []
#         for i in range(self.nl):
#             s = 2 ** i

#             if i != 0:
#                 slices_i = F.max_pool2d(slices, s, stride=s, padding=0)
#                 slices_i = F.max_pool2d(slices_i, 3, stride=1, padding=1)  # expansion
#             else:
#                 slices_i = slices

#             indices_per_layer.append(torch.nonzero(slices_i[:, 0, :, :]))

#         ###################
        
#         # indices_per_layer = []
#         # for i in range(self.nl):
#         #     s = 2 ** i
#         #     if s > 1:
#         #         # TODO: size-adaptive?
#         #         mask_i = F.avg_pool2d(mask, s, stride=s, padding=0)
#         #         # mask_i = F.max_pool2d(mask, s, stride=s, padding=0)
#         #     else:
#         #         mask_i = mask
            
#         #     maxima = F.max_pool2d(mask_i, 3, stride=1, padding=1) == mask_i
#         #     response = mask_i > thresh
#         #     indices = (maxima & response).float()
#         #     indices = F.max_pool2d(indices, 3, stride=1, padding=1)  # expansion for 3x3 conv

#         #     sw, sh = patch_w // s, patch_h // s
#         #     ob, ox, oy = (ob1 + gb[:,:sh,:sw]).view(-1), (ox1//s + gx[:,:sh,:sw]).view(-1), (oy1//s + gy[:,:sh,:sw]).view(-1)
#         #     slices = indices[ob, 0, oy, ox].view(offsets.shape[0], sh, sw)

#         #     indices_per_layer.append(torch.nonzero(slices))

#         return indices_per_layer

#     def forward(self, x):
#         # x = x.copy()  # for profiling
#         masks, offsets, indices_per_layer = None, None, None
#         if isinstance(x, tuple):
#             if len(x) == 2:
#                 x, offsets = x  # offsets(bi,x1,y1,x2,y2)
#             else:
#                 x, offsets, masks = x
#                 assert len(masks) == 1 and not isinstance(masks, torch.Tensor)
#                 if offsets is not None and hasattr(self, 'sparse') and self.sparse:
#                     indices_per_layer = self.get_indices(offsets, masks[0])
#             if offsets is not None:
#                 img_bs = torch.max(offsets[:, 0]).int().item() + 1
#             else:
#                 img_bs = x[0].shape[0]
#         else:
#             img_bs = x[0].shape[0]
        
#         device = x[0].device
#         z = []  # inference output
#         patch_offsets = []
#         for i in range(self.nl):
#             # if len(x) > self.nl:
#             #     hid_feat_i = F.max_pool2d(x[self.nl * 2 - 1 - i], kernel_size=8, stride=8, padding=0)  # 这里有一个倒序关系
#             #     x[i] = torch.cat((x[i], hid_feat_i), dim=1)
            
#             bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
#             if offsets is not None:
#                 r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
#                 patch_off = torch.cat((offsets[:, :1], offsets[:, 1:] / r), dim=1)  # TODO: from 4 to 32
#                 patch_off_xy = patch_off[:, 1:3].view(-1, 1, 1, 1, 2)
#                 patch_offsets.append(patch_off)
            
#             if indices_per_layer is not None:
#                 sp_x = self.m[i](x[i], indices_per_layer[i])  # sparse conv
#                 # x[i] = sp_x.dense(channels_first=True) deprecated # for training
#             else:
#                 sp_x = None
#                 x[i] = self.m[i](x[i])  # conv
#                 x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
#                 if self.class_graph_refiner is not None:
#                     x[i][..., 5:] = self.class_graph_refiner(x[i][..., 5:])

#             if not self.training:  # inference
#                 if self.grid[i].shape[2:4] != (ny, nx) or self.onnx_dynamic:
#                     self.grid[i] = self._make_grid(nx, ny).to(device)

#                 if sp_x is not None:
#                     _logits = sp_x.features.view(-1, self.na, self.no)
#                     if self.class_graph_refiner is not None:
#                         _cls = self.class_graph_refiner(_logits[..., 5:])
#                         y = _logits.clone()
#                         y[..., :5] = y[..., :5].sigmoid()
#                         y[..., 5:] = _cls.sigmoid()
#                     else:
#                         y = _logits.sigmoid()
#                     bi, yi, xi = sp_x.indices.long().T
#                     assert offsets is not None
#                     grid_off = self.grid[i][0, 0, yi, xi].view(-1, 1, 2) + patch_off_xy[bi, ...].view(-1, 1, 2)
#                     anch_wh = self.anchor_grid[i].view(1, self.na, 2)
#                     batch_ind = offsets[bi, 0]  # [num_patches, 5] --> [num_objects, 5], compatible for box concat
#                 else:
#                     _logits = x[i]
#                     if self.class_graph_refiner is not None:
#                         _cls = self.class_graph_refiner(_logits[..., 5:])
#                         y = _logits.clone()
#                         y[..., :5] = y[..., :5].sigmoid()
#                         y[..., 5:] = _cls.sigmoid()
#                     else:
#                         y = _logits.sigmoid()
#                     anch_wh = self.anchor_grid[i].view(1, self.na, 1, 1, 2)
#                     if offsets is not None:
#                         grid_off = self.grid[i] + patch_off_xy
#                         batch_ind = offsets[:, 0]   
#                     else:
#                         grid_off = self.grid[i]
#                         batch_ind = None

#                 if self.inplace:
#                     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_off) * self.stride[i]  # xy
#                     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anch_wh  # wh
#                 else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
#                     xy = (y[..., 0:2] * 2. - 0.5 + grid_off) * self.stride[i]  # xy
#                     wh = (y[..., 2:4] * 2) ** 2 * anch_wh  # wh
#                     y = torch.cat((xy, wh, y[..., 4:]), -1)
#                 # y[..., 4] = 1.0
                
#                 if offsets is not None:
#                     pbox = []
#                     for bi in range(img_bs):
#                         pbox_bi = y[batch_ind == bi]
#                         np = len(pbox_bi)
#                         if np:
#                             pbox.append(pbox_bi.view(-1, self.no))
#                         else:
#                             pbox.append(torch.zeros((0, self.no), device=device))
#                     max_pnum = max([len(boxes) for boxes in pbox])
#                     z.append(torch.stack(
#                         [torch.cat((boxes, torch.zeros((max_pnum - len(boxes), self.no), device=device))) for boxes in pbox]
#                     ))
#                 else:
#                     z.append(y.view(bs, -1, self.no))
            
#         if offsets is not None:
#             x = (x, patch_offsets)
#         return x if self.training else (torch.cat(z, 1), x)

#     @staticmethod
#     def _make_grid(nx=20, ny=20):
#         yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#         return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# class Segmenter(nn.Module):
#     def __init__(self, nc=10, ch=()):
#         super(Segmenter, self).__init__()
#         self.m = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)  # output conv
    
#     def forward(self, x):
#         return [self.m[i](x[i]) for i in range(len(x))]
    

# class Center(nn.Module):
#     def __init__(self, nc=80, ch=()):  # detection layer
#         super(Center, self).__init__()
#         self.nc = nc  # number of classes
#         self.no = nc + 4  # number of outputs
#         self.nl = 1
#         self.na = 0
#         self.anchors = None
#         self.anchor_grid = None
#         self.grid = torch.zeros(1)
#         self.b = torch.zeros(0)
#         self.c = torch.zeros(0)
#         self.stride = torch.tensor([4, 32])  # fake

#         assert len(ch) == 1
#         ch = ch[0]
#         self.m = nn.ModuleList([
#             nn.Sequential(  # hm
#                 nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(ch, nc, kernel_size=1, stride=1, padding=0, bias=True),
#             ),
#             nn.Sequential(  # wh
#                 nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(ch, 2, kernel_size=1, stride=1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#             ),
#             nn.Sequential(  # reg
#                 nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(ch, 2, kernel_size=1, stride=1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#             ),
#         ])  # output convs
#         self.m[0][-1].bias.data.fill_(-2.19)  # hm, expect 0.01
    
#     def forward(self, x):
#         assert isinstance(x, tuple)
#         x, offsets = x  # offsets(bi,x1,y1,x2,y2)
#         assert len(x) == 1
#         x = x[0]
#         hm, wh, reg = [self.m[i](x) for i in range(3)]
        
#         # offsets = torch.cat((offsets[:, :1], offsets[:, 1:] * 2.), dim=1)
            
#         if not self.training:  # inference
#             nb, nc, ny, nx = hm.shape
#             device = hm.device
            
#             wh_ = wh.permute(0, 2, 3, 1).contiguous()
#             reg_ = reg.permute(0, 2, 3, 1).contiguous()
#             if self.grid.shape[1:2] != wh_.shape[1:2]:
#                 self.grid = self._make_grid(nx, ny).to(x.device)
#             # [nb, ny, nx, 4], absolute pixel relative to input size
#             if offsets is not None:
#                 offsets_xy = offsets[:, 1:3].view(-1, 1, 1, 2) * 2
#                 bbox = torch.cat([self.grid + offsets_xy + reg_ - wh_ / 2.,
#                                   self.grid + offsets_xy + reg_ + wh_ / 2.], dim=-1)
#             else:
#                 bbox = torch.cat([self.grid + reg_ - wh_ / 2., self.grid + reg_ + wh_ / 2.], dim=-1)
#             # no clamp
#             # bbox[..., [0, 2]].clamp_(0, nx)
#             # bbox[..., [1, 3]].clamp_(0, ny)
#             # [nb, nc, ny, nx, 4]
#             # print(self.grid.shape, wh_.shape, hm.shape, offsets_xy.shape, bbox.shape)
#             bbox = bbox.view(nb, 1, ny, nx, 4).repeat((1, nc, 1, 1, 1)) * self.stride[0]
            
#             if self.c.shape[:-1] != hm.shape:
#                 # [nb, nc, ny, nx, 1]
#                 self.c = torch.arange(nc).to(x.device).view(1, nc, 1, 1, 1).repeat((nb, 1, ny, nx, 1))
#             self.b = offsets[:, :1].view(nb, 1, 1, 1, 1).repeat((1, nc, ny, nx, 1))
            
#             hm_ = hm.sigmoid()
#             hmax = F.max_pool2d(hm_, 3, stride=1, padding=1)
#             maxima = hmax == hm_
#             # hmax_cls = torch.argmax(hm_, dim=1, keepdim=True)
#             # maxima_class = hmax_cls == self.c[..., 0]
#             # maxima &= maxima_class
            
#             # [n, 6]
#             preds = torch.cat([bbox[maxima], hm_[maxima].view(-1, 1), self.c[maxima]], dim=1)
#             bi = self.b[maxima][:, 0]

#             pbox = []
#             for i in range(torch.max(bi).int().item() + 1):
#                 pbox_bi = preds[bi == i]
#                 # topk_indices = torch.argsort(pbox_bi[:, 4], descending=True)[:500]
#                 if len(pbox_bi):
#                     pbox.append(
#                         torch.cat((xyxy2xywh(pbox_bi[:, :4]), pbox_bi[:, 4:5],
#                                    F.one_hot(pbox_bi[:, 5].long(), self.nc)), dim=1)
#                     )
#                 else:
#                     pbox.append(torch.zeros((0, 5 + self.nc), device=device))
#             max_pnum = max([len(boxes) for boxes in pbox])
#             predictions = torch.stack(
#                 [torch.cat((boxes, torch.zeros((max_pnum - len(boxes), 5 + self.nc), device=device))) for boxes in pbox]
#             )
#         else:
#             predictions = None

#         x = ((hm, wh, reg), [offsets])  # for consistency when testing
            
#         return x if self.training else (predictions, x)
    
#     @staticmethod
#     def _make_grid(nx=20, ny=20):
#         yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#         return torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float()
    

# class Model(nn.Module):
#     def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
#         super(Model, self).__init__()
#         if isinstance(cfg, dict):
#             self.yaml = cfg  # model dict
#         else:  # is *.yaml
#             import yaml  # for torch hub
#             self.yaml_file = Path(cfg).name
#             with open(cfg) as f:
#                 self.yaml = yaml.safe_load(f)  # model dict

#         # Define model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
#         if nc and nc != self.yaml['nc']:
#             logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc  # override yaml value
#         if anchors:
#             logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
#             self.yaml['anchors'] = round(anchors)  # override yaml value
#         self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
#         self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
#         self.inplace = self.yaml.get('inplace', True)
#         # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

#         # Build strides, anchors
#         m = self.model[-1]  # Detect()
#         if isinstance(m, Detect):
#             s = 256  # 2x min stride
#             m.inplace = self.inplace
#             # TODO
#             # m.stride = torch.tensor([ 4., 8., 16., 32.])
#             m.stride = torch.tensor([ 8., 16., 32.])
#             # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
#             m.anchors /= m.stride.view(-1, 1, 1)
#             check_anchor_order(m)
#             self.stride = m.stride
#             self._initialize_biases()  # only run once
#             # logger.info('Strides: %s' % m.stride.tolist())
#         elif isinstance(m, Center):
#             # m.stride = torch.tensor(m.stride)  # no forward
#             self.stride = m.stride
#         # elif isinstance(m, Detect2):  deprecated
#         #     s = 256  # 2x min stride
#         #     m.inplace = self.inplace
#         #     # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
#         #     self.forward(torch.zeros(1, ch, s, s))  # forward
#         #     m.stride = torch.tensor(m.stride)  # no forward
#         #     # m.anchors /= m.stride.view(-1, 1, 1) 这里不归一化
#         #     self.stride = m.stride

#         # Init weights, biases
#         initialize_weights(self)
#         try:
#             self.info()
#         except:
#             logger.info('Failed to capture the model info')
#         logger.info('')

#     def forward(self, x, augment=False, profile=False, hm_only=False):
#         if augment:
#             return self.forward_augment(x)  # augmented inference, None
#         else:
#             return self.forward_once(x, profile, hm_only=hm_only)  # single-scale inference, train

#     def forward_augment(self, x):
#         img_size = x.shape[-2:]  # height, width
#         s = [1, 0.83, 0.67]  # scales
#         f = [None, 3, None]  # flips (2-ud, 3-lr)
#         y = []  # outputs
#         for si, fi in zip(s, f):
#             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#             yi = self.forward_once(xi)[0]  # forward
#             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#             yi = self._descale_pred(yi, fi, si, img_size)
#             y.append(yi)
#         return torch.cat(y, 1), None  # augmented inference, train

#     def forward_once(self, x, profile=False, hm_only=False):
#         y, dt = [], []  # outputs
        
#         masks, pred_masks, offsets = None, None, None
#         heatmap = None
#         if isinstance(x, tuple):
#             x, masks = x  # ground-truth masks
#         x0 = x
#         B, C, H, W = x.shape
#         for mi, m in enumerate(self.model):
#             if m.f != -1:  # if not from previous layer
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else (x0 if j == -100 else y[j]) for j in m.f]  # from earlier layers

#             if profile:
#                 o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
#                 t = time_synchronized()
#                 for _ in range(10):
#                     _ = m(x)
#                 dt.append((time_synchronized() - t) * 100)
#                 if m == self.model[0]:
#                     logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
#                 logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

#             if isinstance(m, HeatMapParser):
#                 # HeatMapParser는 항상 Segmenter의 예측 heatmap을 받아야 함
#                 assert pred_masks is not None, "Segmenter output(pred_masks) must exist before HeatMapParser"
#                 x = (x[0], pred_masks)
#             elif type(m) in [Detect, Center] and offsets is not None:
#                 x = (x, offsets)
#                 x = (*x, masks if masks is not None else pred_masks)
#             elif isinstance(m, MaskedC3TR):
#                 x = (x, heatmap)
#             elif isinstance(m, Token2Image):
#                 x = [x, (H, W)]
            
#             x = m(x)  # run

#             if isinstance(m, Segmenter):
#                 pred_masks = x
#                 if hm_only:
#                     return (None, None), pred_masks
#                 if masks is None:
#                     masks = pred_masks
#             elif isinstance(m, HeatMapParser):
#                 if isinstance(x, torch.Tensor):
#                     offsets = x
#                     if offsets.size(0) == 0:
#                         return (None, None), pred_masks
#                 elif isinstance(x[1], torch.Tensor):
#                     x, offsets = x
#                     if len(x) == 0:
#                         return (None, None), pred_masks
#                 else:
#                     x, thresh = x
#                     heatmap = pred_masks[0].detach().sigmoid()
#                     heatmap = heatmap > thresh
#             y.append(x if m.i in self.save else None)  # save output

#         if profile:
#             logger.info('%.1fms total' % sum(dt))
            
#         return x, pred_masks

#     def _descale_pred(self, p, flips, scale, img_size):
#         # de-scale predictions following augmented inference (inverse operation)
#         if self.inplace:
#             p[..., :4] /= scale  # de-scale
#             if flips == 2:
#                 p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
#             elif flips == 3:
#                 p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
#         else:
#             x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
#             if flips == 2:
#                 y = img_size[0] - y  # de-flip ud
#             elif flips == 3:
#                 x = img_size[1] - x  # de-flip lr
#             p = torch.cat((x, y, wh, p[..., 4:]), -1)
#         return p

#     def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
#         # https://arxiv.org/abs/1708.02002 section 3.3
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
#         m = self.model[-1]  # Detect() module
#         try:
#             for mi, s in zip(m.m, m.stride):  # from YOLOv5 head
#                 b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
#                 b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#                 b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
#                 mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#         except AttributeError:
#             for mi, s in zip(m.m, m.stride):  # from decoupled head
#                 mi.obj_pred.bias.data.fill_(math.log(8 / (640 / s) ** 2))  # obj (8 objects per 640 image)
#                 mi.cls_pred.bias.data.fill_(math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum()))

#         for m_ in self.model:
#             if str(m_.type) == 'models.yolo.Segmenter':  # stupid
#                 for mi in m_.m:
#                     b = mi.bias.view(-1)
#                     b.data += math.log(0.6 / (m.nc - 0.99) if cf is None else torch.log(cf / cf.sum()))  # cls
#                     mi.bias = torch.nn.Parameter(b, requires_grad=True)
#                 break

#     def _print_biases(self):
#         m = self.model[-1]  # Detect() module
#         for mi in m.m:  # from
#             b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
#             logger.info(
#                 ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

#     # def _print_weights(self):
#     #     for m in self.model.modules():
#     #         if type(m) is Bottleneck:
#     #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

#     def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
#         logger.info('Fusing layers... ')
#         for m in self.model.modules():
#             if type(m) is Conv and hasattr(m, 'bn'):
#                 m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#                 delattr(m, 'bn')  # remove batchnorm
#                 m.forward = m.fuseforward  # update forward
#         try:
#             self.info()
#         except:
#             print('Failed to capture the model info')
#         return self

#     def nms(self, mode=True):  # add or remove NMS module
#         present = type(self.model[-1]) is NMS  # last layer is NMS
#         if mode and not present:
#             logger.info('Adding NMS... ')
#             m = NMS()  # module
#             m.f = -1  # from
#             m.i = self.model[-1].i + 1  # index
#             self.model.add_module(name='%s' % m.i, module=m)  # add
#             self.eval()
#         elif not mode and present:
#             logger.info('Removing NMS... ')
#             self.model = self.model[:-1]  # remove
#         return self

#     def autoshape(self):  # add AutoShape module
#         logger.info('Adding AutoShape... ')
#         m = AutoShape(self)  # wrap model
#         copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
#         return m

#     def info(self, verbose=False, img_size=640):  # print model information
#         model_info(self, verbose, img_size)


# def parse_model(d, ch):  # model_dict, input_channels(3)
#     logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
#     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             try:
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#             except:
#                 pass

#         n = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in [nn.Conv2d, Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, ASPP, SPPF,
#                  DWConv, DCN, RepLKConv, MixConv2d, Focus, Blur, CrossConv,
#                  BottleneckCSP, C3, C3TR, MaskedC3TR, C2f, ResBlockLayer, RTMDetCSPLayer,
#                  HeatMapParser]:
#             c2 = args[0]
#             if c2 != no and 'GPViTAdapterSingleStageESOD' not in [_x[2] for _x in d['backbone']]:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)

#             if m is HeatMapParser:
#                 args = [c2, *args[1:]]
#             else:
#                 args = [ch[f], c2, *args[1:]]
#                 if m in [BottleneckCSP, C3, C3TR, MaskedC3TR, C2f, RTMDetCSPLayer]:
#                     args.insert(2, n)  # number of repeats
#                     n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum([ch[x] for x in f])
#         elif m in [Add, nn.Identity]:
#             pass
#         elif m in [Detect, Segmenter]:  # Detect2 deprecated
#             args.append([ch[x] for x in f])
#             if len(args) > 1 and isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Center:
#             args.append([ch[x] for x in f])
#         elif m is SpatialPriorModule:
#             c2 = args[0] * 2
#         elif m is GPViTAdapterSingleStageESOD:
#             pass
#         elif m is Indexer:
#             c2 = args[0]
#             args = args[1:]
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         else:
#             c2 = ch[f]

#         m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum([x.numel() for x in m_.parameters()])  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x not in [-1, -100])  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     return nn.Sequential(*layers), sorted(save)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     opt = parser.parse_args()
#     opt.cfg = check_file(opt.cfg)  # check file
#     set_logging()
#     device = select_device(opt.device)

#     # Create model
#     model = Model(opt.cfg).to(device)
#     model.train()

#     # Profile
#     # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
#     # y = model(img, profile=True)

#     # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
#     # from torch.utils.tensorboard import SummaryWriter
#     # tb_writer = SummaryWriter('.')
#     # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
#     # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
#     # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard

# # ===========================
# # Class-Graph control APIs (monkey-patched onto Model)
# # ===========================
# def _classgraph_enable(self, class_names, clip_model_name='ViT-B-32',
#                        clip_pretrained=None, alpha=0.3, top_r=2, ema_beta=0.9,
#                        lambda_start=0.4, lambda_end=0.1,
#                        warmup_epochs=5, device=None):
#     """Enable hybrid sparse class-graph refinement on Detect head (cls logits only)."""
#     try:
#         nc = int(self.nc) if hasattr(self, 'nc') else len(class_names)
#         refiner = HybridSparseClassGraphRefiner(
#             nc=nc, clip_pretrained=clip_pretrained, alpha=alpha, top_r=top_r, ema_beta=ema_beta,
#             lambda_start=lambda_start, lambda_end=lambda_end,
#             warmup_epochs=warmup_epochs, clip_model_name=clip_model_name,
#             device=device or next(self.parameters()).device
#         )
#         # build semantic prior once
#         refiner.build_semantic_prior(class_names, device=device or next(self.parameters()).device)
#         refiner.rebuild_graph(epoch=0, total_epochs=1)
#         # attach to detect head(s)
#         for m in self.model.modules():
#             if isinstance(m, Detect):
#                 m.class_graph_refiner = refiner
#         self._class_graph_refiner = refiner
#         self._class_graph_enabled = True
#         print(f"[ClassGraph] enabled (nc={nc}, top_r={top_r}, alpha={alpha}, clip={clip_model_name})")
#         return True
#     except Exception as e:
#         print(f"[ClassGraph] enable failed: {e}")
#         self._class_graph_enabled = False
#         return False


# @torch.no_grad()
# def _classgraph_update(self, confusion_matrix, epoch: int, total_epochs: int):
#     """Update EMA confusion prior and rebuild hybrid sparse graph."""
#     if not getattr(self, '_class_graph_enabled', False):
#         return False
#     refiner = getattr(self, '_class_graph_refiner', None)
#     if refiner is None:
#         return False
#     refiner.update_confusion_ema(confusion_matrix, device=next(self.parameters()).device)
#     refiner.rebuild_graph(epoch=int(epoch), total_epochs=int(total_epochs))
#     return True


# # Monkey-patch (robust against indentation/merge issues)
# try:
#     if not hasattr(Model, 'enable_class_graph'):
#         Model.enable_class_graph = _classgraph_enable
#     if not hasattr(Model, 'update_class_graph'):
#         Model.update_class_graph = _classgraph_update
# except Exception:
#     pass