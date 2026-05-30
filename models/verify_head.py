"""
verify_head.py — propose-verify 구조의 verify head.

역할
----
ESOD detector가 낸 검출을 입력받아 "이 검출이 TP냐 FP냐"를 판정한다.
propose head(heatmap)와 *다른 stage*(검출 뒤) · *다른 target*(GT mask가 아니라
detector의 TP/FP 결과)에서 동작 → heatmap과의 redundancy ceiling 회피.

binary Evidential Deep Learning
-------------------------------
2-class(FP, TP) evidence  e = [e_fp, e_tp] = softplus(logits) ≥ 0
alpha = e + 1 ,  S = Σ alpha
  verify score  b_tp = alpha_tp / S   ∈ (0,1)   ← re-scoring·랭킹용
  vacuity       u    = 2 / S          ∈ (0,1)   ← reliability(신뢰도) 출력
EDL이 본연의 일(불확실도 동반 분류)을 하는 위치.

설계 근거
- MetaDetect (Schubert et al., IJCNN 2021): 검출 meta-feature로 TP/FP·품질 추정.
- EDL (Sensoy et al., NeurIPS 2018) / R-EDL (Chen et al., ICLR 2024).

용도
- 학습/추론, offline proxy(고정 검출로 학습)와 1-stage 통합(ESOD와 함께 학습)
  모두 이 클래스 하나를 공유한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_in_dim(nc):
    """build_verify_features 차원: geometry6 + obj1 + conf1 + cls(nc) + shape3."""
    return 11 + int(nc)


def build_verify_features(dets, img_wh, nc):
    """검출 텐서 → verify head 입력 feature.

    dets   : [N, 5+nc]  (cx, cy, w, h, obj, cls_0..cls_{nc-1})  — 입력공간 픽셀,
             obj·cls 는 (0,1) 확률 (YOLO decode 후)
    img_wh : (W, H)
    반환    : [N, 11+nc]   geometry(6) + obj(1) + conf(1) + cls(nc) + cls분포모양(3)
    """
    eps = 1e-6
    W = float(img_wh[0]); H = float(img_wh[1])
    cx, cy, w, h = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    obj = dets[:, 4:5]
    cls = dets[:, 5:5 + nc]

    geom = torch.stack([
        (cx / W).clamp(0.0, 1.0), (cy / H).clamp(0.0, 1.0),
        torch.log(w.clamp(min=eps)), torch.log(h.clamp(min=eps)),
        torch.log((w * h).clamp(min=eps)),
        torch.log(w.clamp(min=eps) / h.clamp(min=eps)),
    ], dim=-1)                                              # [N,6]

    cmax = cls.max(dim=-1, keepdim=True).values             # 최고 클래스 점수
    conf = obj * cmax                                       # = raw YOLO conf (헤드가 최소한 conf 재현 가능)
    pc = cls.clamp(eps, 1.0)
    ent = -(pc * pc.log()).sum(dim=-1, keepdim=True)        # 분포 entropy
    if nc >= 2:
        top2 = cls.topk(2, dim=-1).values
        gap = top2[:, :1] - top2[:, 1:2]                    # 1등-2등 차
    else:
        gap = torch.zeros_like(cmax)
    shape = torch.cat([cmax, ent, gap], dim=-1)             # [N,3]

    return torch.cat([geom, obj, conf, cls, shape], dim=-1)  # [N, 11+nc]


class VerifyHead(nn.Module):
    """검출별 TP/FP 판정 binary-EDL head.

    forward(feat) — feat [N, in_dim] 검출별 feature
      반환 dict(score=b_tp[N], vacuity=u[N], evidence=e[N,2], alpha=alpha[N,2])
    """

    def __init__(self, in_dim, hidden=64, layers=2, drop=0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        blocks, d = [], self.in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if drop > 0:
                blocks += [nn.Dropout(drop)]
            d = hidden
        self.mlp = nn.Sequential(*blocks)
        self.evi = nn.Linear(d, 2)                          # [logit_fp, logit_tp]
        # evidence head 작게 시작 → 초기 b_tp≈0.5 (학습 안정)
        nn.init.zeros_(self.evi.bias)
        nn.init.normal_(self.evi.weight, std=1e-3)

    def forward(self, feat):
        if feat.shape[0] == 0:
            z = feat.new_zeros((0,))
            return dict(score=z, vacuity=z, evidence=feat.new_zeros((0, 2)),
                        alpha=feat.new_zeros((0, 2)))
        evidence = F.softplus(self.evi(self.mlp(feat)))     # [N,2] ≥ 0
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)                 # [N,1]
        return dict(score=alpha[:, 1] / S[:, 0],            # b_tp
                    vacuity=2.0 / S[:, 0],                  # u
                    evidence=evidence, alpha=alpha)


def verify_edl_loss(alpha, tp_label, pos_weight=1.0, reduction='mean'):
    """binary EDL (Dirichlet MSE Bayes risk) loss.

    alpha     : [N,2]  Dirichlet 파라미터 (index 0=FP, 1=TP)
    tp_label  : [N]    1=TP, 0=FP
    pos_weight: TP:FP 가중비 (검출은 FP가 절대다수 — Step0: 86.5% FP)

    L_i = Σ_k (y_k - p_k)^2 + p_k(1-p_k)/(S+1) ,  y = onehot(tp_label)
    (R-EDL: KL 분산-최소화 항 제거)
    """
    S = alpha.sum(dim=-1, keepdim=True)                     # [N,1]
    p = alpha / S                                          # [N,2]
    t = tp_label.to(p.dtype)
    y = torch.stack([1.0 - t, t], dim=-1)                  # [N,2]
    err = (y - p).pow(2).sum(dim=-1)                       # [N]
    var = (p * (1.0 - p) / (S + 1.0)).sum(dim=-1)          # [N]
    loss_i = err + var                                     # [N]

    if reduction != 'mean':
        return loss_i if reduction == 'none' else loss_i.sum()
    pos = t > 0.5
    neg = ~pos
    n_pos = pos.sum().clamp(min=1)
    n_neg = neg.sum().clamp(min=1)
    l_pos = (loss_i * pos).sum() / n_pos
    l_neg = (loss_i * neg).sum() / n_neg
    return (pos_weight * l_pos + l_neg) / (pos_weight + 1.0)
