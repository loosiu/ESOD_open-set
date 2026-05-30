"""
Detector-stage Evidential Deep Learning (EDL) for ESOD.

논문 근거
---------
- R-EDL : Chen et al., "R-EDL: Relaxing Nonessential Settings of Evidential
          Deep Learning", ICLR 2024 (Spotlight).
          → prior weight 를 고정 1 이 아닌 hyperparameter `lam` 으로,
            분산-최소화 KL 정규화 항 제거 (overconfidence 완화).
- E-DETR: "E-DETR: Evidential Deep Learning for End-to-End Uncertainty
          Estimation in Object Detection".
          → IoU-aware loss 로 objectness 의 evidential 타겟을 IoU(국소화 품질)로.
            ⇒ p_obj 가 localization quality 반영 ⇒ score = p_obj × cls 가
              잘-국소화된 검출을 상위 랭킹 (GFL, Li et al. NeurIPS'20 의 mAP 메커니즘).
- EDL  : Sensoy et al., NeurIPS 2018 (기반 정식).

1채널 logit-evidential 파라미터화
---------------------------------
ESOD detector 의 objectness 는 anchor 당 1채널(logit)이다. detection 파이프라인
구조(채널 수·레이아웃·sparse 경로)를 **전혀 바꾸지 않기 위해** 그 1채널 logit z 를
binary evidential 로 재해석한다:

    e_obj = softplus(z),   e_bg = softplus(-z)          # 둘 다 ≥ 0
    alpha = [ e_bg + lam ,  e_obj + lam ]               # R-EDL prior weight lam
    S       = e_obj + e_bg + 2·lam
    p_obj   = (e_obj + lam) / S
    vacuity = 2·lam / S

z≫0 → 객체 확신, z≪0 → 배경 확신, z≈0 → 중간(불확실).
한계: 완전 무지(vacuity→1)는 표현 못 함 (vacuity 상한 ≈ 2lam/(2·softplus(0)+2lam)).
      자유 2채널 evidence 는 Phase-2 확장 사항. Phase-1 점수(p_obj×cls)엔 무관.
"""

import os
import torch
import torch.nn.functional as F

# R-EDL prior weight (tunable hyperparameter; vanilla EDL = 1.0)
LAM = 1.0


def fuse_dual_seg(heat_logit, edl_logit, gate_logit=None):
    """Dual segmenter fusion 통일 helper. 5+ 위치에서 호출 (HeatMapParser/get_indices/calib/...).

    환경변수 분기:
      ESOD_ROLE_DUAL=1 → 3-C noisy-OR:        F = 1 - (1 - σ(heat)) * (1 - p_e_obj)
                                              (role-separated, heat-or-edl propose)
      else (default)   → calibrated gating:   F = (1-w)·σ(heat) + w·p_e_obj
                                              w = σ(gate) * (1 - u)
                                              (메모리상 exp83 calib-gating)

    Inputs (all logits, 4D [B,C,H,W]):
      heat_logit : [B,1,H,W]
      edl_logit  : [B,2,H,W]
      gate_logit : [B,1,H,W] (optional, calib-gating only)
    Output: F ∈ [0,1] calibrated objectness probability map [B,1,H,W]
    """
    prob = torch.sigmoid(heat_logit)                       # [B,1,H,W]
    ev = F.softplus(edl_logit); al = ev + 1.0
    S = al.sum(dim=1, keepdim=True)                        # [B,1,H,W]
    p_e = al[:, 1:2] / S                                   # EDL p(obj)
    u = 2.0 / S                                            # vacuity
    if os.environ.get('ESOD_ROLE_DUAL', '').strip() == '1':
        return 1.0 - (1.0 - prob) * (1.0 - p_e)            # noisy-OR
    if gate_logit is None:
        return torch.max(prob, p_e)
    g = torch.sigmoid(gate_logit)
    w = g * (1.0 - u)
    return (1.0 - w) * prob + w * p_e

# detector-stage EDL 스위치 (True → objectness head를 evidential R-EDL로 학습·추론)
EDL_DET = False


def redl_p_obj(z, lam: float = LAM):
    """1채널 logit z → objectness 확률 p_obj (R-EDL). 추론 점수용."""
    e_obj = F.softplus(z)
    e_bg = F.softplus(-z)
    S = e_obj + e_bg + 2.0 * lam
    return (e_obj + lam) / S


def redl_binary(z, lam: float = LAM):
    """1채널 logit z → (p_obj, vacuity, S). 모니터링/분석용."""
    e_obj = F.softplus(z)
    e_bg = F.softplus(-z)
    S = e_obj + e_bg + 2.0 * lam
    p_obj = (e_obj + lam) / S
    vacuity = (2.0 * lam) / S
    return p_obj, vacuity, S


def redl_2ch(ev, lam: float = LAM):
    """2채널 evidence (e_bg, e_obj) → (p_obj, vacuity, S). post-hoc / inference 용."""
    e_bg = F.softplus(ev[..., 0])
    e_obj = F.softplus(ev[..., 1])
    a_bg = e_bg + lam; a_obj = e_obj + lam
    S = a_bg + a_obj
    return a_obj / S, (2.0 * lam) / S, S


def iou_aware_edl_loss_2ch(ev, target, lam: float = LAM,
                           pos_mask=None, pos_weight: float = 1.0):
    """2채널 evidence (e_bg, e_obj) IoU-aware R-EDL loss (NLL+없는 MSE form).

    ev      : Tensor[..., 2]  마지막 차원 = (e_bg_logit, e_obj_logit)
    target  : 같은 shape의 evidential 타겟 ∈ [0,1] (pos=IoU, neg=0)
    pos_mask: bool tensor (positive anchor — class-balanced 평균)

    R-EDL relaxation (Chen et al., ICLR 2024 Spotlight): KL 정규화 제거.
    Dirichlet MSE Bayes risk on binary [bg, obj]:
        L_i = (1-t-p_bg)^2 + (t-p_obj)^2 + [p_obj(1-p_obj)+p_bg(1-p_bg)]/(S+1)
    """
    e_bg = F.softplus(ev[..., 0])
    e_obj = F.softplus(ev[..., 1])
    a_bg = e_bg + lam; a_obj = e_obj + lam
    S = a_bg + a_obj
    p_obj = a_obj / S
    p_bg = a_bg / S
    t = target.clamp(0.0, 1.0)
    err = (t - p_obj).pow(2) + ((1.0 - t) - p_bg).pow(2)
    var = (p_obj * (1.0 - p_obj) + p_bg * (1.0 - p_bg)) / (S + 1.0)
    loss_i = err + var                                  # anchor별
    if pos_mask is None:
        return loss_i.mean()
    pos = pos_mask.bool(); neg = ~pos
    n_pos = pos.sum().clamp(min=1); n_neg = neg.sum().clamp(min=1)
    l_pos = (loss_i * pos).sum() / n_pos
    l_neg = (loss_i * neg).sum() / n_neg
    return (pos_weight * l_pos + l_neg) / (pos_weight + 1.0)


def redl_Kch(cls_logit, lam: float = LAM):
    """K-class evidential reading. cls_logit [..., K] → (belief [..., K], vacuity [...], S [...]).

    R-EDL (Chen et al., ICLR 2024 Spotlight) parameterization:
        α_k = softplus(z_k) + λ
        S = Σ_k α_k
        belief_k = α_k / S
        vacuity = K·λ / S
    Reference: EOD (Wang et al., AAAI 2024) for OWOD class-evidential head.
    """
    ev = F.softplus(cls_logit)
    alpha = ev + lam
    S = alpha.sum(-1)                                  # [...]
    K = float(cls_logit.shape[-1])
    belief = alpha / S.unsqueeze(-1)                   # [..., K]
    vacuity = (K * lam) / S                            # [...]
    return belief, vacuity, S


def edl_cls_loss_Kch(cls_logit, target_class, lam: float = LAM,
                     pos_mask=None, pos_weight: float = 1.0,
                     label_smoothing: float = 0.0):
    """EOD-style K-class evidential classification loss (R-EDL Dirichlet MSE, KL-free).

    cls_logit  : Tensor[N, K]  positive anchor 의 K-class evidence logits
    target_class : Tensor[N]   long, true class id (0..K-1) — positive anchor only
    pos_mask   : Tensor[N] bool  positive anchor mask (class-balanced 평균용; None=모두 pos)
    label_smoothing: smooth target [0..1] (보통 0; ESOD는 0.0)

    Loss (R-EDL Dirichlet MSE, no KL):
        α = softplus(z) + λ ;  S = Σα ;  p = α / S
        one_hot y = (1-eps)·δ_{y_true} + eps/K  (label smoothing)
        L = Σ_k [ (y_k - p_k)^2 + p_k(1-p_k)/(S+1) ]
    """
    if cls_logit.numel() == 0:
        return cls_logit.new_zeros(())
    K = cls_logit.shape[-1]
    ev = F.softplus(cls_logit)
    alpha = ev + lam
    S = alpha.sum(-1, keepdim=True)                    # [N, 1]
    p = alpha / S                                      # [N, K]

    # target one-hot (with optional label smoothing)
    eps = float(label_smoothing)
    y = F.one_hot(target_class.long(), num_classes=K).to(cls_logit.dtype)
    if eps > 0:
        y = y * (1.0 - eps) + eps / K

    err = (y - p).pow(2).sum(-1)                        # [N]
    var = (p * (1.0 - p)).sum(-1) / (S.squeeze(-1) + 1.0)  # [N]
    loss_i = err + var                                  # [N]

    if pos_mask is None:
        return loss_i.mean()
    pos = pos_mask.bool(); neg = ~pos
    n_pos = pos.sum().clamp(min=1); n_neg = neg.sum().clamp(min=1)
    l_pos = (loss_i * pos).sum() / n_pos
    l_neg = (loss_i * neg).sum() / n_neg if neg.any() else loss_i.new_zeros(())
    return (pos_weight * l_pos + l_neg) / (pos_weight + 1.0)


def iou_aware_edl_loss(z, target, lam: float = LAM,
                       pos_mask=None, pos_weight: float = 1.0):
    """
    IoU-aware R-EDL objectness 손실 (E-DETR style, 1채널).

    Parameters
    ----------
    z          : Tensor[N]   objectness logit
    target     : Tensor[N]   evidential 타겟 ∈ [0,1]
                             - positive anchor : IoU-ratio  (국소화 품질)
                             - negative anchor : 0
    lam        : R-EDL prior weight
    pos_mask   : Tensor[N] bool  positive anchor 마스크 (class-balanced 평균용)
    pos_weight : positive:negative 가중비 (anchor imbalance 보정)

    Notes
    -----
    R-EDL relaxation: **KL 정규화 항 없음**. Dirichlet MSE Bayes risk:
        L_i = Σ_k [ (y_k - p_k)^2 + p_k(1-p_k)/(S+1) ] ,  y = [1-t, t]
    """
    e_obj = F.softplus(z)
    e_bg = F.softplus(-z)
    a_obj = e_obj + lam
    a_bg = e_bg + lam
    S = a_obj + a_bg
    p_obj = a_obj / S
    p_bg = a_bg / S

    t = target.clamp(0.0, 1.0)
    err = (t - p_obj).pow(2) + ((1.0 - t) - p_bg).pow(2)
    var = (p_obj * (1.0 - p_obj) + p_bg * (1.0 - p_bg)) / (S + 1.0)
    loss_i = err + var                                  # [N]  anchor별 R-EDL MSE

    if pos_mask is None:
        return loss_i.mean()

    # class-balanced 평균 (negative anchor 가 절대다수)
    pos = pos_mask.bool()
    neg = ~pos
    n_pos = pos.sum().clamp(min=1)
    n_neg = neg.sum().clamp(min=1)
    l_pos = (loss_i * pos).sum() / n_pos
    l_neg = (loss_i * neg).sum() / n_neg
    return (pos_weight * l_pos + l_neg) / (pos_weight + 1.0)
