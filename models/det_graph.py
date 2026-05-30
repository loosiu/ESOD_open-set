"""
det_graph.py — Evidential Cross-Patch graph refinement (ECPR의 핵심 모듈).

동기 (miss-diag + Step 0)
------------------------
ESOD는 패치를 *고립* 검출 → cross-patch 맥락 상실. Step 0 분해상 mAP headroom의
+0.14는 "검출은 났는데 detector가 약한" 영역 — 이웃(같은 클래스 군집·규칙적 기하)
정보로 보완 가능. 그래프를 한 이미지의 모든 패치 검출 위에 세워 그 정보를 복원한다.

EDL이 핵심에 들어간 형태 (Evidential)
-------------------------------------
- evidential objectness head : 2채널 evidence → Dirichlet(α=e+1) → belief + vacuity
- uncertainty-gated message passing : 이웃 j의 메시지를 (1 - u_j)로 감쇠
  (불확실한 이웃이 좋은 검출을 오염시키지 않게)
- vacuity 출력 : self.last_u — 검출별 신뢰도 신호

설계 노트
---------
- obj/cls/box head 는 zero-init → 학습 시작 시 ≈ 항등 (입력 검출 보존, ablation 정합).
  evidential obj 는 입력 obj logit 을 seed 로 둔 residual → 시작 시 belief≈σ(obj).
- 연산: top-K 노드 × k 엣지 × GNN(hidden, layers). 파라미터 수십 K, 희소 → 경량.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ECPR 스위치: True → Detect 가 cross-patch graph refinement 모듈을 생성·사용
# 2026-05-25 OFF: ecpr3 mAP −0.046 (joint-train backbone 교란). EDL-OOD path로 pivot.
ENABLE = False


def knn_edges(centers, k):
    """centers [N,2] → 각 노드의 k-최근접 이웃 인덱스 [N,k] (자기 제외)."""
    N = centers.shape[0]
    if N <= 1:
        return centers.new_zeros((N, 0), dtype=torch.long)
    d = torch.cdist(centers, centers)                 # [N,N]
    d.fill_diagonal_(float('inf'))
    k = min(int(k), N - 1)
    return d.topk(k, dim=1, largest=False).indices    # [N,k]


class DetGraphRefine(nn.Module):
    """검출 위 희소 인스턴스 그래프로 obj·cls·box 를 evidential 하게 보정.

    forward(boxes, obj, cls) — 한 이미지의 검출들 (배치는 호출측 loop).
      boxes [N,4] xywh(픽셀), obj [N] objectness logit, cls [N,C] class logit
    반환: ref_obj [N] logit, ref_cls [N,C] logit, ref_box [N,4] xywh
    부수: self.last_u [N] — per-detection vacuity (reliability 신호)
    """

    def __init__(self, nc, k=8, hidden=64, layers=2, box_scale=0.1):
        super().__init__()
        self.nc, self.k, self.layers = int(nc), int(k), int(layers)
        self.box_scale = float(box_scale)
        self.last_u = None                              # 최근 forward 의 vacuity

        in_dim = 4 + 1 + nc                             # box(xywh) + obj + cls
        edge_dim = 5                                    # dx, dy, dlogw, dlogh, iou
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden))
        self.msg = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(inplace=True),
                          nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.att = nn.ModuleList(
            nn.Linear(2 * hidden + edge_dim, 1) for _ in range(layers))
        self.vac_head = nn.Linear(hidden, 2)            # per-node evidence → uncertainty gate
        self.obj_head = nn.Linear(hidden, 2)            # evidential objectness (e_bg, e_obj)
        self.cls_head = nn.Linear(hidden, nc)           # cls logit residual
        self.box_head = nn.Linear(hidden, 4)
        for h in (self.obj_head, self.cls_head, self.box_head, self.vac_head):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)   # zero-init → 시작 시 ≈ 항등

    @staticmethod
    def _pair_iou(boxes_xywh, idx):
        """각 박스와 그 k 이웃 간 IoU [N,k]."""
        cx, cy, w, h = boxes_xywh.unbind(-1)
        bi = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)  # [N,4] xyxy
        bj = bi[idx]                                    # [N,k,4]
        bi_ = bi[:, None, :]                            # [N,1,4]
        ix1 = torch.max(bi_[..., 0], bj[..., 0]); iy1 = torch.max(bi_[..., 1], bj[..., 1])
        ix2 = torch.min(bi_[..., 2], bj[..., 2]); iy2 = torch.min(bi_[..., 3], bj[..., 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        ai = (bi_[..., 2] - bi_[..., 0]) * (bi_[..., 3] - bi_[..., 1])
        aj = (bj[..., 2] - bj[..., 0]) * (bj[..., 3] - bj[..., 1])
        return inter / (ai + aj - inter + 1e-6)         # [N,k]

    @staticmethod
    def _dirichlet(evi_logit, seed=None):
        """2채널 evidence-logit → (belief_obj [N], vacuity [N]). seed: residual base."""
        z = evi_logit if seed is None else evi_logit + seed
        alpha = F.softplus(z) + 1.0                     # Dirichlet α = evidence + 1
        S = alpha.sum(-1)                               # [N]
        return alpha[:, 1] / S, 2.0 / S                 # belief(obj), vacuity

    def forward(self, boxes, obj, cls):
        N = boxes.shape[0]
        if N == 0:
            self.last_u = obj.new_zeros((0,))
            return obj, cls, boxes
        cx, cy, w, h = boxes.unbind(-1)
        eps = 1e-6

        # ── 노드 feature ──
        nrm = boxes.detach().abs().max().clamp(min=1.0)
        nf = torch.cat([boxes / nrm, obj[:, None], cls], dim=-1)   # [N, 4+1+C]
        x = self.node_enc(nf)                                      # [N, H]

        # ── per-node uncertainty (uncertainty-gated MP 용) ──
        _, u_node = self._dirichlet(self.vac_head(x))              # [N]

        idx = knn_edges(torch.stack([cx, cy], -1), self.k)         # [N,k]
        if idx.shape[1] == 0:
            self.last_u = u_node.detach()
            return obj, cls, boxes

        # ── 엣지 feature (상대 기하) ──
        dx = (cx[idx] - cx[:, None]) / (w[:, None] + eps)
        dy = (cy[idx] - cy[:, None]) / (h[:, None] + eps)
        dlw = torch.log((w[idx] + eps) / (w[:, None] + eps))
        dlh = torch.log((h[idx] + eps) / (h[:, None] + eps))
        iou = self._pair_iou(boxes, idx)
        edge = torch.stack([dx, dy, dlw, dlh, iou], dim=-1)        # [N,k,5]

        # ── uncertainty-gated message passing ──
        gate = (1.0 - u_node[idx])                                 # [N,k] 이웃 j 신뢰도
        for l in range(self.layers):
            xj = x[idx]                                            # [N,k,H]
            xi = x[:, None, :].expand_as(xj)
            cat = torch.cat([xi, xj, edge], dim=-1)                # [N,k,2H+5]
            m = self.msg[l](cat)                                   # [N,k,H]
            a = torch.softmax(self.att[l](cat).squeeze(-1), dim=-1)  # [N,k]
            a = a * gate                                           # 불확실한 이웃 감쇠
            x = x + (a[..., None] * m).sum(dim=1)                  # residual

        # ── evidential objectness (입력 obj logit 을 seed 로 한 residual) ──
        obj_seed = torch.stack([-obj, obj], dim=-1) * 0.5          # [N,2] belief(seed)=σ(obj)
        belief_obj, u_obj = self._dirichlet(self.obj_head(x), seed=obj_seed)
        self.last_u = u_obj.detach()                               # [N] vacuity 출력 (deepcopy 안전)
        b = belief_obj.clamp(eps, 1 - eps)
        ref_obj = torch.log(b / (1 - b))                           # belief → logit

        # ── cls·box residual 보정 ──
        ref_cls = cls + self.cls_head(x)                           # [N,C] logit
        d_box = torch.tanh(self.box_head(x)) * self.box_scale      # [N,4] bounded
        ref_box = torch.stack([
            cx + d_box[:, 0] * w, cy + d_box[:, 1] * h,
            w * torch.exp(d_box[:, 2]), h * torch.exp(d_box[:, 3]),
        ], dim=-1)
        return ref_obj, ref_cls, ref_box
