"""
OWOD evaluation (ORE/OW-DETR style).

표준 OWOD metrics:
  - mAP_K       : known class mAP (test.py 자체 mAP에서 추출)
  - U-Recall    : unknown class GT 중 detector가 'unknown'으로 잘 잡은 비율
  - WI          : Wilderness Impact — unknown 영역이 known precision 깎는 정도
  - A-OSE       : Absolute Open-Set Error — unknown GT를 known class로 잘못 분류한 절대 개수

Reference:
  Joseph et al., "Towards Open World Object Detection" (CVPR 2021)
  Gupta et al., "OW-DETR: Open-world Detection Transformer" (CVPR 2022)
  Wang et al., "Evidential Open-set Detection (EOD)" (AAAI 2024)

Unknown identification methods (post-hoc, no detector retraining):
  conf        : detection conf < tau → unknown
  msp         : 1 - max(cls_softmax) > tau → unknown  (Hendrycks ICLR'17)
  energy      : -logsumexp(cls_logits) > tau → unknown  (Liu NeurIPS'20)
  vacuity     : EDL Dirichlet vacuity > tau → unknown  (EDL model 한정)
"""

import numpy as np
import torch


def box_iou_np(boxes1, boxes2):
    """xyxy IoU [N, M]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    a = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    b = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (a[:, None] + b[None, :] - inter + 1e-9)


class OWODEval:
    """OWOD evaluation accumulator (ORE/OW-DETR/EOD style).

    Inputs per image:
      pred       : Tensor[N, 6] (xyxy, conf, cls) — post-NMS detections in input space
      gts_xyxy   : Tensor[M, 5] (cls, x1, y1, x2, y2) — GT in input space (10-class)
      vac_per_det: optional Tensor[N] — EDL vacuity per detection (method='vacuity')

    Hold-out class semantics:
      known   : 학습 포함 class indices
      unknown : 학습 제외 class indices (training labels 제거됨, eval만 있음)

    Method 'conf':     pred.conf < tau         → label as unknown
    Method 'msp':      1 - max(softmax) > tau  → unknown (top-class confidence)
    Method 'energy':   -T·logsumexp / T > tau  → unknown (Liu NeurIPS'20)
    Method 'vacuity':  EDL vacuity > tau       → unknown (EDL model only)

    Metrics:
      U-Recall = |{u_gt : ∃ unknown-labeled det with IoU≥0.5}| / |u_gt|
      A-OSE    = |{known-labeled det : IoU(det, any unknown_gt) ≥ 0.5}|
      WI       = (P_K - P_K∪U) / P_K
                  P_K   : 알려진 GT 만 평가했을 때 known 검출의 정밀도
                  P_K∪U : 추가로 unknown GT 도 평가에 포함했을 때 (unknown 영역 검출이
                          known 으로 라벨됐다면 FP 로 count)
    """

    def __init__(self, known_classes, unknown_classes,
                 method='conf', tau=0.1, iou_thresh=0.5):
        self.known = set(int(c) for c in known_classes)
        self.unknown = set(int(c) for c in unknown_classes)
        self.method = method.lower()
        self.tau = float(tau)
        self.iou_t = float(iou_thresh)
        self._records = []

    def _classify_unknown(self, pn, vac_per_det=None):
        """N detections → unknown_mask [N] bool."""
        if len(pn) == 0:
            return np.zeros(0, dtype=bool)
        if self.method == 'conf':
            return pn[:, 4] < self.tau
        if self.method == 'msp':
            # top-cls confidence (post-NMS conf 활용 - softmax는 NMS 전에만 있음)
            # 여기서 pn[:,4] = obj × max(cls) = top score. 1 - pn[:,4]를 MSP proxy로 사용.
            msp_score = 1.0 - pn[:, 4]
            return msp_score > self.tau
        if self.method == 'energy':
            # post-NMS detection의 cls logit이 없으므로 conf 자체로 근사
            # Energy = -log(conf) ≈ MSP의 단조 변환. 실용상 MSP와 비슷
            energy_score = -np.log(pn[:, 4] + 1e-9)
            return energy_score > self.tau
        if self.method == 'vacuity' and vac_per_det is not None:
            vn = (vac_per_det.detach().cpu().numpy() if isinstance(vac_per_det, torch.Tensor)
                  else np.asarray(vac_per_det))
            return vn > self.tau
        return np.zeros(len(pn), dtype=bool)

    def add_image(self, pred, gts_xyxy, vac_per_det=None):
        if pred is None:
            pred = torch.zeros((0, 6))
        pn = (pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor)
              else np.asarray(pred))
        gn = (gts_xyxy.detach().cpu().numpy() if isinstance(gts_xyxy, torch.Tensor)
              else np.asarray(gts_xyxy))

        unknown_mask = self._classify_unknown(pn, vac_per_det)

        det_known = pn[~unknown_mask] if len(pn) else pn
        det_unknown = pn[unknown_mask] if len(pn) else pn

        if len(gn):
            gt_known_mask = np.array([int(g[0]) in self.known for g in gn])
            gt_known = gn[gt_known_mask]
            gt_unknown = gn[~gt_known_mask]
        else:
            gt_known, gt_unknown = gn, gn

        self._records.append(dict(det_known=det_known, det_unknown=det_unknown,
                                  gt_known=gt_known, gt_unknown=gt_unknown))

    def summarize(self, save_dir=None):
        if not self._records:
            print('\n[OWODEval] no records.')
            return None

        # ===== U-Recall =====
        n_unk_gt = 0
        n_unk_matched = 0
        for r in self._records:
            gu = r['gt_unknown']
            du = r['det_unknown']
            n_unk_gt += len(gu)
            if len(gu) == 0 or len(du) == 0:
                continue
            iou = box_iou_np(du[:, :4], gu[:, 1:5])  # [N, M]
            best_iou_per_gt = iou.max(axis=0) if iou.size else np.zeros(len(gu))
            n_unk_matched += int((best_iou_per_gt >= self.iou_t).sum())
        u_recall = n_unk_matched / max(n_unk_gt, 1)

        # ===== A-OSE =====
        a_ose = 0
        for r in self._records:
            dk = r['det_known']
            gu = r['gt_unknown']
            if len(dk) == 0 or len(gu) == 0:
                continue
            iou = box_iou_np(dk[:, :4], gu[:, 1:5])
            if iou.size == 0:
                continue
            best_per_det = iou.max(axis=1)
            a_ose += int((best_per_det >= self.iou_t).sum())

        # ===== WI =====
        # known-labeled detection 분류:
        #   TP_K   : matches known GT with same class
        #   FP_K_only : doesn't match known GT and doesn't match unknown GT (true background FP)
        #   FP_KU   : doesn't match known GT but matches unknown GT (unknown contamination)
        n_tp_K, n_fp_K_only, n_fp_KU = 0, 0, 0
        for r in self._records:
            dk = r['det_known']
            gk = r['gt_known']
            gu = r['gt_unknown']
            if len(dk) == 0:
                continue
            iou_k = box_iou_np(dk[:, :4], gk[:, 1:5]) if len(gk) else np.zeros((len(dk), 0))
            iou_u = box_iou_np(dk[:, :4], gu[:, 1:5]) if len(gu) else np.zeros((len(dk), 0))
            for i in range(len(dk)):
                matched_known = False
                if iou_k.shape[1]:
                    for j in range(iou_k.shape[1]):
                        if iou_k[i, j] >= self.iou_t and int(gk[j, 0]) == int(dk[i, 5]):
                            matched_known = True; break
                if matched_known:
                    n_tp_K += 1
                else:
                    matches_unknown = bool(iou_u.shape[1] and iou_u[i].max() >= self.iou_t)
                    if matches_unknown:
                        n_fp_KU += 1
                    else:
                        n_fp_K_only += 1
        denom_K = n_tp_K + n_fp_K_only
        denom_KU = n_tp_K + n_fp_K_only + n_fp_KU
        p_K = n_tp_K / max(denom_K, 1)
        p_KU = n_tp_K / max(denom_KU, 1)
        wi = (p_K - p_KU) / max(p_K, 1e-9)

        # ===== Print =====
        W = 78
        print('\n' + '=' * W)
        print(' OWOD Evaluation '.center(W, '='))
        print('=' * W)
        print(f'  known   = {sorted(self.known)}')
        print(f'  unknown = {sorted(self.unknown)}')
        print(f'  method  = {self.method}, tau = {self.tau}, IoU thr = {self.iou_t}')
        print('-' * W)
        print(f'  U-Recall (↑) = {u_recall:.4f}    '
              f'({n_unk_matched}/{n_unk_gt} unknown GT recovered as "unknown")')
        print(f'  WI       (↓) = {wi:.4f}        '
              f'(P_K={p_K:.4f} → P_KU={p_KU:.4f})')
        print(f'  A-OSE    (↓) = {a_ose}            '
              f'(known det matched to unknown GT @ IoU≥{self.iou_t})')
        print('=' * W + '\n')

        if save_dir is not None:
            try:
                from pathlib import Path
                sd = Path(save_dir)
                np.savez(sd / 'owod_eval.npz',
                         u_recall=u_recall, wi=wi, a_ose=a_ose,
                         p_K=p_K, p_KU=p_KU,
                         n_unknown_gt=n_unk_gt, n_unknown_matched=n_unk_matched,
                         known=np.array(sorted(self.known)),
                         unknown=np.array(sorted(self.unknown)),
                         method=self.method, tau=self.tau)
                print(f'  [saved] {sd / "owod_eval.npz"}')
            except Exception as e:
                print(f'  [save 경고] {e}')
        return dict(u_recall=u_recall, wi=wi, a_ose=a_ose,
                    p_K=p_K, p_KU=p_KU,
                    n_unknown_gt=n_unk_gt, n_unknown_matched=n_unk_matched)
