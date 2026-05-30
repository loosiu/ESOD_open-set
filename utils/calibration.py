"""
Calibration / reliability evaluation for ESOD+EDL  (contribution C1).

핵심 질문
---------
EDL vacuity가 detector의 raw confidence보다 '더 믿을 만한 신뢰도 신호'인가?
per-detection 단위로 — 모델이 출력한 박스 중 어느 것이 오검(FP)인지를
어느 신호가 더 잘 가려내는지 측정한다.

측정 항목
---------
  - ECE (Expected Calibration Error) : confidence ↔ 실제 정답률 일치도 (↓ better)
  - Reliability diagram               : 위 관계의 시각화
  - Failure detection AUROC           : 불확실도가 오검을 분리하는가 (↑ better)
  - Risk-Coverage AURC                : 불확실한 검출부터 버릴 때의 위험   (↓ better)

비교 신호
---------
  raw            = detector confidence (ESOD가 원래 주는 것 = baseline)
  conf·(1-vac)   = confidence를 vacuity로 할인 (EDL 기여 검증)
  vacuity        = EDL 인식 불확실도 (실패 검출용)

vacuity는 segmenter(ObjSeeker) 단계 신호 → 영역 불확실도임. 검출별 calibration
관점에서 raw confidence 대비 *추가 이득*이 있는지가 C1의 성패를 가른다.
"""

import numpy as np
import torch

__all__ = ['uncertainty_maps', 'CalibrationEval', 'OODEval']


# ──────────────────────────────────────────────────────────────────────────
# segmenter 출력 → per-pixel 신호 맵
# ──────────────────────────────────────────────────────────────────────────
def uncertainty_maps(p_seg):
    """model의 segmenter 출력 p_seg → per-pixel 신호 맵 dict.

    p_seg 형태
      [heat(1ch), edl(2ch), gate(1ch)]  : dual (calib-gating)
      [heat(1ch), edl(2ch)]             : dual + max
      [seg(2ch)]                        : EDL-only
    반환: {'vac','p_e','heat','fused'} 각 [B,H,W] float tensor, 또는 None
    """
    if not isinstance(p_seg, (list, tuple)) or len(p_seg) == 0:
        return None
    Fn = torch.nn.functional

    def _edl(edl):
        ev = Fn.softplus(edl.float())
        al = ev + 1.0
        S = al.sum(dim=1, keepdim=True)
        return al[:, 1:2] / S, 2.0 / S          # p_e (객체확률 ch1), vacuity

    if len(p_seg) == 3 and p_seg[0].shape[1] == 1 and p_seg[1].shape[1] == 2:
        heat = p_seg[0].float().sigmoid()
        p_e, vac = _edl(p_seg[1])
        from utils import edl_det as _edl_det_mod
        fused = _edl_det_mod.fuse_dual_seg(
            p_seg[0].float(), p_seg[1].float(), p_seg[2].float())[:, 0:1]
    elif len(p_seg) == 2 and p_seg[0].shape[1] == 1 and p_seg[1].shape[1] == 2:
        heat = p_seg[0].float().sigmoid()
        p_e, vac = _edl(p_seg[1])
        fused = torch.max(heat, vac)
    elif p_seg[0].shape[1] == 2:
        p_e, vac = _edl(p_seg[0])
        heat, fused = p_e, p_e
    else:
        return None                              # heat-only: EDL 신호 없음
    return {'vac': vac[:, 0], 'p_e': p_e[:, 0], 'heat': heat[:, 0], 'fused': fused[:, 0]}


def _sample(umap, boxes, stride):
    """umap [H,W] → boxes [N,4] xyxy(입력공간) footprint 평균. 반환 [N] np."""
    umap = umap.numpy() if isinstance(umap, torch.Tensor) else np.asarray(umap)
    H, W = umap.shape
    out = np.empty(len(boxes), np.float32)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        gx1 = int(np.clip(x1 / stride, 0, W - 1)); gx2 = int(np.clip(x2 / stride, 0, W - 1))
        gy1 = int(np.clip(y1 / stride, 0, H - 1)); gy2 = int(np.clip(y2 / stride, 0, H - 1))
        out[i] = umap[gy1:gy2 + 1, gx1:gx2 + 1].mean()
    return out


# ──────────────────────────────────────────────────────────────────────────
# calibration metrics
# ──────────────────────────────────────────────────────────────────────────
def _ece(conf, correct, n_bins=15):
    """Expected Calibration Error + per-bin (conf, acc, count)."""
    conf = np.asarray(conf, np.float64); correct = np.asarray(correct, np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N, table = 0.0, len(conf), []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            table.append((0.5 * (lo + hi), np.nan, np.nan, 0)); continue
        c_avg, a_avg = conf[m].mean(), correct[m].mean()
        ece += (cnt / N) * abs(c_avg - a_avg)
        table.append((0.5 * (lo + hi), c_avg, a_avg, cnt))
    return ece, table


def _auroc(score, label):
    """score 높을수록 label=1. tie-averaged rank 기반 AUROC."""
    score = np.asarray(score, np.float64); label = np.asarray(label, np.int64)
    n_pos = int(label.sum()); n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    s = score[order]
    ranks = np.empty(len(score), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _risk_coverage(uncertainty, correct):
    """불확실도 낮은 순 유지 → coverage별 risk(=error rate). 반환 (cov, risk, aurc)."""
    unc = np.asarray(uncertainty, np.float64)
    err = 1.0 - np.asarray(correct, np.float64)
    order = np.argsort(unc, kind='mergesort')      # 낮은 불확실도 먼저
    cum = np.cumsum(err[order])
    k = np.arange(1, len(err) + 1)
    risk = cum / k
    return k / len(err), risk, float(risk.mean())  # AURC = mean risk


# ──────────────────────────────────────────────────────────────────────────
# accumulator
# ──────────────────────────────────────────────────────────────────────────
class CalibrationEval:
    """per-detection (conf, vacuity, p_e, heat, fused, correct@.5) 누적 → 요약."""

    def __init__(self):
        self._conf, self._corr = [], []
        self._vac, self._pe, self._heat, self._fused = [], [], [], []

    def add_image(self, pred, correct, umaps, si, img_h):
        """pred [n,6] xyxy/conf/cls(입력공간), correct [n] bool@.5, umaps dict, si 배치인덱스."""
        if umaps is None or pred is None or len(pred) == 0:
            return
        boxes = pred[:, :4].detach().cpu().numpy()
        self._conf.append(pred[:, 4].detach().cpu().numpy().astype(np.float32))
        self._corr.append(correct.detach().cpu().numpy().astype(np.float32))
        vac = umaps['vac'][si].detach().cpu()
        stride = img_h / vac.shape[0]
        self._vac.append(_sample(vac, boxes, stride))
        self._pe.append(_sample(umaps['p_e'][si].detach().cpu(), boxes, stride))
        self._heat.append(_sample(umaps['heat'][si].detach().cpu(), boxes, stride))
        self._fused.append(_sample(umaps['fused'][si].detach().cpu(), boxes, stride))

    def summarize(self, save_dir=None, conf_floor=0.05, n_bins=15):
        if not self._conf:
            print('\n[Calibration] 수집된 예측 없음 — skip (EDL 모델인지 확인).')
            return None
        conf = np.concatenate(self._conf); corr = np.concatenate(self._corr)
        vac = np.concatenate(self._vac);   p_e = np.concatenate(self._pe)
        heat = np.concatenate(self._heat); fused = np.concatenate(self._fused)

        # 운영 집합: 실제로 쓰일 검출 (conf >= floor) — conf≈0 FP 홍수 제외
        keep = conf >= conf_floor
        n_all, n_op = len(conf), int(keep.sum())
        if n_op < 50:
            print('\n[Calibration] 운영 집합(conf>=%.2f)이 너무 작음(%d) — 전체로 평가.'
                  % (conf_floor, n_op))
            keep = np.ones_like(conf, bool); n_op = n_all
        conf, corr, vac = conf[keep], corr[keep], vac[keep]

        prec = float(corr.mean())                      # 예측 중 TP 비율
        cv = conf * (1.0 - vac)                        # vacuity-할인 confidence

        ece_raw, tbl_raw = _ece(conf, corr, n_bins)
        ece_cv,  tbl_cv  = _ece(cv,   corr, n_bins)

        # failure detection: uncertainty 높을수록 오검(error=1-correct)
        sigs = {'1 - raw conf': 1.0 - conf,
                'vacuity': vac,
                '1 - conf·(1-vac)': 1.0 - cv}
        rows = {}
        for name, unc in sigs.items():
            au = _auroc(unc, 1 - corr)
            cov, risk, aurc = _risk_coverage(unc, corr)
            rows[name] = (au, aurc, cov, risk)

        # ── print ──
        W = 82
        print('\n' + '=' * W)
        print(' Calibration / Reliability  (contribution C1)'.center(W))
        print('=' * W)
        print('  검출 수: 전체 %d / 운영집합 conf>=%.2f → %d' % (n_all, conf_floor, n_op))
        print('  운영집합 precision(TP비율): %.4f   vacuity: [%.3f, %.3f] mean %.3f'
              % (prec, vac.min(), vac.max(), vac.mean()))
        print('\n  -- ECE (confidence ↔ 정답률 일치도, ↓ better) --')
        print('    %-22s ECE = %.4f' % ('raw conf', ece_raw))
        print('    %-22s ECE = %.4f   (Δ %+.4f vs raw)' % ('conf·(1-vacuity)', ece_cv, ece_cv - ece_raw))
        print('\n  -- Failure detection (오검 분리력) --')
        print('    %-22s %8s %8s' % ('uncertainty 신호', 'AUROC↑', 'AURC↓'))
        for name, (au, aurc, _, _) in rows.items():
            print('    %-22s %8.4f %8.4f' % (name, au, aurc))

        # ── verdict ──
        au_raw = rows['1 - raw conf'][0]; au_cv = rows['1 - conf·(1-vac)'][0]
        au_vac = rows['vacuity'][0]
        ar_raw = rows['1 - raw conf'][1]; ar_cv = rows['1 - conf·(1-vac)'][1]
        print('\n  -- VERDICT --')
        better_ece = ece_cv < ece_raw - 1e-4
        better_fd = (au_cv > au_raw + 1e-3) or (ar_cv < ar_raw - 1e-3)
        if better_ece and better_fd:
            print('  ✔ vacuity 할인이 ECE·failure-detection 모두 개선 → C1 신호 있음.')
        elif better_ece or better_fd:
            print('  △ vacuity가 일부 지표만 개선 → C1 약함 (추가 분석 필요).')
        else:
            print('  ✘ vacuity가 raw conf 대비 추가 이득 없음 → 현 형태로는 C1 미성립.')
        print('    (vacuity 단독 AUROC %.4f vs 1-conf %.4f)' % (au_vac, au_raw))
        print('=' * W + '\n')

        # ── save ──
        if save_dir is not None:
            try:
                from pathlib import Path
                sd = Path(save_dir)
                np.savez(sd / 'calibration.npz',
                         conf=conf, correct=corr, vacuity=vac, p_e=p_e[keep],
                         heat=heat[keep], fused=fused[keep])
                _plot_reliability(tbl_raw, tbl_cv, ece_raw, ece_cv, sd / 'reliability_diagram.png')
                _plot_risk_coverage(rows, sd / 'risk_coverage.png')
                print('  [saved] %s , reliability_diagram.png , risk_coverage.png' %
                      (sd / 'calibration.npz'))
            except Exception as e:
                print('  [save 경고] %s' % e)
        return {'ece_raw': ece_raw, 'ece_cv': ece_cv,
                'auroc': {k: v[0] for k, v in rows.items()},
                'aurc': {k: v[1] for k, v in rows.items()}}


# ──────────────────────────────────────────────────────────────────────────
# OOD evaluation (held-out class)
# ──────────────────────────────────────────────────────────────────────────
class OODEval:
    """Held-out class를 OOD로 두고 GT box region에서 OOD score → AUROC.

    측정 신호 (모두 OOD에서 *값이 커야* 좋음):
      vacuity      : segmenter vacuity (region 평균)  ← EDL contribution
      msp          : 1 - max_conf in region          ← MSP baseline (Hendrycks ICLR'17)
      maxlogit_neg : -max_logit(conf) in region      ← MaxLogit baseline (ICML'20)
      combined     : (1 - max_conf) * vacuity        ← 단순 조합

    VOS (ICLR'22), OpenDet (CVPR'22), STUD (CVPR'22) 등이 사용하는 region-level OOD eval.
    held-out region에 detection이 없으면 max_conf=0 → MSP=1, 즉 baseline이 자동 보호됨 (fair).
    """

    def __init__(self, held_out_classes):
        self.held_out = set(int(c) for c in held_out_classes)
        self._records = []           # list of dict (ood, vacuity, msp, maxlogit_neg, combined)

    def add_image(self, gts_xyxy, pred, vac_map, img_h, det_vac_map=None):
        """
        gts_xyxy    : Tensor[M, 5+] (cls, x1, y1, x2, y2 ...) — 입력공간
        pred        : Tensor[N, 6] xyxy/conf/cls — 입력공간 (post-NMS)
        vac_map     : Tensor[H, W] — segmenter vacuity for this image
        img_h       : input image height (vacuity stride 계산용)
        det_vac_map : Tensor[Hd, Wd] — Detect parallel ev_pred vacuity (Phase 2)
        """
        if gts_xyxy is None or len(gts_xyxy) == 0:
            return
        gts = (gts_xyxy.detach().cpu().numpy() if isinstance(gts_xyxy, torch.Tensor)
               else np.asarray(gts_xyxy))
        if pred is not None and len(pred) > 0:
            pn = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
            pb = pn[:, :4]; pc = pn[:, 4]
        else:
            pb, pc = np.zeros((0, 4)), np.zeros((0,))
        if vac_map is not None:
            vm = (vac_map.detach().cpu().numpy() if isinstance(vac_map, torch.Tensor)
                  else np.asarray(vac_map))
            H, W = vm.shape
            stride = float(img_h) / float(H)
        else:
            vm, stride = None, None
        if det_vac_map is not None:
            dvm = (det_vac_map.detach().cpu().numpy() if isinstance(det_vac_map, torch.Tensor)
                   else np.asarray(det_vac_map))
            dH, dW = dvm.shape
            dstride = float(img_h) / float(dH)
        else:
            dvm, dstride = None, None

        for g in gts:
            cls = int(g[0])
            x1, y1, x2, y2 = float(g[1]), float(g[2]), float(g[3]), float(g[4])
            ood = 1 if cls in self.held_out else 0

            # segmenter vacuity in GT footprint
            if vm is not None:
                gx1 = int(np.clip(x1 / stride, 0, W - 1)); gx2 = int(np.clip(x2 / stride, 0, W - 1))
                gy1 = int(np.clip(y1 / stride, 0, H - 1)); gy2 = int(np.clip(y2 / stride, 0, H - 1))
                vac = float(vm[gy1:gy2 + 1, gx1:gx2 + 1].mean())
            else:
                vac = float('nan')

            # Detect parallel evidence vacuity in GT footprint
            if dvm is not None:
                dgx1 = int(np.clip(x1 / dstride, 0, dW - 1)); dgx2 = int(np.clip(x2 / dstride, 0, dW - 1))
                dgy1 = int(np.clip(y1 / dstride, 0, dH - 1)); dgy2 = int(np.clip(y2 / dstride, 0, dH - 1))
                det_vac = float(dvm[dgy1:dgy2 + 1, dgx1:dgx2 + 1].mean())
            else:
                det_vac = float('nan')

            # detector signals: max conf among detections overlapping the GT box (IoG > 0.3)
            if len(pb):
                ix1 = np.maximum(pb[:, 0], x1); iy1 = np.maximum(pb[:, 1], y1)
                ix2 = np.minimum(pb[:, 2], x2); iy2 = np.minimum(pb[:, 3], y2)
                inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                gt_area = max(1e-6, (x2 - x1) * (y2 - y1))
                m = (inter / gt_area) > 0.3
                max_conf = float(pc[m].max()) if m.any() else 0.0
            else:
                max_conf = 0.0
            max_conf = float(np.clip(max_conf, 1e-6, 1.0 - 1e-6))

            self._records.append({
                'ood': ood,
                'vacuity': vac,
                'det_vac': det_vac,
                'msp': 1.0 - max_conf,
                'maxlogit_neg': -np.log(max_conf / (1.0 - max_conf)),
                'combined': (1.0 - max_conf) * (vac if not np.isnan(vac) else 0.5),
                'det_combined': (1.0 - max_conf) * (det_vac if not np.isnan(det_vac) else 0.5),
            })

    def summarize(self, save_dir=None):
        if not self._records:
            print('\n[OODEval] no records.')
            return None
        oods = np.array([r['ood'] for r in self._records], np.int64)
        n_id, n_ood = int((oods == 0).sum()), int((oods == 1).sum())
        if n_ood == 0 or n_id == 0:
            print(f'\n[OODEval] no contrast: ID={n_id}, OOD={n_ood} — held_out_classes 설정 확인.')
            return None
        rows = {}
        for key in ('vacuity', 'det_vac', 'msp', 'maxlogit_neg', 'combined', 'det_combined'):
            sc = np.array([r[key] for r in self._records], np.float64)
            v = ~np.isnan(sc)
            rows[key] = _auroc(sc[v], oods[v]) if v.sum() >= 10 else float('nan')

        W = 72
        print('\n' + '=' * W)
        print((f' OOD Detection  (held-out classes: {sorted(self.held_out)})').center(W))
        print('=' * W)
        print(f'  GT boxes: ID={n_id}, OOD={n_ood}')
        print('\n  -- OOD-AUROC (↑ better) --')
        for k, au in rows.items():
            tag = '  (EDL seg)' if k == 'vacuity' else ('  (EDL det)' if k == 'det_vac' else '')
            print(f'    {k:<22} AUROC = {au:.4f}{tag}')

        au_v, au_msp = rows.get('vacuity', float('nan')), rows.get('msp', float('nan'))
        au_ml = rows.get('maxlogit_neg', float('nan'))
        best_base = max(x for x in (au_msp, au_ml) if not np.isnan(x))
        print('\n  -- VERDICT --')
        if not np.isnan(au_v):
            if au_v > best_base + 1e-3:
                print(f'  ✔ vacuity ({au_v:.4f}) > best baseline ({best_base:.4f}) → EDL OOD 신호 있음.')
            elif abs(au_v - best_base) <= 1e-3:
                print(f'  △ vacuity ({au_v:.4f}) ≈ best baseline ({best_base:.4f}) → TIE.')
            else:
                print(f'  ✘ vacuity ({au_v:.4f}) < best baseline ({best_base:.4f}) → EDL OOD 우위 없음.')
        print('=' * W + '\n')

        if save_dir is not None:
            try:
                from pathlib import Path
                sd = Path(save_dir)
                np.savez(sd / 'ood_eval.npz',
                         ood=oods,
                         vacuity=np.array([r['vacuity'] for r in self._records]),
                         det_vac=np.array([r['det_vac'] for r in self._records]),
                         msp=np.array([r['msp'] for r in self._records]),
                         maxlogit_neg=np.array([r['maxlogit_neg'] for r in self._records]),
                         combined=np.array([r['combined'] for r in self._records]),
                         det_combined=np.array([r['det_combined'] for r in self._records]))
                print(f'  [saved] {sd / "ood_eval.npz"}')
            except Exception as e:
                print(f'  [save 경고] {e}')
        return rows


# ──────────────────────────────────────────────────────────────────────────
# plots
# ──────────────────────────────────────────────────────────────────────────
def _plot_reliability(tbl_raw, tbl_cv, ece_raw, ece_cv, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, tbl, ece, title in [(axes[0], tbl_raw, ece_raw, 'raw conf'),
                                (axes[1], tbl_cv, ece_cv, 'conf·(1-vacuity)')]:
        xs = [t[1] for t in tbl if t[3] > 0]
        ys = [t[2] for t in tbl if t[3] > 0]
        ax.plot([0, 1], [0, 1], '--', color='gray', label='ideal')
        ax.plot(xs, ys, 'o-', color='C0')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('confidence'); ax.set_ylabel('accuracy')
        ax.set_title('%s   (ECE=%.4f)' % (title, ece))
        ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_risk_coverage(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (au, aurc, cov, risk) in rows.items():
        ax.plot(cov, risk, label='%s (AURC=%.4f)' % (name, aurc))
    ax.set_xlabel('coverage'); ax.set_ylabel('risk (error rate)')
    ax.set_title('Risk-Coverage  (불확실도 낮은 순 유지)')
    ax.legend(loc='upper left'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
