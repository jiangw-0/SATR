from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor


@dataclass
class SamConfigV6:
    ckpt: str
    model_type: str = "vit_h"
    device: str = "cuda"


@dataclass
class SamAttnPromptCfg:
    """Generic, attention-guided SAM prompt configuration (方案 1)."""

    # number of points
    n_pos: int = 8
    n_neg: int = 12
    min_dist: int = 18

    # attention sampling quantiles within the bbox
    pos_q: float = 0.85  # take points from top (1-pos_q) attention
    neg_q: float = 0.15  # take points from bottom neg_q attention

    # add some deterministic bbox-structure points
    add_center_point: bool = True
    add_grid_points: bool = True

    # if provided, treat other objects' attention peaks as negative hints
    n_other_neg: int = 6
    other_pos_q: float = 0.85

    # When segmenting *back* objects, we can pass an avoid_mask (union of front
    # objects) to avoid placing positive points on the foreground, and to sample
    # extra negative points from it.
    n_avoid_neg: int = 8
    avoid_dilate: int = 9  # pixels (odd int recommended); 0 disables


@dataclass
class SamGenericScoreCfg:
    """Mask scoring weights for generic mode (replaces shuttle prior)."""

    w_sam: float = 1.0
    w_pos: float = 0.20
    w_neg: float = 0.10
    w_attn: float = 0.70
    w_area: float = 0.20
    # Penalize overlap with an "avoid" mask (typically the union of *front* objects).
    # We also add a hard-reject threshold below (avoid_max_overlap).
    w_avoid: float = 0.90

    # Hard reject: if overlap(mask, avoid_mask) / area(mask) exceeds this value,
    # treat the candidate as invalid. This is crucial for overlapping objects
    # (e.g., a foreground object sitting on a background object), where SAM can
    # otherwise "snap" to the foreground.
    avoid_max_overlap: float = 0.30

    # area sanity range within bbox
    area_min_ratio: float = 0.004
    area_max_ratio: float = 0.96


def _to_box_xyxy_pixels(bbox_y1y2x1x2: List[float], H: int, W: int, pad: float = 0.0) -> np.ndarray:
    y1, y2, x1, x2 = [float(v) for v in bbox_y1y2x1x2]
    # normalized?
    if max(abs(y1), abs(y2), abs(x1), abs(x2)) <= 2.0:
        x1, x2 = x1 * W, x2 * W
        y1, y2 = y1 * H, y2 * H

    if pad > 0:
        bw, bh = (x2 - x1), (y2 - y1)
        px, py = pad * bw, pad * bh
        x1 -= px
        x2 += px
        y1 -= py
        y2 += py

    x1 = float(np.clip(x1, 0, W - 2))
    y1 = float(np.clip(y1, 0, H - 2))
    x2 = float(np.clip(x2, x1 + 2, W))
    y2 = float(np.clip(y2, y1 + 2, H))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def soften_mask(mask01: np.ndarray, k: int = 21, sigma: float = 7.0) -> np.ndarray:
    m = mask01.astype(np.float32)
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=sigma)
    m = m / (float(m.max()) + 1e-6)
    return np.clip(m, 0.0, 1.0)


def _nms_pick_points(
    xs: np.ndarray,
    ys: np.ndarray,
    scores: np.ndarray,
    n: int,
    min_dist: int,
) -> List[Tuple[float, float]]:
    """Pick up to n points with NMS-like min-distance constraint.

    Input arrays must already be sorted by score (descending for positive,
    ascending for negative).
    """
    picked: List[Tuple[float, float]] = []
    md2 = float(min_dist * min_dist)
    for x, y in zip(xs.tolist(), ys.tolist()):
        if len(picked) >= n:
            break
        ok = True
        for px, py in picked:
            if (x - px) ** 2 + (y - py) ** 2 < md2:
                ok = False
                break
        if ok:
            picked.append((float(x), float(y)))
    return picked


def _bbox_to_int(box_xyxy: np.ndarray, H: int, W: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = int(np.clip(round(x1), 0, W - 2))
    y1 = int(np.clip(round(y1), 0, H - 2))
    x2 = int(np.clip(round(x2), x1 + 2, W))
    y2 = int(np.clip(round(y2), y1 + 2, H))
    return x1, y1, x2, y2



def _clamp_bbox_xyxy(bbox_xyxy, W: int, H: int):
    """Clamp bbox (x1,y1,x2,y2) into image size and return (x1,x2,y1,y2) ints.

    Some postprocess utilities historically used this return order.
    """
    if bbox_xyxy is None:
        return 0, 0, 0, 0
    try:
        # bbox may be list/tuple/np array/tensor
        b = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if b.size >= 4:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        else:
            return 0, 0, 0, 0
    except Exception:
        return 0, 0, 0, 0

    # reuse the same rounding/clamping policy as _bbox_to_int
    x1i, y1i, x2i, y2i = _bbox_to_int(np.array([x1, y1, x2, y2], dtype=np.float32), H, W)
    # return legacy order: x1, x2, y1, y2
    return x1i, x2i, y1i, y2i

def _attn_crop(attn_map: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    if attn_map is None:
        return None
    H, W = attn_map.shape[:2]
    x1, y1, x2, y2 = _bbox_to_int(box_xyxy, H, W)
    crop = attn_map[y1:y2, x1:x2].astype(np.float32)
    if crop.size == 0:
        return None
    mx = float(crop.max())
    if mx > 1e-6:
        crop = crop / mx
    return np.clip(crop, 0.0, 1.0)


def _largest_cc(bin01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return bin01.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return (labels == k).astype(np.uint8)


def _fill_holes(bin01: np.ndarray) -> np.ndarray:
    bin01 = bin01.astype(np.uint8)
    h, w = bin01.shape
    flood = bin01.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return (bin01 | holes).astype(np.uint8)


def _mask_bbox(bin01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(bin01 > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _mask_center(bin01: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(bin01 > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def _width_profile_score(bin01: np.ndarray) -> float:
    ys, xs = np.where(bin01 > 0)
    if len(xs) < 80:
        return 0.0

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]           # major axis
    u = np.array([-v[1], v[0]], dtype=np.float32)     # orth axis

    t = pts @ v
    s = pts @ u
    tmin, tmax = float(t.min()), float(t.max())
    if tmax - tmin < 5:
        return 0.0

    bins = 24
    edges = np.linspace(tmin, tmax, bins + 1)
    widths = []
    counts = []
    for i in range(bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        cnt = int(m.sum())
        counts.append(cnt)
        if cnt < 15:
            widths.append(0.0)
        else:
            widths.append(float(s[m].max() - s[m].min()))

    widths = np.array(widths, dtype=np.float32)
    counts = np.array(counts, dtype=np.int32)
    if widths.max() <= 1e-6:
        return 0.0

    wnorm = widths / (widths.max() + 1e-6)
    thin = (wnorm < 0.18) & (counts > 10)
    thin_ratio = float(thin.sum()) / float(bins)
    return float(max(0.0, 1.0 - 1.8 * thin_ratio))


def _cut_thin_tail(bin01: np.ndarray, width_ratio: float = 0.30) -> np.ndarray:
    ys, xs = np.where(bin01 > 0)
    if len(xs) < 100:
        return bin01.astype(np.uint8)

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0, keepdims=True)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    vals, vecs = np.linalg.eig(cov)
    v = vecs[:, np.argmax(vals)]
    v = v / (np.linalg.norm(v) + 1e-6)
    u = np.array([-v[1], v[0]], dtype=np.float32)

    t = pts0 @ v
    s = pts0 @ u

    tmin, tmax = t.min(), t.max()
    bins = 32
    edges = np.linspace(tmin, tmax, bins + 1)

    widths = []
    for i in range(bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() < 5:
            widths.append(0.0)
        else:
            widths.append(float(np.percentile(s[m], 95) - np.percentile(s[m], 5)))

    widths = np.array(widths, dtype=np.float32)
    wmax = float(widths.max()) + 1e-6
    thr = float(width_ratio) * wmax

    # 连续细段 -> 判定为“杆子”
    left = 0
    while left < bins and widths[left] < thr:
        left += 1
    right = bins - 1
    while right >= 0 and widths[right] < thr:
        right -= 1

    keep_t1 = edges[max(left - 1, 0)]
    keep_t2 = edges[min(right + 2, bins)]

    keep = (t >= keep_t1) & (t <= keep_t2)
    out = np.zeros_like(bin01, dtype=np.uint8)
    out[ys[keep], xs[keep]] = 1
    return _largest_cc(out)


def _remove_base_stick(
    m: np.ndarray,
    thickness_ratio: float = 0.55,
    extend_ratio: float = 0.12,
    min_removed_area: int = 30,
) -> np.ndarray:
    """Remove a small rod/stick-like protrusion extending out of the cork/base.

    - DT 找“最厚点”（多半是软木底座中心）
    - 以 mask 质心 -> 最厚点 的方向作为“底座方向”
    - 删除“更往底座方向延伸”且“明显更薄”的区域（典型：底部伸出一截杆/把手）
    """
    if m is None or int(m.sum()) == 0:
        return m

    m_u8 = (m > 0).astype(np.uint8)
    dt = cv2.distanceTransform(m_u8, cv2.DIST_L2, 5)
    max_dt = float(dt.max())
    if max_dt < 2.0:
        return m

    # thickest point
    flat_idx = int(dt.argmax())
    yb, xb = divmod(flat_idx, dt.shape[1])
    base_dt = float(dt[yb, xb])

    cy, cx = _mask_center(m_u8)
    vx, vy = float(xb - cx), float(yb - cy)
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-6:
        return m
    vx, vy = vx / norm, vy / norm

    ys, xs = np.nonzero(m_u8)
    proj = (xs - cx) * vx + (ys - cy) * vy
    if proj.size == 0:
        return m

    proj_min, proj_max = float(proj.min()), float(proj.max())
    base_proj = (xb - cx) * vx + (yb - cy) * vy
    span = max(1e-6, proj_max - proj_min)

    beyond = base_proj + float(extend_ratio) * span
    thin_th = max(1.0, float(thickness_ratio) * base_dt)

    remove = np.zeros_like(m_u8)
    rm = (proj > beyond) & (dt[ys, xs] < thin_th)
    if rm.any():
        ry = ys[rm]
        rx = xs[rm]
        remove[ry, rx] = 1
        if int(remove.sum()) >= int(min_removed_area):
            keep = m_u8.copy()
            keep[remove > 0] = 0
            keep = _largest_cc(keep)
            keep = _fill_holes(keep)
            return keep

    return m


def shuttle_shape_score(bin01: np.ndarray, box_xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    crop = bin01[y1:y2, x1:x2].astype(np.uint8)
    area = float(crop.sum())
    if area < 50:
        return 0.0

    bbox_area = float((y2 - y1) * (x2 - x1))
    ar = area / (bbox_area + 1e-6)
    area_score = np.exp(-((ar - 0.22) ** 2) / (2 * 0.18 * 0.18))

    bb = _mask_bbox(crop)
    if bb is None:
        return 0.0
    mx1, my1, mx2, my2 = bb
    mw, mh = (mx2 - mx1), (my2 - my1)
    aspect = max(mw, mh) / (min(mw, mh) + 1e-6)
    aspect_score = 1.0 / (1.0 + max(0.0, aspect - 3.0) ** 2)

    cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    per = float(cv2.arcLength(max(cnts, key=cv2.contourArea), True))
    comp = (per * per) / (4.0 * np.pi * area + 1e-6)
    comp_score = 1.0 / (1.0 + max(0.0, comp - 6.0) ** 2)

    rod_score = _width_profile_score(crop)
    return float(area_score * aspect_score * comp_score * rod_score)


def _add_base_prior_adaptive(bin01: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    # 在“粗的一端”补一个小椭圆，避免底座缺失；不假设方向（更稳）
    ys, xs = np.where(bin01 > 0)
    if len(xs) < 200:
        return bin01

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0, keepdims=True)
    pts0 = pts - mu
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]       # major axis
    u = np.array([-v[1], v[0]], dtype=np.float32)

    t = pts0 @ v
    s = pts0 @ u
    tmin, tmax = float(t.min()), float(t.max())
    if tmax - tmin < 10:
        return bin01

    rng = (tmax - tmin)
    band = 0.18 * rng
    left = t <= (tmin + band)
    right = t >= (tmax - band)
    if left.sum() < 50 or right.sum() < 50:
        return bin01

    w_left = float(np.percentile(s[left], 95) - np.percentile(s[left], 5))
    w_right = float(np.percentile(s[right], 95) - np.percentile(s[right], 5))
    use_left = (w_left >= w_right)

    end_pts = pts[left] if use_left else pts[right]
    cx, cy = end_pts.mean(axis=0)

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = (x2 - x1), (y2 - y1)
    rx = int(np.clip(0.10 * bw, 18, 70))
    ry = int(np.clip(0.10 * bh, 18, 70))

    out = bin01.copy().astype(np.uint8)
    cv2.ellipse(out, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 1, -1)
    return out


class SamMaskerV6:
    def __init__(self, cfg: SamConfigV6):
        if cfg.device == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
        self.cfg = cfg
        sam = sam_model_registry[cfg.model_type](checkpoint=cfg.ckpt)
        sam.to(device=cfg.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def set_image_bgr(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    def predict_best(
        self,
        box_xyxy: np.ndarray,
        pos_xy: List[Tuple[float, float]],
        neg_xy: List[Tuple[float, float]],
        mode: str = "generic",
        multimask: bool = True,
        attn_map: Optional[np.ndarray] = None,
        avoid_mask: Optional[np.ndarray] = None,
        score_cfg: Optional[SamGenericScoreCfg] = None,
    ) -> Tuple[Optional[np.ndarray], float]:
        pts = np.array(pos_xy + neg_xy, dtype=np.float32)
        lbl = np.array([1] * len(pos_xy) + [0] * len(neg_xy), dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            box=box_xyxy.astype(np.float32),
            multimask_output=multimask,
        )
        if masks is None:
            return None, -1e9

        best = 0
        best_score = -1e9

        if score_cfg is None:
            score_cfg = SamGenericScoreCfg()

        # pre-crop attn / avoid for speed
        attn_crop = _attn_crop(attn_map, box_xyxy) if attn_map is not None else None
        if avoid_mask is not None:
            Hm, Wm = masks.shape[1], masks.shape[2]
            x1, y1, x2, y2 = _bbox_to_int(box_xyxy, Hm, Wm)
            avoid_crop = (avoid_mask[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        else:
            avoid_crop = None
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.uint8)

            pos_ok = 0.0
            for (x, y) in pos_xy:
                xi = int(np.clip(round(x), 0, m.shape[1] - 1))
                yi = int(np.clip(round(y), 0, m.shape[0] - 1))
                pos_ok += float(m[yi, xi] > 0)
            pos_ok /= max(1, len(pos_xy))

            neg_ok = 0.0
            for (x, y) in neg_xy:
                xi = int(np.clip(round(x), 0, m.shape[1] - 1))
                yi = int(np.clip(round(y), 0, m.shape[0] - 1))
                neg_ok += float(m[yi, xi] == 0)
            neg_ok /= max(1, len(neg_xy))

            # ----- extra priors -----
            if mode == "shuttle":
                # legacy: keep shuttle prior for the badminton demo
                shape = shuttle_shape_score(m, box_xyxy)
                s = float(scores[i]) * (0.35 + 0.65 * pos_ok) * (0.40 + 0.60 * neg_ok) * (0.25 + 0.75 * shape)
            else:
                # generic scoring (方案 1): use attention alignment + overlap penalty
                Hm, Wm = m.shape
                x1, y1, x2, y2 = _bbox_to_int(box_xyxy, Hm, Wm)
                mc = m[y1:y2, x1:x2]

                area = float(mc.sum())
                box_area = float((y2 - y1) * (x2 - x1))
                ar = area / (box_area + 1e-6)
                # area_ok in [0,1]
                if ar < score_cfg.area_min_ratio:
                    area_ok = float(ar / max(score_cfg.area_min_ratio, 1e-6))
                elif ar > score_cfg.area_max_ratio:
                    area_ok = float((1.0 - ar) / max(1.0 - score_cfg.area_max_ratio, 1e-6))
                else:
                    area_ok = 1.0
                area_ok = float(np.clip(area_ok, 0.0, 1.0))

                # attention alignment in [0,1]
                attn_score = 0.0
                if attn_crop is not None and attn_crop.shape == mc.shape and area > 5:
                    a = attn_crop
                    a_sum = float(a.sum()) + 1e-6
                    attn_mass = float((a * mc.astype(np.float32)).sum()) / a_sum
                    attn_in = float(a[mc > 0].mean()) if int(mc.sum()) > 0 else 0.0
                    attn_out = float(a[mc == 0].mean()) if int((mc == 0).sum()) > 0 else 0.0
                    attn_sep = float(np.clip(attn_in - attn_out, 0.0, 1.0))
                    attn_score = 0.55 * attn_mass + 0.45 * attn_sep

                # overlap penalty
                overlap = 0.0
                hard_pen = 0.0
                if avoid_crop is not None and avoid_crop.shape == mc.shape and area > 5:
                    overlap = float(((mc > 0) & (avoid_crop > 0)).sum()) / (area + 1e-6)
                    overlap = float(np.clip(overlap, 0.0, 1.0))

                    # hard-reject style penalty when overlap is too large
                    thr = float(getattr(score_cfg, "avoid_max_overlap", 1.0))
                    if 0.0 < thr < 1.0 and overlap > thr:
                        # This penalty dominates all other terms (whose scale is ~[0, 2]).
                        hard_pen = 100.0 * float(overlap - thr) / max(1e-6, 1.0 - thr)

                # final weighted sum
                s = (
                    score_cfg.w_sam * float(scores[i])
                    + score_cfg.w_pos * float(pos_ok)
                    + score_cfg.w_neg * float(neg_ok)
                    + score_cfg.w_attn * float(attn_score)
                    + score_cfg.w_area * float(area_ok)
                    - score_cfg.w_avoid * float(overlap)
                    - hard_pen
                )

            if s > best_score:
                best_score = s
                best = i

        return masks[best].astype(np.uint8), float(best_score)

    # =====================
    # Generic attention-guided prompting (方案 1)
    # =====================

    @staticmethod
    def points_from_attn(
        box_xyxy: np.ndarray,
        attn_map: Optional[np.ndarray],
        other_attn_maps: Optional[List[np.ndarray]] = None,
        cfg: Optional[SamAttnPromptCfg] = None,
        extra_pos: Optional[List[Tuple[float, float]]] = None,
        extra_neg: Optional[List[Tuple[float, float]]] = None,
        avoid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Build SAM (pos, neg) points for a bbox.

        If attn_map is provided, sample positive points from high-attention
        regions and negative points from low-attention / other-objects.
        """
        if cfg is None:
            cfg = SamAttnPromptCfg()

        # Prepare an optional "avoid" mask (typically: union of already-segmented
        # front objects). This makes point prompting more robust in heavy
        # overlaps: we avoid placing positive points on the foreground and add
        # some negative points on it.
        avoid_full: Optional[np.ndarray] = None
        if avoid_mask is not None:
            # Choose a target size consistent with attention map if possible.
            if attn_map is not None:
                th, tw = int(attn_map.shape[0]), int(attn_map.shape[1])
            else:
                th, tw = int(avoid_mask.shape[0]), int(avoid_mask.shape[1])
            a = avoid_mask
            if a.shape[0] != th or a.shape[1] != tw:
                a = cv2.resize(a.astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST)
            avoid_full = (a > 0.5).astype(np.uint8)
            if getattr(cfg, "avoid_dilate", 0) and int(cfg.avoid_dilate) > 1:
                k = int(cfg.avoid_dilate)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                avoid_full = cv2.dilate(avoid_full, kernel, iterations=1)

        def _in_avoid(x: float, y: float) -> bool:
            if avoid_full is None:
                return False
            Hm, Wm = avoid_full.shape[:2]
            xi = int(np.clip(round(x), 0, Wm - 1))
            yi = int(np.clip(round(y), 0, Hm - 1))
            return bool(avoid_full[yi, xi] > 0)

        def _sample_from_bin(bin_mask: np.ndarray, n: int, min_dist: int) -> List[Tuple[float, float]]:
            """Deterministic-ish sampling of points from a binary mask (crop coords)."""
            if bin_mask is None or n <= 0:
                return []
            ys, xs = np.where(bin_mask > 0)
            if xs.size == 0:
                return []
            stride = max(1, int(xs.size // max(n * 8, 1)))
            cand = list(zip(xs[::stride], ys[::stride]))
            if not cand:
                cand = list(zip(xs, ys))
            picked: List[Tuple[float, float]] = []
            for (x, y) in cand:
                if len(picked) >= n:
                    break
                ok = True
                for (px, py) in picked:
                    if (x - px) ** 2 + (y - py) ** 2 < float(min_dist) ** 2:
                        ok = False
                        break
                if ok:
                    picked.append((float(x), float(y)))
            # pad
            i = 0
            while len(picked) < n and i < len(cand):
                x, y = cand[i]
                picked.append((float(x), float(y)))
                i += 1
            return picked[:n]

        # base geometry (deterministic)
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = (x2 - x1), (y2 - y1)

        pos: List[Tuple[float, float]] = []
        neg: List[Tuple[float, float]] = []

        # Always include bbox center as a robust positive hint.
        # But if the center lies in avoid region (common when foreground sits on
        # top of the background), skip it.
        if cfg.add_center_point and not _in_avoid(cx, cy):
            pos.append((cx, cy))

        m = max(2.0, 0.06 * min(w, h))
        neg_corners = [
            (x1 + m, y1 + m),
            (x2 - m, y1 + m),
            (x1 + m, y2 - m),
            (x2 - m, y2 - m),
        ]
        neg_edges = [
            (cx, y1 + m),
            (cx, y2 - m),
            (x1 + m, cy),
            (x2 - m, cy),
        ]

        # attention-guided sampling
        neg_attn: List[Tuple[float, float]] = []
        if attn_map is not None:
            H, W = attn_map.shape[:2]
            xi1, yi1, xi2, yi2 = _bbox_to_int(box_xyxy, H, W)
            crop = _attn_crop(attn_map, box_xyxy)
            if crop is not None and crop.size > 0:
                flat = crop.reshape(-1)
                if flat.size > 0:
                    # positive: top quantile
                    thr_pos = float(np.quantile(flat, cfg.pos_q))
                    yy, xx = np.where(crop >= thr_pos)
                    if yy.size > 0:
                        sc = crop[yy, xx]
                        order = np.argsort(-sc)  # desc
                        yy = yy[order] + yi1
                        xx = xx[order] + xi1
                        pos_attn = _nms_pick_points(xx, yy, sc[order], n=cfg.n_pos * 2, min_dist=cfg.min_dist)
                        # filter out avoid region
                        pos += [(x, y) for (x, y) in pos_attn if not _in_avoid(x, y)]

                    # negative: bottom quantile
                    thr_neg = float(np.quantile(flat, cfg.neg_q))
                    yy, xx = np.where(crop <= thr_neg)
                    if yy.size > 0:
                        sc = crop[yy, xx]
                        order = np.argsort(sc)  # asc
                        yy = yy[order] + yi1
                        xx = xx[order] + xi1
                        neg_attn = _nms_pick_points(xx, yy, sc[order], n=cfg.n_neg, min_dist=cfg.min_dist)

        # deterministic fallback positives
        if cfg.add_grid_points:
            grid = [
                (x1 + 0.35 * w, y1 + 0.35 * h),
                (x1 + 0.65 * w, y1 + 0.35 * h),
                (x1 + 0.35 * w, y1 + 0.65 * h),
                (x1 + 0.65 * w, y1 + 0.65 * h),
            ]
            pos += [(x, y) for (x, y) in grid if not _in_avoid(x, y)]

        # other objects' attention -> extra negative points
        neg_other: List[Tuple[float, float]] = []
        if other_attn_maps is not None and cfg.n_other_neg > 0:
            for om in other_attn_maps:
                if om is None:
                    continue
                H, W = om.shape[:2]
                xi1, yi1, xi2, yi2 = _bbox_to_int(box_xyxy, H, W)
                crop = _attn_crop(om, box_xyxy)
                if crop is None or crop.size == 0:
                    continue
                flat = crop.reshape(-1)
                if flat.size == 0:
                    continue
                thr = float(np.quantile(flat, cfg.other_pos_q))
                yy, xx = np.where(crop >= thr)
                if yy.size == 0:
                    continue
                sc = crop[yy, xx]
                order = np.argsort(-sc)
                yy = yy[order] + yi1
                xx = xx[order] + xi1
                neg_other += _nms_pick_points(xx, yy, sc[order], n=cfg.n_other_neg, min_dist=cfg.min_dist)

        # Assemble negatives with priority:
        #   1) bbox corners (stability)
        #   2) avoid-mask points (separation from foreground)
        #   2) other objects' attention peaks (separation)
        #   3) low-attention points of current object (separation)
        #   4) bbox edge midpoints (stability)
        neg = []
        neg += neg_corners
        # extra negatives sampled from avoid_mask within bbox
        if avoid_full is not None and getattr(cfg, "n_avoid_neg", 0) > 0:
            Hm, Wm = avoid_full.shape[:2]
            xi1, yi1, xi2, yi2 = _bbox_to_int(box_xyxy, Hm, Wm)
            avoid_crop = avoid_full[yi1:yi2, xi1:xi2]
            neg_avoid = _sample_from_bin(avoid_crop, n=int(cfg.n_avoid_neg), min_dist=int(cfg.min_dist))
            neg += [(float(xi1 + x), float(yi1 + y)) for (x, y) in neg_avoid]
        neg += neg_other
        neg += neg_attn
        neg += neg_edges

        if extra_pos:
            pos += list(extra_pos)
        if extra_neg:
            neg += list(extra_neg)

        # de-duplicate a bit (keep order)
        def _dedup(pts: List[Tuple[float, float]]):
            out = []
            seen = set()
            for (x, y) in pts:
                k = (int(round(x)), int(round(y)))
                if k in seen:
                    continue
                seen.add(k)
                out.append((float(x), float(y)))
            return out

        pos = _dedup(pos)
        # If we used an avoid mask and ended up with no positive points,
        # pick the point farthest away from avoid within the bbox.
        if len(pos) == 0 and avoid_full is not None:
            Hm, Wm = avoid_full.shape[:2]
            xi1, yi1, xi2, yi2 = _bbox_to_int(box_xyxy, Hm, Wm)
            avoid_crop = avoid_full[yi1:yi2, xi1:xi2]
            free = (avoid_crop == 0).astype(np.uint8)
            if int(free.sum()) > 0:
                dt = cv2.distanceTransform(free, cv2.DIST_L2, 3)
                yy, xx = np.unravel_index(int(dt.argmax()), dt.shape)
                pos.append((float(xi1 + xx), float(yi1 + yy)))
            else:
                pos.append((float(cx), float(cy)))

        pos = pos[: cfg.n_pos]
        neg = _dedup(neg)[: cfg.n_neg]
        return pos, neg

    @staticmethod
    def points_shuttle(
        box_xyxy: np.ndarray,
        extra_neg: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        w, h = (x2 - x1), (y2 - y1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        pos = [
            (cx, cy),
            (cx, y1 + 0.35 * h),
            (cx, y1 + 0.70 * h),
            (x1 + 0.40 * w, cy),
            (x1 + 0.60 * w, cy),
        ]

        m = max(2.0, 0.06 * min(w, h))
        neg = [
            (x1 + m, y1 + m),
            (x2 - m, y1 + m),
            (x1 + m, y2 - m),
            (x2 - m, y2 - m),
            (cx, y1 + m),
            (cx, y2 - m),
            (x1 + m, cy),
            (x2 - m, cy),
        ]

        if extra_neg:
            neg += list(extra_neg)

        return pos, neg

    @staticmethod
    def points_racket(box_xyxy: np.ndarray) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        w, h = (x2 - x1), (y2 - y1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        pos = [(cx, cy), (x1 + 0.35 * w, y1 + 0.55 * h), (x1 + 0.65 * w, y1 + 0.45 * h)]
        m = max(2.0, 0.06 * min(w, h))
        neg = [(x1 + m, y1 + m), (x2 - m, y1 + m), (x1 + m, y2 - m), (x2 - m, y2 - m)]
        return pos, neg

    @staticmethod
    def handle_mask_from_racket(
        racket_mask_bin: np.ndarray,
        racket_box_xyxy: np.ndarray,
        handle_ratio: float = 0.55,
        dilate_k: int = 19,
        dist_ratio: float = 0.06,
        edge_ratio: float = 0.12,
    ) -> Optional[np.ndarray]:
        """Infer handle mask from racket mask.

        优先：DT 找“更薄的一侧”作为把手侧，在薄区域里取连接到边缘的 CC；
        失败回退：x-cut (handle_ratio)。
        """
        if racket_mask_bin is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in racket_box_xyxy]
        rm = np.zeros_like(racket_mask_bin, dtype=np.uint8)
        rm[y1:y2, x1:x2] = (racket_mask_bin[y1:y2, x1:x2] > 0).astype(np.uint8)
        rm = _largest_cc(rm)
        if rm.sum() == 0:
            return None

        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        band = max(6, int(edge_ratio * bw))

        dt = cv2.distanceTransform(rm, cv2.DIST_L2, 5)

        left_patch = rm[y1:y2, x1:min(x2, x1 + band)] > 0
        right_patch = rm[y1:y2, max(x1, x2 - band):x2] > 0
        left_vals = dt[y1:y2, x1:min(x2, x1 + band)][left_patch]
        right_vals = dt[y1:y2, max(x1, x2 - band):x2][right_patch]

        def _med(a: np.ndarray) -> float:
            return float(np.median(a)) if a is not None and a.size > 0 else 1e9

        side = "left" if _med(left_vals) <= _med(right_vals) else "right"

        th = max(2.5, float(dist_ratio) * float(min(bw, bh)))
        thin = ((rm > 0) & (dt <= th)).astype(np.uint8)

        touch = np.zeros_like(thin)
        if side == "left":
            touch[y1:y2, x1:min(x2, x1 + band)] = 1
        else:
            touch[y1:y2, max(x1, x2 - band):x2] = 1

        touch = (thin > 0) & (touch > 0)

        handle = None
        if touch.sum() > 0:
            num, lab = cv2.connectedComponents(thin)
            keep = np.zeros_like(thin)
            hit = set(int(v) for v in np.unique(lab[touch]))
            hit.discard(0)
            for lbl in hit:
                keep[lab == lbl] = 1
            if keep.sum() > 0:
                handle = keep

        if handle is None or handle.sum() == 0:
            cut = int(x1 + float(handle_ratio) * bw)
            handle = np.zeros_like(rm, dtype=np.uint8)
            handle[y1:y2, x1:cut] = rm[y1:y2, x1:cut]
            handle = _largest_cc(handle)

        if dilate_k % 2 == 0:
            dilate_k += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        handle = cv2.dilate(handle.astype(np.uint8), k, iterations=1)
        return handle

    @staticmethod
    def postprocess_shuttle(
        shuttle_mask_bin: np.ndarray,
        shuttle_box_xyxy: np.ndarray,
        handle_mask_bin: Optional[np.ndarray] = None,
        close_k: int = 17,
        dilate_k: int = 13,
        feather_k: int = 21,
        feather_sigma: float = 7.0,
        add_base_prior: bool = True,
        remove_base_stick: bool = True,
    ) -> Optional[np.ndarray]:
        if shuttle_mask_bin is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in shuttle_box_xyxy]
        m = np.zeros_like(shuttle_mask_bin, dtype=np.uint8)
        m[y1:y2, x1:x2] = (shuttle_mask_bin[y1:y2, x1:x2] > 0).astype(np.uint8)

        m = _largest_cc(m)
        m = _fill_holes(m)

        # 0) Early remove handle overlap (避免底部扩展直接粘把手)
        if handle_mask_bin is not None:
            hm = (handle_mask_bin > 0).astype(np.uint8)
            mm = m.copy()
            mm[hm > 0] = 0
            if mm.sum() > max(200, int(0.12 * m.sum())):
                m = mm

        # 1) close
        if close_k % 2 == 0:
            close_k += 1
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k1, iterations=1)

        # 2) small dilate (别太大，避免桥接到球拍)
        if dilate_k % 2 == 0:
            dilate_k += 1
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        m = cv2.dilate(m, k2, iterations=1)

        # 3) remove handle again (dilate 后容易又连上)
        if handle_mask_bin is not None:
            hm = (handle_mask_bin > 0).astype(np.uint8)
            m[hm > 0] = 0
            m = _largest_cc(m)
            m = _fill_holes(m)

        # 4) cut thin tail (细长杆)
        m = _cut_thin_tail(m, width_ratio=0.34)

        # 5) anti-stick (粗一点的小杆)
        if remove_base_stick:
            m2 = _remove_base_stick(m, thickness_ratio=0.55, extend_ratio=0.12)
            if m2 is not None and m2.sum() > 0:
                m = m2

        # 6) base prior (保守补底座)，然后再切一次
        if add_base_prior:
            m = _add_base_prior_adaptive(m, shuttle_box_xyxy)

        if handle_mask_bin is not None:
            hm = (handle_mask_bin > 0).astype(np.uint8)
            m[hm > 0] = 0

        m = _largest_cc(m)

        # final tighten
        m = _cut_thin_tail(m, width_ratio=0.30)
        if remove_base_stick:
            m2 = _remove_base_stick(m, thickness_ratio=0.58, extend_ratio=0.10)
            if m2 is not None and m2.sum() > 0:
                m = m2

        m = _largest_cc(m)
        m_soft = soften_mask(m.astype(np.float32), k=feather_k, sigma=feather_sigma)
        return m_soft

    @staticmethod
    def postprocess_generic(
        mask_bin: np.ndarray,
        box_xyxy: np.ndarray,
        close_k: int = 11,
        dilate_k: int = 0,
        feather_k: int = 21,
        feather_sigma: float = 7.0,
        keep_largest: bool = True,
        fill_holes: bool = True,
    ) -> Optional[np.ndarray]:
        """A conservative, category-agnostic postprocess.

        - crop to bbox
        - keep largest CC
        - optional close/dilate
        - soften edges
        """
        if mask_bin is None:
            return None
        H, W = mask_bin.shape[:2]
        x1, y1, x2, y2 = _bbox_to_int(box_xyxy, H, W)
        m = np.zeros((H, W), dtype=np.uint8)
        m[y1:y2, x1:x2] = (mask_bin[y1:y2, x1:x2] > 0).astype(np.uint8)

        if keep_largest:
            m = _largest_cc(m)
        if fill_holes:
            m = _fill_holes(m)

        if close_k and close_k > 1:
            if close_k % 2 == 0:
                close_k += 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

        if dilate_k and dilate_k > 1:
            if dilate_k % 2 == 0:
                dilate_k += 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            m = cv2.dilate(m, k, iterations=1)

        if keep_largest:
            m = _largest_cc(m)
        if fill_holes:
            m = _fill_holes(m)

        return soften_mask(m.astype(np.float32), k=feather_k, sigma=feather_sigma)


def postprocess_thin_edges(
    mask_bin: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    image_bgr: Optional[np.ndarray] = None,
    attn_map: Optional[np.ndarray] = None,
    attn_thresh: float = 0.25,
    attn_dilate_k: int = 0,
    canny1: int = 50,
    canny2: int = 150,
    close_k: int = 0,
    dilate_k: int = 0,
    max_thick: int = 5,
    keep_thick_blobs: bool = False,
    remove_large: bool = False,
    large_cc_frac: float = 0.08,
    feather_k: int = 9,
    feather_sigma: float = 2.0,
    restrict_to_mask: bool = True,
    orient_keep_deg: Optional[List[float]] = None,
    orient_tol_deg: float = 18.0,
    fallback_to_generic: bool = True,
    **_unused,
) -> np.ndarray:
    """Convert a (usually filled) mask into a *porous* thin-edge mask.

    Intended for occluders with holes / gaps (e.g. chain-link fence, spider web).

    Key behaviors
    - Extract Canny edges on the crop and thicken them slightly.
    - If `attn_map` is provided, filter edges by an attention ROI so we keep only
      edges relevant to this object (reduces "stealing" subject edges).
    - If `restrict_to_mask=True`, also intersect edges with the original SAM mask.
      For porous occluders that overlap other objects, set it to False and rely on
      attention (SAM often misses thin structures over the subject).
    - `orient_keep_deg` (optional) filters by *gradient orientation* in degrees
      (0-180). Useful to suppress vertical/horizontal bars when you want diagonal
      chain-link patterns.

    Notes
    - `close_k`, `dilate_k`, `keep_thick_blobs`, `remove_large`, `large_cc_frac` are
      kept for backward compatibility with older scene configs.
    """

    H, W = mask_bin.shape[:2]
    x1, x2, y1, y2 = _clamp_bbox_xyxy(bbox_xyxy, W, H)
    if x2 <= x1 or y2 <= y1:
        return mask_bin.astype(np.float32)

    base_crop = mask_bin[y1:y2, x1:x2].astype(bool)

    # attention ROI (optional) — do NOT modify base_crop (SAM) with ROI,
    # otherwise overlap regions can get wiped out.
    roi = None
    if attn_map is not None:
        am = attn_map.astype(np.float32)
        am = am - float(am.min())
        denom = float(am.max() + 1e-6)
        am = am / denom
        roi = (am > float(attn_thresh))
        roi = roi[y1:y2, x1:x2]
        if attn_dilate_k and int(attn_dilate_k) > 0:
            k = int(attn_dilate_k)
            if k % 2 == 0:
                k += 1
            roi = cv2.dilate(roi.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)
        # If ROI is too small (e.g. missing token/attention), ignore it
        if roi is not None and int(roi.sum()) < 32:
            roi = None

    # edges
    if image_bgr is None:
        if fallback_to_generic:
            return SamMaskerV6.postprocess_generic(mask_bin, bbox_xyxy)
        # fallback: boundary of the base mask
        m = (base_crop.astype(np.uint8) * 255)
        edges = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edges_bin = edges > 0
        gray = None
    else:
        crop = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, int(canny1), int(canny2))
        edges_bin = edges > 0

        # optional orientation filter
        if orient_keep_deg is not None and len(orient_keep_deg) > 0:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0  # [0, 180)
            keep = np.zeros_like(edges_bin, dtype=bool)
            tol = float(orient_tol_deg)
            for d in orient_keep_deg:
                if d is None:
                    continue
                target = float(d) % 180.0
                diff = np.abs(ang - target)
                diff = np.minimum(diff, 180.0 - diff)
                keep |= (diff <= tol)
            edges_bin &= keep

    # Filter by attention ROI first (if provided)
    if roi is not None:
        edges_bin &= roi

    # Optionally restrict to SAM mask
    if restrict_to_mask:
        edges_bin &= base_crop

    # optional close to connect nearby edges
    if close_k and int(close_k) > 1:
        ck = int(close_k)
        if ck % 2 == 0:
            ck += 1
        edges_bin = cv2.morphologyEx(edges_bin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8)) > 0

    # thicken slightly (prefer dilate_k if provided)
    thick = int(max_thick) if int(max_thick) > 0 else 5
    if dilate_k and int(dilate_k) > 0:
        thick = max(thick, int(dilate_k))
    thick = max(1, thick)
    if thick > 1:
        edges_bin = cv2.dilate(edges_bin.astype(np.uint8), np.ones((thick, thick), np.uint8), iterations=1) > 0

    # optional: remove overly-large connected component (helps suppress "solid" fences)
    if remove_large:
        num_labels, labels = cv2.connectedComponents(edges_bin.astype(np.uint8), connectivity=8)
        if num_labels > 1:
            # area of each component id
            counts = np.bincount(labels.reshape(-1))
            areas = counts[1:]  # drop background
            if areas.size > 0:
                max_area = int(areas.max())
                if max_area / float(edges_bin.size + 1e-6) > float(large_cc_frac):
                    i_max = 1 + int(areas.argmax())
                    edges_bin = edges_bin & (labels != i_max)

    # NOTE: keep_thick_blobs is kept for backward compatibility.
    # In practice, for porous occluders you typically want it False.
    _ = keep_thick_blobs

    edges_soft = edges_bin.astype(np.float32)
    if feather_k and int(feather_k) > 1:
        edges_soft = soften_mask(edges_soft, k=int(feather_k), sigma=float(feather_sigma))


    # -----------------------------
    # Fallback: prevent "thin missing" (branches/web vanish)
    # If edges are too sparse, blend a faint softened SAM mask as existence prior.
    # -----------------------------
    try:
        min_edge_frac = float(_unused.get("min_edge_frac", 0.002)) if isinstance(_unused, dict) else 0.002
    except Exception:
        min_edge_frac = 0.002
    bb_area = float(max(1, (x2 - x1) * (y2 - y1)))
    if float(edges_soft.sum()) < float(min_edge_frac) * bb_area:
        try:
            fallback_alpha = float(_unused.get("fallback_alpha", 0.25)) if isinstance(_unused, dict) else 0.25
        except Exception:
            fallback_alpha = 0.25
        base = (mask_bin[y1:y2, x1:x2] > 0).astype(np.float32)
        if feather_k and int(feather_k) > 1:
            base = soften_mask(base, k=int(feather_k), sigma=float(feather_sigma))
        edges_soft = np.maximum(edges_soft, fallback_alpha * base)

    out = np.zeros_like(mask_bin, dtype=np.float32)
    out[y1:y2, x1:x2] = edges_soft
    return out.astype(np.float32)


# Simple plugin registry for demo configuration.

# ---- Compatibility: ensure thin_edges postprocess is bound to the class ----
# In some historical versions this function was defined at module scope (decorated with @staticmethod)
# but referenced as SamMaskerV6.postprocess_thin_edges in the registry, causing AttributeError.
if not hasattr(SamMaskerV6, 'postprocess_thin_edges') and 'postprocess_thin_edges' in globals():
    SamMaskerV6.postprocess_thin_edges = staticmethod(globals()['postprocess_thin_edges'])



# ---- Backward-compat: attach module-level thin-edge postprocess to class ----
# Some earlier revisions accidentally defined postprocess_thin_edges at module scope.
# We attach it here so SamMaskerV6.postprocess_thin_edges is always available.
if not hasattr(SamMaskerV6, "postprocess_thin_edges"):
    SamMaskerV6.postprocess_thin_edges = staticmethod(postprocess_thin_edges)
POSTPROCESS_REGISTRY: Dict[str, Any] = {
    "none": lambda m, box_xyxy, **kw: m.astype(np.float32) if m is not None else None,
    "generic": SamMaskerV6.postprocess_generic,
    "shuttle": SamMaskerV6.postprocess_shuttle,
    # Thin / holey foreground occluders (fence / lace / blinds / web / branches)
    "thin_edges": postprocess_thin_edges,
    "lace": postprocess_thin_edges,
    "wire": SamMaskerV6.postprocess_thin_edges,
    "mesh": SamMaskerV6.postprocess_thin_edges,
}
