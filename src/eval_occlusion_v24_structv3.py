#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Occlusion evaluation (v24 struct-v3)

- Supports --presence_mode {abs,bbox}
- Supports --image_glob auto
- Outputs per-object area_bbox + thin metrics (thin_blob / thin_thick_mean) + bbox_iou
- Computes scalar as:
    scalar = pair_pen_mean + presence_pen + quality_pen   (default)
  where quality_pen adds penalties for:
    - person too small in bbox (head-only)
    - person mask bbox IoU too small
    - thin occluder missing / blobby / too thick
    - badminton shuttle sticking to racket (if detectable)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV (cv2) is required. Install opencv-python.") from e

try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("PyTorch is required.") from e

# SAM
try:
    from segment_anything import SamPredictor, sam_model_registry
except Exception as e:
    raise RuntimeError("segment-anything is required.") from e

# Optional CLIP
try:
    from transformers import CLIPModel, CLIPProcessor
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# Optional Depth
try:
    from transformers import DPTForDepthEstimation
    try:
        from transformers import DPTImageProcessor as _DPTProcessor
    except Exception:
        from transformers import DPTFeatureExtractor as _DPTProcessor
    _HAS_DPT = True
except Exception:
    _HAS_DPT = False


_REL_DIRECTIVE_RE = re.compile(r"<<\s*(front_of|behind)\s*:\s*([a-zA-Z0-9_\-]+)\s*>>")


def strip_relation_directives(prompt: str) -> str:
    return _REL_DIRECTIVE_RE.sub("", prompt).strip()


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


_BBOX_FMT = "y1y2x1x2"  # default for this project (scene bbox = [y0,y1,x0,x1])

def xyxy_from_rel_bbox(rel_bbox: List[float], w: int, h: int, fmt: Optional[str] = None) -> np.ndarray:
    """Convert scene bbox in [0,1] to pixel xyxy.

    fmt:
      - y1y2x1x2 : [y0, y1, x0, x1]  (default)
      - x0x1y0y1 : [x0, x1, y0, y1]  (legacy)
    Return: [x0, y0, x1, y1] in pixels.
    """
    a, b, c, d = [float(v) for v in rel_bbox]
    fmt = (fmt or _BBOX_FMT or "y1y2x1x2").lower()
    if fmt == "x0x1y0y1":
        x0, x1, y0, y1 = a, b, c, d
    else:
        y0, y1, x0, x1 = a, b, c, d

    x0i = int(round(clamp01(x0) * w))
    x1i = int(round(clamp01(x1) * w))
    y0i = int(round(clamp01(y0) * h))
    y1i = int(round(clamp01(y1) * h))
    x0i, x1i = sorted((max(0, x0i), min(w, x1i)))
    y0i, y1i = sorted((max(0, y0i), min(h, y1i)))
    return np.array([x0i, y0i, x1i, y1i], dtype=np.int32)


def pad_xyxy(xyxy: np.ndarray, pad: float, w: int, h: int) -> np.ndarray:
    if pad <= 0:
        return xyxy
    x0, y0, x1, y1 = xyxy.tolist()
    bw = x1 - x0
    bh = y1 - y0
    px = int(round(bw * pad))
    py = int(round(bh * pad))
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w, x1 + px)
    y1 = min(h, y1 + py)
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def area_xyxy(xyxy: np.ndarray) -> int:
    x0, y0, x1, y1 = xyxy.tolist()
    return int(max(0, x1 - x0) * max(0, y1 - y0))


def intersect_xyxy(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    ax0, ay0, ax1, ay1 = a.tolist()
    bx0, by0, bx1, by1 = b.tolist()
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = a.tolist()
    bx0, by0, bx1, by1 = b.tolist()
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    bb = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter) / float(aa + bb - inter + 1e-6)


def mask_tight_bbox_in_roi(mask_roi: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask_roi)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def thin_edges_postprocess(
    image_rgb: np.ndarray,
    sam_mask: np.ndarray,
    xyxy: np.ndarray,
    canny_low: int = 60,
    canny_high: int = 140,
    dilate_iter: int = 1,
    restrict_to_mask: bool = True,
) -> np.ndarray:
    x0, y0, x1, y1 = xyxy.tolist()
    if x1 <= x0 or y1 <= y0:
        return sam_mask.astype(bool)

    crop = image_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return sam_mask.astype(bool)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    edges_bool = edges > 0
    sam_crop = sam_mask[y0:y1, x0:x1].astype(bool)
    thin_crop = (edges_bool & sam_crop) if restrict_to_mask else edges_bool

    out = np.zeros_like(sam_mask, dtype=bool)
    out[y0:y1, x0:x1] = thin_crop
    return out


@dataclass
class ObjEval:
    idx: int
    name: str
    present: bool
    area_ratio_in_bbox: float
    mask_area: int
    bbox_area: int
    bbox_iou: Optional[float] = None
    clip_score: Optional[float] = None
    depth_median: Optional[float] = None
    thin_blob_frac: Optional[float] = None
    thin_thick_mean: Optional[float] = None
    cc_count: Optional[int] = None
    hole_count: Optional[int] = None
    fill_ratio: Optional[float] = None
    mixed_cell_ratio: Optional[float] = None
    span_lr: Optional[int] = None
    span_tb: Optional[int] = None
    transition_h: Optional[float] = None
    transition_v: Optional[float] = None
    transition_strength: Optional[float] = None
    line_complexity: Optional[float] = None


@dataclass
class PairEval:
    back: int
    front: int
    mode: str
    overlap_area: int
    front_occ: float
    back_occ: float
    intersection_ratio: float
    occ_pen: float
    back_pen: float
    pair_penalty: float
    touch_ratio: Optional[float] = None
    depth_order_correct: Optional[bool] = None
    depth_front: Optional[float] = None
    depth_back: Optional[float] = None


class OcclusionEvaluator:
    def __init__(
        self,
        sam_ckpt: str,
        sam_model_type: str = "vit_h",
        device: str = "cuda",
        clip_model_name: Optional[str] = None,
        depth_model_name: Optional[str] = None,
        depth_closer: str = "higher",
    ) -> None:
        self.device = device

        sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
        sam.to(device)
        self.sam_predictor = SamPredictor(sam)

        self.clip_model = None
        self.clip_processor = None
        if clip_model_name:
            if not _HAS_CLIP:
                raise RuntimeError("transformers CLIP is missing. Install/upgrade transformers.")
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model.eval()

        self.depth_model = None
        self.depth_processor = None
        self.depth_closer = depth_closer
        if depth_model_name:
            if not _HAS_DPT:
                raise RuntimeError("transformers DPT is missing. Install/upgrade transformers.")
            self.depth_model = DPTForDepthEstimation.from_pretrained(depth_model_name).to(device)
            self.depth_processor = _DPTProcessor.from_pretrained(depth_model_name)
            self.depth_model.eval()

    @torch.inference_mode()
    def clip_similarity(self, image: Image.Image, text: str) -> float:
        assert self.clip_model is not None and self.clip_processor is not None
        max_len = 77
        tok = getattr(self.clip_processor, "tokenizer", None)
        if tok is not None:
            try:
                max_len = int(getattr(tok, "model_max_length", 77))
            except Exception:
                max_len = 77

        inputs = self.clip_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.clip_model(**inputs)
        img_emb = out.image_embeds
        txt_emb = out.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum(dim=-1).detach().cpu().float().item()
        return float(sim)

    @torch.inference_mode()
    def predict_depth(self, image: Image.Image) -> np.ndarray:
        assert self.depth_model is not None and self.depth_processor is not None
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.depth_model(**inputs)
        depth = out.predicted_depth
        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(image.height, image.width), mode="bicubic", align_corners=False)
        depth = depth.squeeze().detach().cpu().float().numpy()
        dmin = float(depth.min())
        dmax = float(depth.max())
        return (depth - dmin) / (dmax - dmin + 1e-6)

    @torch.inference_mode()
    def sam_masks_for_scene(
        self,
        image_rgb: np.ndarray,
        objects_cfg: List[Dict[str, Any]],
        sam_pad: List[float],
        postprocess: List[str],
        sam_mode: List[str],
    ) -> List[np.ndarray]:
        h, w = image_rgb.shape[:2]
        self.sam_predictor.set_image(image_rgb)

        masks: List[np.ndarray] = []
        for i, obj in enumerate(objects_cfg):
            bbox_rel = obj.get("bbox")
            if bbox_rel is None:
                masks.append(np.zeros((h, w), dtype=bool))
                continue

            xyxy = xyxy_from_rel_bbox(bbox_rel, w, h)
            xyxy = pad_xyxy(xyxy, sam_pad[i] if i < len(sam_pad) else 0.0, w, h)

            box = xyxy.astype(np.float32)
            m, scores, _ = self.sam_predictor.predict(box=box[None, :], multimask_output=True)
            best = int(np.argmax(scores))
            mask = m[best].astype(bool)

            pp = (postprocess[i] if i < len(postprocess) else "none") or "none"
            sm = (sam_mode[i] if i < len(sam_mode) else "") or ""
            if pp in ["thin_edges", "lace"] or sm in ["fence", "web", "lace"]:
                cfg = obj.get("postprocess_cfg", {}) or {}
                # compat keys: canny_low/canny_high/dilate_iter (legacy) or canny1/canny2/dilate_k (scenes)
                canny_low = int(cfg.get("canny_low", cfg.get("canny1", 60)))
                canny_high = int(cfg.get("canny_high", cfg.get("canny2", 140)))
                dilate_iter = int(cfg.get("dilate_iter", cfg.get("dilate_k", 1)))
                restrict = bool(cfg.get("restrict_to_mask", True))
                mask = thin_edges_postprocess(image_rgb, mask, xyxy, canny_low, canny_high, dilate_iter, restrict_to_mask=restrict)

            masks.append(mask)

        return masks


def gather_images(scene_dir: Path, image_glob: str) -> List[Path]:
    ig = (image_glob or "").strip().lower()
    if ig == "auto":
        patterns = [
            "refine_best.jpg", "refine_best.png",
            "refine_*.jpg", "refine_*.png", "refine_*.jpeg",
            "final_*.jpg", "final_*.png",
            "best_*.jpg", "best_*.png",
            "*.jpg", "*.png", "*.jpeg",
        ]
        for pat in patterns:
            xs = sorted(scene_dir.glob(pat))
            if xs:
                return xs
        return []
    xs = sorted(scene_dir.glob(image_glob))
    if not xs and image_glob.endswith(".jpg"):
        xs = sorted(scene_dir.glob(image_glob.replace(".jpg", ".png")))
    return xs


def find_scene_json(scenes_dir: Path, scene_name: str) -> Optional[Path]:
    p = scenes_dir / f"{scene_name}.json"
    if p.exists():
        return p
    cands = sorted(scenes_dir.glob(f"{scene_name}*.json"))
    return cands[0] if cands else None


def detect_shuttle_and_racket(objects_cfg: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    shuttle_idx = None
    racket_idx = None
    for i, o in enumerate(objects_cfg):
        prompt = str(o.get("prompt", "")).lower()
        name = str(o.get("name", "")).lower()
        sm = str(o.get("sam_mode", "")).lower()
        if shuttle_idx is None and (sm == "shuttle" or "shuttlecock" in prompt or "birdie" in prompt or "shuttle" in name):
            shuttle_idx = i
        if racket_idx is None and ("racket" in prompt or "racket" in name):
            racket_idx = i
    return shuttle_idx, racket_idx


def shuttle_stick_ratio(m_shuttle: np.ndarray, m_racket: np.ndarray, box: np.ndarray, dilate_k: int = 11) -> float:
    x0, y0, x1, y1 = box.tolist()
    sh = (m_shuttle[y0:y1, x0:x1] > 0).astype(np.uint8)
    rk = (m_racket[y0:y1, x0:x1] > 0).astype(np.uint8)
    if sh.sum() < 20:
        return 0.0
    k = int(dilate_k)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    rk_d = cv2.dilate(rk, kernel, iterations=1)
    ov = float((rk_d & sh).sum())
    return ov / float(sh.sum() + 1e-6)




POROUS_KEYWORDS = [
    "fence", "web", "net", "wire", "lace", "blinds", "branches",
    "cracked glass", "cracked_glass", "fracture", "fractured glass", "shattered glass",
    "barbed wire", "chain link", "chainlink", "mesh", "curtain", "beaded curtain",
    "grille", "grating", "screen", "railing", "necklace", "chain"
]


def is_porous_like_object(obj: Dict[str, Any]) -> bool:
    postp = str(obj.get("postprocess", "none")).lower()
    smode = str(obj.get("sam_mode", "")).lower()
    name = str(obj.get("name", "")).lower()
    prompt = str(obj.get("prompt", "")).lower()
    category = str(obj.get("category", "")).lower()
    tags = " ".join([str(t).lower() for t in obj.get("tags", [])])

    if postp in ["thin_edges", "lace"]:
        return True
    if smode in ["fence", "web", "net", "wire", "lace", "blinds", "branches", "cracked_glass", "glass", "mesh", "screen", "grille", "chain", "necklace"]:
        return True

    blob = " ".join([name, prompt, category, tags])
    return any(k in blob for k in POROUS_KEYWORDS)


def count_cc_and_holes(mask_roi_bin: np.ndarray) -> Tuple[int, int]:
    """Return (connected_components_count, hole_count) in ROI."""
    m = (mask_roi_bin > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return 0, 0
    num_cc, _ = cv2.connectedComponents(m, connectivity=8)
    cc_count = max(0, int(num_cc) - 1)

    inv = (m == 0).astype(np.uint8)
    hh, ww = inv.shape
    flood = inv.copy()
    ffmask = np.zeros((hh + 2, ww + 2), np.uint8)
    seed = None
    for x in range(ww):
        if flood[0, x] == 1: seed = (x, 0); break
        if flood[hh - 1, x] == 1: seed = (x, hh - 1); break
    if seed is None:
        for y in range(hh):
            if flood[y, 0] == 1: seed = (0, y); break
            if flood[y, ww - 1] == 1: seed = (ww - 1, y); break
    if seed is not None:
        cv2.floodFill(flood, ffmask, seedPoint=seed, newVal=0)
    holes = (flood > 0).astype(np.uint8)
    if int(holes.sum()) == 0:
        return cc_count, 0
    num_h, _ = cv2.connectedComponents(holes, connectivity=8)
    hole_count = max(0, int(num_h) - 1)
    return cc_count, hole_count




def porous_stats_v2(mask_roi_bin: np.ndarray, grid: int = 12) -> Dict[str, float]:
    """Stable porous/high-frequency structure stats.

    Returns:
      - fill_ratio: mask occupancy inside ROI
      - mixed_cell_ratio: fraction of grid cells with mixed occupancy (captures interleaved strands/holes)
      - span_lr / span_tb: number of connected components spanning left-right / top-bottom
      - cc_count / hole_count: legacy stats retained for debugging
    """
    m = (mask_roi_bin > 0).astype(np.uint8)
    h, w = m.shape[:2]
    if h == 0 or w == 0 or int(m.sum()) == 0:
        return dict(fill_ratio=0.0, mixed_cell_ratio=0.0, span_lr=0, span_tb=0, cc_count=0, hole_count=0)

    fill_ratio = float(m.mean())

    # Mixed-cell ratio is more stable than raw hole counting for porous structures.
    g = max(2, int(grid))
    mixed = 0
    total = 0
    for gy in range(g):
        for gx in range(g):
            y0 = int(round(gy * h / g))
            y1 = int(round((gy + 1) * h / g))
            x0 = int(round(gx * w / g))
            x1 = int(round((gx + 1) * w / g))
            if y1 <= y0 or x1 <= x0:
                continue
            cell = m[y0:y1, x0:x1]
            fr = float(cell.mean())
            total += 1
            if 0.15 < fr < 0.85:
                mixed += 1
    mixed_cell_ratio = float(mixed) / float(max(1, total))

    num_cc, labels = cv2.connectedComponents(m, connectivity=8)
    cc_count = max(0, int(num_cc) - 1)
    span_lr = 0
    span_tb = 0
    for cid in range(1, num_cc):
        ys, xs = np.where(labels == cid)
        if xs.size == 0:
            continue
        touch_l = bool((xs == 0).any())
        touch_r = bool((xs == w - 1).any())
        touch_t = bool((ys == 0).any())
        touch_b = bool((ys == h - 1).any())
        if touch_l and touch_r:
            span_lr += 1
        if touch_t and touch_b:
            span_tb += 1

    _, hole_count = count_cc_and_holes(m)
    return dict(
        fill_ratio=fill_ratio,
        mixed_cell_ratio=mixed_cell_ratio,
        span_lr=int(span_lr),
        span_tb=int(span_tb),
        cc_count=int(cc_count),
        hole_count=int(hole_count),
    )





def porous_stats_v3(mask_roi_bin: np.ndarray, grid: int = 12) -> Dict[str, float]:
    """Stronger porous/high-frequency structure stats.

    Keeps v2 signals (fill/mixed/span) and adds transition-based complexity:
      - transition_h / transition_v: normalized boundary-transition density along rows / cols
      - transition_strength: max(transition_h, transition_v), robust to directional patterns
      - line_complexity: fraction of active scanlines containing multiple foreground segments

    These are more discriminative for blinds / fences / webs / chains than raw holes alone.
    """
    base = porous_stats_v2(mask_roi_bin, grid=grid)
    m = (mask_roi_bin > 0).astype(np.uint8)
    h, w = m.shape[:2]
    if h == 0 or w == 0 or int(m.sum()) == 0:
        base.update(dict(transition_h=0.0, transition_v=0.0, transition_strength=0.0, line_complexity=0.0))
        return base

    mm = m.astype(np.int16)

    row_counts = np.abs(np.diff(mm, axis=1)).sum(axis=1) if w > 1 else np.zeros((h,), dtype=np.int16)
    col_counts = np.abs(np.diff(mm, axis=0)).sum(axis=0) if h > 1 else np.zeros((w,), dtype=np.int16)

    active_rows = m.any(axis=1)
    active_cols = m.any(axis=0)

    if active_rows.any():
        transition_h = float(row_counts[active_rows].mean()) / 6.0
        row_multi = float(np.mean(row_counts[active_rows] >= 4))
    else:
        transition_h = 0.0
        row_multi = 0.0

    if active_cols.any():
        transition_v = float(col_counts[active_cols].mean()) / 6.0
        col_multi = float(np.mean(col_counts[active_cols] >= 4))
    else:
        transition_v = 0.0
        col_multi = 0.0

    transition_h = float(np.clip(transition_h, 0.0, 1.0))
    transition_v = float(np.clip(transition_v, 0.0, 1.0))
    transition_strength = float(max(transition_h, transition_v))
    line_complexity = float(max(row_multi, col_multi))

    base.update(
        dict(
            transition_h=transition_h,
            transition_v=transition_v,
            transition_strength=transition_strength,
            line_complexity=line_complexity,
        )
    )
    return base


def pair_touch_ratio(m_front: np.ndarray, m_back: np.ndarray, R_xyxy: np.ndarray, dilate_k: int = 11) -> float:
    """Dilation touch ratio in overlap ROI: dilate(front) ∩ back / back."""
    if R_xyxy is None:
        return 0.0
    x0, y0, x1, y1 = [int(v) for v in R_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0:
        return 0.0
    Mf = (m_front[y0:y1, x0:x1] > 0).astype(np.uint8)
    Mb = (m_back[y0:y1, x0:x1] > 0).astype(np.uint8)
    if int(Mb.sum()) < 20:
        return 0.0
    k = int(dilate_k)
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    Mf_d = cv2.dilate(Mf, ker, iterations=1)
    return float((Mf_d & Mb).sum()) / float(Mb.sum() + 1e-6)



def eval_objects(
    image_pil: Image.Image,
    image_rgb: np.ndarray,
    objects_cfg: List[Dict[str, Any]],
    masks: List[np.ndarray],
    depth_map: Optional[np.ndarray],
    presence_mode: str = "abs",
    area_min_default: float = 0.005,
    area_max_default: float = 0.45,
    clip_fn=None,
) -> List[ObjEval]:
    h, w = image_rgb.shape[:2]
    out: List[ObjEval] = []

    for i, obj in enumerate(objects_cfg):
        name = obj.get("name", f"obj{i}")
        bbox_rel = obj.get("bbox", [0, 1, 0, 1])
        xyxy = xyxy_from_rel_bbox(bbox_rel, w, h)
        x0, y0, x1, y1 = xyxy.tolist()
        bbox_area = max(1, area_xyxy(xyxy))

        mask = masks[i].astype(bool)
        mask_roi = mask[y0:y1, x0:x1]
        mask_area = int(mask_roi.sum())
        area_ratio = float(mask_area) / float(bbox_area)
        mask_abs_ratio = float(mask.sum()) / float(h * w)

        ti = mask_tight_bbox_in_roi(mask_roi)
        biou = None
        if ti is not None:
            tx0, ty0, tx1, ty1 = ti
            tight_xyxy = np.array([x0 + tx0, y0 + ty0, x0 + tx1, y0 + ty1], dtype=np.int32)
            biou = bbox_iou(tight_xyxy, xyxy)

        if presence_mode == "bbox":
            area_min_b = obj.get("area_min_bbox", None)
            area_max_b = obj.get("area_max_bbox", None)
            postp = str(obj.get("postprocess", "none")).lower()
            smode = str(obj.get("sam_mode", "")).lower()
            if area_min_b is None:
                area_min_b = 0.001 if (postp in ["thin_edges", "lace"] or smode in ["fence", "web", "lace"]) else 0.02
            if area_max_b is None:
                area_max_b = 0.999
            present = (area_ratio >= float(area_min_b)) and (area_ratio <= float(area_max_b))
        else:
            area_min = float(obj.get("area_min", area_min_default))
            area_max = float(obj.get("area_max", area_max_default))
            present = (mask_abs_ratio >= area_min) and (mask_abs_ratio <= area_max)

        clip_score = None
        if clip_fn is not None:
            crop = image_pil.crop((x0, y0, x1, y1))
            clip_score = float(clip_fn(crop, strip_relation_directives(obj.get("prompt", name))))

        depth_median = None
        if depth_map is not None:
            dm = depth_map[mask]
            if dm.size > 0:
                depth_median = float(np.median(dm))

        thin_blob_frac = None
        thin_thick_mean = None
        postp = str(obj.get("postprocess", "none")).lower()
        smode = str(obj.get("sam_mode", "")).lower()
        if postp in ["thin_edges", "lace"] or smode in ["fence", "web", "lace"]:
            mc = mask_roi.astype(np.uint8)
            if mc.sum() > 20:
                dt = cv2.distanceTransform(mc, cv2.DIST_L2, 3)
                thin_thick_mean = float(dt[mc > 0].mean()) * 2.0
                thick = int((dt > 5.0).sum())
                thin_blob_frac = float(thick) / float(mc.sum() + 1e-6)


        
        # structure metrics (for porous/high-frequency occluders)
        cc_count = None
        hole_count = None
        fill_ratio = None
        mixed_cell_ratio = None
        span_lr = None
        span_tb = None
        transition_h = None
        transition_v = None
        transition_strength = None
        line_complexity = None
        if is_porous_like_object(obj):
            pst = porous_stats_v3(mask_roi.astype(np.uint8), grid=int(obj.get("struct_grid", 12)))
            cc_count = int(pst["cc_count"])
            hole_count = int(pst["hole_count"])
            fill_ratio = float(pst["fill_ratio"])
            mixed_cell_ratio = float(pst["mixed_cell_ratio"])
            span_lr = int(pst["span_lr"])
            span_tb = int(pst["span_tb"])
            transition_h = float(pst["transition_h"])
            transition_v = float(pst["transition_v"])
            transition_strength = float(pst["transition_strength"])
            line_complexity = float(pst["line_complexity"])
        out.append(
            ObjEval(
                idx=i,
                name=name,
                present=bool(present),
                area_ratio_in_bbox=area_ratio,
                mask_area=mask_area,
                bbox_area=bbox_area,
                bbox_iou=biou,
                clip_score=clip_score,
                depth_median=depth_median,
                thin_blob_frac=thin_blob_frac,
                thin_thick_mean=thin_thick_mean,
                cc_count=cc_count,
                hole_count=hole_count,
                fill_ratio=fill_ratio,
                mixed_cell_ratio=mixed_cell_ratio,
                span_lr=span_lr,
                span_tb=span_tb,
                transition_h=transition_h,
                transition_v=transition_v,
                transition_strength=transition_strength,
                line_complexity=line_complexity,
            )
        )

    return out


def eval_pairs(
    objects_cfg: List[Dict[str, Any]],
    masks: List[np.ndarray],
    occlusion_pairs: List[Dict[str, Any]],
    w: int,
    h: int,
    depth_map: Optional[np.ndarray],
    depth_closer: str,
    confusion_weight: float = 1.0,
    occl_weight: float = 1.0,
    compute_touch: bool = False,
    touch_dilate_k_default: int = 11,
) -> List[PairEval]:
    bxy = [xyxy_from_rel_bbox(o.get("bbox", [0, 1, 0, 1]), w, h) for o in objects_cfg]

    out: List[PairEval] = []
    for p in occlusion_pairs:
        back = int(p["back"])
        front = int(p["front"])
        mode = str(p.get("mode", "opaque")).lower()

        R = intersect_xyxy(bxy[back], bxy[front])
        if R is None:
            out.append(PairEval(back, front, mode, 0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
            continue

        x0, y0, x1, y1 = R.tolist()
        area = max(1, area_xyxy(R))

        Mf = masks[front].astype(bool)[y0:y1, x0:x1]
        Mb = masks[back].astype(bool)[y0:y1, x0:x1]

        front_occ = float(Mf.sum()) / float(area)
        back_occ = float(Mb.sum()) / float(area)
        inter = float(np.logical_and(Mf, Mb).sum()) / float(area)

        if mode == "holey":
            fmin = float(p.get("front_occupancy_min", 0.02))
            fmax = float(p.get("front_occupancy_max", 0.22))
            bmin = float(p.get("back_density_min", 0.18))
            occ_pen = max(0.0, fmin - front_occ) + max(0.0, front_occ - fmax)
            back_pen = max(0.0, bmin - back_occ)
        else:
            tmin = float(p.get("min_occl", 0.08))
            tmax = float(p.get("max_occl", 0.35))
            occ_pen = max(0.0, tmin - front_occ) + max(0.0, front_occ - tmax)
            back_pen = max(0.0, back_occ - float(p.get("back_occ_max", 0.05)))

        pair_pen = occl_weight * (occ_pen + back_pen) + confusion_weight * inter
        touch_ratio = None
        if compute_touch:
            try:
                k = int(p.get("touch_dilate_k", touch_dilate_k_default))
                touch_ratio = float(pair_touch_ratio(masks[front], masks[back], R, dilate_k=k))
            except Exception:
                touch_ratio = None

        depth_order_correct = None
        df = db = None
        if depth_map is not None:
            dm_f = depth_map[masks[front].astype(bool)]
            dm_b = depth_map[masks[back].astype(bool)]
            if dm_f.size > 10 and dm_b.size > 10:
                df = float(np.median(dm_f))
                db = float(np.median(dm_b))
                depth_order_correct = bool(df > db) if depth_closer == "higher" else bool(df < db)

        out.append(
            PairEval(
                back=back,
                front=front,
                mode=mode,
                overlap_area=area,
                front_occ=front_occ,
                back_occ=back_occ,
                intersection_ratio=inter,
                occ_pen=occ_pen,
                back_pen=back_pen,
                pair_penalty=pair_pen,
                touch_ratio=touch_ratio,
                depth_order_correct=depth_order_correct,
                depth_front=df,
                depth_back=db,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", type=str, required=True)
    ap.add_argument("--scenes_dir", type=str, required=True)
    ap.add_argument("--sam_ckpt", type=str, required=True)
    ap.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--clip_model", type=str, default="", help="CLIP model path/name; set '' to disable")
    ap.add_argument("--depth_model", type=str, default="", help="DPT model path/name; set '' to disable")
    ap.add_argument("--depth_closer", type=str, default="higher", choices=["higher", "lower"])

    ap.add_argument(
    "--bbox_format",
    type=str,
    default="y1y2x1x2",
    choices=["y1y2x1x2", "x0x1y0y1"],
    help="Scene bbox format. Default y1y2x1x2 means [y0,y1,x0,x1] (used by this project).",
    )

    ap.add_argument("--scalar_mode", type=str, default="scalar50", choices=["scalar", "scalar50"])
    ap.add_argument("--w_depth", type=float, default=0.0)
    ap.add_argument("--w_occl", type=float, default=1.0)
    ap.add_argument("--w_quality", type=float, default=1.0)

    # Optional generic anti-touch (OFF by default)
    ap.add_argument("--w_touch", type=float, default=0.0, help="Weight of generic touch penalty (0 disables).")
    ap.add_argument("--touch_max_default", type=float, default=-1.0, help="Default touch_max if pair does not specify. <0 disables.")
    ap.add_argument("--touch_dilate_k", type=int, default=11)

    # Optional structure penalties (OFF by default)
    ap.add_argument("--w_struct", type=float, default=0.0, help="Weight of structure penalty (0 disables).")
    ap.add_argument("--struct_mode", type=str, default="porous_v3", choices=["legacy", "porous_v2", "porous_v3"])
    ap.add_argument("--struct_grid", type=int, default=12)
    ap.add_argument("--fill_min_default", type=float, default=0.08)
    ap.add_argument("--fill_max_default", type=float, default=0.65)
    ap.add_argument("--mixed_cell_min_default", type=float, default=0.12)
    ap.add_argument("--span_min_default", type=int, default=1)
    ap.add_argument("--transition_min_default", type=float, default=0.18)
    ap.add_argument("--w_struct_fill", type=float, default=0.20)
    ap.add_argument("--w_struct_mixed", type=float, default=0.25)
    ap.add_argument("--w_struct_span", type=float, default=0.15)
    ap.add_argument("--w_struct_transition", type=float, default=0.40)
    ap.add_argument("--cc_min_default", type=int, default=-1)
    ap.add_argument("--cc_max_default", type=int, default=-1)
    ap.add_argument("--holes_min_default", type=int, default=-1)
    ap.add_argument("--holes_max_default", type=int, default=-1)

    ap.add_argument("--presence_mode", type=str, default="abs", choices=["abs", "bbox"])
    ap.add_argument("--image_glob", type=str, default="refine_*.jpg")

    # quality penalties (default ON)
    ap.add_argument("--quality_pen", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--w_person_small", type=float, default=0.8)
    ap.add_argument("--person_area_min", type=float, default=0.15)
    ap.add_argument("--w_person_iou", type=float, default=0.6)
    ap.add_argument("--person_iou_min", type=float, default=0.25)

    ap.add_argument("--w_thin_missing", type=float, default=0.8)
    ap.add_argument("--thin_area_min", type=float, default=0.02)
    ap.add_argument("--w_thin_blob", type=float, default=0.5)
    ap.add_argument("--thin_blob_max", type=float, default=0.35)
    ap.add_argument("--w_thin_thick", type=float, default=0.3)
    ap.add_argument("--thin_thick_max", type=float, default=7.0)

    ap.add_argument("--w_shuttle_stick", type=float, default=0.8)
    ap.add_argument("--shuttle_stick_max", type=float, default=0.06)
    ap.add_argument("--shuttle_dilate_k", type=int, default=11)

    ap.add_argument("--out_csv", type=str, default="eval_results.csv")
    ap.add_argument("--out_json", type=str, default="eval_details.json")

    args = ap.parse_args()
    global _BBOX_FMT
    _BBOX_FMT = str(args.bbox_format).lower()

    images_root = Path(args.images_root)
    scenes_dir = Path(args.scenes_dir)

    clip_model = args.clip_model.strip()
    depth_model = args.depth_model.strip()
    evaluator = OcclusionEvaluator(
        sam_ckpt=args.sam_ckpt,
        sam_model_type=args.sam_model_type,
        device=args.device,
        clip_model_name=(clip_model if clip_model else None),
        depth_model_name=(depth_model if depth_model else None),
        depth_closer=args.depth_closer,
    )

    scene_dirs = [p for p in sorted(images_root.iterdir()) if p.is_dir()]
    if not scene_dirs:
        raise SystemExit(f"No scene directories under: {images_root}")

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    for sd in scene_dirs:
        scene_name = sd.name
        scene_json = find_scene_json(scenes_dir, scene_name)
        if scene_json is None:
            print(f"[WARN] scene json not found for {scene_name}, skipping")
            continue

        scene = json.loads(scene_json.read_text(encoding="utf-8"))
        objects_cfg: List[Dict[str, Any]] = scene["objects"]

        sam_mode = [str(o.get("sam_mode", "")) for o in objects_cfg]
        postprocess = [str(o.get("postprocess", "none")) for o in objects_cfg]
        sam_pad = [float(o.get("sam_pad", 0.02)) for o in objects_cfg]

        occlusion_pairs = scene.get("occlusion_pairs", []) or []
        confusion_weight = float(scene.get("eval_confusion_weight", 1.0))
        occl_weight = float(scene.get("eval_occl_weight", 1.0))

        image_paths = gather_images(sd, args.image_glob)
        if not image_paths:
            print(f"[WARN] no images in {sd} (image_glob={args.image_glob}), skipping")
            continue

        details[scene_name] = {"scene_json": str(scene_json), "images": {}}

        for ip in image_paths:
            im = Image.open(ip).convert("RGB")
            image_rgb = np.array(im)
            h, w = image_rgb.shape[:2]

            depth_map = evaluator.predict_depth(im) if evaluator.depth_model is not None else None

            masks = evaluator.sam_masks_for_scene(
                image_rgb=image_rgb,
                objects_cfg=objects_cfg,
                sam_pad=sam_pad,
                postprocess=postprocess,
                sam_mode=sam_mode,
            )

            obj_evals = eval_objects(
                image_pil=im,
                image_rgb=image_rgb,
                objects_cfg=objects_cfg,
                masks=masks,
                depth_map=depth_map,
                presence_mode=args.presence_mode,
                clip_fn=(evaluator.clip_similarity if evaluator.clip_model is not None else None),
            )

            pair_evals = eval_pairs(
                 objects_cfg=objects_cfg,
                 masks=masks,
                 occlusion_pairs=occlusion_pairs,
                 w=w,
                 h=h,
                 depth_map=depth_map,
                 depth_closer=args.depth_closer,
                 confusion_weight=confusion_weight,
                 occl_weight=occl_weight,
                 compute_touch=(args.w_touch > 0.0 or args.touch_max_default >= 0 or any(("touch_max" in p) for p in occlusion_pairs)),
                 touch_dilate_k_default=int(args.touch_dilate_k),
             )

            presence_pen = float(sum(0 if oe.present else 1 for oe in obj_evals[1:]))
            pair_pen_mean = float(np.mean([pe.pair_penalty for pe in pair_evals])) if pair_evals else 0.0
            inter_mean = float(np.mean([pe.intersection_ratio for pe in pair_evals])) if pair_evals else 0.0

            depth_order_acc = None
            if depth_map is not None and pair_evals:
                vals = [pe.depth_order_correct for pe in pair_evals if pe.depth_order_correct is not None]
                if vals:
                    depth_order_acc = float(np.mean([1.0 if v else 0.0 for v in vals]))

            clip_overall = None
            if evaluator.clip_model is not None:
                full_prompt = ", ".join(strip_relation_directives(o.get("prompt", "")) for o in objects_cfg)
                clip_overall = float(evaluator.clip_similarity(im, full_prompt))

            quality_pen = 0.0
            dbg_q: Dict[str, Any] = {}

            if args.quality_pen == "on":
                person_small_pen = 0.0
                person_iou_pen = 0.0
                for oe in obj_evals[1:]:
                    n = oe.name.lower()
                    if any(k in n for k in ["person", "man", "woman", "human"]):
                        if oe.area_ratio_in_bbox < args.person_area_min:
                            person_small_pen += (args.person_area_min - oe.area_ratio_in_bbox) / max(1e-6, args.person_area_min)
                        if oe.bbox_iou is not None and oe.bbox_iou < args.person_iou_min:
                            person_iou_pen += (args.person_iou_min - oe.bbox_iou) / max(1e-6, args.person_iou_min)

                thin_missing_pen = 0.0
                thin_blob_pen = 0.0
                thin_thick_pen = 0.0
                for oe in obj_evals[1:]:
                    obj = objects_cfg[oe.idx]
                    postp = str(obj.get("postprocess", "none")).lower()
                    smode = str(obj.get("sam_mode", "")).lower()
                    if postp in ["thin_edges", "lace"] or smode in ["fence", "web", "lace"]:
                        if oe.area_ratio_in_bbox < args.thin_area_min:
                            thin_missing_pen += (args.thin_area_min - oe.area_ratio_in_bbox) / max(1e-6, args.thin_area_min)
                        if oe.thin_blob_frac is not None and oe.thin_blob_frac > args.thin_blob_max:
                            thin_blob_pen += (oe.thin_blob_frac - args.thin_blob_max) / max(1e-6, 1.0 - args.thin_blob_max)
                        if oe.thin_thick_mean is not None and oe.thin_thick_mean > args.thin_thick_max:
                            thin_thick_pen += (oe.thin_thick_mean - args.thin_thick_max) / max(1e-6, args.thin_thick_max)

                shuttle_pen = 0.0
                shuttle_idx, racket_idx = detect_shuttle_and_racket(objects_cfg)
                if shuttle_idx is not None and racket_idx is not None:
                    sh_box = xyxy_from_rel_bbox(objects_cfg[shuttle_idx].get("bbox", [0, 1, 0, 1]), w, h)
                    stick = shuttle_stick_ratio(masks[shuttle_idx], masks[racket_idx], sh_box, dilate_k=args.shuttle_dilate_k)
                    dbg_q["shuttle_stick_ratio"] = stick
                    if stick > args.shuttle_stick_max:
                        shuttle_pen = (stick - args.shuttle_stick_max) / max(1e-6, 1.0 - args.shuttle_stick_max)

                quality_pen = (
                    args.w_person_small * person_small_pen +
                    args.w_person_iou * person_iou_pen +
                    args.w_thin_missing * thin_missing_pen +
                    args.w_thin_blob * thin_blob_pen +
                    args.w_thin_thick * thin_thick_pen +
                    args.w_shuttle_stick * shuttle_pen
                )
                dbg_q.update(
                    person_small_pen=person_small_pen,
                    person_iou_pen=person_iou_pen,
                    thin_missing_pen=thin_missing_pen,
                    thin_blob_pen=thin_blob_pen,
                    thin_thick_pen=thin_thick_pen,
                    shuttle_pen=shuttle_pen,
                )

            # generic touch penalty (optional)


            touch_pen = 0.0


            touch_ratio_mean = None


            if args.w_touch > 0.0 or args.touch_max_default >= 0:


                trs = []


                for pe, p in zip(pair_evals, occlusion_pairs):


                    tmax = p.get("touch_max", None)


                    if tmax is None and args.touch_max_default >= 0:


                        tmax = float(args.touch_max_default)


                    if tmax is None or pe.touch_ratio is None:


                        continue


                    tr = float(pe.touch_ratio)


                    trs.append(tr)


                    if tr > float(tmax):


                        touch_pen += (tr - float(tmax)) / (1.0 - float(tmax) + 1e-6)


                if trs:


                    touch_ratio_mean = float(np.mean(trs))




            
            # structure penalty (optional)
            struct_pen = 0.0
            struct_dbg = []
            if args.w_struct > 0.0:
                for oe in obj_evals[1:]:
                    obj = objects_cfg[oe.idx]
                    if not is_porous_like_object(obj):
                        continue

                    if str(args.struct_mode).lower() == "legacy":
                        cc = int(oe.cc_count or 0)
                        holes = int(oe.hole_count or 0)
                        cc_min = int(obj.get("cc_min", args.cc_min_default))
                        cc_max = int(obj.get("cc_max", args.cc_max_default))
                        h_min = int(obj.get("holes_min", args.holes_min_default))
                        h_max = int(obj.get("holes_max", args.holes_max_default))
                        obj_pen = 0.0
                        if cc_min >= 0 and cc < cc_min:
                            obj_pen += (cc_min - cc) / max(1.0, float(cc_min))
                        if cc_max >= 0 and cc > cc_max:
                            obj_pen += (cc - cc_max) / max(1.0, float(cc_max))
                        if h_min >= 0 and holes < h_min:
                            obj_pen += (h_min - holes) / max(1.0, float(h_min))
                        if h_max >= 0 and holes > h_max:
                            obj_pen += (holes - h_max) / max(1.0, float(h_max))
                        struct_pen += obj_pen
                        struct_dbg.append({"idx": oe.idx, "name": oe.name, "obj_pen": obj_pen, "cc": cc, "holes": holes})
                    else:
                        fill_ratio = float(oe.fill_ratio or 0.0)
                        mixed_cell_ratio = float(oe.mixed_cell_ratio or 0.0)
                        span_lr = int(oe.span_lr or 0)
                        span_tb = int(oe.span_tb or 0)
                        transition_h = float(oe.transition_h or 0.0)
                        transition_v = float(oe.transition_v or 0.0)
                        transition_strength = float(oe.transition_strength or 0.0)
                        line_complexity = float(oe.line_complexity or 0.0)

                        fill_min = float(obj.get("fill_min", args.fill_min_default))
                        fill_max = float(obj.get("fill_max", args.fill_max_default))
                        mixed_min = float(obj.get("mixed_cell_min", args.mixed_cell_min_default))
                        span_min = int(obj.get("span_min", args.span_min_default))

                        fill_pen = max(0.0, fill_min - fill_ratio) + max(0.0, fill_ratio - fill_max)
                        mixed_pen = max(0.0, mixed_min - mixed_cell_ratio) / max(1e-6, mixed_min)
                        span_pen = max(0.0, span_min - max(span_lr, span_tb)) / max(1.0, float(span_min))

                        if str(args.struct_mode).lower() == "porous_v2":
                            transition_pen = 0.0
                            obj_pen = (
                                float(args.w_struct_fill) * fill_pen
                                + float(args.w_struct_mixed) * mixed_pen
                                + float(args.w_struct_span) * span_pen
                            )
                        else:
                            transition_min = float(obj.get("transition_min", args.transition_min_default))
                            complexity_strength = max(line_complexity, transition_strength)
                            transition_pen = max(0.0, transition_min - complexity_strength) / max(1e-6, transition_min)
                            obj_pen = (
                                float(args.w_struct_fill) * fill_pen
                                + float(args.w_struct_mixed) * mixed_pen
                                + float(args.w_struct_span) * span_pen
                                + float(args.w_struct_transition) * transition_pen
                            )

                        struct_pen += obj_pen
                        struct_dbg.append({
                            "idx": oe.idx,
                            "name": oe.name,
                            "obj_pen": obj_pen,
                            "fill_ratio": fill_ratio,
                            "mixed_cell_ratio": mixed_cell_ratio,
                            "span_lr": span_lr,
                            "span_tb": span_tb,
                            "transition_h": transition_h,
                            "transition_v": transition_v,
                            "transition_strength": transition_strength,
                            "line_complexity": line_complexity,
                            "fill_pen": fill_pen,
                            "mixed_pen": mixed_pen,
                            "span_pen": span_pen,
                            "transition_pen": transition_pen,
                        })

            quality_pen = quality_pen + float(args.w_touch) * float(touch_pen) + float(args.w_struct) * float(struct_pen)


            dbg_q.update(touch_pen=touch_pen, touch_ratio_mean=touch_ratio_mean, struct_pen=struct_pen, struct_mode=str(args.struct_mode).lower(), struct_details=struct_dbg)



            scalar_base = pair_pen_mean + presence_pen


            scalar_legacy = scalar_base + quality_pen



            depth_pen = 0.0


            if depth_order_acc is not None:


                depth_pen = 1.0 - float(depth_order_acc)


            occl_term = scalar_base + float(args.w_depth) * float(depth_pen)


            scalar50 = 0.5 * (float(args.w_occl) * occl_term) + 0.5 * (float(args.w_quality) * quality_pen)



            scalar = scalar50 if str(args.scalar_mode).lower() == "scalar50" else scalar_legacy
            row: Dict[str, Any] = {
                "scene": scene_name,
                "image": ip.name,
                "scalar": scalar,
                "scalar50": scalar50,
                "scalar_legacy": scalar_legacy,
                "occl_term": occl_term,
                "depth_pen": depth_pen,
                "scalar_base": scalar_base,
                "quality_pen": quality_pen,
                "presence_pen": presence_pen,
                "pair_pen_mean": pair_pen_mean,
                "inter_mean": inter_mean,
                "clip_overall": clip_overall,
                "depth_order_acc": depth_order_acc,
            }
            for k, v in dbg_q.items():
                row[k] = v

            for oe in obj_evals[1:]:
                row[f"present_obj_{oe.idx}_{oe.name}"] = int(oe.present)
                row[f"area_bbox_{oe.idx}_{oe.name}"] = oe.area_ratio_in_bbox
                if oe.bbox_iou is not None:
                    row[f"bbox_iou_{oe.idx}_{oe.name}"] = oe.bbox_iou
                if oe.thin_blob_frac is not None:
                    row[f"thin_blob_{oe.idx}_{oe.name}"] = oe.thin_blob_frac
                if oe.thin_thick_mean is not None:
                    row[f"thin_thick_mean_{oe.idx}_{oe.name}"] = oe.thin_thick_mean
                if oe.clip_score is not None:
                    row[f"clip_obj_{oe.idx}_{oe.name}"] = oe.clip_score
                if oe.depth_median is not None:
                    row[f"depth_obj_{oe.idx}_{oe.name}"] = oe.depth_median

            rows.append(row)

            details[scene_name]["images"][ip.name] = {
                "scalar": scalar,
                "scalar_base": scalar_base,
                "quality_pen": quality_pen,
                "presence_pen": presence_pen,
                "pair_pen_mean": pair_pen_mean,
                "inter_mean": inter_mean,
                "clip_overall": clip_overall,
                "depth_order_acc": depth_order_acc,
                "quality": dbg_q if dbg_q else None,
                "objects": [oe.__dict__ for oe in obj_evals],
                "pairs": [pe.__dict__ for pe in pair_evals],
            }

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
    except Exception:
        with out_csv.open("w", encoding="utf-8") as f:
            if rows:
                keys = list(rows[0].keys())
                f.write(",".join(keys) + "\n")
                for r in rows:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    out_json.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved CSV : {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
