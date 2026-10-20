# demos_occlusion_sam_attn.py (v8: avoid-aware attn-guided SAM + joint masks + fusion schedule + refine selection)
from __future__ import annotations

import argparse
import os
import time
import json
import copy
import inspect
import cv2
import numpy as np
from PIL import Image
import torch
import re

from larender_sdxl_sam_attn import LaRender
from sam_masker_v6 import (
    SamMaskerV6,
    SamConfigV6,
    SamAttnPromptCfg,
    SamGenericScoreCfg,
    POSTPROCESS_REGISTRY,
    _to_box_xyxy_pixels,
    shuttle_shape_score,
)
from utils import infer_project_root, ensure_dir, token_indices_for_keywords, pretty_selected_tokens

torch.use_deterministic_algorithms(False)


def save_masks_debug(masks, save_dir: str):
    ensure_dir(save_dir)
    for i, m in enumerate(masks):
        if m is None:
            continue
        m = np.asarray(m)
        m8 = np.clip(m * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"mask_{i:02d}.png"), m8)
        
def _json_default(o):
    """Make json.dump robust to numpy / torch objects."""
    try:
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.bool_,)):
            return bool(o)
    except Exception:
        pass

    try:
        import torch as _torch
        if isinstance(o, _torch.Tensor):
            return o.detach().cpu().tolist()
    except Exception:
        pass

    if hasattr(o, "__dict__"):
        return dict(o.__dict__)
    return str(o)


def _parse_occlusion_pair(p):
    """Parse one occlusion pair config from scene JSON.

    Supported formats:
      - {"back": i, "front": j, ...}
      - {"back_id": i, "front_id": j, ...}
      - [i, j] / (i, j)
      - {"a": i, "b": j, ...}  (legacy fallback)
    """
    if isinstance(p, (list, tuple)):
        if len(p) < 2:
            raise ValueError(f"Invalid occlusion pair (need 2 ids): {p}")
        return int(p[0]), int(p[1]), {}

    if not isinstance(p, dict):
        raise TypeError(f"Invalid occlusion pair type: {type(p)} -> {p}")

    def _first(*keys):
        for k in keys:
            if k in p and p[k] is not None:
                return p[k]
        return None

    back = _first("back", "back_id", "back_idx", "bg", "behind", "occluded", "target")
    front = _first("front", "front_id", "front_idx", "fg", "ahead", "occluder")

    if (back is None or front is None) and ("a" in p and "b" in p):
        back = p.get("a")
        front = p.get("b")

    if back is None or front is None:
        raise KeyError(
            "Occlusion pair must contain ids. Supported keys: back/front or back_id/front_id (or list [back,front]). "
            f"Got: {p}"
        )

    return int(back), int(front), p


def _pair_minmax(pdict, default_min=0.0, default_max=1.0):
    if not isinstance(pdict, dict):
        return float(default_min), float(default_max)
    minv = pdict.get("min_occlusion", pdict.get("min_occl", pdict.get("min", default_min)))
    maxv = pdict.get("max_occlusion", pdict.get("max_occl", pdict.get("max", default_max)))
    return float(minv), float(maxv)


_REL_DIRECTIVE_RE = re.compile(r"<<\s*(front_of|in_front_of|behind|back_of)\s*:\s*([A-Za-z0-9_\-]+)\s*>>", re.IGNORECASE)

def _strip_relation_directives(text: str):
    """Strip relation directives from a prompt and return (clean_text, relations).

    Supported directives (case-insensitive):
      - <<front_of:NAME>> / <<in_front_of:NAME>>  => this object should be in front of NAME
      - <<behind:NAME>> / <<back_of:NAME>>        => this object should be behind NAME

    NAME is matched against object['name'] (after we auto-fill missing names).
    """
    if not isinstance(text, str) or not text:
        return text, []
    rels = []
    def _repl(m):
        rel = (m.group(1) or '').lower()
        target = (m.group(2) or '').strip()
        if rel and target:
            rels.append((rel, target))
        return ''  # remove the directive from the prompt string

    cleaned = _REL_DIRECTIVE_RE.sub(_repl, text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned, rels

def _stable_toposort(nodes, edges, base_order):
    """Stable Kahn topological sort (falls back to base_order if cyclic)."""
    nodes = list(nodes)
    order_rank = {n: i for i, n in enumerate(base_order)}
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for u, v in edges:
        if u not in indeg or v not in indeg:
            continue
        if u == v:
            continue
        adj[u].append(v)
        indeg[v] += 1

    avail = [n for n in nodes if indeg[n] == 0]
    avail.sort(key=lambda n: order_rank.get(n, 10**9))

    out = []
    while avail:
        u = avail.pop(0)
        out.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                avail.append(v)
        avail.sort(key=lambda n: order_rank.get(n, 10**9))

    if len(out) != len(nodes):
        # cycle or disconnected weirdness
        return list(base_order)
    return out

def normalize_scene_for_occlusion(scene: dict):
    """Normalize a scene dict so occlusion order can be controlled by prompts/pairs.

    - auto-fills missing object['name']
    - parses prompt directives <<front_of:NAME>> / <<behind:NAME>> and adds them as occlusion_pairs
    - optionally reorders objects (except background index 0) using a stable topological sort
      according to the (merged) occlusion_pairs, so changing directives changes actual layering.
    - remaps occlusion_pairs in scene and refine_select to match the new order
    """
    if not isinstance(scene, dict):
        return scene, None

    scene = copy.deepcopy(scene)

    objects = list(scene.get('objects', []) or [])
    n = len(objects)
    if n == 0:
        return scene, None

    # Fill names
    for i, o in enumerate(objects):
        if not isinstance(o, dict):
            o = {'prompt': str(o)}
            objects[i] = o
        name = o.get('name', None)
        if not isinstance(name, str) or not name.strip():
            kws = o.get('keywords') or []
            if isinstance(kws, list) and len(kws) > 0 and isinstance(kws[0], str) and kws[0].strip():
                base = kws[0].strip().lower()
            else:
                base = f'obj{i}'
            base = re.sub(r'[^a-z0-9_\-]+', '_', base)
            o['name'] = base

    name2idx = {str(o.get('name')): i for i, o in enumerate(objects)}

    # Parse directives and strip them from prompts
    auto_pairs = []
    for i, o in enumerate(objects):
        pr, rels = _strip_relation_directives(o.get('prompt', ''))
        o['prompt'] = pr
        neg, _ = _strip_relation_directives(o.get('negative_prompt', ''))
        o['negative_prompt'] = neg

        for rel, target in rels:
            tid = name2idx.get(target) or name2idx.get(str(target).lower())
            if tid is None:
                continue
            if rel in ('front_of', 'in_front_of'):
                auto_pairs.append({'back': tid, 'front': i, 'mode': 'auto'})
            elif rel in ('behind', 'back_of'):
                auto_pairs.append({'back': i, 'front': tid, 'mode': 'auto'})

    # Merge pairs (scene + refine_select + auto)
    def _pairs_as_list(x):
        return list(x) if isinstance(x, (list, tuple)) else []

    scene_pairs = _pairs_as_list(scene.get('occlusion_pairs', []))
    rs = scene.get('refine_select', {}) if isinstance(scene.get('refine_select', {}), dict) else {}
    rs_pairs = _pairs_as_list(rs.get('occlusion_pairs', []))

    merged_pairs = []
    seen = set()
    for p in (scene_pairs + rs_pairs + auto_pairs):
        back, front, pd = _parse_occlusion_pair(p)
        if back is None or front is None:
            continue
        if not (0 <= back < n and 0 <= front < n):
            continue
        key = (back, front)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(p, dict):
            pd2 = dict(p)
            pd2.pop('pair', None)
            pd2.pop('back', None)
            pd2.pop('front', None)
            pd2.pop('back_id', None)
            pd2.pop('front_id', None)
            pd2.pop('back_idx', None)
            pd2.pop('front_idx', None)
        else:
            pd2 = {}
        pd2['back'] = back
        pd2['front'] = front
        merged_pairs.append(pd2)

    # Update scene_pairs/rs_pairs if missing (so directives can drive selection too)
    if not scene_pairs and auto_pairs:
        scene_pairs = auto_pairs
    else:
        # keep existing + auto (dedup)
        scene_pairs = merged_pairs

    if not rs_pairs and auto_pairs:
        rs_pairs = auto_pairs
    else:
        rs_pairs = merged_pairs

    scene['occlusion_pairs'] = scene_pairs
    if 'refine_select' in scene and isinstance(scene['refine_select'], dict):
        scene['refine_select']['occlusion_pairs'] = rs_pairs

    # Optional: reorder objects to satisfy occlusion order
    if scene.get('auto_reorder', True):
        bg = 0
        nodes = list(range(n))
        nodes_wo_bg = [i for i in nodes if i != bg]
        edges = []
        for p in merged_pairs:
            back = int(p.get('back'))
            front = int(p.get('front'))
            if back == bg or front == bg:
                continue
            if 0 <= back < n and 0 <= front < n and back != front:
                edges.append((back, front))

        base_order = nodes_wo_bg[:]  # stable
        sorted_wo_bg = _stable_toposort(nodes_wo_bg, edges, base_order)
        new_order = [bg] + sorted_wo_bg

        if new_order != nodes:
            objects_new = [objects[i] for i in new_order]
            old2new = {old: new for new, old in enumerate(new_order)}
            scene['objects'] = objects_new

            def _remap_pairs(pairs):
                out = []
                for p in _pairs_as_list(pairs):
                    back, front, pd = _parse_occlusion_pair(p)
                    if back is None or front is None:
                        continue
                    if back not in old2new or front not in old2new:
                        continue
                    # preserve extra fields
                    if isinstance(p, dict):
                        d = dict(p)
                        d.pop('pair', None)
                    else:
                        d = dict(pd)
                    d['back'] = old2new[back]
                    d['front'] = old2new[front]
                    out.append(d)
                return out

            scene['occlusion_pairs'] = _remap_pairs(scene.get('occlusion_pairs', []))
            if 'refine_select' in scene and isinstance(scene['refine_select'], dict):
                scene['refine_select']['occlusion_pairs'] = _remap_pairs(scene['refine_select'].get('occlusion_pairs', []))

            return scene, old2new

    scene['objects'] = objects
    return scene, None
def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert("RGB"))
    return arr[:, :, ::-1].copy()


def load_scene(scene_path: str) -> dict:
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
        # Optional: allow <<front_of:NAME>> / <<behind:NAME>> directives in prompts,
        # and reorder objects to match occlusion pairs (so prompt edits can change occlusion order).
        scene, _order_map = normalize_scene_for_occlusion(scene)
        if _order_map is not None:
            print('[Scene] auto_reorder applied (old_index->new_index):', _order_map)
    assert "objects" in scene and isinstance(scene["objects"], list) and len(scene["objects"]) >= 2, \
        "scene.json must contain a non-empty 'objects' list (back -> front)."
    return scene


def build_indices_from_scene(objects: list, tokenizer) -> list:
    indices = []
    for obj in objects:
        prompt = obj["prompt"]
        kws = obj.get("keywords", None) or []
        idx = token_indices_for_keywords(prompt, kws, tokenizer) if kws else []
        # fallback: try last noun-ish keyword, else just use token 1
        if len(idx) == 0:
            # try to use the first non-empty word as a weak anchor
            first_word = str(prompt).strip().split(" ")[0] if str(prompt).strip() else ""
            idx = token_indices_for_keywords(prompt, [first_word], tokenizer) if first_word else []
        if len(idx) == 0:
            idx = [1]  # token 1 usually corresponds to the first real token after BOS
        indices.append(idx)
        print(f"[TOKEN] prompt='{prompt}' keywords={kws} -> {pretty_selected_tokens(prompt, tokenizer, idx)}")
    return indices


def joint_mask_correction_soft(masks_soft: list) -> list:
    """方案2：多对象联合修正。

    Enforce (soft) non-overlap by subtracting the union of front objects from
    each back object's mask.

    Assumes objects are ordered back -> front.
    """
    if not masks_soft:
        return masks_soft
    H, W = None, None
    for m in masks_soft:
        if m is not None:
            H, W = m.shape[:2]
            break
    if H is None:
        return masks_soft

    union_front = np.zeros((H, W), dtype=np.float32)
    out = list(masks_soft)
    # iterate front -> back
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if m is None:
            continue
        m2 = np.clip(m.astype(np.float32) * (1.0 - union_front), 0.0, 1.0)
        out[i] = m2
        union_front = np.maximum(union_front, m2)
    return out


def extract_masks_attention_guided(
    masker: SamMaskerV6,
    bgr: np.ndarray,
    locations: list,
    use_sam: list,
    attn_maps: list,
    sam_mode: list,
    postprocess: list,
    postprocess_kwargs: list,
    sam_pad: list,
    prompt_cfg: SamAttnPromptCfg,
    score_cfg: SamGenericScoreCfg,
) -> Tuple[list, list]:
    """Extract per-object masks with attention-guided SAM (方案1).

    Returns:
        masks_soft: list of soft masks (float32 in [0,1]) or None
        scores: list of float scores (for draft selection)
    """
    H, W = bgr.shape[:2]
    n = len(locations)
    masks_soft = [None for _ in range(n)]
    scores = [0.0 for _ in range(n)]

    # segment front -> back so that back objects can avoid front union
    union_front_bin = np.zeros((H, W), dtype=np.uint8)

    masker.set_image_bgr(bgr)
    for i in range(n - 1, -1, -1):
        if not bool(use_sam[i]):
            continue

        box = _to_box_xyxy_pixels(locations[i], H=H, W=W, pad=float(sam_pad[i]))
        attn_self = attn_maps[i] if (attn_maps is not None and i < len(attn_maps)) else None
        other_attn = []
        if attn_maps is not None:
            for k in range(n):
                if k == i:
                    continue
                if k < len(attn_maps) and attn_maps[k] is not None:
                    other_attn.append(attn_maps[k])

        # Use union of front objects as an "avoid" hint for both point prompting
        # and mask scoring.
        avoid_mask = union_front_bin if int(union_front_bin.sum()) > 0 else None

        # Pass-A: attention-guided points
        pos_a, neg_a = SamMaskerV6.points_from_attn(
            box_xyxy=box,
            attn_map=attn_self,
            other_attn_maps=other_attn,
            cfg=prompt_cfg,
            extra_neg=None,
            avoid_mask=avoid_mask,
        )
        mask_a, s_a = masker.predict_best(
            box_xyxy=box,
            pos_xy=pos_a,
            neg_xy=neg_a,
            mode=str(sam_mode[i] or "generic"),
            multimask=True,
            attn_map=attn_self,
            avoid_mask=avoid_mask,
            score_cfg=score_cfg,
        )

        # Pass-B: fallback (box-geometry) points, still scored with attn_self.
        pos_b, neg_b = SamMaskerV6.points_from_attn(
            box_xyxy=box,
            attn_map=None,
            other_attn_maps=other_attn,
            cfg=prompt_cfg,
            extra_neg=None,
            avoid_mask=avoid_mask,
        )
        mask_b, s_b = masker.predict_best(
            box_xyxy=box,
            pos_xy=pos_b,
            neg_xy=neg_b,
            mode=str(sam_mode[i] or "generic"),
            multimask=True,
            attn_map=attn_self,
            avoid_mask=avoid_mask,
            score_cfg=score_cfg,
        )

        if float(s_b) > float(s_a):
            mask_bin, s = mask_b, s_b
        else:
            mask_bin, s = mask_a, s_a

        scores[i] = float(s)

        if mask_bin is None:
            masks_soft[i] = None
            continue

        pp_name = str(postprocess[i] or "generic").lower()
        pp_fn = POSTPROCESS_REGISTRY.get(pp_name, POSTPROCESS_REGISTRY["generic"])
        kw = postprocess_kwargs[i] if postprocess_kwargs is not None else {}
        try:
            # Filter postprocess kwargs by function signature to avoid unexpected-arg errors,
            # and optionally pass image/attn when supported (needed for thin structures).
            pp_kwargs = dict(kw or {})
            try:
                sig = inspect.signature(pp_fn)
                params = sig.parameters
                accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                allowed = set(params.keys())
                if not accepts_varkw:
                    pp_kwargs = {k: v for k, v in pp_kwargs.items() if k in allowed}
                if (accepts_varkw or 'image_bgr' in allowed) and 'image_bgr' not in pp_kwargs:
                    pp_kwargs['image_bgr'] = bgr
                if (accepts_varkw or 'attn_map' in allowed) and 'attn_map' not in pp_kwargs:
                    if attn_self is not None:
                        if hasattr(attn_self, 'detach'):
                            pp_kwargs['attn_map'] = attn_self.detach().float().cpu().numpy()
                        else:
                            pp_kwargs['attn_map'] = attn_self
            except Exception:
                pp_kwargs = dict(kw or {})
            mask_soft = pp_fn(mask_bin.astype(np.uint8), box, **pp_kwargs)
        except TypeError:
            # fallback if kwargs mismatch
            mask_soft = pp_fn(mask_bin.astype(np.uint8), box)

        masks_soft[i] = mask_soft

        if mask_soft is not None:
            upd = (mask_soft > 0.5).astype(np.uint8)
            # Slightly dilate to make avoidance more robust for back objects.
            k = int(getattr(prompt_cfg, "avoid_dilate", 0) or 0)
            k = int(np.clip(k, 0, 21))
            if k and k > 1:
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                upd = cv2.dilate(upd, kernel, iterations=1)
            union_front_bin = np.maximum(union_front_bin, upd)

    return masks_soft, scores


def run_scene(scene_path: str, root: str, out_dir: str):
    scene = load_scene(scene_path)

    # dataset metadata (optional; does not affect rendering unless used below)
    scene_tags = set([str(x).lower() for x in (scene.get("tags") or [])])
    scene_cat = str(scene.get("category", "")).lower()

    # basic
    # allow shell scripts to override resolution without editing json
    height = int(os.environ.get("OCCLUSION_HEIGHT", scene.get("height", 1024)))
    width = int(os.environ.get("OCCLUSION_WIDTH", scene.get("width", 1024)))
    seed = int(scene.get("seed", 0))
    negative_prompt = str(scene.get("negative_prompt", ""))
        # sampling budget
    # K refers to the number of *refine* candidates for best-of-K selection (to match baseline best-of-16).
    # You can override by scene.json ("k_candidates"/"K"/"num_refine") or env OCCLUSION_K.
    K = int(os.environ.get("OCCLUSION_K", scene.get("k_candidates", scene.get("K", 16))))
    num_draft = int(os.environ.get("OCCLUSION_NUM_DRAFT", scene.get("num_draft", 6)))
    num_refine = int(scene.get("num_refine", K))
    if num_refine < K:
        print(f"[WARN] num_refine({num_refine}) < K({K}); bumping num_refine to {K} for fair best-of-K.")
        num_refine = K

    # model
    pretrain = str(scene.get("pretrain", "IterComp"))
    density_scheduler = str(scene.get("density_scheduler", "inverse_proportional"))

    # objects (back -> front)
    objects_cfg = scene["objects"]
    objects = [o["prompt"] for o in objects_cfg]
    locations = [o["bbox"] for o in objects_cfg]
    alpha = [float(o.get("alpha", 0.8)) for o in objects_cfg]
    use_sam = [bool(o.get("use_sam", True)) for o in objects_cfg]
    use_attn = [bool(o.get("use_attn", True)) for o in objects_cfg]
    sam_mode = [str(o.get("sam_mode", "generic")) for o in objects_cfg]
    postprocess = [str(o.get("postprocess", "generic")) for o in objects_cfg]
    postprocess_kwargs = [dict(o.get("postprocess_kwargs") or o.get("postprocess_cfg") or {}) for o in objects_cfg]
    sam_pad = [float(o.get("sam_pad", 0.05)) for o in objects_cfg]

    # -----------------------------
    # Thin-structure typing + default postprocess params
    # -----------------------------
    # We split "thin occluders" into:
    #   - porous: fence/web (holey masks) -> keep *soft* attention and preserve holes
    #   - sheet: lace curtain / branches -> keep *boosted coverage* so they don't vanish
    def _name_has(obj: dict, keys: list) -> bool:
        s = (str(obj.get("name", "")) + " " + str(obj.get("prompt", ""))).lower()
        return any(k in s for k in keys)

    holey_fronts_pairs = set()
    for _p in (scene.get("occlusion_pairs") or []):
        try:
            _b, _f, _pd = _parse_occlusion_pair(_p)
            if str(_pd.get("mode", "opaque")).lower() in ("holey", "porous", "mesh", "thin", "transparent"):
                holey_fronts_pairs.add(int(_f))
        except Exception:
            pass

    porous_cats = {"fence","web","net","beads","glass","wire"}
    sheet_cats = {"lace","branches","blinds"}
    scene_is_sheet = ("thin_sheet" in scene_tags) or (scene_cat in sheet_cats)
    scene_is_porous = ("thin_porous" in scene_tags) or (scene_cat in porous_cats)

    thin_porous = set()
    thin_sheet = set()
    for _i, _o in enumerate(objects_cfg):
        pp = str(postprocess[_i] or "").lower()
        sm = str(sam_mode[_i] or "").lower()
        is_thin = (pp in ("thin_edges", "lace")) or (sm in ("fence", "web", "lace")) or _name_has(_o, ["lace", "curtain", "branches", "branch", "twig", "blinds"])
        if not is_thin:
            continue
        # porous vs sheet: prefer dataset tags/category when available
        is_porous = False
        if not scene_is_sheet:
            if (_i in holey_fronts_pairs) or (sm in ("fence", "web")) or (scene_is_porous and pp in ("thin_edges", "lace")):
                is_porous = True
        if is_porous:
            thin_porous.add(_i)
        else:
            thin_sheet.add(_i)

    scene_name = os.path.splitext(os.path.basename(scene_path))[0].lower()
    for _i in sorted(list(thin_porous | thin_sheet)):
        pp = str(postprocess[_i] or "").lower()
        if pp not in ("thin_edges", "lace"):
            continue
        kw = postprocess_kwargs[_i] if isinstance(postprocess_kwargs[_i], dict) else {}
        # Fill defaults only when missing, so scene.json can override.
        if _i in thin_porous:
            defaults = dict(
                restrict_to_mask=False,
                attn_thresh=0.22,
                attn_dilate_k=5,
                canny1=50,
                canny2=150,
                max_thick=5,
                dilate_k=0,
                feather_k=9,
                feather_sigma=2.0,
                min_edge_frac=0.0018,
                fallback_alpha=0.22,
            )
        else:
            # sheet-like thin: lace curtain / branches / blinds
            is_branches = ("branches" in scene_name) or _name_has(objects_cfg[_i], ["branches", "branch", "twig"])
            is_blinds = ("blinds" in scene_name) or _name_has(objects_cfg[_i], ["blinds", "window blinds"])
            defaults = dict(
                restrict_to_mask=False,   # crucial: keep thin structure over overlapped regions
                attn_thresh=0.15 if is_branches else (0.18 if is_blinds else 0.17),
                attn_dilate_k=9 if is_branches else (9 if is_blinds else 7),
                canny1=20 if is_branches else (30 if is_blinds else 25),
                canny2=110 if is_branches else (140 if is_blinds else 120),
                max_thick=7,
                dilate_k=3 if is_branches else (2 if is_blinds else 2),
                feather_k=11 if is_branches else (9 if is_blinds else 13),
                feather_sigma=2.5 if is_branches else (2.0 if is_blinds else 3.0),
                min_edge_frac=0.0026 if is_branches else (0.0030 if is_blinds else 0.0030),
                fallback_alpha=0.40 if is_branches else (0.30 if is_blinds else 0.35),
            )
        for k, v in defaults.items():
            if k not in kw or kw.get(k) is None:
                kw[k] = v
        postprocess_kwargs[_i] = kw

    # Number of objects in this scene (used in later loops).
    n_obj = len(objects_cfg)

    # -----------------------------
    # Draft-stage alpha softening for porous/holey foregrounds
    # -----------------------------
    # Stage-1 uses bbox-only masks. If a porous/holey occluder (e.g., fence/web/lace)
    # has a high alpha and a large bbox, it can fully block the back layer in drafts,
    # making SAM mask extraction for the back object fail. We therefore soften those
    # occluders ONLY during Stage-1 (draft) to reveal back layers. Stage-2 keeps the
    # original alpha values.
    alpha_draft = list(alpha)
    draft_holey_alpha_scale = float(scene.get("draft_holey_alpha_scale", 0.45))
    holey_fronts = set()
    for p in (scene.get("occlusion_pairs") or []):
        if str(p.get("mode", "opaque")).lower() in ["holey", "porous", "mesh", "thin"]:
            try:
                holey_fronts.add(int(p.get("front")))
            except Exception:
                pass
    for i in range(len(objects_cfg)):
        if (
            i in holey_fronts
            or (postprocess[i] in ["thin_edges", "lace"])
            or (sam_mode[i] in ["fence", "web", "lace"])
        ):
            alpha_draft[i] = min(alpha_draft[i], alpha_draft[i] * draft_holey_alpha_scale)

    # fusion config (方案3)
    mf = dict(scene.get("mask_fusion", {}))
    mask_fusion_mode = str(mf.get("mode", "sam_attn_blend"))
    fusion_lambda_start = mf.get("lambda_start", 0.90)
    fusion_lambda_end = mf.get("lambda_end", 0.20)
    fusion_schedule = str(mf.get("schedule", "linear"))
    fusion_t_start = float(mf.get("t_start", 0.0))
    fusion_t_end = float(mf.get("t_end", 1.0))
    attn_power = float(mf.get("attn_power", 1.5))
    attn_floor = float(mf.get("attn_floor", 0.10))
    attn_power_per_object = mf.get("attn_power_per_object", None)
    attn_floor_per_object = mf.get("attn_floor_per_object", None)
    attn_cfg_mode = str(mf.get("attn_cfg_mode", "cond"))


    # -----------------------------
    # Thin attention strategy (porous vs sheet)
    # -----------------------------
    # Default is "auto":
    #   porous thin (fence/web) -> soft (power ~= 1.05, floor >= 0.18)
    #   sheet-like thin (lace/branches/blinds) -> boost coverage (power < 1.0, floor >= 0.25)
    thin_attn_strategy = str(scene.get("thin_attn_strategy", "auto")).lower()
    porous_attn_power = float(scene.get("porous_attn_power", 1.05))
    porous_attn_floor = float(scene.get("porous_attn_floor", 0.18))
    sheet_attn_power = float(scene.get("sheet_attn_power", 0.85))
    sheet_attn_floor = float(scene.get("sheet_attn_floor", 0.25))

    if thin_attn_strategy in ("skip", "disable"):
        for _i in (thin_porous | thin_sheet):
            use_attn[_i] = False
    else:
        # Ensure per-object arrays are editable lists
        if not isinstance(attn_power_per_object, (list, tuple)):
            attn_power_per_object = [None for _ in range(n_obj)]
        else:
            attn_power_per_object = list(attn_power_per_object) + [None] * max(0, n_obj - len(attn_power_per_object))
        if not isinstance(attn_floor_per_object, (list, tuple)):
            attn_floor_per_object = [None for _ in range(n_obj)]
        else:
            attn_floor_per_object = list(attn_floor_per_object) + [None] * max(0, n_obj - len(attn_floor_per_object))

        for _i in sorted(list(thin_porous)):
            if not bool(use_attn[_i]):
                continue
            cur_p = attn_power_per_object[_i] if attn_power_per_object[_i] is not None else attn_power
            cur_f = attn_floor_per_object[_i] if attn_floor_per_object[_i] is not None else attn_floor
            attn_power_per_object[_i] = float(min(float(cur_p), porous_attn_power))
            attn_floor_per_object[_i] = float(max(float(cur_f), porous_attn_floor))

        for _i in sorted(list(thin_sheet)):
            if not bool(use_attn[_i]):
                continue
            cur_p = attn_power_per_object[_i] if attn_power_per_object[_i] is not None else attn_power
            cur_f = attn_floor_per_object[_i] if attn_floor_per_object[_i] is not None else attn_floor
            # power < 1 spreads attention mass -> prevents sheets/branches vanishing
            attn_power_per_object[_i] = float(min(float(cur_p), sheet_attn_power))
            attn_floor_per_object[_i] = float(max(float(cur_f), sheet_attn_floor))

    # attention export range for Stage-1 (draft) SAM prompting
    ae = dict(scene.get("attn_export", {}))
    attn_export_start = float(ae.get("start", 0.35))
    attn_export_end = float(ae.get("end", 1.0))

    # SAM prompting cfg (方案1)
    sp = dict(scene.get("sam_prompt", {}))
    prompt_cfg = SamAttnPromptCfg(**{k: v for k, v in sp.items() if k in SamAttnPromptCfg.__annotations__})

    ss = dict(scene.get("sam_score", {}))
    score_cfg = SamGenericScoreCfg(**{k: v for k, v in ss.items() if k in SamGenericScoreCfg.__annotations__})

    joint_enable = bool(dict(scene.get("joint_mask", {})).get("enable", True))

    # init LaRender
    LR = LaRender(
        pretrain=pretrain,
        transmittance_use_bbox=True,
        transmittance_use_attn_map=True,
        density_scheduler=density_scheduler,
        cache_dir=os.environ.get("HF_HUB_CACHE", None),
        local_files_only=(os.environ.get("LOCAL_FILES_ONLY", "0") == "1"),
    )

    indices = build_indices_from_scene(objects_cfg, LR.pipe.tokenizer)

    # init SAM
    sam_ckpt = os.environ.get("SAM_CHECKPOINT", os.path.join(root, "models", "sam", "sam_vit_h_4b8939.pth"))
    if not os.path.exists(sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}")
    sam_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    sam_dev = os.environ.get("SAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    masker = SamMaskerV6(SamConfigV6(ckpt=sam_ckpt, model_type=sam_type, device=sam_dev))

    ensure_dir(out_dir)

    # ===================== Stage 1: draft selection =====================
    print(f"[Stage-1] Draft x{num_draft} (bbox-only render; export attn for SAM scoring)...")
    best = {
        "score": -1e9,
        "draft_idx": -1,
        "image": None,
        "masks": None,
        "scores": None,
    }

    for d in range(num_draft):
        t0 = time.time()
        (imgs, attn_maps) = LR.run(
            objects=objects,
            locations=locations,
            alpha=alpha_draft,
            indices=indices,
            masks=None,
            mask_fusion="sam_attn",  # unused since masks=None
            attn_power=attn_power,
            attn_floor=0.0,
            skip_attn_indices=list(range(len(objects))),  # bbox only
            export_attn=True,
            return_attn=True,
            attn_cfg_mode=attn_cfg_mode,
            attn_export_start=attn_export_start,
            attn_export_end=attn_export_end,
            num_images_per_prompt=1,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            seed=seed + d,
        )
        img = imgs[0]
        img.save(os.path.join(out_dir, f"draft_{d:02d}.jpg"))
        bgr = pil_to_bgr(img)

        masks_soft, scores = extract_masks_attention_guided(
            masker=masker,
            bgr=bgr,
            locations=locations,
            use_sam=use_sam,
            attn_maps=attn_maps,
            sam_mode=sam_mode,
            postprocess=postprocess,
            postprocess_kwargs=postprocess_kwargs,
            sam_pad=sam_pad,

            prompt_cfg=prompt_cfg,
            score_cfg=score_cfg,
        )
        if joint_enable:
            masks_soft = joint_mask_correction_soft(masks_soft)

        # total score: sum over objects that use_sam
        total = 0.0
        for i in range(len(objects)):
            if use_sam[i]:
                total += float(scores[i])

        dt = time.time() - t0
        print(f"  draft[{d}] score={total:.4f} time={dt:.2f}s")
        if total > float(best["score"]):
            best.update(score=total, draft_idx=d, image=img, masks=masks_soft, scores=scores)

        # save draft masks debug
# # save_masks_debug(masks_soft, os.path.join(out_dir, f"draft_{d:02d}_masks"))

    best_img = best["image"]
    best_img.save(os.path.join(out_dir, "draft_best.jpg"))
    print(f"[Stage-1] best draft = {best['draft_idx']} score={best['score']:.4f}")

    # ===================== Stage 2: refine =====================
    masks = best["masks"]
    skip_attn_indices = [i for i, ua in enumerate(use_attn) if not ua]

    print(f"[Stage-2] Refine x{num_refine} (mask_fusion={mask_fusion_mode} + schedule)...")
    t1 = time.time()

    refine_bs = int(os.environ.get("OCCLUSION_REFINE_BATCH", "1"))  # 1最稳，2更快但更吃显存
    images = []

    for start in range(0, num_refine, refine_bs):
        bs = min(refine_bs, num_refine - start)
        imgs = LR.run(
            objects=objects,
            locations=locations,
            alpha=alpha,
            indices=indices,
            masks=masks,
            mask_fusion=mask_fusion_mode,
            fusion_lambda_start=fusion_lambda_start,
            fusion_lambda_end=fusion_lambda_end,
            fusion_schedule=fusion_schedule,
            fusion_t_start=fusion_t_start,
            fusion_t_end=fusion_t_end,
            attn_power=attn_power,
            attn_power_per_object=attn_power_per_object,
            attn_floor=attn_floor,
            attn_floor_per_object=attn_floor_per_object,
            skip_attn_indices=skip_attn_indices,
            num_images_per_prompt=bs,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            seed=seed + 10_000 + start,  # 关键：每个小批次 seed 不同，避免重复
        )
        images.extend(list(imgs))
        torch.cuda.empty_cache()
    print(f"[Stage-2] Done in {time.time()-t1:.2f}s")

    for j, img in enumerate(images):
        img.save(os.path.join(out_dir, f"refine_{j:02d}.jpg"))

        # Optional: pick the best refine sample (recommended).
    # v21 upgrades:
    #   1) Always run refine selection when enabled (not only when metric == "leakage")
    #   2) Default selector = scalar50 (50/50 occlusion vs quality) + depth penalty
    #   3) Thin/person/shuttle "rescue": if a candidate clearly misses lace/branches/person, resample that slot (keeps K fixed)
    rs = dict(scene.get("refine_select", {}))
    select_enable = bool(rs.get("enable", True))
    select_metric = str(rs.get("metric", "scalar50"))
    force_scalar50 = bool(rs.get("force_scalar50", True))
    if force_scalar50 and select_metric.lower() in ("leakage", "pair_penalty", "pair", "occlusion"):
        select_metric = "scalar50"

    select_conf_w = float(rs.get("confusion_weight", 0.35))
    select_fg_w = float(rs.get("foreground_weight", 0.12))
    select_presence_w = float(rs.get('presence_weight', rs.get('presence_w', rs.get('select_presence_w', 0.5))))
    occl_pairs = rs.get("occlusion_pairs", None) or scene.get("occlusion_pairs", None)

    # 50/50 selection weights: occlusion vs quality
    w_occl = float(rs.get("w_occl", 1.0))
    w_quality = float(rs.get("w_quality", 1.0))
    w_depth = float(rs.get("w_depth", 0.60))
    use_depth = bool(rs.get("use_depth", True))
    depth_model_name = str(rs.get("depth_model", "Intel/dpt-hybrid-midas"))

    # quality thresholds (aligned with eval_occlusion.py)
    person_area_min = float(rs.get("person_area_min", 0.12))
    person_iou_min  = float(rs.get("person_iou_min", 0.45))
    thin_area_min   = float(rs.get("thin_area_min", 0.05))
    thin_blob_max   = float(rs.get("thin_blob_max", 0.12))
    thin_thick_max  = float(rs.get("thin_thick_max", 6.0))
    blob_dt_thresh  = float(rs.get("blob_dt_thresh", 3.0))
    shuttle_stick_max = float(rs.get("shuttle_stick_max", 0.10))
    shuttle_dilate_k  = int(rs.get("shuttle_dilate_k", 11))

    # quality sub-weights
    w_person_small = float(rs.get("w_person_small", 1.0))
    w_person_iou   = float(rs.get("w_person_iou", 1.0))
    w_thin_missing = float(rs.get("w_thin_missing", 1.0))
    w_thin_blob    = float(rs.get("w_thin_blob", 1.0))
    w_thin_thick   = float(rs.get("w_thin_thick", 1.0))
    w_shuttle_stick= float(rs.get("w_shuttle_stick", 1.0))


    # Optional (OFF by default): generic anti-touch / structure penalties (align with eval_occlusion_v22)
    w_touch = float(rs.get("w_touch", 0.0))
    touch_max_default = float(rs.get("touch_max_default", -1.0))
    touch_dilate_k_default = int(rs.get("touch_dilate_k", 11))

    w_struct = float(rs.get("w_struct", 0.0))
    cc_min_default = int(rs.get("cc_min_default", -1))
    cc_max_default = int(rs.get("cc_max_default", -1))
    holes_min_default = int(rs.get("holes_min_default", -1))
    holes_max_default = int(rs.get("holes_max_default", -1))

    # ---- Rescue (keep K fixed, but resample bad slots) ----
    rescue_enable = bool(rs.get("rescue_enable", True))
    rescue_max_tries_per_slot = int(rs.get("rescue_max_tries_per_slot", 3))

    # Per-type thin presence thresholds for rescue gating
    thin_area_min_porous   = float(rs.get("thin_area_min_porous", 0.012))
    thin_area_min_sheet    = float(rs.get("thin_area_min_sheet", thin_area_min))
    thin_area_min_branches = float(rs.get("thin_area_min_branches", 0.030))

    def _intersect_xyxy(b1, b2):
        x1 = max(int(b1[0]), int(b2[0]))
        y1 = max(int(b1[1]), int(b2[1]))
        x2 = min(int(b1[2]), int(b2[2]))
        y2 = min(int(b1[3]), int(b2[3]))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _infer_pairs_by_overlap(boxes_xyxy, min_area=32 * 32):
        pairs = []
        n = len(boxes_xyxy)
        for back in range(n):
            for front in range(back + 1, n):
                inter = _intersect_xyxy(boxes_xyxy[back], boxes_xyxy[front])
                if inter is None:
                    continue
                x1, y1, x2, y2 = inter
                if (x2 - x1) * (y2 - y1) < int(min_area):
                    continue
                pairs.append({"back": back, "front": front})
        return pairs

    def _mask_bbox_xyxy(mask_bin):
        if mask_bin is None:
            return None
        ys, xs = np.where(mask_bin > 0)
        if xs.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))

    def _bbox_iou_xyxy(b1, b2):
        if b1 is None or b2 is None:
            return 0.0
        x1 = max(int(b1[0]), int(b2[0]))
        y1 = max(int(b1[1]), int(b2[1]))
        x2 = min(int(b1[2]), int(b2[2]))
        y2 = min(int(b1[3]), int(b2[3]))
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(1, int(b1[2]) - int(b1[0])) * max(1, int(b1[3]) - int(b1[1]))
        a2 = max(1, int(b2[2]) - int(b2[0])) * max(1, int(b2[3]) - int(b2[1]))
        return float(inter) / float(a1 + a2 - inter + 1e-6)

    def _area_ratio_in_bbox(mask_bin, box_xyxy):
        if mask_bin is None:
            return 0.0
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        roi = (mask_bin[y1:y2, x1:x2] > 0).astype(np.uint8)
        area = float(roi.sum())
        bbox_area = float(max(1, (x2 - x1) * (y2 - y1)))
        return area / bbox_area

    def _is_person(obj_cfg):
        n = str(obj_cfg.get("name", "")).lower()
        p = str(obj_cfg.get("prompt", "")).lower()
        return any(k in n for k in ["person","man","woman","human"]) or any(k in p for k in ["person","man","woman","human"])

    def _is_branches(obj_cfg):
        s = (str(obj_cfg.get("name","")) + " " + str(obj_cfg.get("prompt",""))).lower()
        return ("branches" in s) or ("branch" in s) or ("twig" in s)

    def _detect_shuttle_and_racket(objects_cfg):
        shuttle_idx, racket_idx = None, None
        for i, o in enumerate(objects_cfg):
            prompt = str(o.get("prompt","")).lower()
            name = str(o.get("name","")).lower()
            sm = str(o.get("sam_mode","")).lower()
            if shuttle_idx is None and (sm == "shuttle" or "shuttlecock" in prompt or "birdie" in prompt or "shuttle" in name):
                shuttle_idx = i
            if racket_idx is None and ("racket" in prompt or "racket" in name):
                racket_idx = i
        return shuttle_idx, racket_idx

    def _shuttle_stick_ratio(m_shuttle, m_racket, box_xyxy, dilate_k=11):
        if m_shuttle is None or m_racket is None:
            return 0.0
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        sh = (m_shuttle[y1:y2, x1:x2] > 0).astype(np.uint8)
        rk = (m_racket[y1:y2, x1:x2] > 0).astype(np.uint8)
        if int(sh.sum()) < 20:
            return 0.0
        k = int(dilate_k)
        if k % 2 == 0:
            k += 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        rk_d = cv2.dilate(rk, ker, iterations=1)
        ov = float((rk_d & sh).sum())
        return ov / float(sh.sum() + 1e-6)

def _count_cc_and_holes(mask_roi_bin: np.ndarray):
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
        if flood[0, x] == 1:
            seed = (x, 0); break
        if flood[hh - 1, x] == 1:
            seed = (x, hh - 1); break
    if seed is None:
        for y in range(hh):
            if flood[y, 0] == 1:
                seed = (0, y); break
            if flood[y, ww - 1] == 1:
                seed = (ww - 1, y); break
    if seed is not None:
        cv2.floodFill(flood, ffmask, seedPoint=seed, newVal=0)
    holes = (flood > 0).astype(np.uint8)
    if int(holes.sum()) == 0:
        return cc_count, 0
    num_h, _ = cv2.connectedComponents(holes, connectivity=8)
    hole_count = max(0, int(num_h) - 1)
    return cc_count, hole_count

def _pair_touch_ratio(m_front: np.ndarray, m_back: np.ndarray, inter_bbox, dilate_k: int = 11) -> float:
    if inter_bbox is None:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in inter_bbox]
    if x2 <= x1 or y2 <= y1:
        return 0.0
    Mf = (m_front[y1:y2, x1:x2] > 0).astype(np.uint8)
    Mb = (m_back[y1:y2, x1:x2] > 0).astype(np.uint8)
    if int(Mb.sum()) < 20:
        return 0.0
    k = int(dilate_k)
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    Mf_d = cv2.dilate(Mf, ker, iterations=1)
    return float((Mf_d & Mb).sum()) / float(Mb.sum() + 1e-6)


    # Optional depth estimator (DPT, same spirit as eval_occlusion.py)
    depth_est = None
    if use_depth:
        try:
            from transformers import DPTForDepthEstimation
            try:
                from transformers import DPTImageProcessor as _DPTProcessor
            except Exception:
                from transformers import DPTFeatureExtractor as _DPTProcessor

            class _DepthEstimator:
                def __init__(self, model_name="Intel/dpt-hybrid-midas", device=None):
                    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                    self.proc = _DPTProcessor.from_pretrained(model_name)
                    self.model = DPTForDepthEstimation.from_pretrained(model_name).to(dev)
                    self.model.eval()
                    self.dev = dev

                @torch.inference_mode()
                def depth_map(self, img_pil):
                    inp = self.proc(images=img_pil.convert("RGB"), return_tensors="pt")
                    inp = {k: v.to(self.dev) for k, v in inp.items()}
                    out = self.model(**inp).predicted_depth
                    out = out.unsqueeze(1)
                    out = torch.nn.functional.interpolate(out, size=(img_pil.height, img_pil.width), mode="bicubic", align_corners=False)
                    d = out.squeeze().detach().cpu().float().numpy()
                    dmin = float(d.min()); dmax = float(d.max())
                    return (d - dmin) / (dmax - dmin + 1e-6)

            depth_est = _DepthEstimator(model_name=depth_model_name)
        except Exception:
            depth_est = None

    if select_enable:
        H, W = height, width
        boxes_xyxy = [_to_box_xyxy_pixels(locations[i], H, W, pad=0.0) for i in range(len(objects))]
        boxes_xyxy_int = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_xyxy]

        holey_backs_eval = set()
        for _pp in (occl_pairs or []):
            try:
                _b, _f, _pd = _parse_occlusion_pair(_pp)
                if str(_pd.get('mode','opaque')).lower() in ('holey','porous','mesh','thin','transparent'):
                    holey_backs_eval.add(int(_b))
            except Exception:
                pass

        if not occl_pairs:
            occl_pairs = _infer_pairs_by_overlap(boxes_xyxy_int)

        if occl_pairs:
            print(f"[Stage-2] Selecting best refine (metric={select_metric}) over {len(occl_pairs)} pairs...")
            sel_reports = []
            best_j = 0
            best_score = 1e9

            # Use a lightweight SAM prompting (no attention) for evaluation.
            eval_prompt_cfg = SamAttnPromptCfg(
                n_pos=min(10, int(prompt_cfg.n_pos)),
                n_neg=min(16, int(prompt_cfg.n_neg)),
                min_dist=int(prompt_cfg.min_dist),
                add_center_point=True,
                add_grid_points=True,
                n_other_neg=0,
                n_avoid_neg=0,
                avoid_dilate=0,
            )
            eval_score_cfg = SamGenericScoreCfg()

            # Resample one candidate (same refine settings, new seed)
            base_refine_seed = int(seed + 10_000)

            def _resample_one(seed_val: int):
                img_new = LR.run(
                    objects=objects,
                    locations=locations,
                    alpha=alpha,
                    indices=indices,
                    masks=masks,
                    mask_fusion=mask_fusion_mode,
                    fusion_lambda_start=fusion_lambda_start,
                    fusion_lambda_end=fusion_lambda_end,
                    fusion_schedule=fusion_schedule,
                    fusion_t_start=fusion_t_start,
                    fusion_t_end=fusion_t_end,
                    attn_power=attn_power,
                    attn_power_per_object=attn_power_per_object,
                    attn_floor=attn_floor,
                    attn_floor_per_object=attn_floor_per_object,
                    skip_attn_indices=skip_attn_indices,
                    num_images_per_prompt=1,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    seed=int(seed_val),
                )[0]
                return img_new

            for j in range(len(images)):
                img = images[j]
                attempt = 0
                while True:
                    bgr = pil_to_bgr(img)
                    masker.set_image_bgr(bgr)

                    # Segment in front->back order; use union_front as avoid mask (except for holey backs).
                    Hm, Wm = bgr.shape[:2]
                    union_front = np.zeros((Hm, Wm), dtype=np.uint8)
                    masks_bin = [None for _ in range(n_obj)]

                    for oi in range(n_obj - 1, -1, -1):
                        box = boxes_xyxy[oi]
                        avoid = None if (oi in holey_backs_eval) else (union_front if int(union_front.sum()) > 0 else None)
                        pos, neg = SamMaskerV6.points_from_attn(box, attn_map=None, cfg=eval_prompt_cfg, avoid_mask=avoid)
                        m, _s = masker.predict_best(
                            box,
                            pos,
                            neg,
                            mode=str(sam_mode[oi] or "generic"),
                            attn_map=None,
                            avoid_mask=avoid,
                            score_cfg=eval_score_cfg,
                        )

                        # Apply the same postprocess as used in refine (important for thin structures).
                        if m is not None:
                            pp_name = postprocess[oi] if oi < len(postprocess) else 'generic'
                            if pp_name and str(pp_name).lower() != 'none':
                                pp_fn = POSTPROCESS_REGISTRY.get(str(pp_name).lower(), SamMaskerV6.postprocess_generic)
                                kw = postprocess_kwargs[oi] if oi < len(postprocess_kwargs) else {}
                                try:
                                    pp_kwargs = dict(kw or {})
                                    sig = inspect.signature(pp_fn)
                                    params = sig.parameters
                                    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                                    allowed = set(params.keys())
                                    if not accepts_varkw:
                                        pp_kwargs = {k: v for k, v in pp_kwargs.items() if k in allowed}
                                    if (accepts_varkw or 'image_bgr' in allowed) and 'image_bgr' not in pp_kwargs:
                                        pp_kwargs['image_bgr'] = bgr
                                    if (accepts_varkw or 'attn_map' in allowed) and 'attn_map' not in pp_kwargs:
                                        pp_kwargs['attn_map'] = None
                                    m_soft = pp_fn(m.astype(np.uint8), boxes_xyxy[oi], **pp_kwargs)
                                    if m_soft is not None:
                                        m = (np.asarray(m_soft) > 0.5).astype(np.uint8)
                                except Exception:
                                    pass

                        masks_bin[oi] = m
                        if m is not None:
                            upd = (m > 0).astype(np.uint8)
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                            upd = cv2.dilate(upd, kernel, iterations=1)
                            union_front = np.maximum(union_front, upd)

                    # presence_pen
                    presence_pen = 0.0
                    for oi in range(n_obj):
                        m = masks_bin[oi]
                        area_frac = float((m > 0).sum()) / float(Hm * Wm) if m is not None else 0.0
                        amin = objects_cfg[oi].get('area_min', None) if oi < len(objects_cfg) else None
                        amax = objects_cfg[oi].get('area_max', None) if oi < len(objects_cfg) else None
                        try:
                            if amin is not None:
                                amin = float(amin)
                                if area_frac < amin:
                                    presence_pen += (amin - area_frac) / (amin + 1e-6)
                            if amax is not None:
                                amax = float(amax)
                                if area_frac > amax:
                                    presence_pen += (area_frac - amax) / (amax + 1e-6)
                        except Exception:
                            pass

                    # fgq proxy
                    fgq = 0.0
                    if float(select_fg_w) != 0.0:
                        try:
                            union_fg = np.zeros((Hm, Wm), dtype=np.uint8)
                            for oi in range(n_obj):
                                m = masks_bin[oi]
                                if m is None:
                                    continue
                                union_fg = np.maximum(union_fg, (m > 0).astype(np.uint8))
                            fgq = float(union_fg.mean())
                        except Exception:
                            fgq = 0.0

                    # pair penalties
                    pair_pen_list = []
                    inter_ratio_list = []
                    touch_ratios = []
                    for p in occl_pairs:
                        back, front, pd = _parse_occlusion_pair(p)
                        if back < 0 or front < 0 or back >= n_obj or front >= n_obj:
                            continue
                        mb = masks_bin[back]
                        mf = masks_bin[front]
                        if mb is None or mf is None:
                            continue
                        inter_bbox = _intersect_xyxy(boxes_xyxy_int[back], boxes_xyxy_int[front])
                        if inter_bbox is None:
                            continue
                        

                        # optional anti-touch / anti-stick (only active if touch_max is specified)
                        

                        tmax = pd.get("touch_max", None)
                        

                        if tmax is None and touch_max_default >= 0:
                        

                            tmax = touch_max_default
                        

                        if tmax is not None:
                        

                            tk = int(pd.get("touch_dilate_k", touch_dilate_k_default))
                        

                            tr = _pair_touch_ratio(mf, mb, inter_bbox, dilate_k=tk)
                        

                            touch_ratios.append(float(tr))
                        

                            if float(tr) > float(tmax):
                        

                                touch_pen += (float(tr) - float(tmax)) / (1.0 - float(tmax) + 1e-6)
                        

                                gate_fail = True
                        x1, y1, x2, y2 = inter_bbox
# -------------------------
# CLI entrypoint (added)
# -------------------------
if __name__ == "__main__":
    import argparse, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Path to a scene json")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--root",
        default=os.environ.get(
            "OCCLUSION_ROOT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ),
        help="Project root (default from OCCLUSION_ROOT or ../)"
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_scene(args.scene, args.root, args.out)
