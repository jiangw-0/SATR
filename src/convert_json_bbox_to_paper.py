#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert scene JSON bboxes from this project's format [y0,y1,x0,x1]
to the paper baseline format [x0,x1,y0,y1].

Usage:
  python convert_json_bbox_to_paper.py \
    --in_dir  /media/jiang/ssd/zly/Occlusion/src/json \
    --out_dir /media/jiang/ssd/zly/Occlusion/src/json_paper \
    --overwrite \
    --reorder_by_pairs

Optional:
  --overwrite            overwrite existing files
  --reorder_by_pairs     reorder objects to satisfy occlusion_pairs (keeps background at index 0)
  --dry_run              print what would be done
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _swap_bbox_yx_to_xy(b: List[float]) -> List[float]:
    # in:  [y0,y1,x0,x1]
    # out: [x0,x1,y0,y1]
    y0, y1, x0, x1 = [float(v) for v in b]
    return [x0, x1, y0, y1]

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _toposort(n: int, edges: List[Tuple[int,int]]) -> List[int] | None:
    # Kahn
    indeg = [0]*n
    g = [[] for _ in range(n)]
    for u,v in edges:
        if u<0 or v<0 or u>=n or v>=n or u==v:
            continue
        g[u].append(v)
        indeg[v]+=1
    q = [i for i in range(n) if indeg[i]==0]
    out = []
    while q:
        u = q.pop(0)
        out.append(u)
        for v in g[u]:
            indeg[v]-=1
            if indeg[v]==0:
                q.append(v)
    if len(out)!=n:
        return None
    return out

def _reorder_objects_by_pairs(scene: Dict[str, Any]) -> Dict[str, Any]:
    objs = scene.get("objects", [])
    if not isinstance(objs, list) or len(objs) <= 2:
        return scene
    pairs = scene.get("occlusion_pairs", []) or []
    edges = []
    for p in pairs:
        if isinstance(p, dict):
            b = p.get("back", None)
            f = p.get("front", None)
            if isinstance(b, int) and isinstance(f, int):
                edges.append((b, f))
    if not edges:
        return scene

    # Keep background at index 0. Sort indices 1..n-1.
    n = len(objs)
    sub_n = n-1
    sub_edges = []
    for u,v in edges:
        if u==0 or v==0:
            continue
        sub_edges.append((u-1, v-1))

    order = _toposort(sub_n, sub_edges)
    if order is None:
        return scene  # cycles; keep original

    # Build permutation: [0] + [i+1 for i in order]
    perm = [0] + [i+1 for i in order]
    new_objs = [objs[i] for i in perm]

    # Remap indices in occlusion_pairs
    inv = {old:new for new,old in enumerate(perm)}
    new_pairs = []
    for p in pairs:
        if not isinstance(p, dict):
            new_pairs.append(p)
            continue
        b = p.get("back", None)
        f = p.get("front", None)
        if isinstance(b, int) and isinstance(f, int) and b in inv and f in inv:
            p2 = dict(p)
            p2["back"] = inv[b]
            p2["front"] = inv[f]
            new_pairs.append(p2)
        else:
            new_pairs.append(p)

    scene2 = dict(scene)
    scene2["objects"] = new_objs
    scene2["occlusion_pairs"] = new_pairs
    return scene2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing .json scenes (bbox=[y0,y1,x0,x1])")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for converted scenes (bbox=[x0,x1,y0,y1])")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing json files in out_dir")
    ap.add_argument("--reorder_by_pairs", action="store_true", help="Reorder objects to satisfy occlusion_pairs (keep bg at index 0)")
    ap.add_argument("--dry_run", action="store_true", help="Do not write files; only print actions")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not in_dir.exists():
        raise SystemExit(f"[ERROR] in_dir not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"[ERROR] no json files under: {in_dir}")

    n_written = 0
    for p in files:
        scene = _load_json(p)
        if args.reorder_by_pairs:
            scene = _reorder_objects_by_pairs(scene)

        objs = scene.get("objects", None)
        if not isinstance(objs, list) or not objs:
            print(f"[WARN] skip (no objects): {p.name}")
            continue

        changed = False
        for o in objs:
            if not isinstance(o, dict):
                continue
            bbox = o.get("bbox", None)
            if bbox is None:
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            o["bbox"] = _swap_bbox_yx_to_xy(bbox)
            changed = True

        if not changed:
            print(f"[WARN] no bbox to convert: {p.name}")

        out_path = out_dir / p.name
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out_path}")
            continue

        if args.dry_run:
            print(f"[DRY] write: {out_path}")
            continue

        _write_json(out_path, scene)
        n_written += 1

    print(f"[OK] Converted {n_written}/{len(files)} scenes -> {out_dir}")

if __name__ == "__main__":
    main()
