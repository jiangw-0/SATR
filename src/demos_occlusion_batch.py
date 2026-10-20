#!/usr/bin/env python3
"""
Batch runner + JSON exporter for the *paper* LaRender implementation.

Goals:
- One command to run 7 example scenes (like your run_examples_v11.sh).
- Export those 7 scenes to evaluation-compatible scene JSONs (same schema as your project).

This script does NOT depend on your v13/v14 pipeline code. It only uses:
- larender_sdxl.py
- utils.py

Scene JSON schema:
- objects[*].bbox is [x0, x1, y0, y1] in [0,1]
- occlusion_pairs entries are optional (used by eval, not required for generation)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from larender_sdxl import LaRender
from utils import ensure_dir


HERE = Path(__file__).resolve().parent
DEFAULT_BUILTIN_SCENES_DIR = HERE / "paper_scenes"  # shipped alongside this script


DEFAULT_SCENE_FILES = [
    "scene_example_badminton_v11.json",
    "scene_example_chainlink_fence_v11.json",
    "scene_example_leaves_book_v11.json",
    "scene_example_tree_branches_v11.json",
    "scene_example_spiderweb_flower_v11.json",
    "scene_example_lace_curtain_v11.json",
    "scene_example_window_blinds_v11.json",
]


def _gray_image(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="L")


def _heat_image(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * arr - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * arr - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * arr - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def _save_maps(maps_by_step: Dict[int, List[np.ndarray]] | None, out_dir: Path, stem: str) -> None:
    if not maps_by_step:
        return
    save_dir = Path(ensure_dir(out_dir / stem))
    for step_idx in sorted(maps_by_step.keys()):
        per_obj = maps_by_step[step_idx]
        if per_obj is None:
            continue
        for obj_i, arr in enumerate(per_obj):
            if arr is None:
                continue
            gray = _gray_image(arr)
            heat = _heat_image(arr)
            prefix = f"step_{int(step_idx):03d}_obj_{int(obj_i):02d}_{stem}"
            gray.save(save_dir / f"{prefix}_gray.png")
            heat.save(save_dir / f"{prefix}_heat.jpg", quality=95)


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_to_loc_y0y1x0x1(bbox_x0x1y0y1: List[float]) -> List[float]:
    # bbox: [x0, x1, y0, y1] -> loc: [y0, y1, x0, x1]
    x0, x1, y0, y1 = bbox_x0x1y0y1
    return [float(y0), float(y1), float(x0), float(x1)]


def _clamp_alpha(a: float) -> float:
    """Clamp alpha into [0, 1) to avoid infinite density (-log(1-alpha))."""
    if a >= 1.0:
        return 0.9999
    if a < 0.0:
        return 0.0
    return float(a)


def scene_to_larender_inputs(
    scene: Dict[str, Any]
) -> Tuple[
    # Newer LaRender API (paper SDXL implementation): background is also an object.
    List[str],
    List[List[float]],
    List[float],
    List[List[int]],
    str,
    str,
    # Legacy API (if your local LaRender expects global prompt + foreground phrases).
    List[str],
    List[List[float]],
    List[float],
    List[int],
]:
    """
    Convert evaluation-style scene json to LaRender inputs.

    We treat objects[0] as background, and use its prompt as the global prompt.
    The rest objects are passed as GLIGEN phrases+boxes via LaRender.
    """
    objs = scene.get("objects", [])
    if not objs or len(objs) < 2:
        raise ValueError("scene json must contain at least 2 objects (background + 1 foreground).")

    bg = objs[0]
    global_prompt = str(bg.get("prompt", "")).strip()
    if not global_prompt:
        global_prompt = str(scene.get("prompt", "")).strip()
    if not global_prompt:
        raise ValueError("Missing background/global prompt in scene json (objects[0].prompt or scene.prompt).")

    negative_prompt = str(scene.get("negative_prompt", "")).strip()

    # -----------------
    # New API inputs
    # -----------------
    objects_all: List[str] = []
    locations_all: List[List[float]] = []
    alpha_all: List[float] = []
    indices_all: List[List[int]] = []

    for i, o in enumerate(objs):
        phrase = str(o.get("prompt") or o.get("name") or "").strip()
        if not phrase:
            raise ValueError(f"object #{i} missing prompt/name.")
        objects_all.append(phrase)

        bbox = o.get("bbox")
        if bbox is None:
            # Background sometimes omits bbox. Default to full canvas.
            bbox = [0.0, 1.0, 0.0, 1.0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"object '{o.get('name', i)}' invalid bbox, expect [x0,x1,y0,y1].")
        locations_all.append(_bbox_to_loc_y0y1x0x1(bbox))

        alpha_all.append(_clamp_alpha(float(o.get("alpha", 0.999))))

        # Token indices used by attention-map based masking.
        # Many JSONs won't include them; fall back to [0].
        idx = o.get("indices") or o.get("subject_indices")
        if isinstance(idx, list) and len(idx) > 0:
            indices_all.append([int(x) for x in idx])
        else:
            indices_all.append([0])

    # -----------------
    # Legacy API inputs
    # -----------------
    phrases_fg: List[str] = []
    locations_fg: List[List[float]] = []
    alpha_fg: List[float] = []
    indices_fg: List[int] = []

    for j, o in enumerate(objs[1:]):
        phrase = str(o.get("prompt") or o.get("name") or "").strip()
        if not phrase:
            raise ValueError(f"object #{j+1} missing prompt/name.")
        phrases_fg.append(phrase)

        bbox = o.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"object '{o.get('name', j+1)}' missing bbox [x0,x1,y0,y1].")
        locations_fg.append(_bbox_to_loc_y0y1x0x1(bbox))

        alpha_fg.append(_clamp_alpha(float(o.get("alpha", 0.999))))
        indices_fg.append(j)

    return (
        objects_all,
        locations_all,
        alpha_all,
        indices_all,
        global_prompt,
        negative_prompt,
        phrases_fg,
        locations_fg,
        alpha_fg,
        indices_fg,
    )


def _make_larender(*, device: str, pretrain: str, transmittance_use_attn_map: bool) -> LaRender:
    """Create LaRender instance while being compatible with multiple signatures."""
    sig = inspect.signature(LaRender.__init__)
    kwargs: Dict[str, Any] = {"pretrain": pretrain}
    if "device" in sig.parameters:
        kwargs["device"] = device
    if "transmittance_use_attn_map" in sig.parameters:
        kwargs["transmittance_use_attn_map"] = transmittance_use_attn_map
    return LaRender(**kwargs)


def run_one(scene_path: Path, out_dir: Path, *, device: str, pretrain: str | None, num_images: int | None,
            steps: int | None, guidance: float | None, seed: int | None,
            height: int | None, width: int | None, disable_dp: bool) -> None:
    scene = _load_json(scene_path)

    # Decide runtime params: CLI overrides > json
    pretrain_val = pretrain or str(scene.get("pretrain", "IterComp"))
    height = int(height if height is not None else scene.get("height", 1024))
    width  = int(width  if width  is not None else scene.get("width", 1024))

    # Prefer "refine_*" if present (because your v13/v14 jsons have draft/refine),
    # but allow CLI overrides.
    steps_val = int(steps if steps is not None else scene.get("refine_steps", scene.get("steps", 50)))
    guidance_val = float(guidance if guidance is not None else scene.get("refine_guidance", scene.get("guidance_scale", 7.5)))
    num_images_val = int(num_images if num_images is not None else scene.get("num_images", 4))
    seed_val = int(seed if seed is not None else scene.get("seed", 0))

    (
        objects_all,
        locations_all,
        alpha_all,
        indices_all,
        prompt,
        neg,
        phrases_fg,
        locations_fg,
        alpha_fg,
        indices_fg,
    ) = scene_to_larender_inputs(scene)

    ensure_dir(str(out_dir))
    print(f"\n=== Running {scene_path.name} -> {out_dir} (pretrain={pretrain_val}, seed={seed_val}) ===")

    # Many scene JSONs don't include token indices for each object; for robustness,
    # we disable attention-map based transmittance by default and rely on bboxes.
    lr = _make_larender(device=device, pretrain=pretrain_val, transmittance_use_attn_map=False)

    run_sig = inspect.signature(lr.run)
    run_params = run_sig.parameters
    transmittance_maps = None
    support_maps = None
    contrib_maps = None
    wants_transmittance = (
        os.environ.get("OCCLUSION_SAVE_TRANSMITTANCE", "1") != "0"
        and ("return_transmittance" in run_params or "export_transmittance" in run_params)
    )
    wants_support = (
        os.environ.get("OCCLUSION_SAVE_SUPPORT", "1") != "0"
        and "export_support" in run_params
    )
    wants_contrib = (
        os.environ.get("OCCLUSION_SAVE_CONTRIB", "1") != "0"
        and "export_contrib" in run_params
    )

    # Legacy LaRender variants may take a global "prompt" and only foreground phrases.
    if "prompt" in run_params:
        use_dp = (not disable_dp) and bool(scene.get("use_dependency_parsing", False))
        kwargs: Dict[str, Any] = dict(
            objects=phrases_fg,
            locations=locations_fg,
            alpha=alpha_fg,
            indices=indices_fg,
            prompt=prompt,
            negative_prompt=neg,
            height=height,
            width=width,
            seed=seed_val,
        )
        if "num_images" in run_params:
            kwargs["num_images"] = num_images_val
        if "num_images_per_prompt" in run_params:
            kwargs["num_images_per_prompt"] = num_images_val
        if "use_dependency_parsing" in run_params:
            kwargs["use_dependency_parsing"] = use_dp
        if "num_inference_steps" in run_params:
            kwargs["num_inference_steps"] = steps_val
        if "guidance_scale" in run_params:
            kwargs["guidance_scale"] = guidance_val
        if wants_transmittance and "export_transmittance" in run_params:
            kwargs["export_transmittance"] = True
        if wants_support and "export_support" in run_params:
            kwargs["export_support"] = True
        if wants_contrib and "export_contrib" in run_params:
            kwargs["export_contrib"] = True
        if wants_transmittance and "return_transmittance" in run_params:
            kwargs["return_transmittance"] = True
        result = lr.run(**kwargs)
    else:
        # Paper SDXL implementation: objects include background; indices are List[List[int]].
        kwargs2: Dict[str, Any] = dict(
            objects=objects_all,
            locations=locations_all,
            alpha=alpha_all,
            indices=indices_all,
            negative_prompt=neg,
            height=height,
            width=width,
            seed=seed_val,
        )
        if "num_images_per_prompt" in run_params:
            kwargs2["num_images_per_prompt"] = num_images_val
        elif "num_images" in run_params:
            kwargs2["num_images"] = num_images_val
        if wants_transmittance and "export_transmittance" in run_params:
            kwargs2["export_transmittance"] = True
        if wants_support and "export_support" in run_params:
            kwargs2["export_support"] = True
        if wants_contrib and "export_contrib" in run_params:
            kwargs2["export_contrib"] = True
        if wants_transmittance and "return_transmittance" in run_params:
            kwargs2["return_transmittance"] = True
        result = lr.run(**kwargs2)

    if wants_transmittance and isinstance(result, tuple):
        images, transmittance_maps = result
    else:
        images = result

    # Pull exported maps from cache when available.
    if wants_transmittance and transmittance_maps is None and hasattr(lr, "get_transmittance_maps"):
        transmittance_maps = lr.get_transmittance_maps(out_h=height, out_w=width)
    if wants_support and hasattr(lr, "get_support_maps"):
        support_maps = lr.get_support_maps(out_h=height, out_w=width)
    if wants_contrib and hasattr(lr, "get_contrib_maps"):
        contrib_maps = lr.get_contrib_maps(out_h=height, out_w=width)

    # Save
    for i, im in enumerate(images):
        im.save(out_dir / f"{i:03d}.png")

    if wants_transmittance:
        _save_maps(transmittance_maps, out_dir, "transmittance")
    if wants_support:
        _save_maps(support_maps, out_dir, "support")
    if wants_contrib:
        _save_maps(contrib_maps, out_dir, "contrib")

    # Save the effective scene+run config for reproducibility
    meta = dict(scene)
    meta["_paper_runner"] = {
        "scene_file": scene_path.name,
        "pretrain": pretrain_val,
        "height": height,
        "width": width,
        "num_images": num_images_val,
        # steps/guidance may be ignored depending on LaRender variant.
        "steps": steps_val,
        "guidance": guidance_val,
        "seed": seed_val,
        "disable_dp": disable_dp,
    }
    (out_dir / "scene_used.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {num_images_val} images to: {out_dir}")


def export_scenes(dst_dir: Path, scenes_dir: Path) -> None:
    ensure_dir(str(dst_dir))
    for fn in DEFAULT_SCENE_FILES:
        src = scenes_dir / fn
        if not src.exists():
            raise FileNotFoundError(f"Missing builtin scene json: {src}")
        (dst_dir / fn).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Exported {len(DEFAULT_SCENE_FILES)} scene JSONs to: {dst_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="outputs/paper_examples",
                    help="Output root directory. Each scene gets its own subfolder.")
    ap.add_argument("--scene", type=str, default="",
                    help="Run a single scene json. If empty, use --run_all.")
    ap.add_argument("--run_all", action="store_true", help="Run all 7 builtin scenes.")
    ap.add_argument("--scenes_dir", type=str, default=str(DEFAULT_BUILTIN_SCENES_DIR),
                    help="Directory that contains the 7 scene json files (default: shipped paper_scenes).")

    ap.add_argument("--export_scenes_dir", type=str, default="",
                    help="If set, export the 7 builtin scenes into this directory and exit (unless --run_all/--scene also set).")

    ap.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    ap.add_argument("--pretrain", type=str, default="", help="Override scene pretrain (e.g., IterComp / gligen)")
    ap.add_argument("--num_images", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--height", type=int, default=None, help="Override scene height")
    ap.add_argument("--width",  type=int, default=None, help="Override scene width")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--disable_dp", action="store_true", help="Disable dependency parsing in LaRender (recommended).")

    args = ap.parse_args()

    scenes_dir = Path(args.scenes_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    ensure_dir(str(out_root))

    if args.export_scenes_dir:
        export_scenes(Path(args.export_scenes_dir).expanduser().resolve(), scenes_dir)
        # If user only wanted export, exit.
        if not args.run_all and not args.scene:
            return

    pretrain = args.pretrain.strip() or None

    if args.run_all:
        for fn in DEFAULT_SCENE_FILES:
            sp = scenes_dir / fn
            run_one(
                sp,
                out_root / sp.stem,
                device=args.device,
                pretrain=pretrain,
                num_images=args.num_images,
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
                height=args.height,
                width=args.width,
                disable_dp=args.disable_dp,
            )
        print(f"\nDone. Outputs in: {out_root}")
        return

    if args.scene:
        sp = Path(args.scene).expanduser()
        if not sp.exists():
            # allow passing just a filename that exists under scenes_dir
            cand = scenes_dir / args.scene
            if cand.exists():
                sp = cand
        sp = sp.resolve()
        run_one(
            sp,
            out_root / sp.stem,
            device=args.device,
            pretrain=pretrain,
            num_images=args.num_images,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            height=args.height,
            width=args.width,
            disable_dp=args.disable_dp,
        )
        return

    ap.error("Nothing to do. Use --run_all or --scene, or --export_scenes_dir.")


if __name__ == "__main__":
    main()