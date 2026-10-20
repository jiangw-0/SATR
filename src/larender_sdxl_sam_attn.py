from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from utils import torch_fix_seed




def _normalize_density_scheduler(density_scheduler):
    """Accept legacy string or v11 dict config.

    Returns:
        (mode: str, cfg: dict)
    """
    cfg = {}
    if isinstance(density_scheduler, dict):
        cfg = dict(density_scheduler)
        density_scheduler = cfg.get("mode") or cfg.get("type") or cfg.get("name")
    if density_scheduler is None:
        density_scheduler = "inverse_proportional"
    if not isinstance(density_scheduler, str):
        print(f"[WARN] density_scheduler should be str or dict, got {type(density_scheduler)}. Falling back to 'inverse_proportional'.")
        return "inverse_proportional", cfg

    mode = density_scheduler.strip().lower()
    mode = mode.replace('-', '_').replace(' ', '_')
    aliases = {
        'inverse': 'inverse_proportional',
        'inv': 'inverse_proportional',
        'inverseproportional': 'inverse_proportional',
        'inv_proportional': 'inverse_proportional',
        'opaque_only': 'opaque',
        'keep': 'unchanged',
        'none': 'unchanged',
        'default': 'inverse_proportional',
        # if someone passes a schedule dict mode like 'linear', we just fall back to inverse_proportional
        'linear': 'inverse_proportional',
        'cosine': 'inverse_proportional',
        'power': 'inverse_proportional',
    }
    mode = aliases.get(mode, mode)

    valid = {'opaque', 'unchanged', 'inverse_proportional'}
    if mode not in valid:
        print(f"[WARN] Unknown density_scheduler='{density_scheduler}'. Valid: {sorted(valid)}. Falling back to 'inverse_proportional'.")
        mode = 'inverse_proportional'

    cfg.setdefault('mode', mode)
    return mode, cfg


class LaRender():
    def __init__(self,
                 pretrain='IterComp',
                 transmittance_use_bbox=True,
                 transmittance_use_attn_map=True,
                 density_scheduler='unchanged',
                 cache_dir=None,
                 local_files_only=None):
        assert pretrain in ['IterComp', 'GLIGEN']
        density_scheduler_mode, density_scheduler_cfg = _normalize_density_scheduler(density_scheduler)
        self.density_scheduler = density_scheduler_mode
        self.density_scheduler_cfg = density_scheduler_cfg

        self.pretrain = pretrain
        self.transmittance_use_bbox = transmittance_use_bbox
        self.transmittance_use_attn_map = transmittance_use_attn_map
        self.density_scheduler_raw = density_scheduler

        if cache_dir is None:
            cache_dir = os.environ.get("HF_HUB_CACHE", None)

        if local_files_only is None:
            # Respect HF_HUB_OFFLINE to avoid network + SSL/mirror failures.
            local_files_only = (
                os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
                or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            )

        def _find_sdxl_base_candidates() -> List[str]:
            """Return possible local SDXL base snapshot paths."""

            cands: List[str] = []
            env_path = os.environ.get("SDXL_BASE_MODEL", None)
            if env_path:
                cands.append(env_path)

            if cache_dir:
                # cache_dir might already be the 'hub' folder, but sometimes users set HF_HOME.
                for base in [cache_dir, os.path.join(cache_dir, "hub")]:
                    snap_root = os.path.join(
                        base,
                        "models--stabilityai--stable-diffusion-xl-base-1.0",
                        "snapshots",
                    )
                    if os.path.isdir(snap_root):
                        try:
                            snaps = sorted(
                                [os.path.join(snap_root, d) for d in os.listdir(snap_root)]
                            )
                            cands.extend([s for s in snaps if os.path.isdir(s)])
                        except Exception:
                            pass
            return cands

        if pretrain == 'IterComp':
            # Prefer an explicit local snapshot path if provided.
            itercomp_path = os.environ.get("ITERCOMP_MODEL", None)
            itercomp_candidates: List[str] = []
            if itercomp_path and os.path.isdir(itercomp_path):
                itercomp_candidates.append(itercomp_path)
            itercomp_candidates.append("comin/IterComp")

            last_err: Optional[Exception] = None
            self.pipe = None
            for cid in itercomp_candidates:
                try:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        cid,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                    break
                except Exception as e:
                    last_err = e

            # Fallback: if IterComp is unavailable/incomplete, try SDXL base if present locally.
            if self.pipe is None:
                sdxl_candidates = _find_sdxl_base_candidates()
                for sp in sdxl_candidates:
                    try:
                        print(
                            "[WARN] IterComp not available; falling back to SDXL base at:",
                            sp,
                        )
                        self.pipe = DiffusionPipeline.from_pretrained(
                            sp,
                            torch_dtype=torch.float16,
                            use_safetensors=True,
                            cache_dir=cache_dir,
                            local_files_only=True,
                        )
                        break
                    except Exception as e:
                        last_err = e

            if self.pipe is None:
                raise last_err if last_err else RuntimeError(
                    "Failed to load IterComp pipeline. If you're offline, set ITERCOMP_MODEL to a local snapshot path."
                )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                "jiuntian/gligen-xl-1024",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(self.device)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True
        )

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

        for name, module in self.pipe.unet.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                module.forward = self.create_larender_forward(module)
                module.name = name

        # ===== attention export cache (for SAM prompts) =====
        # Will be re-initialized per `run()`.
        self._attn_cache: Optional[List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]]] = None
        self._attn_export_enabled: bool = False
        self._attn_export_start: float = 0.0
        self._attn_export_end: float = 1.0
        self._attn_cfg_mode: str = "cond"  # "cond" | "all"

    def create_larender_forward(self, module):
        def forward(hidden_states, encoder_hidden_states=None, **kwargs):
            return self.larender_core_process(module, hidden_states, encoder_hidden_states)
        return forward

    def _forward_cross_attention(self, module, hidden_states, encoder_hidden_states):
        query = module.to_q(hidden_states)
        key = module.to_k(encoder_hidden_states)
        value = module.to_v(encoder_hidden_states)

        query = module.head_to_batch_dim(query)
        key = module.head_to_batch_dim(key)
        value = module.head_to_batch_dim(value)

        attention_probs = module.get_attention_scores(query, key)
        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = module.batch_to_head_dim(hidden_states)
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)
        return hidden_states, attention_probs

    def _bboxes_to_mask(self, bboxes, h, w, dtype, device):
        masks = []
        for bbox in bboxes:
            M = torch.zeros((h, w), dtype=dtype, device=device)
            upper = int(h * bbox[0])
            lower = int(h * bbox[1])
            left = int(w * bbox[2])
            right = int(w * bbox[3])
            assert lower > upper and right > left, f"Invalid bbox: {bbox}"
            M[upper:lower, left:right] = 1
            masks.append(M)
        return masks

    @staticmethod
    def _resolve_per_object(value, i: int, default=None):
        """Resolve a scalar/list/dict hyperparam for object i."""
        if value is None:
            return default
        try:
            if isinstance(value, (list, tuple)):
                if i < len(value) and value[i] is not None:
                    return value[i]
            elif isinstance(value, dict):
                if i in value and value[i] is not None:
                    return value[i]
            else:
                return value
        except Exception:
            return default
        return default

    @staticmethod
    def _schedule(progress: float, start: float, end: float, mode: str = "linear") -> float:
        """Simple scalar schedule in [0,1]."""
        p = float(np.clip(progress, 0.0, 1.0))
        if mode == "linear":
            return float(start * (1.0 - p) + end * p)
        if mode == "cosine":
            # smooth start/end
            t = 0.5 - 0.5 * math.cos(math.pi * p)
            return float(start * (1.0 - t) + end * t)
        return float(start * (1.0 - p) + end * p)

    def _reset_attn_cache(self, num_objects: int):
        self._attn_cache = [dict() for _ in range(num_objects)]

    def _export_attn(self, obj_i: int, attn_hw: torch.Tensor, progress: float):
        """Accumulate an attention map (H,W) for object i."""
        if not self._attn_export_enabled or self._attn_cache is None:
            return
        if progress < self._attn_export_start or progress > self._attn_export_end:
            return

        key = (int(attn_hw.shape[0]), int(attn_hw.shape[1]))
        slot = self._attn_cache[obj_i].get(key, None)
        if slot is None:
            slot = {
                "sum": torch.zeros_like(attn_hw, dtype=torch.float32),
                "cnt": torch.zeros((), device=attn_hw.device, dtype=torch.float32),
            }
            self._attn_cache[obj_i][key] = slot

        slot["sum"] = slot["sum"] + attn_hw.detach().to(dtype=torch.float32)
        slot["cnt"] = slot["cnt"] + 1.0

    def get_attn_maps(
        self,
        out_h: Optional[int] = None,
        out_w: Optional[int] = None,
        pick: str = "max",  # "max" resolution
    ) -> Optional[List[np.ndarray]]:
        """Get the last run's exported attention maps as numpy arrays.

        Returns:
            List[np.ndarray] of length num_objects, each in [0,1] float32,
            resized to (out_h,out_w) if specified.
        """
        if not self._attn_cache:
            return None
        if out_h is None:
            out_h = int(getattr(self, "height", 0) or 0)
        if out_w is None:
            out_w = int(getattr(self, "width", 0) or 0)

        maps: List[np.ndarray] = []
        for i in range(len(self._attn_cache)):
            if not self._attn_cache[i]:
                maps.append(None)
                continue

            # pick the highest resolution by default
            keys = list(self._attn_cache[i].keys())
            if pick == "max":
                key = max(keys, key=lambda k: int(k[0]) * int(k[1]))
            else:
                key = keys[0]

            slot = self._attn_cache[i][key]
            cnt = float(slot["cnt"].item()) if slot["cnt"].numel() else 0.0
            if cnt <= 0:
                maps.append(None)
                continue
            attn = (slot["sum"] / max(cnt, 1.0)).clamp(0, 1)

            if out_h > 0 and out_w > 0 and (attn.shape[0] != out_h or attn.shape[1] != out_w):
                t = attn[None, None, ...]
                t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
                attn = t[0, 0]

            attn_np = attn.detach().float().cpu().numpy().astype(np.float32)
            # normalize again defensively
            mx = float(attn_np.max())
            if mx > 1e-6:
                attn_np = attn_np / mx
            maps.append(attn_np)

        return maps

    def _prompt_embedding(self, objects, negative_prompt):
        prompt_embeds_list = []
        negative_prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        negative_pooled_prompt_embeds_list = []

        for p in objects:
            (prompt_embeds,
             negative_prompt_embeds,
             pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                prompt=p,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt
            )
            prompt_embeds_list.append(prompt_embeds)
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            negative_pooled_prompt_embeds_list.append(negative_pooled_prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=1)

        (_, _, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=', '.join(objects),
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def larender_core_process(self, module, hidden_states, encoder_hidden_states):
        locations = self.larender_inputs['locations']
        indices = self.larender_inputs['indices']
        densities = self.larender_inputs['densities']
        masks_input = self.larender_inputs.get('masks', None)
        mask_fusion = self.larender_inputs.get('mask_fusion', 'sam_attn')

        attn_power_global = float(self.larender_inputs.get('attn_power', 1.0))
        attn_power_per_object = self.larender_inputs.get('attn_power_per_object', None)

        skip_attn_indices = set(self.larender_inputs.get('skip_attn_indices', []))

        attn_floor_global = float(self.larender_inputs.get('attn_floor', 0.0))
        attn_floor_per_object = self.larender_inputs.get('attn_floor_per_object', None)

        # mask fusion schedule (SAM mask + attention)
        fusion_lambda_global = self.larender_inputs.get("fusion_lambda", None)
        fusion_lambda_per_object = self.larender_inputs.get("fusion_lambda_per_object", None)
        fusion_lambda_start = self.larender_inputs.get("fusion_lambda_start", None)
        fusion_lambda_end = self.larender_inputs.get("fusion_lambda_end", None)
        fusion_schedule = str(self.larender_inputs.get("fusion_schedule", "linear"))
        fusion_t_start = float(self.larender_inputs.get("fusion_t_start", 0.0))
        fusion_t_end = float(self.larender_inputs.get("fusion_t_end", 1.0))

        height = self.height
        width = self.width
        latent_h = int(math.sqrt(hidden_states.size()[1] * height / width))
        assert hidden_states.size()[1] % latent_h == 0
        latent_w = int(hidden_states.size()[1] / latent_h)

        num_inference_steps = len(self.pipe.scheduler.timesteps)
        current_count = self.pipe.scheduler.step_index if getattr(self.pipe.scheduler, "step_index", None) is not None else 0
        progress = float(current_count) / float(max(1, num_inference_steps - 1))

        if self.density_scheduler == 'inverse_proportional':
            density_multiplier = num_inference_steps / (current_count + 1)
            densities_multiplied = [x * density_multiplier for x in densities]
        elif self.density_scheduler == 'opaque':
            densities_multiplied = [x * num_inference_steps for x in densities]
        else:
            densities_multiplied = densities

        token_max_length = self.pipe.tokenizer.model_max_length
        assert encoder_hidden_states.shape[1] % token_max_length == 0

        dtype = hidden_states.dtype
        device = hidden_states.device

        # Precompute bbox masks once (also used to restrict attention-fusion to the object bbox)
        if self.transmittance_use_bbox:
            bbox_M_all = self._bboxes_to_mask(locations, latent_h, latent_w, dtype, device)
        else:
            bbox_M_all = [torch.ones((latent_h, latent_w), dtype=dtype, device=device) for _ in locations]

        # ===== init masks M (allow per-object None) =====
        if masks_input is not None:
            assert len(masks_input) == self.num_objects, \
                f"masks length must equal num_objects ({self.num_objects}), got {len(masks_input)}"

            M = []
            bbox_M = bbox_M_all

            for i in range(self.num_objects):
                m = masks_input[i]
                if m is None:
                    M.append(bbox_M[i])
                    continue

                if not torch.is_tensor(m):
                    m = torch.from_numpy(m)
                m = m.to(device=device, dtype=torch.float32)

                if m.dim() == 2:
                    m = m[None, None, :, :]
                elif m.dim() == 3:
                    m = m.mean(dim=0, keepdim=True)[None, ...]

                m = F.interpolate(m, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
                m = m[0, 0].to(dtype=dtype)
                m = torch.clamp(m, 0, 1)
                if float(m.mean()) < 1e-6:
                    M.append(bbox_M[i])
                else:
                    m = m / (m.max() + 1e-6)
                    M.append(m)
        else:
            M = bbox_M_all

        # ===== object-wise attention and fusion =====
        R = []
        for i in range(self.num_objects):
            context = encoder_hidden_states[:, i * token_max_length: (i + 1) * token_max_length, :]
            R_i, attn_probs = self._forward_cross_attention(module, hidden_states, context)
            R.append(R_i.reshape(R_i.shape[0], latent_h, latent_w, R_i.shape[2]))

            # We may want attention maps even if we "skip" using them for transmittance,
            # because they can guide SAM prompts (方案 1).
            if not self.transmittance_use_attn_map:
                continue

            # ---- build an object attention map from cross-attention probs ----
            eps = 1e-6

            # attn_probs: [B*heads, Q, K]
            bs_total = int(hidden_states.shape[0])
            heads = int(getattr(module, "heads", 1))
            q = int(latent_h * latent_w)
            k = int(attn_probs.shape[-1])
            # Defensive: some attention implementations may change shapes.
            if attn_probs.dim() != 3 or attn_probs.shape[1] != q:
                continue

            try:
                attn_bhqk = attn_probs.reshape(bs_total, heads, q, k)
            except Exception:
                # fallback to old behaviour
                attn_bhqk = attn_probs.reshape(-1, 1, q, k)

            # CFG handling: keep conditional half only for sharper maps.
            # (Original LaRender averages unconditional+conditional, which blurs.)
            if self._attn_cfg_mode == "cond" and bs_total % 2 == 0 and bs_total >= 2:
                attn_bhqk = attn_bhqk[bs_total // 2 :]

            # mean over heads -> [B,Q,K]
            attn_bqk = attn_bhqk.mean(dim=1)

            if not indices[i]:
                continue

            sel = attn_bqk[..., indices[i]]  # [B,Q,Ksel]
            if sel.dim() == 2:
                # [B,Q]
                attn_sel = sel
            else:
                # max over selected tokens -> [B,Q]
                attn_sel = sel.max(dim=-1).values

            # mean over batch -> [Q]
            attn_sel = attn_sel.mean(dim=0)
            attn_map = attn_sel.reshape(latent_h, latent_w)

            # normalize to [0,1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + eps)
            attn_map = torch.clamp(attn_map, 0, 1)
            attn_map = attn_map / (attn_map.max() + eps)

            # --- per-object power (v6+) ---
            power_i = float(self._resolve_per_object(attn_power_per_object, i, attn_power_global))
            power_i = max(0.2, min(float(power_i), 8.0))
            attn_map_sharp = attn_map ** power_i

            # Export *sharp but no-floor* map for SAM prompts.
            self._export_attn(i, attn_map_sharp, progress)

            # --- per-object floor ---
            floor_i = float(self._resolve_per_object(attn_floor_per_object, i, attn_floor_global))

            floor_i = max(0.0, min(float(floor_i), 0.95))
            attn_map_fusion = attn_map_sharp
            if floor_i > 0:
                attn_map_fusion = floor_i + (1.0 - floor_i) * attn_map_fusion

            # Decide whether we apply attn to transmittance for this object.
            use_attn_for_M = (i not in skip_attn_indices)

            if use_attn_for_M:
                # restrict attention to bbox (prevents attention bleed outside the object region)
                attn_box = attn_map_fusion * bbox_M_all[i]

                # resolve fusion lambda (can be scheduled)
                lam_i = self._resolve_per_object(fusion_lambda_per_object, i, None)
                if lam_i is None:
                    lam_i = fusion_lambda_global

                # schedule override
                if fusion_lambda_start is not None and fusion_lambda_end is not None:
                    # remap progress to [0,1] within [t_start,t_end]
                    if fusion_t_end <= fusion_t_start + 1e-6:
                        p_local = 1.0
                    else:
                        p_local = (progress - fusion_t_start) / (fusion_t_end - fusion_t_start)
                    p_local = float(np.clip(p_local, 0.0, 1.0))
                    lam_i = self._schedule(p_local, float(fusion_lambda_start), float(fusion_lambda_end), fusion_schedule)

                # defaults
                if lam_i is None:
                    lam_i = 0.0
                lam_i = float(np.clip(float(lam_i), 0.0, 1.0))

                if masks_input is not None:
                    if mask_fusion in ["sam_attn", "multiply", "sam*attn", "sam_attn_mul"]:
                        # legacy: pure multiplication
                        M[i] = torch.clamp(M[i] * attn_box, 0, 1)
                    elif mask_fusion in ["sam_attn_blend", "sam_gate", "sam*blend"]:
                        # NEW (方案3): keep SAM support, but make the Attn influence adjustable.
                        # lam=1 -> SAM only; lam=0 -> SAM*Attn
                        M[i] = torch.clamp(M[i] * (lam_i + (1.0 - lam_i) * attn_box), 0, 1)
                    elif mask_fusion in ["blend", "lerp"]:
                        # convex blend between SAM and Attn
                        M[i] = torch.clamp(lam_i * M[i] + (1.0 - lam_i) * attn_box, 0, 1)
                    elif mask_fusion == "attn_only":
                        M[i] = attn_box
                    elif mask_fusion == "sam_only":
                        pass
                    else:
                        M[i] = torch.clamp(M[i] * attn_box, 0, 1)
                else:
                    # no SAM mask provided -> bbox mask shaped by attention
                    M[i] = torch.clamp(M[i] * attn_box, 0, 1)

            M[i] = M[i] / (M[i].max() + eps)

        # ===== transmittance accumulate =====
        T = [torch.ones((latent_h, latent_w), dtype=dtype, device=device) for _ in range(self.num_objects)]
        for i in range(self.num_objects - 1, -1, -1):
            for j in range(i + 1, self.num_objects):
                T[i] *= torch.exp(-torch.tensor(densities_multiplied[j], device=device, dtype=dtype) * M[j])

        # ===== rendering =====
        S = 0
        R_out = 0
        for i in range(self.num_objects):
            contrib = T[i] * (1 - math.exp(-densities_multiplied[i])) * M[i]
            contrib = contrib[None, :, :, None]
            R_out += contrib * R[i]
            S += contrib

        S = torch.clamp(S, min=1e-6)
        R_out = R_out / S
        return R_out.reshape(*hidden_states.shape)

    def run(self,
            objects,
            locations,
            alpha,
            indices,
            masks=None,
            mask_fusion='sam_attn',
            fusion_lambda=None,
            fusion_lambda_per_object=None,
            fusion_lambda_start=None,
            fusion_lambda_end=None,
            fusion_schedule: str = "linear",
            fusion_t_start: float = 0.0,
            fusion_t_end: float = 1.0,
            attn_power=1.5,
            attn_power_per_object=None,   # v6+
            attn_floor=0.15,
            attn_floor_per_object=None,   # v6
            skip_attn_indices=None,
            export_attn: bool = False,
            attn_export_start: float = 0.35,
            attn_export_end: float = 1.0,
            attn_cfg_mode: str = "cond",
            return_attn: bool = False,
            num_images_per_prompt=1,
            height=1024,
            width=1024,
            negative_prompt='',
            seed=0):
        # Convert alpha (opacity in [0,1]) to density via -log(1-a).
        # Some scenes may use a==1.0 for fully opaque layers (especially background).
        # That would yield infinite density and break downstream computations, so we clamp.
        if alpha is None:
            # Default: near-opaque for every object to mimic the original intent.
            alpha = [0.999 for _ in objects]

        eps = 1e-4
        max_density = 20.0
        alpha_safe = []
        for i, a in enumerate(alpha):
            try:
                a = float(a)
            except Exception:
                print(f"[WARN] alpha[{i}] is not a number ({a}); setting to 0.")
                a = 0.0
            if a >= 1.0:
                print(f"[WARN] alpha[{i}]={a} >= 1; clamping to {1.0 - eps} to avoid infinite density.")
                a = 1.0 - eps
            elif a < 0.0:
                print(f"[WARN] alpha[{i}]={a} < 0; clamping to 0.")
                a = 0.0
            alpha_safe.append(a)

        densities = []
        for a in alpha_safe:
            d = -np.log(max(1e-12, 1.0 - a))
            if d > max_density:
                d = max_density
            densities.append(float(d))
        self.num_objects = len(objects)

        assert len(locations) == len(objects) == len(densities)

        if skip_attn_indices is None:
            skip_attn_indices = []

        # attention export config
        self._attn_export_enabled = bool(export_attn or return_attn)
        self._attn_export_start = float(attn_export_start)
        self._attn_export_end = float(attn_export_end)
        self._attn_cfg_mode = str(attn_cfg_mode or "cond").lower()
        self._reset_attn_cache(self.num_objects)

        self.larender_inputs = dict(
            locations=locations,
            indices=indices,
            densities=densities,
            masks=masks,
            mask_fusion=mask_fusion,
            fusion_lambda=fusion_lambda,
            fusion_lambda_per_object=fusion_lambda_per_object,
            fusion_lambda_start=fusion_lambda_start,
            fusion_lambda_end=fusion_lambda_end,
            fusion_schedule=fusion_schedule,
            fusion_t_start=fusion_t_start,
            fusion_t_end=fusion_t_end,
            attn_power=attn_power,
            attn_power_per_object=attn_power_per_object,
            attn_floor=attn_floor,
            attn_floor_per_object=attn_floor_per_object,
            skip_attn_indices=skip_attn_indices
        )

        self.height = height
        self.width = width

        (prompt_embeds,
         negative_prompt_embeds,
         pooled_prompt_embeds,
         negative_pooled_prompt_embeds) = self._prompt_embedding(objects, negative_prompt)

        torch_fix_seed(seed)

        common_kwargs = dict(
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            guidance_scale=7.0,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
        )

        if self.pretrain == 'IterComp':
            images = self.pipe(num_inference_steps=25, **common_kwargs).images
        else:
            images = self.pipe(
                num_inference_steps=50,
                gligen_scheduled_sampling_beta=0.4,
                gligen_boxes=[[loc[2], loc[0], loc[3], loc[1]] for loc in locations],
                gligen_phrases=objects,
                **common_kwargs
            ).images

        if return_attn:
            attn_maps = self.get_attn_maps(out_h=height, out_w=width)
            return images, attn_maps
        return images
