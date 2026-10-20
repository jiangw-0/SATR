import math
import numpy as np
import os
import re
from typing import Dict, List, Optional, Tuple

from diffusers.utils import logging as diffusers_logging
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from utils import torch_fix_seed

_OCC_TAG_RE = re.compile(r"<<[^<>]*>>")

def strip_occlusion_tags(text: str) -> str:
    """Remove occlusion directives like <<behind:...>> / <<front_of:...>> from prompt text."""
    if not isinstance(text, str):
        text = str(text)
    text = _OCC_TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



class LaRender():
    def __init__(self,#加载模型 + 替换 UNet cross-attn
                 pretrain='IterComp',
                 transmittance_use_bbox=True,
                 transmittance_use_attn_map=True,
                 density_scheduler='unchanged',
                 device='cuda'):
        '''
        Args:
            pretrain ('str', *optional*):
                Pretrain model type, support 'IterComp' or 'GLIGEN' now. 'IterComp' is T2I without location control, while 'GLIGEN' is a pre-trained location contorl model.
            transmittance_use_bbox (bool, *optional*):
                The flag whether to use bounding boxes in approximating transmittance maps. If set to False, objects in the back usually disappear.
            transmittance_use_attn_map (bool, *optional*):
                The flag whether to use attention maps in approximating transmittance maps.
            density_scheduler (str, *optional*):
                The type of density scheduler, support 'opaque', 'unchanged', 'inverse_proportional'.

        '''
        assert pretrain in ['IterComp', 'GLIGEN']
        assert density_scheduler in [
            'opaque', 'unchanged', 'inverse_proportional']
        self.pretrain = pretrain
        self.transmittance_use_bbox = transmittance_use_bbox
        self.density_scheduler = density_scheduler
        self.device = device

        # Optional: suppress diffusers "prompt too long" warnings (does NOT change truncation behavior).
        if os.environ.get('LARENDER_SUPPRESS_LONG_PROMPT_WARN', '0') == '1':
            try:
                diffusers_logging.set_verbosity_error()
            except Exception:
                pass

        # Respect offline mode if user exports HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE.
        local_only = (os.getenv('HF_HUB_OFFLINE', '0') == '1') or (os.getenv('TRANSFORMERS_OFFLINE', '0') == '1')

        if pretrain == 'IterComp':
            # Allow pointing to a local snapshot directory, e.g.
            #   export ITERCOMP_PATH=$HF_HUB_CACHE/models--comin--IterComp/snapshots/<sha>
            model_id = os.getenv('ITERCOMP_PATH') or os.getenv('ITERCOMP_MODEL') or 'comin/IterComp'
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=local_only,
            )
        else:
            gligen_id = os.getenv('GLIGEN_PATH', 'jiuntian/gligen-xl-1024')
            self.pipe = DiffusionPipeline.from_pretrained(
                gligen_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                local_files_only=local_only,
            )

        # Move pipeline to requested device (e.g. "cuda", "cuda:0", "cpu").
        self.pipe.to(device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True)
        # 原来可能是直接调用（没有 try/except）
        # self.pipe.enable_xformers_memory_efficient_attention()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[LaRender] xformers enabled.")
        except Exception as e:
            print(f"[LaRender] xformers not available, fallback to PyTorch attention. ({e})")
            # 可选：进一步省显存（不是必须）
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

        self.isvanilla = True
        self.transmittance_use_attn_map = transmittance_use_attn_map

        # Cache exported transmittance maps for optional visualization/debug.
        # Structure: {step_idx: [ {(h,w): {'sum': Tensor, 'cnt': Tensor}}, ... ]}
        self._transmittance_cache: Optional[Dict[int, List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]]]] = None
        self._transmittance_export_enabled: bool = False
        self._support_cache: Optional[Dict[int, List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]]]] = None
        self._contrib_cache: Optional[Dict[int, List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]]]] = None
        self._support_export_enabled: bool = False
        self._contrib_export_enabled: bool = False

        for name, module in self.pipe.unet.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                module.forward = self.create_larender_forward(module)#把这个 attention 模块的 forward 函数替换成 LaRender 自己的版本
                module.name = name

    def _forward_cross_attention(self, module, hidden_states, encoder_hidden_states):#“手写一遍”cross-attn 并返回注意力
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

    def _bboxes_to_mask(self, bboxes, h, w, dtype, device):#把 bbox 转成 latent 网格上的 0/1 mask
        '''
        Args:
            bboxes (List[List[y1, y2, x1, x2]]):
                The bounding boxes in y1, y2, x1, x2 format.
        '''
        masks = []
        for bbox in bboxes:
            M = torch.zeros((h, w), dtype=dtype, device=device)
            upper = int(h * bbox[0])
            lower = int(h * bbox[1])
            left = int(w * bbox[2])
            right = int(w * bbox[3])
            assert lower > upper and right > left, f"Invalid bounding box: {bbox}"
            M[upper:lower, left:right] = 1
            masks.append(M)
        return masks

    def create_larender_forward(self, module):#生成替换用的 forward 闭包
        def forward(hidden_states, encoder_hidden_states=None, **kwargs):#定义闭包函数 forward
            return self.larender_core_process(module, hidden_states, encoder_hidden_states)#真正执行 LaRender 的核心逻辑：多对象渲染融合
        return forward

    def _reset_transmittance_cache(self, num_objects: int):
        self._transmittance_cache = {}

    def _reset_support_cache(self, num_objects: int):
        self._support_cache = {}

    def _reset_contrib_cache(self, num_objects: int):
        self._contrib_cache = {}

    def _export_transmittance(self, step_idx: int, obj_i: int, trans_hw: torch.Tensor):
        """Accumulate one transmittance map (H,W) for object i at a denoising step."""
        if not self._transmittance_export_enabled or self._transmittance_cache is None:
            return

        step_idx = int(step_idx)
        if step_idx not in self._transmittance_cache:
            self._transmittance_cache[step_idx] = [dict() for _ in range(self.num_objects)]

        key = (int(trans_hw.shape[0]), int(trans_hw.shape[1]))
        slot = self._transmittance_cache[step_idx][obj_i].get(key, None)
        if slot is None:
            slot = {
                'sum': torch.zeros_like(trans_hw, dtype=torch.float32),
                'cnt': torch.zeros((), device=trans_hw.device, dtype=torch.float32),
            }
            self._transmittance_cache[step_idx][obj_i][key] = slot

        slot['sum'] = slot['sum'] + trans_hw.detach().to(dtype=torch.float32)
        slot['cnt'] = slot['cnt'] + 1.0

    def _export_support(self, step_idx: int, obj_i: int, support_hw: torch.Tensor):
        if not self._support_export_enabled or self._support_cache is None:
            return

        step_idx = int(step_idx)
        if step_idx not in self._support_cache:
            self._support_cache[step_idx] = [dict() for _ in range(self.num_objects)]

        key = (int(support_hw.shape[0]), int(support_hw.shape[1]))
        slot = self._support_cache[step_idx][obj_i].get(key, None)
        if slot is None:
            slot = {
                'sum': torch.zeros_like(support_hw, dtype=torch.float32),
                'cnt': torch.zeros((), device=support_hw.device, dtype=torch.float32),
            }
            self._support_cache[step_idx][obj_i][key] = slot

        slot['sum'] = slot['sum'] + support_hw.detach().to(dtype=torch.float32)
        slot['cnt'] = slot['cnt'] + 1.0

    def _export_contrib(self, step_idx: int, obj_i: int, contrib_hw: torch.Tensor):
        if not self._contrib_export_enabled or self._contrib_cache is None:
            return

        step_idx = int(step_idx)
        if step_idx not in self._contrib_cache:
            self._contrib_cache[step_idx] = [dict() for _ in range(self.num_objects)]

        key = (int(contrib_hw.shape[0]), int(contrib_hw.shape[1]))
        slot = self._contrib_cache[step_idx][obj_i].get(key, None)
        if slot is None:
            slot = {
                'sum': torch.zeros_like(contrib_hw, dtype=torch.float32),
                'cnt': torch.zeros((), device=contrib_hw.device, dtype=torch.float32),
            }
            self._contrib_cache[step_idx][obj_i][key] = slot

        slot['sum'] = slot['sum'] + contrib_hw.detach().to(dtype=torch.float32)
        slot['cnt'] = slot['cnt'] + 1.0

    def get_transmittance_maps(
        self,
        out_h: Optional[int] = None,
        out_w: Optional[int] = None,
        pick: str = 'max',
    ) -> Optional[Dict[int, List[np.ndarray]]]:
        """Get the last run's exported transmittance maps as {step_idx: [np.ndarray, ...]}."""
        if not self._transmittance_cache:
            return None
        if out_h is None:
            out_h = int(getattr(self, 'height', 0) or 0)
        if out_w is None:
            out_w = int(getattr(self, 'width', 0) or 0)

        out: Dict[int, List[np.ndarray]] = {}
        for step_idx in sorted(self._transmittance_cache.keys()):
            per_obj = self._transmittance_cache[step_idx]
            maps: List[np.ndarray] = []
            for i in range(len(per_obj)):
                if not per_obj[i]:
                    maps.append(None)
                    continue

                keys = list(per_obj[i].keys())
                key = max(keys, key=lambda k: int(k[0]) * int(k[1])) if pick == 'max' else keys[0]

                slot = per_obj[i][key]
                cnt = float(slot['cnt'].item()) if slot['cnt'].numel() else 0.0
                if cnt <= 0:
                    maps.append(None)
                    continue

                trans = (slot['sum'] / max(cnt, 1.0)).clamp(0, 1)
                if out_h > 0 and out_w > 0 and (trans.shape[0] != out_h or trans.shape[1] != out_w):
                    t = trans[None, None, ...]
                    t = F.interpolate(t, size=(out_h, out_w), mode='bilinear', align_corners=False)
                    trans = t[0, 0]

                trans_np = trans.detach().float().cpu().numpy().astype(np.float32)
                trans_np = np.nan_to_num(trans_np, nan=0.0, posinf=1.0, neginf=0.0)
                trans_np = np.clip(trans_np, 0.0, 1.0)
                maps.append(trans_np)
            out[int(step_idx)] = maps
        return out

    def get_support_maps(
        self,
        out_h: Optional[int] = None,
        out_w: Optional[int] = None,
        pick: str = 'max',
    ) -> Optional[Dict[int, List[np.ndarray]]]:
        if not self._support_cache:
            return None
        if out_h is None:
            out_h = int(getattr(self, 'height', 0) or 0)
        if out_w is None:
            out_w = int(getattr(self, 'width', 0) or 0)

        out: Dict[int, List[np.ndarray]] = {}
        for step_idx in sorted(self._support_cache.keys()):
            per_obj = self._support_cache[step_idx]
            maps: List[np.ndarray] = []
            for i in range(len(per_obj)):
                if not per_obj[i]:
                    maps.append(None)
                    continue

                keys = list(per_obj[i].keys())
                key = max(keys, key=lambda k: int(k[0]) * int(k[1])) if pick == 'max' else keys[0]

                slot = per_obj[i][key]
                cnt = float(slot['cnt'].item()) if slot['cnt'].numel() else 0.0
                if cnt <= 0:
                    maps.append(None)
                    continue

                support = (slot['sum'] / max(cnt, 1.0)).clamp(0, 1)
                if out_h > 0 and out_w > 0 and (support.shape[0] != out_h or support.shape[1] != out_w):
                    t = support[None, None, ...]
                    t = F.interpolate(t, size=(out_h, out_w), mode='bilinear', align_corners=False)
                    support = t[0, 0]

                support_np = support.detach().float().cpu().numpy().astype(np.float32)
                support_np = np.nan_to_num(support_np, nan=0.0, posinf=1.0, neginf=0.0)
                support_np = np.clip(support_np, 0.0, 1.0)
                maps.append(support_np)
            out[int(step_idx)] = maps
        return out

    def get_contrib_maps(
        self,
        out_h: Optional[int] = None,
        out_w: Optional[int] = None,
        pick: str = 'max',
    ) -> Optional[Dict[int, List[np.ndarray]]]:
        if not self._contrib_cache:
            return None
        if out_h is None:
            out_h = int(getattr(self, 'height', 0) or 0)
        if out_w is None:
            out_w = int(getattr(self, 'width', 0) or 0)

        out: Dict[int, List[np.ndarray]] = {}
        for step_idx in sorted(self._contrib_cache.keys()):
            per_obj = self._contrib_cache[step_idx]
            maps: List[np.ndarray] = []
            for i in range(len(per_obj)):
                if not per_obj[i]:
                    maps.append(None)
                    continue

                keys = list(per_obj[i].keys())
                key = max(keys, key=lambda k: int(k[0]) * int(k[1])) if pick == 'max' else keys[0]

                slot = per_obj[i][key]
                cnt = float(slot['cnt'].item()) if slot['cnt'].numel() else 0.0
                if cnt <= 0:
                    maps.append(None)
                    continue

                contrib = (slot['sum'] / max(cnt, 1.0)).clamp(0, 1)
                if out_h > 0 and out_w > 0 and (contrib.shape[0] != out_h or contrib.shape[1] != out_w):
                    t = contrib[None, None, ...]
                    t = F.interpolate(t, size=(out_h, out_w), mode='bilinear', align_corners=False)
                    contrib = t[0, 0]

                contrib_np = contrib.detach().float().cpu().numpy().astype(np.float32)
                contrib_np = np.nan_to_num(contrib_np, nan=0.0, posinf=1.0, neginf=0.0)
                contrib_np = np.clip(contrib_np, 0.0, 1.0)
                maps.append(contrib_np)
            out[int(step_idx)] = maps
        return out

    def _prompt_embedding(self, objects, negative_prompt):#把每个对象 prompt 编成“拼接的 token 序列”
        prompt_embeds_list = []
        # Optional: strip occlusion tags like <<behind:...>> from the *text encoder* input to save 77-token budget.
        if os.environ.get('LARENDER_STRIP_OCC_TAGS', '1') == '1':
            objects = [strip_occlusion_tags(p) for p in objects]
            negative_prompt = strip_occlusion_tags(negative_prompt)
        negative_prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        negative_pooled_prompt_embeds_list = []
        for p in objects:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(#调用 diffusers pipeline 的 encode_prompt：把文本 prompt 编成 embeddings
                prompt=p,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_list.append(prompt_embeds)
            negative_prompt_embeds_list.append(negative_prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            negative_pooled_prompt_embeds_list.append(
                negative_pooled_prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1)#把多个对象的 token 序列在 token 维（dim=1） 拼接
        negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=1)
        pooled_prompt_embeds = sum(
            pooled_prompt_embeds_list) / len(pooled_prompt_embeds)
        negative_pooled_prompt_embeds = sum(
            negative_pooled_prompt_embeds_list) / len(negative_pooled_prompt_embeds)

        (
            _,
            _,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=', '.join(objects),
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def larender_core_process(self, module, hidden_states, encoder_hidden_states):#核心：对象级 cross-attn + 遮挡透射率 + 渲染融合
        # preparation，（准备输入、推 latent 尺寸、密度调度）
        locations = self.larender_inputs['locations']
        indices = self.larender_inputs['indices']
        densities = self.larender_inputs['densities']
        height = self.height
        width = self.width
        latent_h = int(math.sqrt(hidden_states.size()[1] * height / width))
        assert hidden_states.size()[1] % latent_h == 0, \
            "Cannot infer latent_h and latent_w, please check the code."
        latent_w = int(hidden_states.size()[1] / latent_h)
        num_inference_steps = len(self.pipe.scheduler.timesteps)
        assert hasattr(self.pipe.scheduler, 'step_index')
        current_count = self.pipe.scheduler.step_index if self.pipe.scheduler.step_index is not None else 0
        self.current_count = current_count  # only used in visualization
        if self.density_scheduler == 'inverse_proportional':
            density_multiplier = num_inference_steps / (current_count + 1)
            densities_multiplied = [x * density_multiplier for x in densities]
        elif self.density_scheduler == 'opaque':
            densities_multiplied = [x * num_inference_steps for x in densities]
        else:
            densities_multiplied = densities

        token_max_length = self.pipe.tokenizer.model_max_length
        assert encoder_hidden_states.shape[1] % token_max_length == 0

        # object-wise cross-attention and transmittance map（对象级 cross-attn 和透射率图（mask））
        R = []
        dtype = hidden_states.dtype
        device = hidden_states.device
        if self.transmittance_use_bbox:
            M = self._bboxes_to_mask(
                locations, latent_h, latent_w, dtype, device)
        else:
            M = [torch.ones((latent_h, latent_w), dtype=dtype,
                            device=device) for _ in locations]
        for i in range(self.num_objects):
            context = encoder_hidden_states[:, i *
                                            token_max_length: (i+1) * token_max_length, :]
            R_i, attn_probs = \
                self._forward_cross_attention(module, hidden_states, context)#图像特征 hidden_states 对文本段 context 做注意力，得到对象特征 R_i + 注意力矩阵 attn_probs
            R.append(R_i.reshape(R_i.shape[0],
                     latent_h, latent_w, R_i.shape[2]))
            if self.transmittance_use_attn_map:
                attn_map = attn_probs.reshape(
                    attn_probs.shape[0], latent_h, latent_w, -1)
                attn_map = attn_map[..., indices[i]].mean(dim=-1).mean(dim=0)#哪里更关注主体词，哪里更可能是该对象的区域
                min_val = attn_map.min()
                max_val = attn_map.max()
                normalized_attn_map = (
                    attn_map - min_val) / (max_val - min_val)
                M[i] *= normalized_attn_map#用注意力图去“软化/细化” bbox mask：bbox 限定大致范围，attention 决定更像轮廓的权重

        if self._support_export_enabled:
            for i in range(self.num_objects):
                self._export_support(current_count, i, M[i])

        # accumulated transmittance maps (visibility of planar i from the virtual camera)，计算累积透射率 T（谁挡住谁）
        T = [torch.ones((latent_h, latent_w), dtype=dtype, device=device)#T[i] 表示对象 i 在每个像素上的“还剩多少能看到”（被前景遮挡会变小）
             for _ in range(self.num_objects)]
        for i in range(self.num_objects - 1, -1, -1):  # top to bottom
            for j in range(i + 1, self.num_objects):
                # TODO
                T[i] *= torch.exp(-torch.tensor(densities_multiplied[j], device=device, dtype=dtype) * M[j])#遮挡衰减公式

        if self._transmittance_export_enabled:
            for i in range(self.num_objects):
                self._export_transmittance(current_count, i, T[i])

        # rendering，渲染融合：按 T、density、M 融合 R_i 得到输出
        S = 0
        R_out = 0
        for i in range(self.num_objects):
            contrib = T[i] * (1 - math.exp(-densities_multiplied[i])) * M[i]
            if self._contrib_export_enabled:
                self._export_contrib(current_count, i, contrib)
            contrib = contrib[None, :, :, None]  # unsqueeze 0, -1
            R_out += contrib * R[i]
            S += contrib
        S = torch.clamp(S, min=1e-6)
        R_out /= S
        return R_out.reshape(*hidden_states.shape)

    def run(self,#对外接口：输入 objects/bbox/alpha/indices → 输出 PIL images
            objects,#对象 prompt 列表（每个对象一段短语/句子）
            locations,#bbox 列表，每个对象一个 bbox
            alpha,#直观表示“强度/不透明度”
            indices,#每个对象主体词 token 的索引（用于取 attention 通道）
            num_images_per_prompt=1,#每个 prompt 生成几张图（diffusers 支持批量出图）
            height=1024,
            width=1024,
            negative_prompt='',
            seed=0,
            export_transmittance=False,
            return_transmittance=False,
            export_support=False,
            export_contrib=False):
        '''
        Args:
            objects (List[str]):
                List of object prompts.
            locations (List[List[float]]):
                List of bounding boxes in [top, bottom, left, right] format.
            alpha ([List[float]):
                List of alpha of each object, values must be in [0, 1).
            indices (List[List[int]]):
                List of indices of subjects in each object prompt.
        Output:
            images (List[PIL.Image]):
                List of generated images.
        '''
        # 把 alpha 转为 density，保存输入给 attention forward 使用
        # 允许 alpha==1（完全不透明）时做数值钳位，避免 density=inf
        _alpha_in = [float(a) for a in alpha]
        alpha = []
        for i, a in enumerate(_alpha_in):
            if a < 0:
                raise ValueError(f"alpha[{i}] must be >=0, got {a}")
            if a >= 1.0:
                print(f"[WARN] alpha[{i}]={a} >= 1; clamping to 0.9999 to avoid infinite density.")
                a = 0.9999
            alpha.append(a)
        densities = [-np.log(1 - a) for a in alpha]  # density = -log(1-alpha)
        self.num_objects = len(objects)
        assert len(locations) == len(objects) and len(
            densities) == len(objects)
        self.larender_inputs = dict(
            locations=locations, indices=indices, densities=densities)

        self._transmittance_export_enabled = bool(export_transmittance or return_transmittance)
        self._support_export_enabled = bool(export_support)
        self._contrib_export_enabled = bool(export_contrib)
        self._reset_transmittance_cache(self.num_objects)
        self._reset_support_cache(self.num_objects)
        self._reset_contrib_cache(self.num_objects)

        self.height = height
        self.width = width

        (prompt_embeds,
         negative_prompt_embeds,
         pooled_prompt_embeds,
         negative_pooled_prompt_embeds) = self._prompt_embedding(objects, negative_prompt)
#token embeddings：拼接后的对象段，pooled embeddings：整体 prompt 段

        torch_fix_seed(seed)

#组装 diffusers 推理参数
        common_kwargs = dict(#构造一个 dict，存放两种模型都会用到的参数
            height=height,
            width=width,
            guidance_scale=7.0,#CFG 强度，固定为 7.0，guidance_scale 越大，越听 prompt，但可能更不自然/更容易过饱和
            prompt_embeds=prompt_embeds,#传入我们自己算好的 token embeddings（而不是传 prompt 文本），这样 cross-attn 才能按对象段切片
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds)

        # ---- Micro-batching to avoid OOM when generating many images at once ----
        # Total images:
        total = int(num_images_per_prompt)
        # Per-call batch size. Default=4; set to 1/2 if still OOM:
        #   export LARENDER_BATCH_SIZE=2
        # Disable chunking:
        #   export LARENDER_NO_CHUNK=1
        chunk = int(os.environ.get("LARENDER_BATCH_SIZE", "4"))
        if os.environ.get("LARENDER_NO_CHUNK", "0") == "1":
            chunk = total
        chunk = max(1, min(chunk, total))

        images = []
        for _start in range(0, total, chunk):
            bs = min(chunk, total - _start)

            # Use per-image generators for reproducibility across different chunk sizes.
            # This makes image i deterministic w.r.t. (seed + i).
            gens = [torch.Generator(device=self.device).manual_seed(int(seed) + _start + k) for k in range(bs)]
            if self.pretrain == 'IterComp':#如果用 IterComp，走普通推理
                out = self.pipe(#调用 diffusers pipeline 推理生成图像
                    num_inference_steps=25,
                    num_images_per_prompt=bs,
                    generator=gens,
                    **common_kwargs).images#.images 取出 pipeline 输出中的 PIL 图像列表
            else:
                out = self.pipe(
                    num_inference_steps=50,
                    num_images_per_prompt=bs,
                    generator=gens,
                    gligen_scheduled_sampling_beta=0.4,#GLIGEN 的 scheduled sampling 超参（具体含义取决于 GLIGEN 实现，但可以把它理解成一个控制条件注入强度/策略的参数）
                    gligen_boxes=[[loc[2], loc[0], loc[3], loc[1]]#把 bbox 从 [y1,y2,x1,x2] 改成 GLIGEN 需要的 [x1,y1,x2,y2]
                                  for loc in locations],  # y1y2x1x2 -> x1y1x2y2
                    gligen_phrases=objects,
                    **common_kwargs).images#把公共参数也传进去，拿到 .images
            images.extend(out)

            if os.environ.get("LARENDER_EMPTY_CACHE", "0") == "1" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_transmittance:
            transmittance_maps = self.get_transmittance_maps(out_h=height, out_w=width)
            return images, transmittance_maps
        return images