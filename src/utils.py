"""Utility helpers used by both the original LaRender code and the extended demos.

The upstream LaRender repo's `utils.py` only contained:
  - DependencyParsing (spaCy-based subject token selection)
  - torch_fix_seed

Your newer SAM×Attn demos additionally expect some small filesystem helpers and
token-selection helpers. They are implemented here to make the project
self-contained and to support *attention-guided* token selection.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


class DependencyParsing:
    """Parse English prompts to find "subject" noun tokens.

    Returns indices within spaCy tokenization.
    NOTE: For CLIP token indices you still need to offset / map separately.
    """

    def __init__(self):
        import spacy

        self.nlp = spacy.load("en_core_web_sm")
        # if it reports cannot find "en_core_web_sm", try to load as below:
        # self.nlp = spacy.load("/path/to/en_core_web_sm")

    def parse(self, text: str, never_empty: bool):
        doc = self.nlp(text)
        indices: List[int] = []
        subjects: List[str] = []
        for token in doc:
            # noun of a sentence or noun in a phrase
            if (token.dep_ in ["nsubj", "nsubjpass"]) or (
                token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]
            ):
                indices.append(int(token.i))
                subjects.append(str(token.text))
        if never_empty and len(indices) == 0:
            indices.append(0)
            subjects.append(str(doc[0].text))
        return indices, subjects


def torch_fix_seed(seed: int = 42, deterministic: bool = False):
    """Best-effort reproducibility.

    Notes:
    - We do **not** force deterministic algorithms by default because many
      diffusion pipelines call CUDA kernels that are non-deterministic (and
      PyTorch may raise errors when `use_deterministic_algorithms(True)`).
    - If you need strict determinism, pass deterministic=True.
    """

    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


# =====================
# File system helpers
# =====================


def ensure_dir(path: str | os.PathLike) -> str:
    """Create directory if missing and return it as a string path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def infer_project_root(fallback: str | None = None) -> str:
    """Infer a project root folder.

    Priority:
    1) Env var `OCCLUSION_ROOT` or `PROJECT_ROOT`
    2) `fallback` if provided
    3) Current working directory
    """

    def _normalize(p: str | os.PathLike) -> str:
        p = str(p)
        try:
            pp = Path(p).expanduser().resolve()
        except Exception:
            pp = Path(p).expanduser()

        # If user points OCCLUSION_ROOT to the *src* folder, treat its parent as
        # the project root. This matches common repo layouts:
        #   Occlusion/
        #     src/
        #     models/
        #     outputs/
        if pp.name == "src" and pp.parent.exists():
            return str(pp.parent)

        # If it's already a repo-like folder, keep as-is.
        if (pp / "src").exists() or (pp / "models").exists() or (pp / "outputs").exists():
            return str(pp)

        # If it's a file within the repo, return its parent.
        if pp.is_file():
            return str(pp.parent)

        return str(pp)

    for k in ("OCCLUSION_ROOT", "PROJECT_ROOT"):
        v = os.environ.get(k, None)
        if v:
            return _normalize(v)
    if fallback is not None:
        return _normalize(fallback)
    return _normalize(os.getcwd())


def clip_tokenizer_max_length(tokenizer, default: int = 77) -> int:
    """Best-effort CLIP max length.

    Some tokenizers expose a very large sentinel value; clamp to a reasonable
    default for SDXL/CLIP.
    """

    try:
        m = int(getattr(tokenizer, "model_max_length", default))
    except Exception:
        m = int(default)
    if m <= 0:
        return int(default)
    # Transformers sometimes uses VERY_LARGE_INTEGER for unlimited.
    if m > 4096:
        return int(default)
    return m


def truncate_prompt_to_max_tokens(
    prompt: str,
    tokenizer,
    max_length: int | None = None,
) -> tuple[str, bool]:
    """Truncate a text prompt so CLIP won't exceed its max token length.

    Returns (new_prompt, was_truncated).

    Why this exists:
    - CLIP text encoders used in SDXL typically have a hard cap (77 tokens).
    - If a prompt exceeds this, Diffusers will truncate internally, but you may
      see warnings and (worse) *lose* important keywords at the end.
    - By truncating ourselves we can keep token-index selection and attention
      export consistent.
    """

    if prompt is None:
        return "", False

    max_len = int(max_length or clip_tokenizer_max_length(tokenizer))

    # Encode once without truncation to detect overflow.
    enc_full = tokenizer(prompt, add_special_tokens=True, truncation=False)
    ids_full = list(enc_full.get("input_ids", []))
    if len(ids_full) <= max_len:
        return prompt, False

    # Encode with truncation, then decode back.
    enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_len)
    ids = list(enc.get("input_ids", []))
    new_prompt = tokenizer.decode(ids, skip_special_tokens=True).strip()
    if not new_prompt:
        # Extremely unlikely, but keep original prompt if decode fails.
        return prompt, True
    return new_prompt, True


# =====================
# Token helpers (attention-guided)
# =====================


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    """Return all start indices where `needle` occurs in `haystack`."""

    if not needle or not haystack or len(needle) > len(haystack):
        return []
    out: List[int] = []
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            out.append(i)
    return out


def token_indices_for_keywords(
    prompt: str,
    keywords: Iterable[str],
    tokenizer,
    max_indices: int | None = 16,
) -> List[int]:
    """Return token indices in `prompt` that match any keyword.

    This is robust to keywords spanning multiple BPE tokens by matching token-id
    subsequences instead of naive string matching.

    Notes:
    - Returned indices are indices within the tokenizer sequence (including BOS).
    - Duplicates are removed and indices are sorted.
    """

    if prompt is None:
        return []
    kws = [str(k).strip() for k in keywords if k and str(k).strip()]
    if not kws:
        return []

    enc = tokenizer(prompt, add_special_tokens=True)
    prompt_ids: List[int] = list(enc["input_ids"])

    hit: set[int] = set()
    for kw in kws:
        kw_ids: List[int] = list(tokenizer(kw, add_special_tokens=False)["input_ids"])
        for s in _find_subsequence(prompt_ids, kw_ids):
            for j in range(s, s + len(kw_ids)):
                hit.add(int(j))

    out = sorted(hit)
    if max_indices is not None and len(out) > int(max_indices):
        out = out[: int(max_indices)]
    return out


def pretty_selected_tokens(prompt: str, tokenizer, indices: Sequence[int]) -> str:
    """Pretty-print selected token indices for debugging."""

    if not indices:
        return "[]"
    enc = tokenizer(prompt, add_special_tokens=True)
    ids: List[int] = list(enc["input_ids"])

    pieces = []
    for i in indices:
        ii = int(i)
        if 0 <= ii < len(ids):
            tok = tokenizer.decode([ids[ii]])
            tok = tok.replace("\n", "\\n")
            pieces.append(f"{ii}:'{tok}'")
    return "[" + ", ".join(pieces) + "]"
