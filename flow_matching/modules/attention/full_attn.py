import math
import os
import warnings
from typing import Literal

import torch

BACKEND: Literal['flash_attn', 'sdpa', 'naive'] = os.environ.get('ATTN_BACKEND', 'flash_attn')
if BACKEND not in {'flash_attn', 'sdpa', 'naive'}:
    raise ValueError(f'Unsupported ATTN_BACKEND: {BACKEND}')


def _flash_attn_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


if BACKEND == 'flash_attn':
    try:
        if not _flash_attn_supported():
            raise RuntimeError(
                f'flash-attn requires GPU compute capability >= 8.0 '
                f'(detected {torch.cuda.get_device_capability() if torch.cuda.is_available() else "no CUDA"})'
            )
        import flash_attn
    except (ImportError, RuntimeError) as e:
        warnings.warn(f'flash_attn unavailable ({e}); falling back to sdpa backend.')
        BACKEND = 'sdpa'

if BACKEND == 'sdpa':
    from torch.nn.functional import scaled_dot_product_attention as _sdpa


def set_backend(backend: Literal['flash_attn', 'sdpa', 'naive']) -> None:
    global BACKEND
    BACKEND = backend


def _naive_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    scale = 1 / math.sqrt(q.size(-1))
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return (attn @ v).permute(0, 2, 1, 3)


def scaled_dot_product_attention(*args) -> torch.Tensor:
    """Run scaled dot-product attention.

    Supports three call signatures (matching the original Trellis style):
      - 1 arg: packed qkv of shape [B, L, 3, H, C]
      - 2 args: q of shape [B, L, H, C] and packed kv of shape [B, L, 2, H, C]
      - 3 args: q, k, v each of shape [B, L, H, C]
    """
    if len(args) == 1:
        qkv = args[0]
        assert qkv.ndim == 5 and qkv.shape[2] == 3
    elif len(args) == 2:
        q, kv = args
        assert q.ndim == 4 and kv.ndim == 5 and kv.shape[2] == 2
    elif len(args) == 3:
        q, k, v = args
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    else:
        raise ValueError(f'Expected 1-3 args, got {len(args)}')

    if BACKEND == 'flash_attn':
        if len(args) == 1:
            orig_dtype = qkv.dtype
            out = flash_attn.flash_attn_qkvpacked_func(qkv.to(torch.bfloat16))
        elif len(args) == 2:
            orig_dtype = q.dtype
            out = flash_attn.flash_attn_kvpacked_func(q.to(torch.bfloat16), kv.to(torch.bfloat16))
        else:
            orig_dtype = q.dtype
            out = flash_attn.flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))
        return out.to(orig_dtype)

    # sdpa / naive both want explicit q, k, v
    if len(args) == 1:
        q, k, v = qkv.unbind(dim=2)
    elif len(args) == 2:
        k, v = kv.unbind(dim=2)

    if BACKEND == 'sdpa':
        q = q.permute(0, 2, 1, 3); k = k.permute(0, 2, 1, 3); v = v.permute(0, 2, 1, 3)
        out = _sdpa(q, k, v)
        return out.permute(0, 2, 1, 3)

    return _naive_sdpa(q, k, v)
