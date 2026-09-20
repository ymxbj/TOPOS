"""Multi-head self/cross attention with selectable backend.

Backend is controlled by ``ATTN_BACKEND``: ``flash_attn`` (default), ``sdpa``,
or ``naive``. ``flash_attn`` requires the ``flash-attn`` package and a GPU with
compute capability >= 8.0 (Ampere or newer); otherwise it auto-falls back to ``sdpa``.
"""
from .modules import MultiHeadAttention
from .full_attn import scaled_dot_product_attention
