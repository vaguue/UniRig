"""Drop-in replacement for flash_attn.modules.mha.MHA.

Used when flash_attn cannot be compiled from source (e.g., cross-compilation
on ARM Mac → amd64 Docker image where NVCC can't run under QEMU).

Key design constraints:
1. Parameter names MUST match flash_attn's MHA exactly (Wq, Wkv, out_proj)
   so pre-trained UniRig weights load without state_dict key mismatches.
2. Constructor API: MHA(embed_dim, num_heads, cross_attn=True)
3. Forward API: forward(x, x_kv=None) → tensor (not tuple like nn.MHA)
4. Uses F.scaled_dot_product_attention which automatically uses FlashAttention
   on supported CUDA GPUs (A100, RTX 4090, etc.) via PyTorch 2.0+.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    """Multi-Head Attention compatible with flash_attn's MHA API."""

    def __init__(self, embed_dim, num_heads, cross_attn=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cross_attn = cross_attn

        if cross_attn:
            self.Wq = nn.Linear(embed_dim, embed_dim)
            self.Wkv = nn.Linear(embed_dim, 2 * embed_dim)
        else:
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_kv=None, **kwargs):
        batch_size = x.shape[0]

        if self.cross_attn and x_kv is not None:
            q = self.Wq(x)
            kv = self.Wkv(x_kv)
            k, v = kv.chunk(2, dim=-1)
        else:
            if hasattr(self, "Wqkv"):
                qkv = self.Wqkv(x)
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                q = self.Wq(x)
                kv = self.Wkv(x)
                k, v = kv.chunk(2, dim=-1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        return self.out_proj(attn_output)
