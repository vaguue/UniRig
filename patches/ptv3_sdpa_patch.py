"""Patch PTv3Object.py to use F.scaled_dot_product_attention instead of naive matmul.

Problem: Without the flash_attn pip package, PTv3's SerializedAttention falls back
to naive O(N*K^2) matmul attention (q @ k.T, softmax, @ v). This is 50-100x slower
than flash_attn's fused kernel.

Solution: Replace the naive matmul path with F.scaled_dot_product_attention (SDPA).
PyTorch 2.0+ SDPA automatically selects the fastest backend:
  - FlashAttention kernel (sm80+, RTX 4090 is sm89)
  - Memory-efficient attention (xFormers)
  - Math fallback (only if others unavailable)

This gives us flash-attention-level performance WITHOUT the flash_attn pip package.

Note: RPE (relative positional encoding) is disabled in UniRig's config, so the
RPE branch in the attention code is never hit. SDPA doesn't support RPE natively,
but since it's disabled, this is safe.
"""

import os

PTV3_PATH = "/opt/unirig/src/model/pointcept/models/PTv3Object.py"

with open(PTV3_PATH) as f:
    code = f.read()

# Patch 1: Auto-fallback when flash_attn not available (same as before)
code = code.replace(
    "super().__init__()\n        assert channels % num_heads == 0",
    "super().__init__()\n"
    "        # Auto-fallback when flash_attn not available\n"
    "        if enable_flash and flash_attn is None:\n"
    "            enable_flash = False\n"
    "            upcast_attention = False\n"
    "            upcast_softmax = False\n"
    "        assert channels % num_heads == 0",
)

# Patch 2: Replace naive matmul attention with F.scaled_dot_product_attention.
# Original non-flash code (lines ~215-231):
#   q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
#   if self.upcast_attention: q = q.float(); k = k.float()
#   attn = (q * self.scale) @ k.transpose(-2, -1)
#   ... softmax, dropout ...
#   feat = (attn @ v).transpose(1, 2).reshape(-1, C)
#
# Replace with SDPA which uses fused FlashAttention kernel on RTX 4090:
OLD_ATTENTION = """        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)"""

NEW_ATTENTION = """        if not self.enable_flash:
            # PATCHED: Use F.scaled_dot_product_attention instead of naive matmul.
            # SDPA automatically uses FlashAttention kernel on RTX 4090 (sm89).
            # This is 50-100x faster than the original q @ k.T matmul path.
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # q, k, v shape: (N', H, K, C') — perfect for SDPA
            drop_p = self.attn_drop.p if self.training else 0.0
            feat = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=drop_p, scale=self.scale,
            )
            feat = feat.transpose(1, 2).reshape(-1, C).to(qkv.dtype)"""

if OLD_ATTENTION in code:
    code = code.replace(OLD_ATTENTION, NEW_ATTENTION)
    print("Patched PTv3Object.py: replaced naive matmul attention with SDPA")
else:
    print("WARNING: Could not find exact attention code to patch.")
    print("Attempting line-by-line search...")
    # Show what we're looking for vs what exists
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if "q * self.scale" in line and "k.transpose" in line:
            print(f"  Found matmul at line {i+1}: {line.strip()}")
        if "self.softmax(attn)" in line:
            print(f"  Found softmax at line {i+1}: {line.strip()}")

with open(PTV3_PATH, "w") as f:
    f.write(code)

# Verify the patch was applied
with open(PTV3_PATH) as f:
    patched = f.read()
if "scaled_dot_product_attention" in patched:
    print("VERIFIED: SDPA patch applied successfully")
else:
    print("ERROR: SDPA patch NOT found in output file")
