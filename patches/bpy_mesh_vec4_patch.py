#!/usr/bin/env python3
"""
Patch bpy's glTF importer to handle 4-component data from Draco+Quantization.

When a GLB uses KHR_draco_mesh_compression + KHR_mesh_quantization,
the Draco decoder can produce 4-component data for attributes that should
be VEC3 (positions, normals, morph targets). bpy's gltf2_blender_mesh.py
initializes arrays as (0,3) then tries to concatenate (N,4), crashing with:
"ValueError: along dimension 1, array at index 0 has size 3 and index 1 has size 4"

Fix: After decode_accessor for POSITION, NORMAL, and morph POSITION,
truncate to expected component count if oversized.
"""

import os
import sys

MESH_FILE = "/opt/conda/lib/python3.11/site-packages/bpy/4.2/scripts/addons_core/io_scene_gltf2/blender/imp/gltf2_blender_mesh.py"

if not os.path.exists(MESH_FILE):
    import glob
    candidates = glob.glob("/opt/conda/**/gltf2_blender_mesh.py", recursive=True)
    if candidates:
        MESH_FILE = candidates[0]
    else:
        print("WARNING: gltf2_blender_mesh.py not found, skipping patch")
        sys.exit(0)

with open(MESH_FILE) as f:
    code = f.read()

patches_applied = 0

# === Patch 1: POSITION (line ~187) ===
OLD_POS = "vs = BinaryData.decode_accessor(gltf, prim.attributes['POSITION'], cache=True)\n        vert_locs = np.concatenate((vert_locs, vs[unique_indices]))"
NEW_POS = """vs = BinaryData.decode_accessor(gltf, prim.attributes['POSITION'], cache=True)
        # PATCH(v40): Truncate VEC4 positions from Draco+Quantization to VEC3
        if vs.ndim == 2 and vs.shape[1] > 3:
            vs = vs[:, :3]
        vert_locs = np.concatenate((vert_locs, vs[unique_indices]))"""

if OLD_POS in code:
    code = code.replace(OLD_POS, NEW_POS)
    patches_applied += 1
    print("  [1/3] Patched POSITION truncation")
elif "PATCH(v40): Truncate VEC4 positions" in code:
    print("  [1/3] POSITION already patched")
    patches_applied += 1
elif "PATCH(v39)" in code:
    # v39 patch exists, replace it with v40
    code = code.replace("PATCH(v39)", "PATCH(v40)")
    patches_applied += 1
    print("  [1/3] Updated v39 POSITION patch to v40")
else:
    print("  [1/3] WARNING: Could not find POSITION target")

# === Patch 2: NORMAL (line ~198) ===
OLD_NORM = """if 'NORMAL' in prim.attributes:
                ns = BinaryData.decode_accessor(gltf, prim.attributes['NORMAL'], cache=True)
                ns = ns[unique_indices]"""
NEW_NORM = """if 'NORMAL' in prim.attributes:
                ns = BinaryData.decode_accessor(gltf, prim.attributes['NORMAL'], cache=True)
                # PATCH(v40): Truncate VEC4 normals from Draco+Quantization to VEC3
                if ns.ndim == 2 and ns.shape[1] > 3:
                    ns = ns[:, :3]
                ns = ns[unique_indices]"""

if OLD_NORM in code:
    code = code.replace(OLD_NORM, NEW_NORM)
    patches_applied += 1
    print("  [2/3] Patched NORMAL truncation")
elif "PATCH(v40): Truncate VEC4 normals" in code:
    print("  [2/3] NORMAL already patched")
    patches_applied += 1
else:
    print("  [2/3] WARNING: Could not find NORMAL target")

# === Patch 3: Morph target POSITION ===
OLD_MORPH = """morph_vs = BinaryData.decode_accessor(gltf, prim.targets[sk]['POSITION'], cache=True)
                morph_vs = morph_vs[unique_indices]"""
NEW_MORPH = """morph_vs = BinaryData.decode_accessor(gltf, prim.targets[sk]['POSITION'], cache=True)
                # PATCH(v40): Truncate VEC4 morph positions from Draco+Quantization
                if morph_vs.ndim == 2 and morph_vs.shape[1] > 3:
                    morph_vs = morph_vs[:, :3]
                morph_vs = morph_vs[unique_indices]"""

if OLD_MORPH in code:
    code = code.replace(OLD_MORPH, NEW_MORPH)
    patches_applied += 1
    print("  [3/3] Patched morph target POSITION truncation")
elif "PATCH(v40): Truncate VEC4 morph positions" in code:
    print("  [3/3] Morph POSITION already patched")
    patches_applied += 1
else:
    print("  [3/3] WARNING: Could not find morph target POSITION target (may not exist)")

with open(MESH_FILE, 'w') as f:
    f.write(code)

print(f"\nPatched {MESH_FILE}: {patches_applied} patches applied")

# Verify
with open(MESH_FILE) as f:
    content = f.read()
    has_pos = "PATCH(v40): Truncate VEC4 positions" in content
    has_norm = "PATCH(v40): Truncate VEC4 normals" in content
    if has_pos and has_norm:
        print("VERIFIED: All critical patches applied successfully")
    elif has_pos:
        print("VERIFIED: POSITION patch applied (NORMAL patch may need manual check)")
    else:
        print("ERROR: Critical patches not found")
        sys.exit(1)
