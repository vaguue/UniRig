"""Patch UniRig's sampler.py to use blended area+uniform sampling.

PROBLEM:
  sample_surface() uses pure area-weighted face sampling. On chibi characters
  with disproportionately large heads/bodies, ~90% of the 65,536 sample points
  concentrate on large surfaces, leaving small limbs (arms/legs) with ~5-10%
  of points. The PointTransformerV3 encoder can't detect features from such
  sparse limb sampling, causing skeleton prediction to miss arm/leg bones entirely.

FIX:
  Replace pure area weighting with blended area+uniform (alpha=0.5):
    face_weight = 0.5 * normalized_area + 0.5 * uniform

  Effect on chibi characters:
    - Small limb faces (0.1x avg area): 5.5x boost (0.1 → 0.55)
    - Large body faces (100x avg area): 2x reduction (100 → 50.5)
    - Limbs go from ~5% to ~20% of the point cloud
"""

import os
import sys

SAMPLER_PATH = "/opt/unirig/src/data/sampler.py"
PATCH_MARKER = "PATCH(sampler-adaptive-v1)"

if not os.path.exists(SAMPLER_PATH):
    print(f"ERROR: {SAMPLER_PATH} not found")
    sys.exit(1)

with open(SAMPLER_PATH) as f:
    code = f.read()

if PATCH_MARKER in code:
    print("Sampler adaptive patch already applied, skipping")
    sys.exit(0)

# Target: the area weight computation in sample_surface()
OLD_CODE = """\
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    face_weight = (face_weight * face_weight).sum(axis=1)

    weight_cum = np.cumsum(face_weight, axis=0)"""

NEW_CODE = """\
    # PATCH(sampler-adaptive-v1): Blend area-weighted + uniform sampling
    # for better limb coverage on chibi characters.
    _area_raw = np.cross(offset_0, offset_1, axis=-1)
    _area_raw = (_area_raw * _area_raw).sum(axis=1)
    _area_sum = _area_raw.sum()
    if _area_sum > 0:
        _n = len(_area_raw)
        _alpha = 0.5
        face_weight = _alpha * (_area_raw / _area_sum) + (1.0 - _alpha) * (1.0 / _n)
        face_weight = face_weight * _area_sum  # rescale for cumsum compatibility
    else:
        face_weight = _area_raw

    weight_cum = np.cumsum(face_weight, axis=0)"""

if OLD_CODE in code:
    code = code.replace(OLD_CODE, NEW_CODE)
    print("Applied sampler adaptive patch (exact match)")
else:
    # Fallback: try matching without exact whitespace
    # Look for the two key lines
    key1 = "face_weight = np.cross(offset_0, offset_1, axis=-1)"
    key2 = "face_weight = (face_weight * face_weight).sum(axis=1)"

    if key1 in code and key2 in code:
        # Replace the two area computation lines
        code = code.replace(
            key2,
            f"""# {PATCH_MARKER}: Blend area-weighted + uniform sampling
    _area_raw = face_weight  # from cross product above
    _area_raw = (_area_raw * _area_raw).sum(axis=1)
    _area_sum = _area_raw.sum()
    if _area_sum > 0:
        _n = len(_area_raw)
        _alpha = 0.5
        face_weight = _alpha * (_area_raw / _area_sum) + (1.0 - _alpha) * (1.0 / _n)
        face_weight = face_weight * _area_sum
    else:
        face_weight = _area_raw""",
        )
        print("Applied sampler adaptive patch (fallback match)")
    else:
        print(f"ERROR: Could not find target code in {SAMPLER_PATH}")
        print(f"  key1 found: {key1 in code}")
        print(f"  key2 found: {key2 in code}")
        sys.exit(1)

with open(SAMPLER_PATH, "w") as f:
    f.write(code)

# Verify
with open(SAMPLER_PATH) as f:
    content = f.read()

if PATCH_MARKER in content:
    print(f"VERIFIED: Sampler adaptive patch applied to {SAMPLER_PATH}")
else:
    print("ERROR: Patch verification failed")
    sys.exit(1)
