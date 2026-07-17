"""
augmentation.py -- dynamic multi-crop + per-crop augmentation preprocessing
for the VFIMamba full-resolution fine-tune pipeline.

This is a PRE-processing stage that runs BEFORE dataset.prepare_datasets():
it reads raw sequence folders out of a source data_root (same layout
dataset.py expects: seq_xxxx/<frame_filenames>) and writes new, expanded
sequence folders -- cropped and augmented -- into a sibling output folder.
Point train_config.yaml's data.data_root at the output of this script (or
merge it alongside your original data) to train on the expanded set.

    python augmentation.py --config augmentation_config.yaml

=== DYNAMIC CROP COUNT ===
For a scene of size (w, h) and a configured crop_size (crop_w, crop_h):

    num_crops = min(w // crop_w, h // crop_h)

i.e. "how many times does the crop fit into this frame" -- a 4096x2048
scene with crop_size=512 gets min(8, 4) = 4 crops; a 600x600 scene with
crop_size=512 gets 1; a scene smaller than crop_size in either dimension
gets 0 (skipped entirely, logged). Optionally capped by
crop.max_crops_per_scene.

=== DIFFERENT AREAS, GUARANTEED ===
The valid crop region [0, w-crop_w] x [0, h-crop_h] is partitioned into a
grid with at least num_crops cells; cells are shuffled (seeded) and one
distinct cell is assigned per crop, so crops can never coincide. Within
its cell, a crop is either randomly jittered (crop.jitter: true, default)
or pinned to the cell's top-left corner (crop.jitter: false, fully
deterministic placement).

=== TEMPORAL CONSISTENCY -- WHY THIS MATTERS ===
Every frame in a scene depicts the same underlying motion. A crop box and
a set of per-frame augmentation parameters (flip flags, color-jitter
factors, blur kernels, ...) are therefore each drawn ONCE per crop
instance and applied IDENTICALLY to every frame of that crop -- never
re-randomized per frame. Doing otherwise would decorrelate img0/gt/img1's
geometry/color from each other and train the model on physically
nonsensical "motion." The two exceptions are Gaussian noise and JPEG
compression, which use an independent random realization per frame (real
sensor noise/compression artifacts differ frame to frame) while still
sharing one drawn "severity" parameter (std / quality) across the whole
crop instance -- see section 4 below.

=== DIFFERING AUGMENTATION PER CROP -- NOW THREE TIERS ===
Each augmentation type in augmentation.types has its own `enabled` +
`prob`, but they are no longer all decided the same way. Every
augmentation type now falls into exactly one of three tiers:

  1. SCENE-LEVEL LENS/OPTICS (_LENS_OPTICS_TYPES: lens_distortion,
     chromatic_aberration, vignetting) -- drawn ONCE per scene and
     reused, unchanged, by every crop and every duplicate of that scene.
     See "LENS / OPTICS EFFECTS ARE SCENE-FIXED" below.
  2. THE MUTUALLY-EXCLUSIVE TONE/EXPOSURE GROUP (_TONE_EXPOSURE_GROUP:
     exposure_adjustment, white_balance_shift, gamma_correction,
     color_jitter) -- at most one of these ever fires on a given crop
     instance. See "TONE / EXPOSURE MUTUAL EXCLUSION" below.
  3. EVERYTHING ELSE (_INDEPENDENT_PER_CROP_TYPES) -- independently
     coin-flipped per crop instance exactly as before: of, say, 4 crops
     taken from one scene, crop 0 might get flip_h + random_erasing,
     crop 1 only random_erasing, crop 2 only gaussian_blur, crop 3
     nothing at all.

This happens automatically via decide_crop_local_augmentations (tiers 2
and 3) plus a separate once-per-scene draw for tier 1 -- there's no need
to hand-author per-crop combos.

=== LENS / OPTICS EFFECTS ARE SCENE-FIXED (BUT CROP-POSITION AWARE) ===
lens_distortion, chromatic_aberration, and vignetting are all physically
a function of distance from the LENS's optical center in the original,
uncropped frame -- not the crop's own center -- AND they are physical
properties of the CAMERA that shot a scene (its distortion coefficients,
its CA fringing amount, its vignetting falloff). Every crop of one scene
came through the same lens, so unlike every other augmentation type,
these three are drawn ONCE per scene (both whether each one fires AND
its continuous parameters) rather than once per crop instance -- that
single draw is then reused for every crop and every duplicate taken from
that scene. The crop-position-aware pixel math in
_apply_lens_distortion/_apply_chromatic_aberration/_apply_vignetting
still runs per crop (each crop's absolute position in the source frame is
different), so two crops of the same scene still look like two different
windows onto the SAME camera, rather than two different cameras. For the
same reason they run BEFORE flip_h/flip_v/rotate90 in the fixed
application order (see section 5) -- those are synthetic dataset-level
transforms with no real optical meaning, and running them first would
leave this crop's pixels in an orientation the lens math doesn't expect.
Because only the already-cropped pixels are in memory, any sample that
would fall outside the crop's own box (possible with strong
distortion/aberration near crop edges) falls back to edge-reflection
rather than true neighboring content -- keep default-scale k1/k2/shift
values, or pre-apply these on full frames before cropping if you need
stronger effects.

=== TONE / EXPOSURE MUTUAL EXCLUSION -- NEW ===
exposure_adjustment, white_balance_shift, gamma_correction, and
color_jitter are four different mechanisms for doing roughly the same
photometric job: shifting a frame's brightness and/or color. Coin-
flipping each one independently (as every other type does) would let all
four stack on the same crop, compounding into an exposure/color
combination that no single real camera + lighting condition would
actually produce. These four are therefore grouped and made mutually
exclusive -- see decide_mutually_exclusive_group: a single group-level
coin flip decides whether ANY of them fires at all, and if so exactly
one member is picked (weighted by that member's own configured `prob`,
which becomes a relative weight inside the group rather than an
independent per-crop firing probability once the group fires).

=== TEMPORAL AUGMENTATIONS (reverse / variable interval) ===
Two more augmentations operate on WHICH raw frames get used and in what
order, rather than on pixels, and are decided under augmentation.temporal:

  - reverse: with configured probability, the crop's frames are used in
    reverse order. The OUTPUT is still written under the original
    frame_filenames names (frame1.jpg, frame2.jpg, ...) -- what changes
    is which content lands in which file, e.g. frame1.jpg ends up
    holding what was originally the last frame. This is the standard
    "time reversal" trick for interpolation/flow training: the result is
    still physically plausible motion, just backwards.
  - variable_interval: with configured probability, instead of the
    scene's default consecutive frame_filenames, a differently-strided
    subset of a larger on-disk frame pool is used (e.g. skipping every
    other frame to simulate a lower effective frame rate). This requires
    the scene folder to actually contain extra frames beyond
    frame_filenames, discovered via temporal.variable_interval.
    frame_pattern (a "{idx}"-templated filename pattern) -- if no valid
    strided subset exists on disk, this augmentation silently falls back
    to the default frames for that crop.

Both are decided ONCE per crop instance (not per output frame) and are
shared by every duplicate of that crop (see below) -- they describe what
content the crop IS, whereas the per-frame augmentations in section 4
describe how it's rendered.

=== CROP DUPLICATION ===
Under augmentation.duplication, a crop instance can, with configured
probability, be additionally duplicated into extra output copies (up to
duplication.max_total_instances copies total, including the original --
the number actually used per triggered crop is itself randomized between
2 and max_total_instances). Every duplicate shares the same crop box, the
same temporal (reverse / variable_interval) content, AND the same
scene-fixed lens/optics draw as its siblings, but draws its OWN
independent crop-local combo (tiers 2 and 3 above, via
decide_crop_local_augmentations). To guarantee duplicates are actually
different from each other, each combo is checked against every combo
already used by a sibling of the same crop (compared by *which
crop-local augmentation types fired*, not their exact continuous
parameters, and deliberately excluding the scene-fixed lens/optics types
-- see decide_unique_augmentations) and redrawn (up to
duplication.max_redraw_attempts times) on a collision. Duplicates are
named "<base_name>_dup01", "_dup02", etc.

=== OUTPUT LAYOUT / NAMING ===
Output scenes are written under data.output_root (default:
"<data_root>_augmented", a sibling of data_root -- never inside it).
A scene that yields exactly one crop is named "<scene>_<size>"; a scene
that yields N>1 crops is named "<scene>_<size>_00", "..._01", ...
("<size>" is "512" for a square crop, "512x384" otherwise) -- a single
"<scene>_<size>" name can't disambiguate multiple crops of the same
scene, so the index suffix is only added when there's more than one.
Duplicate copies of a crop append "_dupNN" to that crop's base name (see
above). Each output scene folder keeps the same frame_filenames as the
source.

A manifest.jsonl is written to output_root recording, per output scene:
source scene, crop box, temporal choices, exactly which augmentations
(with their drawn parameters) were applied, and duplicate bookkeeping --
a durable, browsable record of what was actually generated, in the same
spirit as dataset.py's on-disk train/val split.
"""
import argparse
import math
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import yaml

from dataset import seed_everything  # reuse -- one seeding implementation, not two


# ============================================================
# 1. PER-SCENE RNG DERIVATION
#    A single global seed would make every scene's crop layout /
#    augmentation choices depend on the order scenes happen to be
#    processed in (whatever call sequence exhausts the global RNG up to
#    that point). Deriving a private RNG per scene from (seed, scene
#    index) instead makes each scene's output reproducible on its own,
#    independent of how many scenes were processed before it -- e.g.
#    re-running on a data_root with one new scene added doesn't change
#    the crops generated for every existing scene.
# ============================================================

def _scene_rng(seed, scene_index):
    # NOTE: random.Random() does not accept a raw tuple as a seed (only
    # None/int/float/str/bytes/bytearray) -- a string seed gets the same
    # "derive independently per (seed, scene_index)" property without
    # tripping that restriction, and is deterministic across interpreters
    # (str seeds are hashed internally via hashlib, not via hash()).
    return random.Random(f'{seed}:{scene_index}')


# ============================================================
# 2. DYNAMIC CROP-BOX COMPUTATION
# ============================================================

def compute_num_crops(w, h, crop_w, crop_h, max_crops=None):
    """
    How many times crop_size fits into (w, h), taking the min across
    both axes so every crop is guaranteed to fully fit regardless of
    which axis is the tight one. 0 means the scene is smaller than the
    crop in at least one dimension and should be skipped.
    """
    if w < crop_w or h < crop_h:
        return 0
    n = min(w // crop_w, h // crop_h)
    if max_crops is not None:
        n = min(n, max_crops)
    return max(0, n)


def compute_crop_boxes(w, h, crop_w, crop_h, num_crops, jitter, rng):
    """
    Returns a list of (x, y) top-left corners, length == num_crops, each
    from a distinct grid cell of the valid placement region
    [0, w-crop_w] x [0, h-crop_h] -- see module docstring "DIFFERENT
    AREAS, GUARANTEED" for why this can't just be num_crops independent
    random draws.
    """
    max_x = w - crop_w
    max_y = h - crop_h

    if num_crops <= 1:
        x = rng.randint(0, max_x) if jitter else 0
        y = rng.randint(0, max_y) if jitter else 0
        return [(x, y)]

    cols = math.ceil(math.sqrt(num_crops))
    rows = math.ceil(num_crops / cols)
    cell_w = max_x / cols
    cell_h = max_y / rows

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    chosen_cells = cells[:num_crops]

    boxes = []
    for (r, c) in chosen_cells:
        x0 = int(round(c * cell_w))
        y0 = int(round(r * cell_h))
        x1 = int(round((c + 1) * cell_w)) if c + 1 < cols else max_x
        y1 = int(round((r + 1) * cell_h)) if r + 1 < rows else max_y
        x1, y1 = max(x1, x0), max(y1, y0)  # last row/col can round below x0/y0 on tiny regions

        if jitter:
            x = rng.randint(x0, x1)
            y = rng.randint(y0, y1)
        else:
            x, y = x0, y0
        boxes.append((x, y))

    return boxes


# ============================================================
# 3. TEMPORAL CONTENT SELECTION -- reverse / variable interval
#    These decide WHICH raw frames a crop instance is built from and in
#    what order, before any cropping or pixel augmentation happens (see
#    module docstring "TEMPORAL AUGMENTATIONS").
# ============================================================

def _decide_temporal_reverse(temporal_cfg, rng):
    spec = (temporal_cfg or {}).get('reverse') or {}
    if not spec.get('enabled', False):
        return False
    return rng.random() < spec.get('prob', 0.5)


def _scan_frame_pool(seq_dir, pattern, max_n):
    """
    Returns a sorted list of (idx, filename) for every idx in
    [1, max_n] where pattern.format(idx=idx) exists on disk in seq_dir.
    pattern is a template like "frame{idx}.jpg".
    """
    pool = []
    for i in range(1, max_n + 1):
        fname = pattern.format(idx=i)
        if os.path.exists(os.path.join(seq_dir, fname)):
            pool.append((i, fname))
    return pool


def _pick_variable_interval_frames(pool, num_needed, stride_choices, rng):
    """
    Tries each stride in a shuffled copy of stride_choices and looks for
    ANY run of num_needed frames, evenly spaced by that stride, that are
    ALL present in the discovered pool. Returns the filenames for one
    such run (chosen at random among valid runs for the first stride
    that has any), or None if no configured stride has a valid run --
    callers should fall back to the scene's default consecutive frames.
    """
    if len(pool) < num_needed:
        return None
    filenames_by_idx = {idx: fn for idx, fn in pool}
    indices = sorted(filenames_by_idx.keys())

    strides = list(stride_choices)
    rng.shuffle(strides)
    for stride in strides:
        candidates = []
        for start in indices:
            wanted = [start + i * stride for i in range(num_needed)]
            if all(w in filenames_by_idx for w in wanted):
                candidates.append(wanted)
        if candidates:
            chosen = rng.choice(candidates)
            return [filenames_by_idx[i] for i in chosen]
    return None


def _decide_variable_interval(temporal_cfg, seq_dir, present, rng):
    """
    Returns a list of filenames (same length as `present`) to use as this
    crop's frame content, or None to fall back to the scene's default
    consecutive frames.
    """
    spec = (temporal_cfg or {}).get('variable_interval') or {}
    if not spec.get('enabled', False):
        return None
    if rng.random() >= spec.get('prob', 0.2):
        return None
    pattern = spec.get('frame_pattern')
    if not pattern:
        return None
    max_n = spec.get('max_available_frames', len(present))
    pool = _scan_frame_pool(seq_dir, pattern, max_n)
    picked = _pick_variable_interval_frames(
        pool, len(present), spec.get('stride_choices', [1]), rng)
    if picked is None or picked == present:
        return None
    return picked


# ============================================================
# 4. PER-FRAME AUGMENTATION PRIMITIVES
#    Each takes a LIST of frames (one crop instance's frames) and
#    returns a new list -- always applied to every frame in the list
#    identically, with two exceptions noted inline (gaussian_noise,
#    jpeg_compression) that use an independent realization per frame
#    while sharing one drawn severity parameter for the whole instance.
#    Every _apply_* function returns NEW arrays and never mutates its
#    input frames in place -- this matters because duplicated crops
#    (section 7) reuse the same base cropped frames across multiple
#    independent augmentation draws.
# ============================================================

def _apply_flip_h(frames):
    return [np.ascontiguousarray(f[:, ::-1]) for f in frames]


def _apply_flip_v(frames):
    return [np.ascontiguousarray(f[::-1]) for f in frames]


def _apply_rotate90(frames):
    return [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]


def _draw_color_jitter_params(spec, rng):
    brightness = spec.get('brightness', 0.0)
    contrast = spec.get('contrast', 0.0)
    saturation = spec.get('saturation', 0.0)
    hue = spec.get('hue', 0.0)
    return {
        'brightness': 1.0 + rng.uniform(-brightness, brightness) if brightness else 1.0,
        'contrast': 1.0 + rng.uniform(-contrast, contrast) if contrast else 1.0,
        'saturation': 1.0 + rng.uniform(-saturation, saturation) if saturation else 1.0,
        'hue_deg': rng.uniform(-hue, hue) * 179 if hue else 0.0,  # cv2 HSV hue channel is 0-179
    }


def _apply_color_jitter(frames, params):
    b, c, s, hue_shift = (params['brightness'], params['contrast'],
                           params['saturation'], params['hue_deg'])
    out = []
    for f in frames:
        img = f.astype(np.float32)
        img = img * b                                    # brightness
        img = (img - img.mean()) * c + img.mean()         # contrast, around this frame's own mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        if s != 1.0 or hue_shift != 0.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
            hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out.append(img)
    return out


def _draw_gamma_params(spec, rng):
    lo, hi = spec.get('gamma_range', [0.7, 1.3])
    return {'gamma': rng.uniform(lo, hi)}


def _apply_gamma(frames, params):
    gamma = max(params['gamma'], 1e-6)
    inv = 1.0 / gamma
    table = np.clip(
        np.array([((i / 255.0) ** inv) * 255 for i in range(256)]), 0, 255
    ).astype(np.uint8)
    return [cv2.LUT(f, table) for f in frames]


def _draw_gaussian_blur_params(spec, rng):
    lo, hi = spec.get('kernel_range', [3, 7])
    choices = [k for k in range(int(lo), int(hi) + 1) if k % 2 == 1] or [3]
    kernel = rng.choice(choices)
    sigma_lo, sigma_hi = spec.get('sigma_range', [0, 0])  # 0,0 -> cv2 auto-derives sigma from kernel
    sigma = rng.uniform(sigma_lo, sigma_hi)
    return {'kernel': kernel, 'sigma': sigma}


def _apply_gaussian_blur(frames, params):
    k, sigma = params['kernel'], params['sigma']
    return [cv2.GaussianBlur(f, (k, k), sigma) for f in frames]


def _draw_motion_blur_params(spec, rng):
    lo, hi = spec.get('kernel_range', [5, 15])
    choices = [k for k in range(int(lo), int(hi) + 1) if k % 2 == 1] or [5]
    kernel = rng.choice(choices)
    angle_lo, angle_hi = spec.get('angle_range', [0, 360])
    angle = rng.uniform(angle_lo, angle_hi)
    return {'kernel': kernel, 'angle': angle}


def _build_motion_blur_kernel(size, angle):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (size, size))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        kernel[size // 2, size // 2] = 1.0  # degenerate rotation -- fall back to identity
    return kernel


def _apply_motion_blur(frames, params):
    kernel = _build_motion_blur_kernel(params['kernel'], params['angle'])
    return [cv2.filter2D(f, -1, kernel) for f in frames]


def _draw_noise_params(spec, rng):
    lo, hi = spec.get('std_range', [3, 15])
    return {'std': rng.uniform(lo, hi), 'seed': rng.randint(0, 2 ** 31 - 1)}


def _apply_gaussian_noise(frames, params):
    # Independent noise REALIZATION per frame (real sensor noise differs
    # frame to frame) but the same std/severity for the whole crop
    # instance -- see module docstring "TEMPORAL CONSISTENCY".
    std = params['std']
    base_seed = params['seed']
    out = []
    for i, f in enumerate(frames):
        nrng = np.random.RandomState((base_seed + i) % (2 ** 32 - 1))
        noise = nrng.normal(0, std, f.shape).astype(np.float32)
        out.append(np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return out


def _draw_defocus_blur_params(spec, rng):
    lo, hi = spec.get('radius_range', [2, 6])
    return {'radius': rng.randint(int(lo), int(hi))}


def _build_defocus_kernel(radius):
    # A disk kernel (not a Gaussian) is what an out-of-focus circular
    # aperture actually produces -- this is what visually distinguishes
    # "defocus blur" from gaussian_blur/motion_blur above.
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        kernel[radius, radius] = 1.0  # radius 0 -- identity
    return kernel


def _apply_defocus_blur(frames, params):
    kernel = _build_defocus_kernel(max(1, int(params['radius'])))
    return [cv2.filter2D(f, -1, kernel) for f in frames]


def _draw_white_balance_params(spec, rng):
    lo, hi = spec.get('gain_range', [0.85, 1.15])
    # R and B gain are drawn independently (not mirrored) -- real AWB
    # error isn't always a clean warm<->cool seesaw, G is left as the
    # reference channel.
    return {'r_gain': rng.uniform(lo, hi), 'b_gain': rng.uniform(lo, hi)}


def _apply_white_balance(frames, params):
    r_gain, b_gain = params['r_gain'], params['b_gain']
    out = []
    for f in frames:
        img = f.astype(np.float32)
        img[..., 2] *= r_gain  # cv2 is BGR -- index 2 is R
        img[..., 0] *= b_gain  # index 0 is B
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def _draw_exposure_params(spec, rng):
    lo, hi = spec.get('stops_range', [-0.5, 0.5])
    return {'stops': rng.uniform(lo, hi)}


def _apply_exposure(frames, params):
    # Stops-based (2**stops) multiplicative gain -- distinct from
    # color_jitter's brightness, which is a linear +/-fraction factor.
    # Since exposure_adjustment and color_jitter now belong to the same
    # mutually-exclusive tone/exposure group (see module docstring
    # "TONE / EXPOSURE MUTUAL EXCLUSION"), they can no longer stack on
    # the same crop instance.
    factor = 2.0 ** params['stops']
    return [np.clip(f.astype(np.float32) * factor, 0, 255).astype(np.uint8) for f in frames]


def _full_frame_radial_grid(crop_w, crop_h, crop_x, crop_y, full_w, full_h):
    """
    Per-pixel (nx, ny) normalized position of THIS crop's pixels relative
    to the true optical center of the ORIGINAL, pre-crop frame.

    lens_distortion / chromatic_aberration / vignetting are physically a
    function of distance from the lens's optical axis in the FULL frame
    -- not from the crop's own center. Computing them post-crop using the
    crop's own center would make every off-center crop look like it was
    shot dead-center (max vignetting in the wrong place, distortion
    curving the wrong way), which is the "is this safe to do to a crop"
    trap for exactly these three augmentations. Passing crop_x/crop_y/
    full_w/full_h through lets each crop compute the SAME physical field
    that its parent full frame would have had, just windowed to its own
    box -- two adjacent crops of the same scene get consistent-looking
    vignetting instead of two independently-centered vignettes. Since
    these three types are now also drawn once per SCENE (see module
    docstring "LENS / OPTICS EFFECTS ARE SCENE-FIXED"), this is what lets
    every crop of a scene look like it came through one shared camera
    rather than each crop getting its own independently-parameterized
    lens.
    """
    ys, xs = np.mgrid[0:crop_h, 0:crop_w]
    abs_x = xs + crop_x
    abs_y = ys + crop_y
    nx = (abs_x - full_w / 2.0) / (full_w / 2.0)
    ny = (abs_y - full_h / 2.0) / (full_h / 2.0)
    return nx.astype(np.float32), ny.astype(np.float32)


def _draw_lens_distortion_params(spec, rng):
    lo1, hi1 = spec.get('k1_range', [-0.15, 0.15])
    lo2, hi2 = spec.get('k2_range', [-0.05, 0.05])
    return {'k1': rng.uniform(lo1, hi1), 'k2': rng.uniform(lo2, hi2)}


def _lens_distortion_maps(params, crop_ctx):
    crop_w, crop_h, crop_x, crop_y, full_w, full_h = crop_ctx
    nx, ny = _full_frame_radial_grid(crop_w, crop_h, crop_x, crop_y, full_w, full_h)
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + params['k1'] * r2 + params['k2'] * (r2 ** 2)
    src_abs_x = (nx * factor) * (full_w / 2.0) + full_w / 2.0
    src_abs_y = (ny * factor) * (full_h / 2.0) + full_h / 2.0
    map_x = (src_abs_x - crop_x).astype(np.float32)
    map_y = (src_abs_y - crop_y).astype(np.float32)
    return map_x, map_y


def _apply_lens_distortion(frames, params, crop_ctx):
    # NOTE: we only have this crop's own pixels in memory (it was already
    # sliced out of the source frame before augmentation runs), so a
    # sample that would physically fall outside the crop's box falls back
    # to edge-reflection (BORDER_REFLECT101) instead of true out-of-crop
    # content. Keep k1/k2 modest (defaults stay under ~0.15/0.05) so this
    # approximation stays imperceptible; for strong distortion, do it as
    # a full-frame pre-pass before cropping instead. `params` here is the
    # single once-per-scene draw (see decide_augmentations /
    # _LENS_OPTICS_TYPES) -- only crop_ctx varies per crop.
    map_x, map_y = _lens_distortion_maps(params, crop_ctx)
    return [cv2.remap(f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101) for f in frames]


def _draw_chromatic_aberration_params(spec, rng):
    lo, hi = spec.get('shift_range', [0.0, 0.006])
    # R and B are pushed in opposite radial directions -- the classic
    # red/cyan or blue/yellow fringing signature -- rather than both
    # scaling the same way (which would look like plain resampling).
    return {'r_shift': rng.uniform(lo, hi), 'b_shift': -rng.uniform(lo, hi)}


def _apply_chromatic_aberration(frames, params, crop_ctx):
    crop_w, crop_h, crop_x, crop_y, full_w, full_h = crop_ctx
    nx, ny = _full_frame_radial_grid(crop_w, crop_h, crop_x, crop_y, full_w, full_h)

    def channel_map(shift):
        factor = 1.0 + shift
        src_abs_x = (nx * factor) * (full_w / 2.0) + full_w / 2.0
        src_abs_y = (ny * factor) * (full_h / 2.0) + full_h / 2.0
        return (src_abs_x - crop_x).astype(np.float32), (src_abs_y - crop_y).astype(np.float32)

    r_map_x, r_map_y = channel_map(params['r_shift'])
    b_map_x, b_map_y = channel_map(params['b_shift'])
    out = []
    for f in frames:
        b, g, r = cv2.split(f)
        # same edge-reflection caveat as lens_distortion above
        r = cv2.remap(r, r_map_x, r_map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)
        b = cv2.remap(b, b_map_x, b_map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101)
        out.append(cv2.merge([b, g, r]))
    return out


def _draw_vignetting_params(spec, rng):
    lo, hi = spec.get('strength_range', [0.15, 0.45])
    return {'strength': rng.uniform(lo, hi), 'falloff': spec.get('falloff', 2.2)}


def _apply_vignetting(frames, params, crop_ctx):
    crop_w, crop_h, crop_x, crop_y, full_w, full_h = crop_ctx
    nx, ny = _full_frame_radial_grid(crop_w, crop_h, crop_x, crop_y, full_w, full_h)
    r = np.sqrt(nx ** 2 + ny ** 2) / math.sqrt(2.0)  # full-frame corner -> r==1
    mask = np.clip(1.0 - params['strength'] * np.clip(r, 0, 1) ** params['falloff'], 0.0, 1.0)
    mask = mask[..., None]
    return [np.clip(f.astype(np.float32) * mask, 0, 255).astype(np.uint8) for f in frames]


def _draw_jpeg_params(spec, rng):
    lo, hi = spec.get('quality_range', [30, 70])
    return {'quality': rng.randint(int(lo), int(hi))}


def _apply_jpeg_compression(frames, params):
    q = int(params['quality'])
    out = []
    for f in frames:
        ok, enc = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, q])
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else f)
    return out


def _draw_random_erasing_params(spec, rng, crop_w, crop_h):
    lo_n, hi_n = spec.get('num_patches_range', [1, 1])
    num_patches = rng.randint(int(lo_n), int(hi_n))
    area_lo, area_hi = spec.get('area_ratio_range', [0.02, 0.08])
    ar_lo, ar_hi = spec.get('aspect_ratio_range', [0.3, 3.3])
    fill_mode = spec.get('fill', 'mean')  # 'mean' (per-frame local mean) or 'random' (fixed noise patch)

    patches = []
    for _ in range(num_patches):
        area = rng.uniform(area_lo, area_hi) * crop_w * crop_h
        aspect = rng.uniform(ar_lo, ar_hi)
        ph = min(max(int(round(math.sqrt(area / aspect))), 1), crop_h)
        pw = min(max(int(round(math.sqrt(area * aspect))), 1), crop_w)
        x0 = rng.randint(0, crop_w - pw)
        y0 = rng.randint(0, crop_h - ph)
        patches.append({'x': x0, 'y': y0, 'w': pw, 'h': ph})

    # Only a seed (a plain int) is stored, not the generated pixels -- this
    # keeps the params dict JSON-serializable for the manifest, and the
    # fill content is regenerated deterministically from it in
    # _apply_random_erasing (once per patch, then reused identically
    # across every frame -- it represents a static occluder, consistent
    # with every other "identical across frames" augmentation here).
    fill_seed = rng.randint(0, 2 ** 31 - 1) if fill_mode == 'random' else None

    return {'patches': patches, 'fill': fill_mode, 'fill_seed': fill_seed}


def _apply_random_erasing(frames, params):
    patches = params['patches']
    fill_mode = params['fill']
    fill_arrays = None
    if fill_mode == 'random':
        nrng = np.random.RandomState(params['fill_seed'])
        fill_arrays = [nrng.randint(0, 256, (p['h'], p['w'], 3), dtype=np.uint8)
                        for p in patches]
    out = []
    for f in frames:
        f = f.copy()
        for i, p in enumerate(patches):
            x, y, pw, ph = p['x'], p['y'], p['w'], p['h']
            if fill_mode == 'random':
                f[y:y + ph, x:x + pw] = fill_arrays[i]
            else:
                f[y:y + ph, x:x + pw] = f[y:y + ph, x:x + pw].mean(axis=(0, 1)).astype(np.uint8)
        out.append(f)
    return out


def _draw_h264_params(spec, rng):
    lo, hi = spec.get('crf_range', [28, 40])
    return {
        'crf': rng.randint(int(lo), int(hi)),
        'fps': spec.get('fps', 24),
        'ffmpeg_path': spec.get('ffmpeg_path', 'ffmpeg'),
    }


def _apply_h264_compression(frames, params):
    """
    Simulates real inter-frame H.264 compression (distinct from the
    single-image JPEG artifact simulation above) by actually encoding
    the crop instance's frames as a short clip via ffmpeg and decoding
    it back. Requires ffmpeg on PATH -- if it's missing, or encoding /
    decoding doesn't cleanly round-trip to the same frame count, this
    silently returns the frames unmodified rather than risk desyncing
    img0/gt/img1.
    """
    ffmpeg_path = params.get('ffmpeg_path', 'ffmpeg')
    if shutil.which(ffmpeg_path) is None:
        return frames

    crf = params['crf']
    fps = params.get('fps', 24)
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(td, f'{i:04d}.png'), f)
        out_path = os.path.join(td, 'out.mp4')
        cmd = [ffmpeg_path, '-y', '-framerate', str(fps), '-i',
               os.path.join(td, '%04d.png'), '-c:v', 'libx264',
               '-crf', str(crf), '-pix_fmt', 'yuv420p', out_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return frames

        cap = cv2.VideoCapture(out_path)
        decoded = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            decoded.append(img)
        cap.release()

    if len(decoded) != len(frames):
        return frames  # round-trip frame count mismatch -- bail out safely
    return decoded


# ============================================================
# 5. PER-CROP AUGMENTATION SELECTION
#    Every augmentation type falls into exactly one of three tiers (see
#    module docstring "DIFFERING AUGMENTATION PER CROP -- NOW THREE
#    TIERS"): scene-fixed lens/optics, the mutually-exclusive
#    tone/exposure group, or ordinary independently-coin-flipped types.
#    decide_augmentations below is the shared low-level primitive (an
#    independent per-type coin flip over whatever `types_to_consider`
#    is given); decide_mutually_exclusive_group and
#    decide_crop_local_augmentations layer the group/tier logic on top
#    of it.
# ============================================================

# Fixed application order, roughly mirroring a real capture + dataset
# pipeline so each stage "sees" the ones before it already baked in:
#
#   1. lens/optics (lens_distortion, chromatic_aberration, vignetting)
#      -- physically happens at the lens, and its math is defined in
#      terms of THIS crop's position within the original full frame
#      (see _full_frame_radial_grid). It must run before flip_h/flip_v/
#      rotate90: those are synthetic dataset-level transforms (a real
#      lens never "sees" a flipped scene), and running them first would
#      leave the crop's pixel positions rotated/mirrored relative to the
#      coordinate system the lens math assumes, corrupting the effect
#      for any crop where both fire together.
#   2. flip_h / flip_v / rotate90 (geometric, synthetic)
#   3. exposure_adjustment / white_balance_shift (sensor/ISP stage)
#   4. gamma_correction / color_jitter (tone/color grading)
#      -- stages 3 and 4 together are the mutually-exclusive
#      tone/exposure group; at most one of the four ever fires.
#   5. random_erasing (synthetic occlusion)
#   6. gaussian_blur / motion_blur / defocus_blur (optical blur)
#   7. gaussian_noise (sensor noise)
#   8. jpeg_compression / h264_compression (codec, last -- sees
#      everything else already baked in)
_AUG_TYPES = (
    'lens_distortion', 'chromatic_aberration', 'vignetting',
    'flip_h', 'flip_v', 'rotate90',
    'exposure_adjustment', 'white_balance_shift',
    'gamma_correction', 'color_jitter',
    'random_erasing',
    'gaussian_blur', 'motion_blur', 'defocus_blur',
    'gaussian_noise',
    'jpeg_compression', 'h264_compression',
)

# lens_distortion / chromatic_aberration / vignetting are physical
# properties of the CAMERA that shot a scene (its distortion
# coefficients, its CA fringing amount, its vignetting falloff) -- they
# don't change from one crop of a scene to the next, because every crop
# of one scene came through the same lens. See "SCENE-LEVEL LENS/OPTICS"
# below: these three are drawn ONCE per scene, not once per crop like
# every other type, and that single draw (both whether each one fires
# AND its continuous parameters) is then reused for every crop and
# every duplicate taken from that scene. The crop-position-aware pixel
# math in _apply_lens_distortion/_apply_chromatic_aberration/
# _apply_vignetting still runs per crop (each crop's absolute position
# in the source frame is different), so two crops of the same scene
# still look like two different windows onto the SAME camera, rather
# than two different cameras.
_LENS_OPTICS_TYPES = ('lens_distortion', 'chromatic_aberration', 'vignetting')

# exposure_adjustment, white_balance_shift, gamma_correction, and
# color_jitter are four different mechanisms for doing roughly the same
# photometric job: shifting a frame's brightness and/or color. Coin-
# flipping each one independently (as every other type does) lets all
# four stack on the same crop, compounding into an exposure/color
# combination that no single real camera + lighting condition would
# actually produce. This group makes them MUTUALLY EXCLUSIVE -- see
# decide_mutually_exclusive_group below: at most one member of this
# group ever fires on a given crop instance.
_TONE_EXPOSURE_GROUP = ('exposure_adjustment', 'white_balance_shift',
                         'gamma_correction', 'color_jitter')


def _draw_params_for_type(name, spec, rng, crop_w, crop_h):
    """Dispatches to the right _draw_*_params function for one
    augmentation type. Factored out of decide_augmentations so the
    mutually-exclusive group logic (decide_mutually_exclusive_group)
    can draw a single member's params without duplicating this
    dispatch table."""
    if name == 'color_jitter':
        return _draw_color_jitter_params(spec, rng)
    elif name == 'gamma_correction':
        return _draw_gamma_params(spec, rng)
    elif name == 'gaussian_blur':
        return _draw_gaussian_blur_params(spec, rng)
    elif name == 'motion_blur':
        return _draw_motion_blur_params(spec, rng)
    elif name == 'gaussian_noise':
        return _draw_noise_params(spec, rng)
    elif name == 'jpeg_compression':
        return _draw_jpeg_params(spec, rng)
    elif name == 'random_erasing':
        return _draw_random_erasing_params(spec, rng, crop_w, crop_h)
    elif name == 'h264_compression':
        return _draw_h264_params(spec, rng)
    elif name == 'defocus_blur':
        return _draw_defocus_blur_params(spec, rng)
    elif name == 'white_balance_shift':
        return _draw_white_balance_params(spec, rng)
    elif name == 'exposure_adjustment':
        return _draw_exposure_params(spec, rng)
    elif name == 'lens_distortion':
        return _draw_lens_distortion_params(spec, rng)
    elif name == 'chromatic_aberration':
        return _draw_chromatic_aberration_params(spec, rng)
    elif name == 'vignetting':
        return _draw_vignetting_params(spec, rng)
    else:
        return {}


def decide_augmentations(aug_types_cfg, rng, crop_w, crop_h, types_to_consider=_AUG_TYPES):
    """
    Independently coin-flips each ENABLED type in `types_to_consider`
    (default: every type in _AUG_TYPES). Returns an ordered dict
    {name: resolved_params}, in _AUG_TYPES order, so that e.g.
    flip_h + random_erasing together compose the same way regardless of
    how the yaml lists them.

    `types_to_consider` lets a caller restrict this to a subset -- used
    for the once-per-scene lens/optics draw (types_to_consider=
    _LENS_OPTICS_TYPES) and for the ordinary per-crop draw of everything
    else (types_to_consider = _AUG_TYPES minus the lens/optics types
    minus the tone/exposure group, which are decided by dedicated logic
    instead -- see process_scene).
    """
    chosen = {}
    for name in types_to_consider:
        spec = aug_types_cfg.get(name) or {}
        if not spec.get('enabled', False):
            continue
        prob = spec.get('prob', 0.5)
        if rng.random() >= prob:
            continue
        chosen[name] = _draw_params_for_type(name, spec, rng, crop_w, crop_h)
    return chosen


def decide_mutually_exclusive_group(aug_types_cfg, member_names, group_prob, rng, crop_w, crop_h):
    """
    Picks AT MOST ONE member of `member_names` to fire, instead of
    coin-flipping each member independently (see _TONE_EXPOSURE_GROUP
    docstring for why exposure/white-balance/gamma/color-jitter are
    grouped this way).

    Two-stage draw:
      1. One group-level coin flip at `group_prob` decides whether ANY
         member fires at all. Below this, the group contributes
         nothing -- same as every disabled/missed type elsewhere.
      2. If the group fires, exactly one member is picked, weighted by
         that member's own configured `prob` in aug_types_cfg (so e.g.
         color_jitter at prob=0.6 is still more likely to be the one
         picked than white_balance_shift at prob=0.3 -- it just can no
         longer stack with its group-mates). Once picked, that member's
         params are drawn normally and it fires with certainty -- its
         own `prob` field is no longer an independent per-crop firing
         probability, it's now only a relative weight inside the group.
         (If you'd rather every member be equally likely when the group
         fires, replace the weights below with a flat [1]*len(members).)

    Returns {name: params} with 0 or 1 entries.
    """
    members = [m for m in member_names
               if (aug_types_cfg.get(m) or {}).get('enabled', False)]
    if not members:
        return {}
    if rng.random() >= group_prob:
        return {}
    weights = [max((aug_types_cfg.get(m) or {}).get('prob', 0.0), 1e-9) for m in members]
    chosen_name = rng.choices(members, weights=weights, k=1)[0]
    spec = aug_types_cfg.get(chosen_name) or {}
    return {chosen_name: _draw_params_for_type(chosen_name, spec, rng, crop_w, crop_h)}


def _merge_in_type_order(*dicts):
    """
    Merges several {name: params} dicts -- assumed to have pairwise
    disjoint keys, e.g. one from the once-per-scene lens/optics draw,
    one from the tone/exposure mutual-exclusion group, one from the
    ordinary independent per-crop draw -- into a single dict re-ordered
    to match _AUG_TYPES. This is what lets those three be decided by
    three separate mechanisms and still always get applied in the one
    fixed order apply_augmentations expects (see module docstring
    section 5): dict insertion order is what apply_augmentations
    actually iterates in, so simply dict-updating them together in
    whatever order they happened to be computed would silently break
    that fixed order (e.g. color_jitter ending up applied after
    random_erasing instead of before it).
    """
    merged = {}
    for name in _AUG_TYPES:
        for d in dicts:
            if name in d:
                merged[name] = d[name]
                break
    return merged


def apply_augmentations(frames, chosen, crop_ctx=None):
    """
    Applies `chosen` (from decide_augmentations / _merge_in_type_order)
    to `frames`, in fixed _AUG_TYPES order regardless of which of the
    three tiers each entry in `chosen` actually came from. crop_ctx is
    (crop_w, crop_h, crop_x, crop_y, full_w, full_h) -- required only if
    lens_distortion / chromatic_aberration / vignetting are enabled (see
    _full_frame_radial_grid for why they need the crop's absolute
    position, not just its size).
    """
    for name, params in chosen.items():
        if name == 'flip_h':
            frames = _apply_flip_h(frames)
        elif name == 'flip_v':
            frames = _apply_flip_v(frames)
        elif name == 'rotate90':
            frames = _apply_rotate90(frames)
        elif name == 'gamma_correction':
            frames = _apply_gamma(frames, params)
        elif name == 'color_jitter':
            frames = _apply_color_jitter(frames, params)
        elif name == 'random_erasing':
            frames = _apply_random_erasing(frames, params)
        elif name == 'gaussian_blur':
            frames = _apply_gaussian_blur(frames, params)
        elif name == 'motion_blur':
            frames = _apply_motion_blur(frames, params)
        elif name == 'defocus_blur':
            frames = _apply_defocus_blur(frames, params)
        elif name == 'gaussian_noise':
            frames = _apply_gaussian_noise(frames, params)
        elif name == 'jpeg_compression':
            frames = _apply_jpeg_compression(frames, params)
        elif name == 'h264_compression':
            frames = _apply_h264_compression(frames, params)
        elif name == 'white_balance_shift':
            frames = _apply_white_balance(frames, params)
        elif name == 'exposure_adjustment':
            frames = _apply_exposure(frames, params)
        elif name == 'lens_distortion':
            frames = _apply_lens_distortion(frames, params, crop_ctx)
        elif name == 'chromatic_aberration':
            frames = _apply_chromatic_aberration(frames, params, crop_ctx)
        elif name == 'vignetting':
            frames = _apply_vignetting(frames, params, crop_ctx)
    return frames


# Every type EXCEPT lens/optics (scene-fixed, see _LENS_OPTICS_TYPES) and
# the tone/exposure group members (drawn together as a single mutually-
# exclusive choice, see _TONE_EXPOSURE_GROUP) -- this is what actually
# varies independently from one crop instance to the next.
_INDEPENDENT_PER_CROP_TYPES = tuple(
    t for t in _AUG_TYPES
    if t not in _LENS_OPTICS_TYPES and t not in _TONE_EXPOSURE_GROUP
)


def _augmentation_signature(chosen):
    """Which augmentation TYPES fired, order-independent -- used for the
    duplicate-uniqueness check in section 7. Exact continuous parameters
    are intentionally ignored: two independent draws landing on the same
    float brightness factor is effectively impossible, so comparing
    params would make the uniqueness check meaningless; comparing the
    set of types is what actually captures "the same augmentation."
    Callers pass only the crop-local portion of `chosen` (see
    decide_unique_augmentations) -- lens/optics are scene-fixed and
    would otherwise pad every signature with the same constant entries
    without helping differentiate duplicates from one another."""
    return tuple(sorted(chosen.keys()))


def decide_crop_local_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h):
    """
    Draws the per-crop-varying portion of one crop instance's
    augmentation combo: the ordinary independently-coin-flipped types
    (_INDEPENDENT_PER_CROP_TYPES) plus at most one member of the
    tone/exposure mutual-exclusion group (_TONE_EXPOSURE_GROUP). Does
    NOT include lens/optics -- those are decided once per scene, see
    process_scene's `scene_lens_optics` and _merge_in_type_order.
    """
    tone_group_spec = (groups_cfg or {}).get('tone_exposure') or {}
    tone_group_prob = tone_group_spec.get('prob', 0.5)

    chosen = decide_augmentations(aug_types_cfg, rng, crop_w, crop_h,
                                   types_to_consider=_INDEPENDENT_PER_CROP_TYPES)
    chosen.update(decide_mutually_exclusive_group(
        aug_types_cfg, _TONE_EXPOSURE_GROUP, tone_group_prob, rng, crop_w, crop_h))
    return chosen


def decide_unique_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h,
                                 existing_signatures, max_attempts):
    """
    Draws a crop-local augmentation combo (decide_crop_local_augmentations),
    redrawing (up to max_attempts times) whenever its signature exactly
    matches one already used by a sibling duplicate of the same crop.
    Lens/optics are deliberately excluded from both the draw and the
    signature here -- they're scene-fixed (identical for every duplicate
    of every crop in this scene), so including them would only pad every
    signature with the same constant entries and never help
    differentiate duplicates from one another. Returns
    (chosen, signature, redraw_attempts_used).
    """
    chosen = decide_crop_local_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h)
    sig = _augmentation_signature(chosen)
    attempts = 0
    while sig in existing_signatures and attempts < max_attempts:
        chosen = decide_crop_local_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h)
        sig = _augmentation_signature(chosen)
        attempts += 1
    return chosen, sig, attempts


# ============================================================
# 6. NAMING
# ============================================================

def _size_str(crop_w, crop_h):
    return str(crop_w) if crop_w == crop_h else f'{crop_w}x{crop_h}'


def output_scene_name(scene_name, crop_w, crop_h, crop_idx, num_crops):
    size = _size_str(crop_w, crop_h)
    if num_crops <= 1:
        return f'{scene_name}_{size}'
    return f'{scene_name}_{size}_{crop_idx:02d}'


def _duplicate_name(base_name, dup_index):
    return base_name if dup_index == 0 else f'{base_name}_dup{dup_index:02d}'


# ============================================================
# 7. PER-SCENE PROCESSING
# ============================================================

def _load_frame(seq_dir, filename, log):
    path = os.path.join(seq_dir, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f'could not read {path} -- check frame_filenames matches your '
            f'actual data layout')
    return img


def process_scene(scene_name, seq_dir, frame_filenames, min_frames_required,
                   crop_w, crop_h, max_crops_per_scene, jitter,
                   aug_types_cfg, groups_cfg, temporal_cfg, dup_cfg,
                   output_root, seed, scene_index,
                   overwrite, log):
    """
    Crops + augments one source scene into 0..num_crops output scene
    folders (each possibly further expanded into duplicates, see module
    docstring "CROP DUPLICATION") under output_root. Returns a list of
    manifest entries (dicts) for whatever it wrote (empty list if the
    scene was skipped).

    Draws the scene-fixed lens/optics combo (`scene_lens_optics`) exactly
    ONCE here and reuses it, unchanged, for every crop and every
    duplicate produced below -- see module docstring "LENS / OPTICS
    EFFECTS ARE SCENE-FIXED". Each duplicate still draws its own
    independent crop-local combo (decide_unique_augmentations); the two
    are merged in fixed _AUG_TYPES order via _merge_in_type_order before
    being applied and recorded.
    """
    present = [f for f in frame_filenames
               if os.path.exists(os.path.join(seq_dir, f))]
    if len(present) < min_frames_required:
        log(f'  [skip] {scene_name}: only {len(present)}/{min_frames_required} '
            f'required frames present')
        return []

    # frame_cache lets variable_interval crops load extra on-disk frames
    # (beyond `present`) on demand without re-reading files already loaded.
    frame_cache = {}
    for f in present:
        frame_cache[f] = _load_frame(seq_dir, f, log)
    frames = [frame_cache[f] for f in present]
    h, w = frames[0].shape[:2]
    for f, arr in zip(present, frames):
        if arr.shape[:2] != (h, w):
            log(f'  [skip] {scene_name}: frame {f} is {arr.shape[:2]}, '
                f'expected {(h, w)} to match {present[0]} -- frames in one '
                f'scene must share a resolution')
            return []

    num_crops = compute_num_crops(w, h, crop_w, crop_h, max_crops_per_scene)
    if num_crops == 0:
        log(f'  [skip] {scene_name}: {w}x{h} is smaller than crop size '
            f'{crop_w}x{crop_h} in at least one dimension')
        return []

    rng = _scene_rng(seed, scene_index)
    boxes = compute_crop_boxes(w, h, crop_w, crop_h, num_crops, jitter, rng)

    # Drawn ONCE for the whole scene -- see module docstring "LENS / OPTICS
    # EFFECTS ARE SCENE-FIXED" and process_scene's own docstring above.
    # Every crop of this scene (and every duplicate of every crop) reuses
    # this exact draw; only the crop-position-aware pixel math in
    # _apply_lens_distortion/_apply_chromatic_aberration/_apply_vignetting
    # varies per crop, via crop_ctx.
    scene_lens_optics = decide_augmentations(
        aug_types_cfg, rng, crop_w, crop_h, types_to_consider=_LENS_OPTICS_TYPES)

    dup_enabled = dup_cfg.get('enabled', False)
    dup_prob = dup_cfg.get('prob', 0.0)
    max_total_instances = max(1, dup_cfg.get('max_total_instances', 1))
    max_redraw_attempts = dup_cfg.get('max_redraw_attempts', 20)

    manifest_entries = []
    for idx, (x, y) in enumerate(boxes):
        base_out_name = output_scene_name(scene_name, crop_w, crop_h, idx, num_crops)

        # --- structural / content choices for this crop, shared by every
        # --- duplicate of it (see module docstring "TEMPORAL AUGMENTATIONS") ---
        reverse_flag = _decide_temporal_reverse(temporal_cfg, rng)
        alt_filenames = _decide_variable_interval(temporal_cfg, seq_dir, present, rng)

        source_filenames = present
        used_alt_interval = False
        if alt_filenames is not None:
            valid = True
            for fn in alt_filenames:
                if fn not in frame_cache:
                    try:
                        frame_cache[fn] = _load_frame(seq_dir, fn, log)
                    except FileNotFoundError:
                        valid = False
                        break
                if frame_cache[fn].shape[:2] != (h, w):
                    valid = False
                    break
            if valid:
                source_filenames = alt_filenames
                used_alt_interval = True
            # else: silently fall back to `present` -- an invalid alt-interval
            # pick shouldn't prevent the crop from being produced at all

        source_frames = [frame_cache[fn] for fn in source_filenames]
        cropped = [f[y:y + crop_h, x:x + crop_w] for f in source_frames]
        if reverse_flag:
            cropped = list(reversed(cropped))

        # --- how many total output copies this crop becomes (1 = no
        # --- duplication) -- see module docstring "CROP DUPLICATION" ---
        num_instances = 1
        if dup_enabled and max_total_instances > 1 and rng.random() < dup_prob:
            num_instances = rng.randint(2, max_total_instances)

        signatures_used = set()
        for inst in range(num_instances):
            out_name = _duplicate_name(base_out_name, inst)
            out_dir = os.path.join(output_root, out_name)

            if os.path.exists(out_dir) and not overwrite:
                log(f'    {out_name}: already exists, skipping (overwrite: false)')
                continue

            crop_local_augs, sig, redraw_attempts = decide_unique_augmentations(
                aug_types_cfg, groups_cfg, rng, crop_w, crop_h,
                signatures_used, max_redraw_attempts)
            signatures_used.add(sig)
            # Scene-fixed lens/optics (same draw for every crop/duplicate of
            # this scene) + this duplicate's own crop-local combo, merged
            # back into the one fixed _AUG_TYPES order apply_augmentations
            # expects -- see _merge_in_type_order.
            chosen_augs = _merge_in_type_order(scene_lens_optics, crop_local_augs)
            crop_ctx = (crop_w, crop_h, x, y, w, h)
            augmented = apply_augmentations(cropped, chosen_augs, crop_ctx)

            os.makedirs(out_dir, exist_ok=True)
            for filename, img in zip(present, augmented):
                cv2.imwrite(os.path.join(out_dir, filename), img)

            applied_summary = ', '.join(chosen_augs.keys()) or 'none'
            dup_note = '' if inst == 0 else f' [duplicate {inst}/{num_instances - 1}]'
            collision_note = '' if redraw_attempts == 0 else f' (redrawn x{redraw_attempts} for uniqueness)'
            log(f'    {out_name}: crop=({x},{y},{crop_w},{crop_h}) '
                f'reverse={reverse_flag} interval={"alt" if used_alt_interval else "default"} '
                f'aug=[{applied_summary}]{dup_note}{collision_note}')

            manifest_entries.append({
                'source_scene': scene_name,
                'source_dir': seq_dir,
                'output_scene': out_name,
                'output_dir': out_dir,
                'crop_box': {'x': x, 'y': y, 'w': crop_w, 'h': crop_h},
                'source_resolution': {'w': w, 'h': h},
                'temporal': {
                    'reverse': reverse_flag,
                    'variable_interval_used': used_alt_interval,
                    'source_frames': source_filenames if used_alt_interval else None,
                },
                'augmentations': chosen_augs,
                'is_duplicate': inst > 0,
                'duplicate_index': inst,
                'duplicate_group_size': num_instances,
            })

    return manifest_entries


# ============================================================
# 8. ORCHESTRATION
# ============================================================

def augment_dataset(C, log=print):
    d_cfg = C['data']
    c_cfg = C['crop']
    a_cfg = C.get('augmentation', {})

    seed = a_cfg.get('seed', 42)
    seed_everything(seed)

    data_root = d_cfg['data_root']
    frame_filenames = list(d_cfg['frame_filenames'])
    min_frames_required = d_cfg.get('min_frames_required', 3)

    output_root = d_cfg.get('output_root') or f'{data_root.rstrip(os.sep)}_augmented'
    overwrite = d_cfg.get('overwrite', False)
    os.makedirs(output_root, exist_ok=True)

    crop_size = c_cfg['crop_size']
    if isinstance(crop_size, (list, tuple)):
        crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
    else:
        crop_w = crop_h = int(crop_size)
    max_crops_per_scene = c_cfg.get('max_crops_per_scene')
    jitter = c_cfg.get('jitter', True)

    aug_types_cfg = a_cfg.get('types', {})
    # groups_cfg carries the mutually-exclusive-group-level settings (e.g.
    # augmentation.groups.tone_exposure.prob) -- see decide_mutually_exclusive_group
    # and module docstring "TONE / EXPOSURE MUTUAL EXCLUSION". This is
    # distinct from each member's own `prob` in aug_types_cfg, which
    # becomes a relative weight *within* the group once it fires.
    groups_cfg = a_cfg.get('groups', {})
    temporal_cfg = a_cfg.get('temporal', {})
    dup_cfg = a_cfg.get('duplication', {})

    h264_spec = aug_types_cfg.get('h264_compression') or {}
    if h264_spec.get('enabled', False):
        ffmpeg_path = h264_spec.get('ffmpeg_path', 'ffmpeg')
        if shutil.which(ffmpeg_path) is None:
            log(f'  [warn] h264_compression is enabled but "{ffmpeg_path}" was not '
                f'found on PATH -- affected frames will be written uncompressed')

    scenes = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    log(f'augmenting {len(scenes)} scene(s) from {data_root}')
    log(f'  crop_size={crop_w}x{crop_h}  max_crops_per_scene={max_crops_per_scene}  '
        f'jitter={jitter}  seed={seed}')
    if dup_cfg.get('enabled', False):
        log(f'  duplication: enabled prob={dup_cfg.get("prob", 0.0)} '
            f'max_total_instances={dup_cfg.get("max_total_instances", 1)}')
    else:
        log('  duplication: disabled')
    log(f'  output_root={output_root}  overwrite={overwrite}')

    manifest_path = os.path.join(output_root, 'manifest.jsonl')
    manifest_mode = 'a' if (overwrite or os.path.exists(manifest_path)) else 'w'
    total_written = 0

    with open(manifest_path, manifest_mode) as mf:
        for scene_index, scene_name in enumerate(scenes):
            seq_dir = os.path.join(data_root, scene_name)
            entries = process_scene(
                scene_name, seq_dir, frame_filenames, min_frames_required,
                crop_w, crop_h, max_crops_per_scene, jitter,
                aug_types_cfg, groups_cfg, temporal_cfg, dup_cfg,
                output_root, seed, scene_index,
                overwrite, log)
            for entry in entries:
                mf.write(json.dumps(entry) + '\n')
            total_written += len(entries)

    log(f'-> wrote {total_written} augmented scene(s) to {output_root}')
    log(f'-> manifest: {manifest_path}')
    return output_root


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dynamic multi-crop + per-crop augmentation preprocessing '
                    'for the VFIMamba fine-tune pipeline')
    parser.add_argument('--config', required=True, type=str,
                         help='path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    augment_dataset(C)


if __name__ == '__main__':
    main()