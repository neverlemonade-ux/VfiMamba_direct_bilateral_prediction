"""
augmentation.py -- pixel-level photometric/blur/noise/compression
augmentation for the VFIMamba fine-tune pipeline.

This is STAGE 2 of a two-stage preprocessing pipeline: run cropping.py
FIRST (dynamic crop placement, motion gating, progressive
sequence-length variants), then point this script's data.data_root at
cropping.py's output_root. This script no longer does any cropping or
frame selection itself -- see module docstring "WHAT MOVED TO
cropping.py" below.

    python augmentation.py --config augmentation_config.yaml

=== WHAT MOVED TO cropping.py ===
Everything about WHICH pixels and WHICH frames go into a training
example -- dynamic crop placement, the motion gate, and progressive
sequence-length variants -- is now cropping.py's job, run as a separate,
earlier pass. This script only decides and applies HOW those pixels
look: photometric adjustments, blur, noise, and compression. The
temporal `reverse` augmentation (play a clip backwards) still lives here
since it's a cheap, purely order-based transform with no interaction
with crop placement; `variable_interval` (picking a differently-strided
subset of extra on-disk frames) was removed -- cropping.py's
sequence-length variants now cover that same "vary the effective
frame rate / duration" need directly, and each of its output folders
only contains exactly the frames for that one variant (no extra pool to
stride through).

=== INPUT LAYOUT / THE CROP MANIFEST ===
data.data_root is cropping.py's output_root: a folder of scene
directories (already cropped, already the right length for whichever
sequence-length variant they represent), each containing that variant's
frames under whatever filenames/extensions cropping.py wrote (frame
naming and extension are auto-discovered here too -- see
discover_scene_frames -- so this script doesn't care whether frames are
000000.png or frame1.jpg).

Alongside that output, cropping.py also writes manifest.jsonl recording
each output folder's source scene, crop box, and source (pre-crop)
resolution. This script reads that manifest (data.crop_manifest, default
"<data_root>/manifest.jsonl") to recover exactly where each folder's
crop sits within its ORIGINAL uncropped frame -- information this script
has no other way to know once cropping already happened, but which the
lens/optics augmentations below genuinely need (see "LENS / OPTICS
EFFECTS ARE SCENE-FIXED"). If the manifest is missing, or a given folder
has no entry (e.g. you point data_root at some other cropped dataset
that didn't come through cropping.py), that folder is treated as if it
IS the full original frame (crop box = (0, 0, w, h) at its own size) --
lens/optics effects still apply, just centered on this folder's own
frame rather than a true off-center position within a larger scene.

=== SCENE-FIXED LENS/OPTICS, ACROSS FOLDERS NOW ===
lens_distortion, chromatic_aberration, and vignetting are still drawn
ONCE per ORIGINAL scene (the manifest's `source_scene` field) and reused
unchanged across every crop index AND every sequence-length variant of
that same scene -- even though those are now separate folders on disk
rather than being generated together in one process_scene call. This
script groups folders by `source_scene` from the manifest (falling back
to the folder's own name if no manifest entry exists) and caches the
scene-level draw the first time it sees each key, so folder processing
order doesn't matter and the draw is still made exactly once per real
camera/scene -- see decide_augmentations calls in augment_dataset.

=== EVERYTHING ELSE IS UNCHANGED ===
The three-tier augmentation model (scene-fixed lens/optics; the
mutually-exclusive tone/exposure group; everything else independently
coin-flipped per folder), crop duplication with uniqueness-checked
combos, and the fixed application order are all identical in spirit to
the original combined script -- see decide_crop_local_augmentations,
decide_mutually_exclusive_group, decide_unique_augmentations,
apply_augmentations, and process_cropped_folder below.

=== OUTPUT LAYOUT / NAMING ===
Output folders are written under data.output_root (default
"<data_root>_augmented", a sibling of data_root). A folder's name is
simply cropping.py's own output folder name, passed through unchanged --
crop position, size, and frame count are already encoded there. Only
duplicates (see CROP DUPLICATION in the original docstring, unchanged
here) add a "_dupNN" suffix on top of that name.

A manifest.jsonl is written to output_root recording, per output
folder: source (cropped) folder, source_scene (for the lens/optics
grouping), which augmentations (with drawn parameters) were applied, and
duplicate bookkeeping.
"""
import argparse
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
# 1. RNG DERIVATION
#    Two independent RNG streams: one keyed by a folder's ORIGINAL
#    source scene (shared by every crop/variant of that scene -- used
#    only for the once-per-scene lens/optics draw), one keyed by the
#    folder's own output name (used for everything else: crop-local
#    augmentation choices, temporal reverse, duplication). Keying by
#    string rather than a processing-order index means results don't
#    depend on what order folders happen to be iterated in.
# ============================================================

def _rng_for_key(seed, key):
    return random.Random(f'{seed}:{key}')


# ============================================================
# 2. CROP MANIFEST (written by cropping.py)
# ============================================================

def load_crop_manifest(manifest_path, log):
    """
    Returns {output_scene_name: entry_dict} from cropping.py's
    manifest.jsonl, or {} (with a logged warning) if the file doesn't
    exist -- this script still runs without it, just with degraded
    lens/optics positioning (see module docstring).
    """
    if not manifest_path or not os.path.exists(manifest_path):
        log(f'  [warn] crop manifest not found at {manifest_path!r} -- '
            f'lens/optics effects (if enabled) will treat every folder as '
            f'an uncropped full frame instead of using its true position '
            f'within the original scene')
        return {}
    by_scene = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            by_scene[entry['output_scene']] = entry
    return by_scene


def _crop_ctx_for_folder(folder_name, frame_w, frame_h, manifest_entry):
    """
    Returns (crop_w, crop_h, crop_x, crop_y, full_w, full_h) for the
    lens/optics pixel math. Uses the crop manifest's recorded box +
    source resolution when available; otherwise falls back to treating
    this folder's own frames as the full, uncropped frame (crop at the
    origin) -- see module docstring "INPUT LAYOUT / THE CROP MANIFEST".
    """
    if manifest_entry is not None:
        box = manifest_entry['crop_box']
        res = manifest_entry['source_resolution']
        return (box['w'], box['h'], box['x'], box['y'], res['w'], res['h'])
    return (frame_w, frame_h, 0, 0, frame_w, frame_h)


# ============================================================
# 3. FRAME DISCOVERY (dynamic -- cropping.py's output folders can use
#    any naming/extension; this script doesn't assume a fixed template)
# ============================================================

_DEFAULT_IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff')


def discover_scene_frames(seq_dir, extensions=_DEFAULT_IMAGE_EXTENSIONS):
    """
    Returns every file in seq_dir whose extension is in `extensions`,
    sorted lexicographically -- correct for cropping.py's zero-padded
    numeric filenames (000000.png, 000001.png, ...) without needing to
    know the padding width or original naming scheme up front.
    """
    exts = tuple(f'.{e.lower().lstrip(".")}' for e in extensions)
    names = [f for f in os.listdir(seq_dir)
             if os.path.splitext(f)[1].lower() in exts]
    return sorted(names)


def _load_frame(seq_dir, filename, log):
    path = os.path.join(seq_dir, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f'could not read {path} -- check data.image_extensions '
            f'matches your actual on-disk naming')
    return img


# ============================================================
# 4. TEMPORAL -- reverse only (see module docstring "WHAT MOVED TO
#    cropping.py" for why variable_interval was removed)
# ============================================================

def _decide_temporal_reverse(temporal_cfg, rng):
    spec = (temporal_cfg or {}).get('reverse') or {}
    if not spec.get('enabled', False):
        return False
    return rng.random() < spec.get('prob', 0.5)


# ============================================================
# 5. PER-FRAME AUGMENTATION PRIMITIVES
#    Unchanged from the original combined script -- pixel-level
#    photometric/blur/noise/compression logic doesn't care whether
#    cropping happened in this process or an earlier one. Each takes a
#    LIST of frames (one folder's frames) and returns a NEW list,
#    applied identically to every frame, with two exceptions noted
#    inline (gaussian_noise, jpeg_compression) that use an independent
#    realization per frame while sharing one drawn severity parameter.
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
    # frame to frame) but the same std/severity for the whole folder --
    # see module docstring "EVERYTHING ELSE IS UNCHANGED".
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
    # aperture actually produces.
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
    # exposure_adjustment and color_jitter belong to the same
    # mutually-exclusive tone/exposure group (see decide_mutually_exclusive_group),
    # so they can never stack on the same folder.
    factor = 2.0 ** params['stops']
    return [np.clip(f.astype(np.float32) * factor, 0, 255).astype(np.uint8) for f in frames]


def _full_frame_radial_grid(crop_w, crop_h, crop_x, crop_y, full_w, full_h):
    """
    Per-pixel (nx, ny) normalized position of THIS folder's pixels
    relative to the true optical center of the ORIGINAL, pre-crop frame
    -- see module docstring "SCENE-FIXED LENS/OPTICS, ACROSS FOLDERS
    NOW" for where crop_x/crop_y/full_w/full_h now come from (the crop
    manifest, or a same-as-crop fallback).
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
    # Only this folder's own pixels are in memory, so a sample that
    # would physically fall outside the crop's box falls back to
    # edge-reflection (BORDER_REFLECT101) instead of true out-of-crop
    # content. Keep k1/k2 modest (defaults stay under ~0.15/0.05).
    map_x, map_y = _lens_distortion_maps(params, crop_ctx)
    return [cv2.remap(f, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT101) for f in frames]


def _draw_chromatic_aberration_params(spec, rng):
    lo, hi = spec.get('shift_range', [0.0, 0.006])
    # R and B are pushed in opposite radial directions -- the classic
    # red/cyan or blue/yellow fringing signature.
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
    r = np.sqrt(nx ** 2 + ny ** 2) / np.sqrt(2.0)  # full-frame corner -> r==1
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
        ph = min(max(int(round((area / aspect) ** 0.5)), 1), crop_h)
        pw = min(max(int(round((area * aspect) ** 0.5)), 1), crop_w)
        x0 = rng.randint(0, crop_w - pw)
        y0 = rng.randint(0, crop_h - ph)
        patches.append({'x': x0, 'y': y0, 'w': pw, 'h': ph})

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
    Simulates real inter-frame H.264 compression by actually encoding
    the folder's frames as a short clip via ffmpeg and decoding it back.
    Requires ffmpeg on PATH -- if missing, or the round-trip doesn't
    cleanly preserve frame count, silently returns the frames unmodified.
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
# 6. PER-FOLDER AUGMENTATION SELECTION
#    Unchanged in spirit from the original combined script -- three
#    tiers (scene-fixed lens/optics; the mutually-exclusive
#    tone/exposure group; everything else independent per folder).
# ============================================================

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

_LENS_OPTICS_TYPES = ('lens_distortion', 'chromatic_aberration', 'vignetting')

_TONE_EXPOSURE_GROUP = ('exposure_adjustment', 'white_balance_shift',
                         'gamma_correction', 'color_jitter')


def _draw_params_for_type(name, spec, rng, crop_w, crop_h):
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
    """Independently coin-flips each ENABLED type in `types_to_consider`.
    Returns an ordered dict {name: resolved_params}, in _AUG_TYPES order."""
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
    """Picks AT MOST ONE member of `member_names` to fire -- a group-level
    coin flip at `group_prob` decides if any fire at all, then one member
    is picked weighted by its own configured `prob`."""
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
    """Merges several {name: params} dicts (pairwise disjoint keys) into
    one, re-ordered to match _AUG_TYPES -- this is what lets the
    scene-fixed / tone-exposure-group / independent tiers be decided by
    three separate mechanisms and still apply in one fixed order."""
    merged = {}
    for name in _AUG_TYPES:
        for d in dicts:
            if name in d:
                merged[name] = d[name]
                break
    return merged


def apply_augmentations(frames, chosen, crop_ctx=None):
    """Applies `chosen` to `frames`, in fixed _AUG_TYPES order. crop_ctx
    is (crop_w, crop_h, crop_x, crop_y, full_w, full_h) -- required only
    if lens_distortion / chromatic_aberration / vignetting are enabled."""
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


_INDEPENDENT_PER_CROP_TYPES = tuple(
    t for t in _AUG_TYPES
    if t not in _LENS_OPTICS_TYPES and t not in _TONE_EXPOSURE_GROUP
)


def _augmentation_signature(chosen):
    """Which augmentation TYPES fired, order-independent -- used for the
    duplicate-uniqueness check. Lens/optics are excluded by callers
    since they're scene-fixed and would pad every signature identically."""
    return tuple(sorted(chosen.keys()))


def decide_crop_local_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h):
    """Draws the per-folder-varying portion of one folder's augmentation
    combo: independent types + at most one tone/exposure group member.
    Does NOT include lens/optics -- those are decided once per SOURCE
    SCENE by the caller (see augment_dataset)."""
    tone_group_spec = (groups_cfg or {}).get('tone_exposure') or {}
    tone_group_prob = tone_group_spec.get('prob', 0.5)

    chosen = decide_augmentations(aug_types_cfg, rng, crop_w, crop_h,
                                   types_to_consider=_INDEPENDENT_PER_CROP_TYPES)
    chosen.update(decide_mutually_exclusive_group(
        aug_types_cfg, _TONE_EXPOSURE_GROUP, tone_group_prob, rng, crop_w, crop_h))
    return chosen


def decide_unique_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h,
                                 existing_signatures, max_attempts):
    """Draws a crop-local combo, redrawing (up to max_attempts times)
    whenever its signature matches one already used by a sibling
    duplicate. Returns (chosen, signature, redraw_attempts_used)."""
    chosen = decide_crop_local_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h)
    sig = _augmentation_signature(chosen)
    attempts = 0
    while sig in existing_signatures and attempts < max_attempts:
        chosen = decide_crop_local_augmentations(aug_types_cfg, groups_cfg, rng, crop_w, crop_h)
        sig = _augmentation_signature(chosen)
        attempts += 1
    return chosen, sig, attempts


# ============================================================
# 7. NAMING -- much simpler now: cropping.py already produced a unique,
#    descriptive folder name (crop position/size/frame count all
#    encoded). This script only appends a duplicate suffix on top.
# ============================================================

def _duplicate_name(base_name, dup_index):
    return base_name if dup_index == 0 else f'{base_name}_dup{dup_index:02d}'


# ============================================================
# 8. PER-FOLDER PROCESSING
# ============================================================

def process_cropped_folder(folder_name, seq_dir, manifest_entry,
                            scene_lens_optics, image_extensions,
                            aug_types_cfg, groups_cfg, temporal_cfg,
                            dup_cfg, output_root, seed, overwrite, log):
    """
    Augments one already-cropped folder (as written by cropping.py):
    decides temporal reverse + a crop-local augmentation combo (redrawn
    per duplicate for uniqueness), merges in the scene-fixed lens/optics
    draw passed in from the caller, and writes 1..N output copies (N>1
    only if duplication triggers). Returns a list of manifest entries.
    """
    filenames = discover_scene_frames(seq_dir, image_extensions)
    if not filenames:
        log(f'  [skip] {folder_name}: no frames found (extensions={image_extensions})')
        return []

    frames = [_load_frame(seq_dir, f, log) for f in filenames]
    h, w = frames[0].shape[:2]
    for fname, arr in zip(filenames, frames):
        if arr.shape[:2] != (h, w):
            log(f'  [skip] {folder_name}: frame {fname} is {arr.shape[:2]}, '
                f'expected {(h, w)}')
            return []

    crop_ctx = _crop_ctx_for_folder(folder_name, w, h, manifest_entry)
    crop_w, crop_h = crop_ctx[0], crop_ctx[1]

    rng = _rng_for_key(seed, folder_name)

    reverse_flag = _decide_temporal_reverse(temporal_cfg, rng)
    ordered = list(reversed(frames)) if reverse_flag else frames

    dup_enabled = dup_cfg.get('enabled', False)
    dup_prob = dup_cfg.get('prob', 0.0)
    max_total_instances = max(1, dup_cfg.get('max_total_instances', 1))
    max_redraw_attempts = dup_cfg.get('max_redraw_attempts', 20)

    num_instances = 1
    if dup_enabled and max_total_instances > 1 and rng.random() < dup_prob:
        num_instances = rng.randint(2, max_total_instances)

    manifest_entries = []
    signatures_used = set()
    for inst in range(num_instances):
        out_name = _duplicate_name(folder_name, inst)
        out_dir = os.path.join(output_root, out_name)

        if os.path.exists(out_dir) and not overwrite:
            log(f'    {out_name}: already exists, skipping (overwrite: false)')
            continue

        crop_local_augs, sig, redraw_attempts = decide_unique_augmentations(
            aug_types_cfg, groups_cfg, rng, crop_w, crop_h,
            signatures_used, max_redraw_attempts)
        signatures_used.add(sig)

        chosen_augs = _merge_in_type_order(scene_lens_optics, crop_local_augs)
        augmented = apply_augmentations(ordered, chosen_augs, crop_ctx)

        os.makedirs(out_dir, exist_ok=True)
        for filename, img in zip(filenames, augmented):
            cv2.imwrite(os.path.join(out_dir, filename), img)

        applied_summary = ', '.join(chosen_augs.keys()) or 'none'
        dup_note = '' if inst == 0 else f' [duplicate {inst}/{num_instances - 1}]'
        collision_note = '' if redraw_attempts == 0 else f' (redrawn x{redraw_attempts} for uniqueness)'
        log(f'    {out_name}: reverse={reverse_flag} aug=[{applied_summary}]'
            f'{dup_note}{collision_note}')

        manifest_entries.append({
            'source_folder': seq_dir,
            'source_scene': (manifest_entry or {}).get('source_scene', folder_name),
            'output_scene': out_name,
            'output_dir': out_dir,
            'reverse': reverse_flag,
            'augmentations': chosen_augs,
            'is_duplicate': inst > 0,
            'duplicate_index': inst,
            'duplicate_group_size': num_instances,
        })

    return manifest_entries


# ============================================================
# 9. ORCHESTRATION
# ============================================================

def augment_dataset(C, log=print):
    d_cfg = C['data']
    a_cfg = C.get('augmentation', {})

    seed = a_cfg.get('seed', 42)
    seed_everything(seed)

    data_root = d_cfg['data_root']
    manifest_path = d_cfg.get('crop_manifest') or os.path.join(data_root, 'manifest.jsonl')
    output_root = d_cfg.get('output_root') or f'{data_root.rstrip(os.sep)}_augmented'
    overwrite = d_cfg.get('overwrite', False)
    image_extensions = d_cfg.get('image_extensions', list(_DEFAULT_IMAGE_EXTENSIONS))
    os.makedirs(output_root, exist_ok=True)

    aug_types_cfg = a_cfg.get('types', {})
    groups_cfg = a_cfg.get('groups', {})
    temporal_cfg = a_cfg.get('temporal', {})
    dup_cfg = a_cfg.get('duplication', {})

    h264_spec = aug_types_cfg.get('h264_compression') or {}
    if h264_spec.get('enabled', False):
        ffmpeg_path = h264_spec.get('ffmpeg_path', 'ffmpeg')
        if shutil.which(ffmpeg_path) is None:
            log(f'  [warn] h264_compression is enabled but "{ffmpeg_path}" was not '
                f'found on PATH -- affected frames will be written uncompressed')

    crop_manifest = load_crop_manifest(manifest_path, log)

    scenes = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    log(f'augmenting {len(scenes)} folder(s) from {data_root}')
    log(f'  crop_manifest={manifest_path}  ({len(crop_manifest)} entries loaded)')
    if dup_cfg.get('enabled', False):
        log(f'  duplication: enabled prob={dup_cfg.get("prob", 0.0)} '
            f'max_total_instances={dup_cfg.get("max_total_instances", 1)}')
    else:
        log('  duplication: disabled')
    log(f'  output_root={output_root}  overwrite={overwrite}')

    manifest_out_path = os.path.join(output_root, 'manifest.jsonl')
    manifest_mode = 'a' if (overwrite or os.path.exists(manifest_out_path)) else 'w'
    total_written = 0

    # Cache of scene-fixed lens/optics draws, keyed by the folder's
    # ORIGINAL source scene (from the crop manifest) -- see module
    # docstring "SCENE-FIXED LENS/OPTICS, ACROSS FOLDERS NOW". Falls
    # back to the folder's own name as the key if no manifest entry
    # exists, so each such folder just gets its own independent draw.
    lens_optics_cache = {}

    with open(manifest_out_path, manifest_mode) as mf:
        for folder_name in scenes:
            seq_dir = os.path.join(data_root, folder_name)
            entry = crop_manifest.get(folder_name)
            scene_key = entry['source_scene'] if entry else folder_name

            if scene_key not in lens_optics_cache:
                lens_rng = _rng_for_key(seed, f'lens:{scene_key}')
                # crop_w/crop_h only matter here for random_erasing's
                # dispatch signature, which lens/optics types never use.
                lens_optics_cache[scene_key] = decide_augmentations(
                    aug_types_cfg, lens_rng, 0, 0,
                    types_to_consider=_LENS_OPTICS_TYPES)
            scene_lens_optics = lens_optics_cache[scene_key]

            entries = process_cropped_folder(
                folder_name, seq_dir, entry, scene_lens_optics,
                image_extensions, aug_types_cfg, groups_cfg, temporal_cfg,
                dup_cfg, output_root, seed, overwrite, log)
            for e in entries:
                mf.write(json.dumps(e) + '\n')
            total_written += len(entries)

    log(f'-> wrote {total_written} augmented folder(s) to {output_root}')
    log(f'-> manifest: {manifest_out_path}')
    return output_root


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pixel-level augmentation stage for the VFIMamba '
                    'fine-tune pipeline (run after cropping.py)')
    parser.add_argument('--config', required=True, type=str,
                         help='path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    augment_dataset(C)


if __name__ == '__main__':
    main()