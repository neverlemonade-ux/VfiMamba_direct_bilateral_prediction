"""
cropping.py -- scene cropping + motion filtering + progressive
sequence-length variant generation for the VFIMamba fine-tune pipeline.

This is STAGE 1 of a two-stage preprocessing pipeline (this script, then
augmentation.py). The jobs are now split like this:

    cropping.py       : WHICH pixels + WHICH frames go into a training
                        example. Dynamic multi-crop placement, a motion
                        gate (reject crops that are just static sky/wall/
                        background), and generation of a family of
                        progressively shorter clips from each crop.
    augmentation.py    : HOW those pixels look. Photometric/blur/noise/
                        compression augmentation, run as a SEPARATE pass
                        over whatever this script writes.

Point augmentation.py's data.data_root at THIS script's data.output_root
(or train on this script's output directly if you don't need pixel
augmentation).

    python cropping.py --config cropping_config.yaml

=== INPUT LAYOUT ===
data.input_root is a folder containing one subfolder per scene, each
subfolder containing that scene's frames, numbered contiguously with a
fixed zero-padded width (e.g. 000000.png, 000001.png, ... -- the
default: data.frame_index_start=0, data.frame_index_digits=6). Frames
are discovered by probing that index sequence and stopping at the first
missing index -- i.e. numbering is assumed contiguous with no gaps.

Multiple file types are supported: for each index, every extension in
data.frame_extensions (default png, jpg, jpeg, bmp, tif, tiff, tried in
that order) is checked. Whichever extension matches at a scene's first
frame is "locked in" and tried first for the rest of that scene (for
speed); if a later index is missing under the locked extension, the
other configured extensions are tried before giving up on that index --
so one scene can be all-png, another all-jpg, without any config change,
and an odd mixed-extension scene still won't wrongly stop early. Scenes
with fewer than data.min_scene_frames discovered frames are skipped.

=== ONE CROP SIZE PER RUN ===
crop.crop_size is a single size (an int for a square crop, or a
[width, height] pair). Every output folder name still encodes the crop
size (see OUTPUT LAYOUT / NAMING) even though only one size is active
per run -- if you want a size ladder (e.g. 128, 256, 320x240, 512,
640x480, 720x480), run this script once per size against the same
input_root with a different crop_size each time; since output names are
disambiguated by size, all runs can safely share the same output_root.

=== DYNAMIC CROP COUNT / PLACEMENT ===
Unchanged from the original combined script: for a scene of size (w, h)
and the configured crop_size (crop_w, crop_h),

    num_crops = min(w // crop_w, h // crop_h)

optionally capped by crop.max_crops_per_scene. The valid placement
region is partitioned into a grid with >= num_crops cells, shuffled
(seeded), and one distinct cell is assigned per crop so crop boxes can
never coincide. See compute_num_crops / compute_crop_boxes.

=== MOTION GATE -- WHY ===
A crop box chosen purely by grid position can easily land on a static
sky, a locked-off background wall, or any other patch with no real
inter-frame motion -- exactly the kind of example that teaches an
interpolation/flow model nothing (or teaches it that "predict the input
frame unchanged" is a fine strategy). Before a crop is ever written to
disk, compute_motion_score samples a handful of frames spanning the
scene, downsamples them, and measures mean grayscale frame-to-frame
difference *within that crop's own box*. If the score is below
motion_check.min_motion_score, the box is redrawn (uniformly at random,
up to motion_check.max_reroll_attempts times) and rechecked. If no
attempt clears the bar, the crop slot is either skipped entirely
(motion_check.skip_on_failure: true, the default) or accepted anyway
with a logged warning (false) -- never silently written without a
review trail either way.

=== SEQUENCE-LENGTH VARIANTS -- WHAT AND WHY ===
Once a crop box passes the motion gate, ALL of the scene's frames are
cropped through it exactly once (cheap: one crop pass, not one per
variant). That single cropped frame list is then written out as several
training examples of DECREASING length, all anchored at frame 1:

    variant 0: frames 1..N            (the full-length crop)
    variant 1: frames 1..floor(0.9*N) (drop the last ~10%)
    variant 2: frames 1..floor(0.9*that)
    ...

i.e. each variant keeps the FRONT of the sequence and drops frames off
the END, and each new length is ~90% of the previous variant's length
(not 90% of the original N) -- so a 100-frame scene yields lengths
100, 90, 81, 72, 64, ... The reduction keeps going only as long as the
last variant actually written still has at least
sequence.min_frames_to_continue frames (default 5); once continuing
would produce something shorter than sequence.min_output_frames
(default 3), generation for that crop stops. See
compute_sequence_lengths for the exact loop -- the effective shortest
variant you end up with therefore depends on N and isn't a fixed number,
it just won't go below min_output_frames or continue past a variant
shorter than min_frames_to_continue.

This gives the model training examples across a whole range of
effective "playback speeds"/durations from the same underlying motion,
without re-deciding the crop position or re-running the motion gate for
each one.

=== OUTPUT LAYOUT / NAMING ===
Output scenes are written under data.output_root (default
"<input_root>_cropped_<size>", a sibling of input_root -- never inside
it -- where <size> is the single active crop.crop_size for this run,
e.g. "train_10k_cropped_128" or "train_10k_cropped_320x240"; this keeps
size-ladder runs against the same input_root from colliding by default
even before a per-scene crop index is appended). Crop-position naming
matches the original combined script: a scene yielding exactly one crop
is named "<scene>_<size>"; one yielding N>1 crops is named
"<scene>_<size>_00", "..._01", .... Every scene, per this task's
requirement, ALSO carries its frame count in the folder name: an
"_f<NNN>" suffix is appended (zero-padded to the digit-width of the
scene's own full frame count, so e.g. a 100-frame scene's variants sort
as ..._f100, ..._f090, ..._f081, ...) -- a separate size string (e.g.
"128", "320x240") is already part of the base name before that suffix,
so different crop_sizes never collide. Each output folder keeps the
same frame filenames (and extensions) as the source (000000.png,
000001.png, ...) -- just fewer of them for the shorter variants; nothing
is renumbered.

A manifest.jsonl is written to output_root recording, per output folder:
source scene, crop box, source resolution, full-scene frame count, this
variant's frame count, motion score, and reroll attempts used -- the
same "durable, browsable record" convention as the rest of this
pipeline.

=== ASSUMPTIONS WORTH KNOWING ABOUT ===
- Frame numbering is contiguous from data.frame_index_start (see INPUT
  LAYOUT above). A scene with gaps (000000, 000001, 000003, ...) will be
  treated as only having 2 usable frames.
- All frames within one scene are assumed to share the same zero-padded
  digit width (data.frame_index_digits) -- only the extension is
  auto-detected per scene, not the width.
- Only one crop size is active per run (see ONE CROP SIZE PER RUN
  above); run the script again with a different crop.crop_size (and the
  same output_root) to build out a size ladder.
- The "~10%" reduction is `floor(current * (1 - sequence.reduction_pct))`
  applied to the PREVIOUS variant's length, not the original count --
  this matches "100, 90, 81, ..." rather than "100, 90, 80, ...".
- The motion gate looks at the WHOLE scene (sampled), not just the
  variant being written -- a crop is a good/bad motion candidate at the
  crop-box level, independent of which length variant you're looking at.
"""
import argparse
import math
import json
import os
import random
import sys

import cv2
import numpy as np
import yaml

from dataset import seed_everything  # reuse -- one seeding implementation, not two


# ============================================================
# 1. PER-SCENE RNG DERIVATION (same rationale as augmentation.py:
#    deriving a private RNG per scene from (seed, scene index) keeps
#    each scene's crop layout reproducible independent of processing
#    order / how many scenes were processed before it)
# ============================================================

def _scene_rng(seed, scene_index, crop_w, crop_h):
    # Keying on crop size too means switching crop.crop_size between
    # runs (e.g. building a size ladder one run at a time) never
    # perturbs a size's own reproducible crop layout.
    return random.Random(f'{seed}:{scene_index}:{crop_w}x{crop_h}')


# ============================================================
# 2. DYNAMIC CROP-BOX COMPUTATION (unchanged logic from the original
#    combined script -- kept here since cropping is now this script's
#    sole responsibility)
# ============================================================

def compute_num_crops(w, h, crop_w, crop_h, max_crops=None):
    if w < crop_w or h < crop_h:
        return 0
    n = min(w // crop_w, h // crop_h)
    if max_crops is not None:
        n = min(n, max_crops)
    return max(0, n)


def compute_crop_boxes(w, h, crop_w, crop_h, num_crops, jitter, rng):
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
        x1, y1 = max(x1, x0), max(y1, y0)

        if jitter:
            x = rng.randint(x0, x1)
            y = rng.randint(y0, y1)
        else:
            x, y = x0, y0
        boxes.append((x, y))

    return boxes


def _redraw_random_box(w, h, crop_w, crop_h, rng):
    """Uniform random re-roll used ONLY by the motion-gate retry loop
    below -- unlike compute_crop_boxes, this doesn't need to guarantee
    distinctness from other crops' boxes (a reroll replaces this one
    crop's own failed attempt, it isn't competing with siblings)."""
    max_x = w - crop_w
    max_y = h - crop_h
    return (rng.randint(0, max_x), rng.randint(0, max_y))


# ============================================================
# 3. FRAME DISCOVERY
# ============================================================

def discover_frames(seq_dir, index_start, digits, extensions, max_probe):
    """
    Returns a sorted list of filenames present in seq_dir, for indices
    index_start, index_start+1, ... zero-padded to `digits` (e.g.
    000000.png, 000001.png, ...), stopping at the first missing index
    (contiguous numbering assumed -- see module docstring "ASSUMPTIONS
    WORTH KNOWING ABOUT").

    Handles multiple file types: `extensions` is tried in order at the
    first index, and whichever one matches is reused first for every
    later index in this scene (fast path, since one scene is normally
    one export batch in one format); if a later index isn't found under
    that locked-in extension, every other configured extension is tried
    before concluding the sequence has ended.
    """
    frames = []
    locked_ext = None
    i = index_start
    probed = 0
    while probed < max_probe:
        base = f'{i:0{digits}d}'
        ext_order = extensions if locked_ext is None else (
            [locked_ext] + [e for e in extensions if e != locked_ext])
        found = None
        for ext in ext_order:
            candidate = f'{base}.{ext}'
            if os.path.exists(os.path.join(seq_dir, candidate)):
                found = candidate
                locked_ext = ext
                break
        if found is None:
            break
        frames.append(found)
        i += 1
        probed += 1
    return frames


def _load_frame(seq_dir, filename):
    path = os.path.join(seq_dir, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f'could not read {path} -- check data.frame_index_digits / '
            f'data.frame_extensions match your actual on-disk naming')
    return img


# ============================================================
# 4. MOTION GATE
#    Rejects/rerolls crop boxes that land on effectively static content
#    (sky, locked-off background, ...) -- see module docstring
#    "MOTION GATE -- WHY".
# ============================================================

def _crop_gray_downsampled(img, box, factor):
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if factor > 1:
        new_w = max(1, gray.shape[1] // factor)
        new_h = max(1, gray.shape[0] // factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray


def compute_motion_score(frames, box, max_frames_to_sample, downsample_factor):
    """
    Samples up to `max_frames_to_sample` frames evenly spaced across the
    whole scene, restricted to `box`, downsamples them for speed, and
    returns the mean absolute grayscale difference between consecutive
    SAMPLED frames -- a cheap, orientation/color-agnostic proxy for "is
    anything actually moving in this window." Higher = more motion.
    """
    n = len(frames)
    n_samples = min(max_frames_to_sample, n)
    if n_samples < 2:
        return 0.0
    idxs = sorted(set(int(round(t)) for t in np.linspace(0, n - 1, n_samples)))
    grays = [_crop_gray_downsampled(frames[i], box, downsample_factor) for i in idxs]
    if len(grays) < 2:
        return 0.0
    diffs = [float(np.abs(grays[i + 1] - grays[i]).mean()) for i in range(len(grays) - 1)]
    return float(np.mean(diffs))


def find_motion_passing_box(frames, w, h, crop_w, crop_h, initial_box,
                             motion_cfg, rng, log, scene_name, crop_label):
    """
    Checks `initial_box` against the motion gate, rerolling a fresh
    random box (up to motion_cfg.max_reroll_attempts times) on failure.
    Returns (box, score, attempts_used, passed).
    """
    if not motion_cfg.get('enabled', True):
        return initial_box, None, 0, True

    min_score = motion_cfg.get('min_motion_score', 4.0)
    max_samples = motion_cfg.get('max_frames_to_sample', 10)
    downsample = motion_cfg.get('downsample_factor', 4)
    max_attempts = motion_cfg.get('max_reroll_attempts', 8)

    box = initial_box
    attempts = 0
    while True:
        x, y = box
        full_box = (x, y, crop_w, crop_h)
        score = compute_motion_score(frames, full_box, max_samples, downsample)
        if score >= min_score:
            return box, score, attempts, True
        if attempts >= max_attempts:
            log(f'  [motion] {scene_name} {crop_label}: no box cleared '
                f'min_motion_score={min_score} after {attempts} reroll(s) '
                f'(best score seen this attempt: {score:.2f})')
            return box, score, attempts, False
        box = _redraw_random_box(w, h, crop_w, crop_h, rng)
        attempts += 1


# ============================================================
# 5. SEQUENCE-LENGTH VARIANTS
#    See module docstring "SEQUENCE-LENGTH VARIANTS -- WHAT AND WHY".
# ============================================================

def compute_sequence_lengths(total_frames, reduction_pct=0.10,
                              min_frames_to_continue=5, min_output_frames=3):
    """
    Returns a strictly-decreasing list of frame counts starting at
    total_frames: [N, floor(0.9*N), floor(0.9*that), ...]. Keeps
    reducing only while the LAST length actually produced is still
    >= min_frames_to_continue; stops before appending anything shorter
    than min_output_frames. Guarantees strict decrease even in
    edge-case rounding (e.g. very small `current`) by falling back to
    `current - 1`.
    """
    lengths = [total_frames]
    current = total_frames
    while current >= min_frames_to_continue:
        nxt = math.floor(current * (1.0 - reduction_pct))
        if nxt >= current:
            nxt = current - 1
        if nxt < min_output_frames:
            break
        lengths.append(nxt)
        current = nxt
    return lengths


# ============================================================
# 6. NAMING
# ============================================================

def _size_str(crop_w, crop_h):
    return str(crop_w) if crop_w == crop_h else f'{crop_w}x{crop_h}'


def crop_base_name(scene_name, crop_w, crop_h, crop_idx, num_crops):
    size = _size_str(crop_w, crop_h)
    if num_crops <= 1:
        return f'{scene_name}_{size}'
    return f'{scene_name}_{size}_{crop_idx:02d}'


def variant_output_name(base_name, length, total_frames):
    """Appends the frame-count suffix required for every output scene
    folder -- zero-padded to the digit-width of the scene's FULL frame
    count so all of one scene's variants line up/sort together
    (f100, f090, f081, ... rather than f100, f90, f81, ...)."""
    width = len(str(total_frames))
    return f'{base_name}_f{length:0{width}d}'


# ============================================================
# 7. PER-SCENE PROCESSING
# ============================================================

def process_scene(scene_name, seq_dir, data_cfg, crop_w, crop_h,
                   max_crops_per_scene, jitter, motion_cfg, seq_cfg,
                   output_root, seed, scene_index, overwrite, log):
    """
    Crops one source scene (0..num_crops crop boxes at the single
    configured crop size, each motion-gated) and, for every box that
    passes, writes the full-length crop plus every shorter
    sequence-length variant computed by compute_sequence_lengths.
    Returns a list of manifest entries.
    """
    index_start = data_cfg.get('frame_index_start', 0)
    digits = data_cfg.get('frame_index_digits', 6)
    extensions = data_cfg.get('frame_extensions',
                               ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'])
    max_frames_to_probe = data_cfg.get('max_frames_to_probe', 2000)
    min_scene_frames = data_cfg.get('min_scene_frames', 3)

    filenames = discover_frames(seq_dir, index_start, digits, extensions,
                                 max_frames_to_probe)
    total_frames = len(filenames)
    if total_frames < min_scene_frames:
        log(f'  [skip] {scene_name}: only {total_frames}/{min_scene_frames} '
            f'required frames present')
        return []

    frames = [_load_frame(seq_dir, f) for f in filenames]
    h, w = frames[0].shape[:2]
    for fname, arr in zip(filenames, frames):
        if arr.shape[:2] != (h, w):
            log(f'  [skip] {scene_name}: frame {fname} is {arr.shape[:2]}, '
                f'expected {(h, w)} to match {filenames[0]} -- frames in one '
                f'scene must share a resolution')
            return []

    num_crops = compute_num_crops(w, h, crop_w, crop_h, max_crops_per_scene)
    if num_crops == 0:
        log(f'  [skip] {scene_name}: {w}x{h} is smaller than crop size '
            f'{crop_w}x{crop_h} in at least one dimension')
        return []

    lengths = compute_sequence_lengths(
        total_frames,
        reduction_pct=seq_cfg.get('reduction_pct', 0.10),
        min_frames_to_continue=seq_cfg.get('min_frames_to_continue', 5),
        min_output_frames=seq_cfg.get('min_output_frames', 3))

    rng = _scene_rng(seed, scene_index, crop_w, crop_h)
    boxes = compute_crop_boxes(w, h, crop_w, crop_h, num_crops, jitter, rng)

    manifest_entries = []
    for idx, initial_box in enumerate(boxes):
        crop_label = f'crop{idx:02d}'
        box, motion_score, reroll_attempts, passed = find_motion_passing_box(
            frames, w, h, crop_w, crop_h, initial_box, motion_cfg, rng, log,
            scene_name, crop_label)

        if not passed and motion_cfg.get('skip_on_failure', True):
            log(f'    {scene_name} {crop_label}: SKIPPED (no motion-passing '
                f'box found, skip_on_failure=true)')
            continue
        if not passed:
            log(f'    {scene_name} {crop_label}: accepting low-motion box '
                f'(score={motion_score:.2f}) -- skip_on_failure=false')

        x, y = box
        base_name = crop_base_name(scene_name, crop_w, crop_h, idx, num_crops)

        # Crop every frame through this box exactly ONCE; every length
        # variant below is a slice of this single cropped list.
        cropped = [f[y:y + crop_h, x:x + crop_w] for f in frames]

        for length in lengths:
            out_name = variant_output_name(base_name, length, total_frames)
            out_dir = os.path.join(output_root, out_name)

            if os.path.exists(out_dir) and not overwrite:
                log(f'    {out_name}: already exists, skipping (overwrite: false)')
                continue

            os.makedirs(out_dir, exist_ok=True)
            for fname, img in zip(filenames[:length], cropped[:length]):
                cv2.imwrite(os.path.join(out_dir, fname), img)

            motion_note = '' if motion_score is None else f' motion={motion_score:.2f}'
            reroll_note = '' if reroll_attempts == 0 else f' (rerolled x{reroll_attempts})'
            log(f'    {out_name}: crop=({x},{y},{crop_w},{crop_h}) '
                f'frames={length}/{total_frames}{motion_note}{reroll_note}')

            manifest_entries.append({
                'source_scene': scene_name,
                'source_dir': seq_dir,
                'output_scene': out_name,
                'output_dir': out_dir,
                'crop_size': {'w': crop_w, 'h': crop_h},
                'crop_box': {'x': x, 'y': y, 'w': crop_w, 'h': crop_h},
                'source_resolution': {'w': w, 'h': h},
                'source_total_frames': total_frames,
                'variant_frame_count': length,
                'motion_score': motion_score,
                'motion_gate_passed': passed,
                'motion_reroll_attempts': reroll_attempts,
            })

    return manifest_entries


# ============================================================
# 8. ORCHESTRATION
# ============================================================

def crop_dataset(C, log=print):
    d_cfg = C['data']
    c_cfg = C['crop']
    motion_cfg = C.get('motion_check', {})
    seq_cfg = C.get('sequence', {})

    seed = C.get('seed', 42)
    seed_everything(seed)

    input_root = d_cfg['input_root']
    overwrite = d_cfg.get('overwrite', False)

    crop_size = c_cfg['crop_size']
    if isinstance(crop_size, (list, tuple)):
        crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
    else:
        crop_w = crop_h = int(crop_size)
    max_crops_per_scene = c_cfg.get('max_crops_per_scene')
    jitter = c_cfg.get('jitter', True)

    # Default output_root now carries the active crop size (e.g.
    # "<input_root>_cropped_128") -- see module docstring "OUTPUT LAYOUT
    # / NAMING". Needs crop_w/crop_h, so this is resolved after crop
    # size parsing above rather than up front.
    output_root = d_cfg.get('output_root') or (
        f'{input_root.rstrip(os.sep)}_cropped_{_size_str(crop_w, crop_h)}')
    os.makedirs(output_root, exist_ok=True)

    scenes = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )
    log(f'cropping {len(scenes)} scene(s) from {input_root}')
    log(f'  crop_size={_size_str(crop_w, crop_h)}  max_crops_per_scene={max_crops_per_scene}  '
        f'jitter={jitter}  seed={seed}')
    log(f'  frame naming: start={d_cfg.get("frame_index_start", 0)} '
        f'digits={d_cfg.get("frame_index_digits", 6)} '
        f'extensions={d_cfg.get("frame_extensions", ["png","jpg","jpeg","bmp","tif","tiff"])}')
    log(f'  motion_check: enabled={motion_cfg.get("enabled", True)} '
        f'min_motion_score={motion_cfg.get("min_motion_score", 4.0)} '
        f'skip_on_failure={motion_cfg.get("skip_on_failure", True)}')
    log(f'  sequence: reduction_pct={seq_cfg.get("reduction_pct", 0.10)} '
        f'min_frames_to_continue={seq_cfg.get("min_frames_to_continue", 5)} '
        f'min_output_frames={seq_cfg.get("min_output_frames", 3)}')
    log(f'  output_root={output_root}  overwrite={overwrite}')

    manifest_path = os.path.join(output_root, 'manifest.jsonl')
    manifest_mode = 'a' if (overwrite or os.path.exists(manifest_path)) else 'w'
    total_written = 0

    with open(manifest_path, manifest_mode) as mf:
        for scene_index, scene_name in enumerate(scenes):
            seq_dir = os.path.join(input_root, scene_name)
            entries = process_scene(
                scene_name, seq_dir, d_cfg, crop_w, crop_h,
                max_crops_per_scene, jitter, motion_cfg, seq_cfg,
                output_root, seed, scene_index, overwrite, log)
            for entry in entries:
                mf.write(json.dumps(entry) + '\n')
            total_written += len(entries)

    log(f'-> wrote {total_written} scene folder(s) to {output_root}')
    log(f'-> manifest: {manifest_path}')
    return output_root


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scene cropping + motion filtering + progressive '
                    'sequence-length variant generation for the VFIMamba '
                    'fine-tune pipeline')
    parser.add_argument('--config', required=True, type=str,
                         help='path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    crop_dataset(C)


if __name__ == '__main__':
    main()