"""
dataset.py

Loads ONE already-built curriculum phase folder (TrainingData/Phase1..4/,
produced by curriculum_builder.py -- run that first) into a torch Dataset
that yields padded (img0, gt, img1, timestep, orig_h, orig_w) tuples, with
a random interior-frame/timestep pick per training item. Also loads
ValidationData/, unbatched, with every interior frame of every val
sequence enumerated once.

> **Revision note (this revision -- real batch loading via BatchUidGroupedSampler):**
> Previously this file only ever produced single-item Datasets and train.py
> forced DataLoader(batch_size=1), implementing "dynamic batch size" purely
> as gradient accumulation over singles. That left GPU utilization on the
> table at low resolutions (Phase 1/2), where a single native-res forward
> pass often doesn't saturate the GPU and per-call overhead (kernel launch,
> autocast setup, cudnn algorithm selection) becomes a real fraction of
> step time.
>
> This revision adds `BatchUidGroupedSampler` and `ragged_collate`, which
> together let train.py load a REAL minibatch -- one physical
> `BatchNNN_<res>/` folder's worth of scenes at once -- onto the GPU in a
> single forward call, while changing nothing about which batches exist,
> how they're ordered, or how accumulation windows are formed (all of that
> is still curriculum_builder.py's job, untouched by this revision).
>
> Two things make this safe:
>   1. Every scene inside one `BatchNNN_<res>/` folder shares EXACTLY one
>      padded resolution by construction (see curriculum_builder.py's
>      group_into_batches docstring) -- so a batch_uid-grouped minibatch
>      never needs cross-resolution padding/cropping, only stacking.
>   2. Scenes within one batch_uid group can still have DIFFERING interior-
>      frame counts (a scene's frame count depends on max_frame_span
>      sub-clipping, which is per-scene, not per-resolution) -- so
>      `ragged_collate` pads gt/timesteps to this minibatch's own
>      T_max = max(num_interior) and returns a `valid_mask` (B, T_max)
>      bool tensor, which train.py's masked loss uses to make sure a
>      shorter-scene item's per-item loss weight stays independent of how
>      many interior frames it happened to have -- exactly the same
>      invariant the old single-item `frame_losses.mean()` already
>      preserved, just now computed per-item-then-across-batch instead of
>      per-item-then-across-steps.
>
> At high resolutions (Phase 3/4), if config.yaml's dynamic_batch table
> already resolves `batch_size=1` for those thresholds, a batch_uid group
> naturally contains exactly one scene -- this file's loading behavior
> degrades to the old one-item-at-a-time path automatically, with no
> phase-conditional branching needed anywhere. What changes phase to phase
> is purely the config table, not this code.
>
> `Extra/` is still skipped entirely (see _PHASE_SKIP_ENTRIES below,
> unchanged) -- it was never covered by accumulation windows and isn't
> covered by batch-uid grouping either; nothing in this revision touches
> that behavior.

NOTE ON Extra/: curriculum_builder.py writes an Extra/ folder inside each
Phase<N>/ for undersized batches (item count below the configured
batch_size for their resolution). This file's directory walk
(_list_train_seq_dirs) only descends into Phase<N>/BatchNNN_<res>/ --
Extra/ is not itself a "BatchNNN_<res>"-shaped folder inside the phase's
own numbering (it's a container of differently-named batch folders), and
since a plain os.listdir() scan visits every dir entry under phase_dir,
Extra/ would otherwise be walked too. It's intentionally skipped here
(see the _PHASE_SKIP_ENTRIES check) -- undersized batches were only ever
a scaffolding by-product of the curriculum build, not something meant to
be trained on directly. Every batch under Extra/ was also built from its
own phase's data and never arrived there via replay -- curriculum_builder.py's
split_retain_export() excludes undersized (is_extra) batches from export
entirely, so Extra/ never contains cross-phase replay content. is_extra
batches also never go through accumulation-window grouping in
curriculum_builder.py (see that file's group_batches_into_accum_windows),
so skipping Extra/ here means every batch this file ever loads is
guaranteed to carry window_id/window_num_items in its batch_metadata.yaml.

=== ON-DISK LAYOUT THIS FILE EXPECTS (see curriculum_builder.py) ===

  Train:  TrainingData/Phase<N>/BatchNNN_<res>/<idx>_<origin>__<dataset>__<scene>/
  Val:    ValidationData/<dataset>/<scene>/

Both are walked with a plain nested sorted() -- no randomness is
re-derived here. For train, batch folders are already numbered in
curriculum order and each scene folder's zero-padded index prefix is
monotonically increasing across the whole phase, so sorting batch names
then scene names within each reproduces the exact curriculum order
curriculum_builder.py computed. For val, order doesn't carry curriculum
meaning, so alphabetical (dataset, then scene) is used.

=== WHY TRAIN ITEMS WITHIN ONE BATCH_UID GROUP CAN BE STACKED, BUT NOT
    ACROSS BATCH_UID GROUPS ===
The curriculum interleaves replay from earlier (lower-resolution) phases
throughout each phase's own (higher-resolution) progression -- so
consecutive BATCH FOLDERS on disk can be different native resolutions by
design (batch folders group RUNS of same-resolution items, not the whole
phase -- see curriculum_builder.py's module docstring). WITHIN one batch
folder, however, every scene shares exactly one padded resolution by
construction. `BatchUidGroupedSampler` (below) exploits exactly this:
it groups dataset indices by batch_uid (== one physical BatchNNN_<res>/
folder), so every minibatch train.py receives is single-resolution and
can be stacked into a real (B, C, H, W) tensor with zero cropping or
cross-resolution padding. Consecutive minibatches (i.e. consecutive
batch_uid groups) can still differ in resolution from each other -- that
is unchanged and is exactly what train.py's cudnn.benchmark=False,
per-step resolution.resolve_dynamic_batch() lookup, and window_id-based
(not batch_uid-based) accumulation flushing are already built to handle.

=== WHY TRAIN BATCHES ARE PROCESSED VIA ragged_collate, NOT DEFAULT COLLATE ===
Even within one batch_uid group (fixed resolution), scenes can have
differing interior-frame counts (frame count depends on max_frame_span
sub-clipping, a per-SCENE draw -- see "PER-PHASE MAX FRAME SPAN" below --
not on resolution). Default collate can't stack ragged gt_stack
(num_interior, C, H, W) tensors of differing num_interior across items in
a batch. `ragged_collate` pads gt/timesteps to this minibatch's own
T_max = max(num_interior over the group) and returns a `valid_mask`
(B, T_max) bool tensor alongside everything else -- see that function's
docstring, and train.py's masked-loss computation which consumes it.

=== WHY THE UNDERLYING MODEL STILL USES DYNAMIC batch_size * accum_steps,
    NOT A SINGLE FIXED PHYSICAL BATCH SIZE ===
The curriculum interleaves replay from earlier (lower-resolution) phases
throughout each phase's own (higher-resolution) progression, and
config.yaml's dynamic_batch.thresholds table assigns a DIFFERENT
batch_size/accum_steps pair per resolution bucket (small resolutions get
a large batch_size and accum_steps=1; large resolutions get batch_size=1
and a large accum_steps) specifically so that VRAM stays roughly constant
across the curriculum despite native-resolution training. Consecutive
batch_uid groups (== consecutive minibatches yielded by
BatchUidGroupedSampler) can therefore have DIFFERING physical batch
sizes B within one phase, and train.py's accumulation-window flushing
(gated on window_id, not on B or resolution) is exactly the mechanism
that makes mixing differing-B minibatches within one accumulation window
mathematically correct -- see train.py's module docstring.

=== WHY train_seq_dirs / val_seq_dirs still get walked one scene at a time ===
The directory walk that discovers scenes (_list_train_seq_dirs) still
returns one entry per scene, in curriculum order -- that part is
unaffected by this revision. What changed is only how those flat items
get GROUPED into DataLoader steps (BatchUidGroupedSampler, keyed on the
batch_uid already present per item) and COLLATED once grouped
(ragged_collate). If you ever want to go back to strict one-item-at-a-
time loading (e.g. for debugging), just build a plain
torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, ...)
against the same train_set -- __getitem__'s per-item return shape is
unchanged, batch-of-1 default collate still works exactly as before.

=== WHY TRAIN BATCHES ARE PROCESSED ONE RESOLUTION AT A TIME ===
The curriculum interleaves replay from earlier (lower-resolution) phases
throughout each phase's own (higher-resolution) progression -- so
consecutive batch_uid GROUPS on disk can be different native resolutions
by design (batch folders group RUNS of same-resolution items, not the
whole phase -- see curriculum_builder.py's module docstring). Stacking
ACROSS batch_uid groups into one bigger physical batch would require
either cropping everyone to a common shape (destroys full-native-
resolution training, the whole point of this pipeline) or trusting the
underlying VFIMamba Trainer.Model to accept mixed-resolution tensors in
one call, which it does not. So this pipeline keeps one physical batch
per batch_uid group (single resolution, variable B), and implements
"dynamic batch_size" and "dynamic grad accumulation" together as ONE
micro-step counter in train.py: effective_batch = batch_size(resolution)
* accum_steps(resolution), looked up per group via
resolution.resolve_dynamic_batch(). This is mathematically identical to
true batching for gradient-accumulation purposes (loss is additive
across samples) and agrees with what curriculum_builder.py already wrote
into each batch folder's batch_metadata.yaml, since every item in a
batch folder shares one resolution by construction. See train.py's
training loop.

=== ACCUMULATION WINDOWS: window_id / window_num_items ===
curriculum_builder.py's group_batches_into_accum_windows() chunks each
resolution run into accumulation windows -- one intended window is
usually SEVERAL consecutive batch folders of the same resolution, not
one -- and stamps every batch in a window with a shared window_id plus
the window's TRUE total item count (window_num_items). Both are read
here straight out of each batch folder's batch_metadata.yaml (see
_list_train_seq_dirs) and returned per item alongside batch_uid; since
every item within one batch_uid group shares window_id/window_num_items
identically (they're stamped per BATCH, not per scene), ragged_collate
takes them from the group's first item without any risk of mismatch.

batch_uid identifies which single physical BatchNNN_<res>/ folder an
item (and, as of this revision, an entire minibatch) came from and
changes at every folder boundary -- including boundaries INSIDE one
window -- so it is exposed here for traceability/logging only, same as
before. window_id is the explicit, authoritative signal train.py flushes
the optimizer on, and window_num_items is what train.py divides the
accumulated loss by. Both full windows (exactly accum_steps(resolution)
batches) and partial windows (a shorter trailing remainder of a
resolution run, appended at the end of the phase's stream by
curriculum_builder.py rather than discarded) carry these fields the same
way -- train.py does not need to know or care which kind a given group's
window is; window_num_items is already exact for either case. See
train.py's training loop.

=== SEEDING: PER-EPOCH, NOT GLOBAL-RANDOM-STATE ===
Each training item's interior-frame/timestep choice is drawn from a
dedicated RNG keyed on (global seed, 'timestep', current epoch, sequence
folder) via seeding.rng_for() -- see __getitem__ and set_epoch() below.
This makes the choice both reproducible (same seed always produces the
same per-epoch pick for a given sequence) AND different from one epoch to
the next (unlike drawing from Python's global `random` module, which
worker_init_fn seeds once per worker and which would otherwise make every
epoch repeat the same picks). train.py MUST call
train_set.set_epoch(epoch) at the start of every epoch for this to work --
see that file's training loop. This per-item seeding is unaffected by
batch_uid grouping -- each scene's RNG is still keyed on its own
sample_id, independent of which other scenes happen to land in the same
minibatch.

=== PER-PHASE MAX FRAME SPAN (clip sampling for long scenes) ===
Some source scenes run to hundreds or thousands of frames. Using a
scene's literal first/last present frame as img0/img1 (the pre-existing
behavior) means the motion between img0 and img1 can span many seconds of
real footage -- far outside what a two-frame flow-based VFI model (cost
volumes, local correlation, linear/accel motion models) has any basis to
interpolate correctly, especially in EARLY curriculum phases where the
whole point is to establish small, well-posed motion before progressing
to harder cases.

data.max_frame_span_by_phase in config.yaml (read by prepare_datasets and
passed into FullResVFIDataset as `max_frame_span`) caps this per phase.
When a scene's frame count exceeds `max_frame_span`, __getitem__ draws a
random contiguous SUB-CLIP of length in [3, max_frame_span] from the
scene (seeded the same way as the timestep pick: keyed on
(seed, epoch, sample_id), so it's reproducible per-epoch and changes from
one epoch to the next -- see set_epoch()) and runs the existing
interior-frame/timestep selection on that clip instead of on the whole
scene. img0/img1 become the clip's first/last frame, not the scene's.
Scenes at or under the cap (or when max_frame_span is None for a phase)
are unaffected and use the whole scene, exactly as before. Because this
draw is per-scene, two scenes in the same batch_uid group (same
resolution) can still end up with different resulting num_interior --
this is exactly the raggedness ragged_collate exists to handle.

This is a per-item CLIP, not a train.py accumulation WINDOW -- window_id/
window_num_items (see above) are a completely separate concept (gradient
accumulation boundaries across many minibatches) and are untouched by
this feature; deliberately different terminology is used here to avoid
confusion between the two.

Validation is NOT clipped by this mechanism -- val always uses the full
scene's first/last frame and enumerates every interior frame up to
val_max_interior_per_seq, same as before. If val scenes are also very
long, this means val motion spans may not match what a given phase was
trained on.
"""
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml

import seeding
from seeding import worker_init_fn  # noqa: F401  (re-exported for train.py's DataLoader)

# curriculum_builder.py names each scene folder
# "<idx>_<origin>__<dataset>__<scene>"; kept here in case any tooling wants
# to parse it back apart (not required for loading -- see the two _list_*
# helpers below, which only rely on directory nesting + sort order).
ORDER_NAME_RE = re.compile(r'^(?P<idx>\d+)_(?P<origin>[^_]+(?:-[^_]+)?)__(?P<dataset>.+)__(?P<scene>[^_]+)$')

# Non-batch entries that can appear directly under a Phase<N>/ folder and
# must not be walked as if they were a BatchNNN_<res>/ folder of scenes.
_PHASE_SKIP_ENTRIES = {'Extra', 'phase_metadata.yaml'}


def pad_to_multiple(x, multiple):
    """x: (B, C, H, W). Pads only bottom/right (replicate), so cropping
    the first h rows / w cols back out later recovers the original."""
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pw, 0, ph), mode='replicate'), h, w


def _pick_val_interior_indices(n, max_per_seq):
    interior = list(range(1, n - 1))
    if max_per_seq is None or len(interior) <= max_per_seq:
        return interior
    idxs = sorted(set(int(round(i)) for i in np.linspace(0, len(interior) - 1, max_per_seq)))
    return [interior[i] for i in idxs]


def _list_train_seq_dirs(phase_dir):
    """Walk Phase<N>/BatchNNN_<res>/<scene>/ in curriculum order.
    Returns a list of (scene_dir, sample_id, batch_uid, window_id,
    window_num_items) tuples.

    batch_uid is the batch folder's own path -- a natural, unique
    identifier for "which physical BatchNNN_<res>/ folder did this scene
    come from." As of this revision it does double duty: it's still used
    for logging/tracing an item back to its source folder, AND it's the
    grouping key BatchUidGroupedSampler uses to build real minibatches
    (see that class below) -- every scene sharing a batch_uid shares the
    same resolution by construction (see curriculum_builder.py's
    group_into_batches), which is exactly what makes stacking them into
    one (B, C, H, W) tensor safe. The reverse does NOT hold within one
    accumulation window: curriculum_builder.py's
    group_batches_into_accum_windows() groups several consecutive
    same-resolution (or same-bucket) batch folders into a single window,
    so batch_uid can change multiple times WITHIN one window -- i.e.
    multiple minibatches can belong to the same window_id.

    window_id and window_num_items are read straight out of this batch
    folder's batch_metadata.yaml (written by curriculum_builder.py's
    _tag_windows/write_phase_folder) -- window_id is the unambiguous,
    authoritative signal train.py flushes the optimizer on; window_id is
    guaranteed to change exactly at accumulation-window boundaries, never
    inside one, and is present on every batch this function returns
    (Extra/ batches, the only ones without it, are skipped below before
    this point is reached). window_num_items is the window's true total
    item count and is what train.py divides the accumulated loss by --
    see that file's training loop.
    """
    seq_dirs = []
    for batch_name in sorted(os.listdir(phase_dir)):
        if batch_name in _PHASE_SKIP_ENTRIES:
            continue
        batch_dir = os.path.join(phase_dir, batch_name)
        if not os.path.isdir(batch_dir):
            continue

        meta_path = os.path.join(batch_dir, 'batch_metadata.yaml')
        with open(meta_path) as f:
            batch_meta = yaml.safe_load(f)

        # Every batch folder reaching this point is a normal (non-Extra)
        # batch, so window_id/window_num_items are always present --
        # written unconditionally for non-extra batches by
        # curriculum_builder.py's write_phase_folder.
        window_id = batch_meta['window_id']
        window_num_items = batch_meta['window_num_items']
        sample_ids = {s['folder']: s['sample_id'] for s in batch_meta['scenes']}

        for scene_name in sorted(os.listdir(batch_dir)):
            scene_dir = os.path.join(batch_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            sample_id = sample_ids.get(scene_name)
            if sample_id is None:
                raise RuntimeError(
                    f'{scene_dir} has no matching entry in '
                    f'{batch_dir}/batch_metadata.yaml -- was this folder written '
                    f'by curriculum_builder.py?')
            # batch_dir doubles as batch_uid -- unique per physical folder,
            # stable across a run, and exactly the key BatchUidGroupedSampler
            # groups on below.
            seq_dirs.append((scene_dir, sample_id, batch_dir, window_id, window_num_items))
    return seq_dirs


def _list_val_seq_dirs(val_dir):
    """ValidationData/<dataset>/<scene>/ -- two levels below val_dir,
    alphabetical (dataset, then scene); order doesn't carry curriculum
    meaning for validation."""
    seq_dirs = []
    for dataset_name in sorted(os.listdir(val_dir)):
        dataset_dir = os.path.join(val_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for scene_name in sorted(os.listdir(dataset_dir)):
            scene_dir = os.path.join(dataset_dir, scene_name)
            if os.path.isdir(scene_dir):
                seq_dirs.append(scene_dir)
    return seq_dirs


class FullResVFIDataset(Dataset):
    """
    mode='train': reads a curriculum Phase folder (nested Batch/Scene, see
    _list_train_seq_dirs) in curriculum order. One item per scene;
    interior frame picked per-epoch (seeded) on every __getitem__ -- see
    set_epoch(). Each item also carries its accumulation window's id and
    true item count (window_id, window_num_items) -- see this module's
    docstring. As of this revision, __getitem__'s per-item return shape
    is UNCHANGED -- batching is applied afterward, by DataLoader, via
    BatchUidGroupedSampler + ragged_collate (see below). This dataset
    class itself has no notion of batching.

    If max_frame_span is set and a scene's frame count exceeds it, a
    random contiguous sub-clip (length in [3, max_frame_span], also
    per-epoch seeded) is drawn first, and interior-frame selection runs
    on that clip instead of the whole scene -- see this module's
    docstring, "PER-PHASE MAX FRAME SPAN".

    mode='val': reads ValidationData/ (nested Dataset/Scene, see
    _list_val_seq_dirs). Every interior frame of every sequence is
    enumerated once at construction (capped via val_max_interior_per_seq),
    fixed order, deterministic across epochs. NOT affected by
    max_frame_span -- val always spans the full scene. Val is also NOT
    affected by batch_uid grouping -- train.py's val_loader still uses a
    plain DataLoader (see make_val_loader in train.py).
    """

    def __init__(self, split_dir, mode, frame_names, image_extensions, pad_multiple,
                 seed, val_max_interior_per_seq=4, max_frame_span=None, log=print):
        assert mode in ('train', 'val')
        if max_frame_span is not None and max_frame_span < 3:
            raise ValueError(f'max_frame_span must be >= 3 (got {max_frame_span})')
        self.max_frame_span = max_frame_span
        self.split_dir = split_dir
        self.mode = mode
        self.frame_names = frame_names
        self.image_extensions = image_extensions
        self.pad_multiple = pad_multiple
        self.seed = seed
        self.epoch = 0

        seq_dirs = _list_train_seq_dirs(split_dir) if mode == 'train' else _list_val_seq_dirs(split_dir)

        # Both modes resolve each frame_name -> actual on-disk filename
        # (frame_name + whichever extension in image_extensions exists)
        # the same way curriculum_builder.py's find_frame_file() does --
        # frame_names in config are extension-less, so a bare
        # os.path.exists(seq_dir / frame_name) check (no extension) would
        # never match anything on disk.
        self.items = []
        skipped = 0
        if mode == 'train':
            for seq_dir, sample_id, batch_uid, window_id, window_num_items in seq_dirs:
                present = [p for p in (
                    self._find_frame_file(seq_dir, name, self.image_extensions)
                    for name in self.frame_names
                ) if p is not None]
                if len(present) < 3:
                    skipped += 1
                    continue
                self.items.append((seq_dir, present, sample_id, batch_uid, window_id, window_num_items))
        else:
            for seq_dir in seq_dirs:
                present = [p for p in (
                    self._find_frame_file(seq_dir, name, self.image_extensions)
                    for name in self.frame_names
                ) if p is not None]
                if len(present) < 3:
                    skipped += 1
                    continue
                n = len(present)
                for k in _pick_val_interior_indices(n, val_max_interior_per_seq):
                    self.items.append((seq_dir, present, k))

        if skipped:
            log(f'  [{mode}] {skipped} folder(s) under {split_dir} had < 3 frames present, skipped')
        log(f'  [{mode}] {len(self.items)} item(s) from {split_dir}')
        if mode == 'train' and self.max_frame_span is not None:
            log(f'  [{mode}] max_frame_span={self.max_frame_span} '
                f'(scenes longer than this will be randomly sub-clipped per epoch)')
        if skipped and not self.items:
            log(f'  [{mode}] WARNING: everything under {split_dir} was skipped -- check '
                f'data.frame_names against the actual files on disk.')

    @staticmethod
    def _find_frame_file(seq_dir, frame_name, extensions):
        """Try frame_name + each extension, in order, return the matched
        filename (with extension) or None. Mirrors curriculum_builder.py's
        find_frame_file so both files resolve the same scene's frames
        identically regardless of per-dataset format."""
        for ext in extensions:
            candidate = os.path.join(seq_dir, f'{frame_name}{ext}')
            if os.path.exists(candidate):
                return f'{frame_name}{ext}'
        return None

    def set_epoch(self, epoch):
        """Call once at the start of every training epoch (train.py's
        training loop, before iterating train_loader). Changes the salt
        fed into the per-item timestep RNG (and, when max_frame_span
        applies, the sub-clip RNG -- same rng instance, see __getitem__)
        in __getitem__, so both the sub-clip and the interior-frame pick
        vary deterministically from one epoch to the next instead of
        being fixed for the whole run.

        NOTE on DataLoader worker processes: this only reaches
        already-running persistent workers if the DataLoader is recreated
        (or persistent_workers=False, the default) each epoch -- a
        persistent-worker pool holds a fork/pickle snapshot of this
        Dataset object taken at first iteration and will NOT see later
        attribute updates made in the parent process. train.py's
        DataLoader explicitly sets persistent_workers=False for this
        reason; if that ever changes, this method stops propagating to
        workers silently. This is unaffected by BatchUidGroupedSampler --
        the sampler only decides which indices get grouped into a step,
        it doesn't change how/when Dataset attributes propagate to
        workers.
        """
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load(seq_dir, filename):
        path = os.path.join(seq_dir, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'could not read {path} -- check frame_names')
        return img

    def _to_padded_tensor(self, im):
        im = np.ascontiguousarray(im)
        t = torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)
        t, h, w = pad_to_multiple(t.unsqueeze(0), self.pad_multiple)
        return t.squeeze(0), h, w

    def __getitem__(self, idx):
        if self.mode == 'train':
            seq_dir, present, sample_id, batch_uid, window_id, window_num_items = self.items[idx]
            n = len(present)
            rng = seeding.rng_for(self.seed, 'timestep', self.epoch, sample_id)

            if self.max_frame_span is not None and n > self.max_frame_span:
                # Still random per epoch -- this controls which SUB-CLIP of a
                # long scene is visible this epoch, not which frame within it.
                clip_len = rng.randint(3, self.max_frame_span)
                start = rng.randint(0, n - clip_len)
                clip = present[start:start + clip_len]
            else:
                clip = present

            cn = len(clip)
            img0_name, img1_name = clip[0], clip[-1]
            # ALL interior frames now, not one random pick.
            interior_ks = list(range(1, cn - 1))
            gt_names = [clip[k] for k in interior_ks]
            timesteps = [k / (cn - 1) for k in interior_ks]
        else:
            seq_dir, present, k = self.items[idx]
            n = len(present)
            img0_name, gt_name, img1_name = present[0], present[k], present[-1]
            timestep = k / (n - 1)

        img0 = self._load(seq_dir, img0_name)
        img0_t, h, w = self._to_padded_tensor(img0)
        img1 = self._load(seq_dir, img1_name)
        img1_t, _, _ = self._to_padded_tensor(img1)

        if self.mode == 'train':
            # (num_interior, C, H, W) -- num_interior varies per item. As of
            # this revision, DataLoader batch_size can be > 1 (see
            # BatchUidGroupedSampler), so ragged_collate (below) -- not
            # default collate -- is responsible for padding this to a
            # per-minibatch T_max and building a valid_mask. This function
            # itself still returns exactly one scene's (possibly ragged)
            # data; it has no knowledge of what else will land in its batch.
            gt_stack = torch.stack(
                [self._to_padded_tensor(self._load(seq_dir, name))[0] for name in gt_names], dim=0)
            timesteps_t = torch.tensor(timesteps, dtype=torch.float32)
            return (img0_t, gt_stack, img1_t, timesteps_t,
                    h, w, batch_uid, window_id, window_num_items)

        gt = self._load(seq_dir, gt_name)
        gt_t, _, _ = self._to_padded_tensor(gt)
        return (img0_t, gt_t, img1_t, torch.tensor(timestep, dtype=torch.float32), h, w)


class BatchUidGroupedSampler(torch.utils.data.Sampler):
    """Groups a train FullResVFIDataset's flat item list into one index
    list per physical batch_uid (== one BatchNNN_<res>/ folder written by
    curriculum_builder.py), in curriculum order both across and within
    groups.

    This is what actually turns "dynamic batch_size" from a purely
    gradient-accumulation-over-singles concept into REAL batching: every
    scene sharing a batch_uid shares exactly one padded resolution by
    construction (curriculum_builder.py's group_into_batches invariant),
    so grouping on batch_uid is always safe to stack into one (B, C, H, W)
    tensor -- no cropping, no cross-resolution padding, ever needed.

    Because curriculum_builder.py's config.yaml dynamic_batch.thresholds
    table typically assigns large batch_size at small resolutions and
    batch_size=1 at large resolutions (to keep VRAM roughly constant
    across native-resolution training), this sampler naturally produces
    large real minibatches at low-resolution phases (where GPU
    utilization benefits most from batching) and single-item "batches" at
    high-resolution phases (where a single native-res forward pass
    already saturates the GPU) -- with zero phase-conditional code
    anywhere. What changes phase to phase is purely the resolution->
    batch_size mapping in config.yaml, not this sampler or train.py's
    loop structure.

    Use as `DataLoader(train_set, batch_sampler=BatchUidGroupedSampler(train_set),
    collate_fn=ragged_collate, ...)` -- batch_sampler (not batch_size +
    sampler) because group sizes vary (they equal each batch folder's own
    scene count, which for a trailing/undersized-adjacent full batch or a
    replay group landing at a phase boundary can be smaller than the
    resolution's configured batch_size).

    Does NOT shuffle -- train.py's curriculum order (own progression +
    interleaved replay + trailing partial windows) IS the intended
    training order; shuffling groups or their contents here would undo
    exactly what curriculum_builder.py spent its whole retain/export/
    interleave pipeline building. This mirrors the shuffle=False rationale
    already present in train.py's DataLoader construction for the
    previous (batch_size=1) loader.
    """

    def __init__(self, dataset):
        if dataset.mode != 'train':
            raise ValueError('BatchUidGroupedSampler is only valid for mode="train" datasets '
                              '(val is loaded natively, unbatched -- see make_val_loader in train.py)')
        groups = {}
        order = []
        for idx, item in enumerate(dataset.items):
            # item == (seq_dir, present, sample_id, batch_uid, window_id, window_num_items)
            batch_uid = item[3]
            if batch_uid not in groups:
                groups[batch_uid] = []
                order.append(batch_uid)
            groups[batch_uid].append(idx)
        # order[] preserves first-appearance order, which -- because
        # dataset.items itself was built by iterating _list_train_seq_dirs's
        # curriculum-ordered output -- is exactly the curriculum order of
        # batch folders. Every idx within one group is similarly already in
        # curriculum (scene-sort) order, since it was appended to
        # dataset.items in that order too.
        self.grouped_indices = [groups[uid] for uid in order]

    def __iter__(self):
        return iter(self.grouped_indices)

    def __len__(self):
        return len(self.grouped_indices)


def ragged_collate(batch):
    """Collate function for a batch_uid-grouped minibatch of TRAIN items
    (see BatchUidGroupedSampler). Every item in `batch` shares exactly one
    resolution (guaranteed by grouping on batch_uid, in turn guaranteed by
    curriculum_builder.py's group_into_batches), so img0/img1 stack
    trivially. What does NOT match across items is num_interior (each
    scene's own frame-count / max_frame_span sub-clip draw is independent
    -- see this module's docstring) -- gt and timesteps are padded to this
    minibatch's own T_max = max(num_interior), and a `valid_mask`
    (B, T_max) bool tensor is returned so train.py's loss computation can
    exclude the padded slots and keep each item's loss weight independent
    of how many interior frames it actually had (mirroring the pre-
    batching `frame_losses.mean()` invariant, just computed per-item-
    then-across-batch instead of per-item-then-across-steps).

    batch_uid, window_id, and window_num_items are identical across every
    item in `batch` by construction (they're stamped per BATCH FOLDER by
    curriculum_builder.py, not per scene -- see _list_train_seq_dirs) --
    taken from batch[0] here with no risk of a mismatched value hiding
    among the rest of the group.

    Returns a tuple matching train.py's unpacking:
        (img0, gt, img1, timesteps, valid_mask, h, w,
         batch_uid, window_id, window_num_items)
    where:
        img0, img1        : (B, C, H, W)
        gt, timesteps      : (B, T_max, C, H, W) / (B, T_max) -- zero-padded
                              past each item's own num_interior
        valid_mask          : (B, T_max) bool -- True where gt/timesteps are
                              real (not padding)
        h, w                : (B,) int64 -- identical across the batch in
                              practice (single resolution per group), kept
                              per-item for symmetry with the val loader and
                              so train.py can sanity-check if it wants to
        batch_uid            : str, this group's source folder path
        window_id            : str, this group's accumulation-window id
        window_num_items      : int, this window's true total item count
    """
    B = len(batch)
    T_max = max(item[1].shape[0] for item in batch)
    C, H, W = batch[0][0].shape

    img0 = torch.stack([item[0] for item in batch], dim=0)
    img1 = torch.stack([item[2] for item in batch], dim=0)

    gt = torch.zeros(B, T_max, C, H, W, dtype=batch[0][1].dtype)
    timesteps = torch.zeros(B, T_max, dtype=torch.float32)
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for b, (_, gt_stack, _, ts, *_rest) in enumerate(batch):
        n = gt_stack.shape[0]
        gt[b, :n] = gt_stack
        timesteps[b, :n] = ts
        valid_mask[b, :n] = True

    h = torch.tensor([item[4] for item in batch], dtype=torch.int64)
    w = torch.tensor([item[5] for item in batch], dtype=torch.int64)

    # Identical across the group by construction -- see docstring above.
    batch_uid = batch[0][6]
    window_id = batch[0][7]
    window_num_items = batch[0][8]

    return img0, gt, img1, timesteps, valid_mask, h, w, batch_uid, window_id, window_num_items


def prepare_datasets(cfg, phase, log=print):
    """
    Build the train Dataset for the given phase (1-4) plus the val
    Dataset. Both are read-only over a curriculum ALREADY BUILT by
    curriculum_builder.py -- this function does not split or shuffle
    anything itself; run curriculum_builder.py first. This function
    itself is unaffected by the batching revision -- it still returns
    plain FullResVFIDataset objects; train.py is responsible for wrapping
    train_set in a DataLoader with BatchUidGroupedSampler + ragged_collate
    (see that file's make_train_loader).

    data.max_frame_span_by_phase (config.yaml) is looked up for this
    phase and passed into the train Dataset as max_frame_span -- a
    missing phase key, a missing max_frame_span_by_phase table entirely,
    or an explicit null/None all mean "no cap" for this phase. The val
    Dataset never receives this value -- see this module's docstring.
    """
    dataset_root = cfg['paths']['dataset_root']
    phase_dir = os.path.join(dataset_root, 'TrainingData', f'Phase{phase}')
    val_dir = os.path.join(dataset_root, 'ValidationData')
    if not os.path.isdir(phase_dir):
        raise RuntimeError(
            f'{phase_dir} does not exist -- run curriculum_builder.py first '
            f'(it builds TrainingData/Phase1..4/ and ValidationData/ under paths.dataset_root).')
    if not os.path.isdir(val_dir):
        raise RuntimeError(f'{val_dir} does not exist -- run curriculum_builder.py first.')

    d_cfg = cfg['data']
    image_extensions = cfg['image_extensions']
    pad_multiple = cfg['pad_multiple']
    seed = cfg['seed']

    max_span_by_phase = d_cfg.get('max_frame_span_by_phase', {}) or {}
    max_frame_span = max_span_by_phase.get(phase)

    train_set = FullResVFIDataset(
        phase_dir, 'train', d_cfg['frame_names'], image_extensions, pad_multiple,
        seed=seed, max_frame_span=max_frame_span, log=log)

    val_set = FullResVFIDataset(
        val_dir, 'val', d_cfg['frame_names'], image_extensions, pad_multiple,
        seed=seed, val_max_interior_per_seq=d_cfg.get('val_max_interior_per_seq', 4), log=log)

    if len(train_set) == 0:
        raise RuntimeError(
            f'{phase_dir} produced 0 usable train items -- check data.frame_names '
            f'against the actual files on disk, and that curriculum_builder.py ran '
            f'successfully for this phase.')
    if len(val_set) == 0:
        log('  WARNING: val set has 0 items -- val_loss/PSNR will be meaningless this run.')

    return train_set, val_set