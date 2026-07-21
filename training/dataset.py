"""
dataset.py

Loads ONE already-built curriculum phase folder (TrainingData/Phase1..4/,
produced by curriculum_builder.py -- run that first) into a torch Dataset
that yields padded (img0, gt, img1, timestep, orig_h, orig_w) tuples, with
a random interior-frame/timestep pick per training item. Also loads
ValidationData/, unbatched, with every interior frame of every val
sequence enumerated once.

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

=== WHY TRAIN BATCHES ARE PROCESSED ONE ITEM AT A TIME, NOT STACKED ===
The curriculum interleaves replay from earlier (lower-resolution) phases
throughout each phase's own (higher-resolution) progression -- so
consecutive items on disk can be different native resolutions by design
(batch folders group RUNS of same-resolution items, not the whole phase --
see curriculum_builder.py's module docstring). Stacking a physical batch
> 1 across differing resolutions would require either cropping everyone
to a common shape (destroys full-native-resolution training, the whole
point of this pipeline) or trusting the underlying VFIMamba Trainer.Model
to accept a per-sample timestep TENSOR for batch > 1 -- which the given
Trainer.py/model_oldRepo code never demonstrates (every call site in the
original train.py passed a single scalar timestep).

So this pipeline keeps DataLoader batch_size=1 (one sequence, its own
native resolution, per physical step) and implements "dynamic batch_size"
and "dynamic grad accumulation" together as ONE micro-step counter in
train.py: effective_batch = batch_size(resolution) * accum_steps(resolution),
looked up per item via resolution.resolve_dynamic_batch(). This is
mathematically identical to true batching for gradient-accumulation
purposes (loss is additive across samples) and makes no assumption about
the model's batching support -- and it agrees with what curriculum_builder.py
already wrote into each batch folder's batch_metadata.yaml, since every
item in a batch folder shares one resolution by construction. See
train.py's training loop.

=== ACCUMULATION WINDOWS: window_id / window_num_items ===
curriculum_builder.py's group_batches_into_accum_windows() chunks each
resolution run into accumulation windows -- one intended window is
usually SEVERAL consecutive batch folders of the same resolution, not
one -- and stamps every batch in a window with a shared window_id plus
the window's TRUE total item count (window_num_items). Both are read
here straight out of each batch folder's batch_metadata.yaml (see
_list_train_seq_dirs) and returned per item alongside batch_uid.

batch_uid identifies which single physical BatchNNN_<res>/ folder an
item came from and changes at every folder boundary -- including
boundaries INSIDE one window -- so it is exposed here for
traceability/logging only, same as before. window_id is the explicit,
authoritative signal train.py flushes the optimizer on, and
window_num_items is what train.py divides the accumulated loss by. Both
full windows (exactly accum_steps(resolution) batches) and partial
windows (a shorter trailing remainder of a resolution run, appended at
the end of the phase's stream by curriculum_builder.py rather than
discarded) carry these fields the same way -- train.py does not need to
know or care which kind a given item's window is; window_num_items is
already exact for either case. See train.py's training loop.

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
see that file's training loop.
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
    come from," useful for logging/tracing an item back to its source
    folder. Every scene sharing a batch_uid shares the same resolution by
    construction (see curriculum_builder.py's group_into_batches), but
    the reverse does NOT hold within one accumulation window:
    curriculum_builder.py's group_batches_into_accum_windows() groups
    several consecutive same-resolution batch folders into a single
    window, so batch_uid can change multiple times WITHIN one window.

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
    docstring.

    mode='val': reads ValidationData/ (nested Dataset/Scene, see
    _list_val_seq_dirs). Every interior frame of every sequence is
    enumerated once at construction (capped via val_max_interior_per_seq),
    fixed order, deterministic across epochs.
    """

    def __init__(self, split_dir, mode, frame_names, image_extensions, pad_multiple,
                 seed, val_max_interior_per_seq=4, log=print):
        assert mode in ('train', 'val')
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
        fed into the per-item timestep RNG in __getitem__, so the
        interior-frame/timestep pick varies deterministically from one
        epoch to the next instead of being fixed for the whole run.

        NOTE on DataLoader worker processes: this only reaches
        already-running persistent workers if the DataLoader is recreated
        (or persistent_workers=False, the default) each epoch -- a
        persistent-worker pool holds a fork/pickle snapshot of this
        Dataset object taken at first iteration and will NOT see later
        attribute updates made in the parent process. train.py's
        DataLoader explicitly sets persistent_workers=False for this
        reason; if that ever changes, this method stops propagating to
        workers silently.
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
            if n == 3:
                k = 1
            else:
                rng = seeding.rng_for(self.seed, 'timestep', self.epoch, sample_id)
                k = rng.randint(1, n - 2)
        else:
            seq_dir, present, k = self.items[idx]
            n = len(present)

        img0 = self._load(seq_dir, present[0])
        gt = self._load(seq_dir, present[k])
        img1 = self._load(seq_dir, present[-1])
        timestep = k / (n - 1)

        img0_t, h, w = self._to_padded_tensor(img0)
        gt_t, _, _ = self._to_padded_tensor(gt)
        img1_t, _, _ = self._to_padded_tensor(img1)

        if self.mode == 'train':
            return (img0_t, gt_t, img1_t, torch.tensor(timestep, dtype=torch.float32),
                    h, w, batch_uid, window_id, window_num_items)
        return (img0_t, gt_t, img1_t, torch.tensor(timestep, dtype=torch.float32), h, w)


def prepare_datasets(cfg, phase, log=print):
    """
    Build the train Dataset for the given phase (1-4) plus the val
    Dataset. Both are read-only over a curriculum ALREADY BUILT by
    curriculum_builder.py -- this function does not split or shuffle
    anything itself; run curriculum_builder.py first.
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

    train_set = FullResVFIDataset(
        phase_dir, 'train', d_cfg['frame_names'], image_extensions, pad_multiple,
        seed=seed, log=log)

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