"""
dataset.py -- ALL data handling for the VFIMamba full-resolution fine-tune
lives here. train.py never touches a raw frame, a random seed, or a pixel
of padding directly; it just calls prepare_datasets() once and gets back
two torch Datasets that yield fully-prepared, padded-to-32
(img0, gt, img1, timestep, scale, orig_h, orig_w) tuples ready to feed
straight into the model. This is the one file to edit if you want to
change how sequences are discovered, split, augmented, or preprocessed.

WHAT LIVES HERE, IN ORDER:
  1. Seeding                     -- seed_everything() / worker_init_fn()
  2. Physical train/val split    -- compute_split() / materialize_split()
  3. Dynamic interior-frame pick -- _pick_val_interior_indices()
  4. Resolution-aware scale      -- resolve_train_scale()
  5. Padding to a multiple of 32 -- pad_to_multiple()
  6. The Dataset itself          -- FullResVFIDataset
  7. Orchestration               -- prepare_datasets(), called by train.py

=== WHY A PHYSICAL SPLIT INSTEAD OF AN IN-MEMORY ONE ===
The previous version of this pipeline split DATA_ROOT into train/val
entirely in memory, every time train.py ran, and never wrote that split to
disk -- there was no durable record of which sequences were actually held
out, and a separate script (split_train_val.py) had to re-implement the
exact same seed/shuffle logic by hand to reconstruct it, with a comment
warning that its SEED/VAL_SPLIT had to be kept manually in sync or you'd
silently get a different split than what the model was validated on.

materialize_split() below now does this splitting itself, once, as part of
prepare_datasets() -- it symlinks (default) or copies each sequence folder
under DATA_ROOT into <run_dir>/split/train/ or <run_dir>/split/val/ (or
wherever data.train_out / data.val_out point in the yaml). That gives you
a browsable, per-run, on-disk record of exactly which sequences were
trained on vs held out, with zero risk of the split logic drifting out of
sync between two files, because there is now only one copy of it.
"""
import os
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. SEEDING
#    Governs BOTH the train/val sequence split (compute_split) and the
#    per-item dynamic interior-frame/timestep choice made inside
#    FullResVFIDataset.__getitem__, since that draws from the global
#    `random` module. Call seed_everything() once, before building the
#    split or the datasets -- prepare_datasets() does this for you.
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """
    Re-derives a per-worker seed from torch's (already-seeded) RNG, so
    DataLoader workers' random frame/timestep choices are reproducible
    too, not just the main process's.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# 2. PHYSICAL TRAIN/VAL SPLIT
# ============================================================

def compute_split(data_root, seed, val_split):
    """
    Which sequence folders go to train vs val, given (seed, val_split).
    Deterministic seeded shuffle, split by SEQUENCE (never by frame/item),
    so overlapping items from one sequence can never leak across the
    train/val boundary.
    """
    all_seqs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    if not all_seqs:
        raise RuntimeError(f'No subfolders found under {data_root}')

    rng = random.Random(seed)
    shuffled = all_seqs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    val_seqs = set(shuffled[:n_val])

    train_seqs = [s for s in all_seqs if s not in val_seqs]
    val_seqs_list = [s for s in all_seqs if s in val_seqs]
    return train_seqs, val_seqs_list


def _place(seq_name, src_dir, dst_root, mode, log):
    dst = os.path.join(dst_root, seq_name)
    if os.path.exists(dst) or os.path.islink(dst):
        return 'skipped (already exists)'

    if mode == 'symlink':
        try:
            os.symlink(os.path.abspath(src_dir), dst, target_is_directory=True)
            return 'linked'
        except OSError as e:
            # WSL symlinks onto NTFS-mounted drives (/mnt/c, /mnt/d, ...)
            # need Windows Developer Mode enabled, or this raises
            # "Operation not permitted" even with correct WSL permissions.
            # Falling back to copy keeps the pipeline usable either way.
            log(f'  symlink failed for {seq_name} ({e}) -- falling back to copy. '
                f'Enable Windows Developer Mode to get symlinks working.')
            shutil.copytree(src_dir, dst)
            return 'copied (symlink fallback)'
    else:
        shutil.copytree(src_dir, dst)
        return 'copied'


def materialize_split(data_root, train_out, val_out, seed, val_split,
                       mode='symlink', log=print):
    """
    Physically places each sequence folder from data_root into train_out/
    or val_out/, so the split is a browsable, durable record instead of
    something only ever computed in memory. Idempotent -- re-running with
    the same settings just skips sequences that are already placed, so
    it's safe to call this at the start of every training run.

    Returns (train_out, val_out) -- the two directories FullResVFIDataset
    should be pointed at.
    """
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    train_seqs, val_seqs = compute_split(data_root, seed, val_split)
    log(f'  data split: {len(train_seqs)} train / {len(val_seqs)} val sequence(s) '
        f'(seed={seed}, val_split={val_split}, mode={mode})')

    for seq in train_seqs:
        result = _place(seq, os.path.join(data_root, seq), train_out, mode, log)
        if result != 'linked':
            log(f'    train/{seq}: {result}')
    for seq in val_seqs:
        result = _place(seq, os.path.join(data_root, seq), val_out, mode, log)
        if result != 'linked':
            log(f'    val/{seq}: {result}')

    log(f'  -> {train_out} ({len(train_seqs)} sequences)')
    log(f'  -> {val_out} ({len(val_seqs)} sequences)')
    return train_out, val_out


# ============================================================
# 3. DYNAMIC INTERIOR-FRAME SELECTION (validation enumeration)
# ============================================================

def _pick_val_interior_indices(n, max_per_seq):
    """
    All interior frame indices (1 .. n-2) for a sequence of length n,
    evenly subsampled down to at most max_per_seq if there would
    otherwise be more -- keeps one very long sequence from dominating (or
    blowing up the runtime of) validation by itself. Deterministic, so
    val composition never changes across epochs. max_per_seq=None means
    "use all interior frames, no cap".
    """
    interior = list(range(1, n - 1))
    if max_per_seq is None or len(interior) <= max_per_seq:
        return interior
    idxs = np.linspace(0, len(interior) - 1, max_per_seq)
    idxs = sorted(set(int(round(i)) for i in idxs))
    return [interior[i] for i in idxs]


# ============================================================
# 4. RESOLUTION-AWARE TRAIN_SCALE
# ============================================================

def resolve_train_scale(h, w, thresholds, fallback):
    """
    Pick a flow-estimation scale for a sample based on its (original,
    pre-padding) resolution. `thresholds` is a list of
    [min_long_edge_px, scale] pairs; the first one the sample's long edge
    (max(h, w)) meets or exceeds wins, checked largest-threshold-first
    regardless of the order given. Falls back to `fallback` if thresholds
    is empty (e.g. train_scale_auto disabled) or doesn't cover this
    sample's resolution.
    """
    if not thresholds:
        return fallback
    long_edge = max(h, w)
    for min_dim, scale in sorted(thresholds, key=lambda t: -t[0]):
        if long_edge >= min_dim:
            return scale
    return fallback


# ============================================================
# 5. PADDING TO A MULTIPLE OF 32
# ============================================================

def pad_to_multiple(x, multiple=32):
    """
    x: (B, C, H, W). Pads only on the bottom/right with replicate padding,
    so cropping the first h rows / w cols back out of the padded tensor
    later exactly recovers the original, unpadded content.
    """
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pw, 0, ph), mode='replicate'), h, w


# ============================================================
# 6. THE DATASET
#    Reads from an ALREADY-SPLIT folder (produced by materialize_split
#    above) -- one instance per train/val folder, not per DATA_ROOT+mode,
#    so there is no split logic left to duplicate or drift.
# ============================================================

class FullResVFIDataset(torch.utils.data.Dataset):
    """
    For a sequence dir with N of frame_filenames present (N >= 3):
      - img0 = first present frame, img1 = last present frame
      - gt   = a DYNAMICALLY CHOSEN interior frame, timestep = k / (N - 1)
    TRAIN mode picks the interior frame at random on every __getitem__
    call (reproducible given the same seed -- see seed_everything /
    worker_init_fn above). A sequence dir always contributes exactly ONE
    training item regardless of how many frames it has; a longer sequence
    supplies more timestep variety ACROSS epochs, not more items within
    one epoch.
    VAL mode enumerates every interior frame of every val sequence once at
    construction time (capped per sequence via val_max_interior_per_seq),
    so val composition -- and therefore val_loss/PSNR comparisons across
    epochs -- stays fixed and reproducible.

    __getitem__ returns frames already padded to a multiple of
    `pad_multiple`, plus the pre-padding (orig_h, orig_w) so predictions
    can be cropped back to the original size before computing PSNR later.
    Padding is appended (bottom/right only), so
    `padded_gt[:, :, :orig_h, :orig_w]` exactly recovers the unpadded gt.
    """

    def __init__(self, split_dir, mode, frame_filenames, flip_aug=False,
                 val_max_interior_per_seq=4, pad_multiple=32,
                 train_scale=0.5, train_scale_auto=False,
                 train_scale_thresholds=None, log=print):
        assert mode in ('train', 'val')
        if len(frame_filenames) < 3:
            raise ValueError('frame_filenames must list at least 3 frames')

        self.split_dir = split_dir
        self.mode = mode
        self.flip_aug = flip_aug and mode == 'train'
        self.pad_multiple = pad_multiple
        self.train_scale = train_scale
        self.train_scale_auto = train_scale_auto
        self.train_scale_thresholds = train_scale_thresholds or []

        seqs = sorted(
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        )

        # train: one item per sequence -> (seq_dir, present_frames)
        # val:   one item per (sequence, interior frame) -> (seq_dir, present_frames, k)
        self.items = []
        skipped = 0
        for seq in seqs:
            seq_dir = os.path.join(split_dir, seq)
            present = [f for f in frame_filenames
                       if os.path.exists(os.path.join(seq_dir, f))]
            if len(present) < 3:
                skipped += 1
                continue
            if mode == 'train':
                self.items.append((seq_dir, present))
            else:
                n = len(present)
                for k in _pick_val_interior_indices(n, val_max_interior_per_seq):
                    self.items.append((seq_dir, present, k))

        if skipped:
            log(f'  [{mode}] {skipped} sequence dir(s) under {split_dir} had fewer '
                f'than 3 of frame_filenames present and were skipped entirely')
        log(f'  [{mode}] {len(seqs) - skipped} sequence(s) -> {len(self.items)} '
            f'item(s) from {split_dir}')

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load(seq_dir, filename):
        path = os.path.join(seq_dir, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(
                f'could not read {path} -- check frame_filenames matches your '
                f'actual data layout'
            )
        return img

    def _resolve_scale(self, h, w):
        if self.train_scale_auto:
            return resolve_train_scale(h, w, self.train_scale_thresholds, self.train_scale)
        return self.train_scale

    def _to_padded_tensor(self, im):
        im = np.ascontiguousarray(im)
        t = torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)
        t, h, w = pad_to_multiple(t.unsqueeze(0), self.pad_multiple)
        return t.squeeze(0), h, w

    def __getitem__(self, idx):
        if self.mode == 'train':
            seq_dir, present = self.items[idx]
            n = len(present)
            # n == 3 -> only one possible interior frame (index 1).
            # n  > 3 -> pick one at random on every call, from the global
            # `random` module (seeded -- see seed_everything()).
            k = 1 if n == 3 else random.randint(1, n - 2)
        else:
            seq_dir, present, k = self.items[idx]
            n = len(present)

        img0 = self._load(seq_dir, present[0])
        gt = self._load(seq_dir, present[k])
        img1 = self._load(seq_dir, present[-1])
        timestep = k / (n - 1)

        if self.flip_aug:
            if random.random() < 0.5:  # horizontal flip
                img0, gt, img1 = img0[:, ::-1], gt[:, ::-1], img1[:, ::-1]
            if random.random() < 0.5:  # vertical flip
                img0, gt, img1 = img0[::-1], gt[::-1], img1[::-1]
            if random.random() < 0.5:  # temporal swap
                img0, img1 = img1, img0
                # Reverses which frame timestep is measured from -- MUST
                # flip timestep too now that it varies, or roughly a sixth
                # of augmented samples would silently train against the
                # wrong target (see module docstring in the old train.py).
                timestep = 1.0 - timestep

        orig_h, orig_w = gt.shape[:2]
        scale = self._resolve_scale(orig_h, orig_w)

        img0_t, h, w = self._to_padded_tensor(img0)
        gt_t, _, _ = self._to_padded_tensor(gt)
        img1_t, _, _ = self._to_padded_tensor(img1)

        return (img0_t, gt_t, img1_t,
                torch.tensor(timestep, dtype=torch.float32),
                torch.tensor(scale, dtype=torch.float32),
                h, w)


# ============================================================
# 7. ORCHESTRATION -- the one thing train.py actually calls
# ============================================================

def prepare_datasets(C, run_dir, log=print):
    """
    Everything train.py needs to go from a yaml config to two ready-to-use
    Datasets: seeds every RNG, materializes the physical train/val split
    on disk, and constructs both FullResVFIDataset instances. train.py
    should not need to import anything else from this file for the common
    case -- worker_init_fn is exposed separately only because DataLoader
    needs a direct reference to it.
    """
    d_cfg = C['data']
    r_cfg = C['run']
    t_cfg = C['train']

    seed = r_cfg.get('seed', 42)
    seed_everything(seed)

    data_root = d_cfg['data_root']
    val_split = d_cfg.get('val_split', 0.2)
    split_mode = d_cfg.get('split_mode', 'symlink')
    train_out = d_cfg.get('train_out') or os.path.join(run_dir, 'split', 'train')
    val_out = d_cfg.get('val_out') or os.path.join(run_dir, 'split', 'val')

    materialize_split(data_root, train_out, val_out, seed, val_split,
                       mode=split_mode, log=log)

    frame_filenames = tuple(d_cfg['frame_filenames'])
    flip_aug = d_cfg.get('flip_aug', False)
    val_max_interior = d_cfg.get('val_max_interior_per_seq', 4)

    pad_multiple = t_cfg.get('pad_multiple', 32)
    train_scale = t_cfg.get('train_scale', 0.5)
    ts_auto_cfg = t_cfg.get('train_scale_auto') or {}
    train_scale_auto = ts_auto_cfg.get('enabled', False)
    train_scale_thresholds = ts_auto_cfg.get('thresholds') or []
    if train_scale_auto and not train_scale_thresholds:
        raise ValueError(
            'train.train_scale_auto.enabled is true but no thresholds were '
            'given -- set train.train_scale_auto.thresholds, or disable '
            'train_scale_auto and rely on the flat train.train_scale instead')

    common_kwargs = dict(
        frame_filenames=frame_filenames,
        val_max_interior_per_seq=val_max_interior,
        pad_multiple=pad_multiple,
        train_scale=train_scale,
        train_scale_auto=train_scale_auto,
        train_scale_thresholds=train_scale_thresholds,
        log=log,
    )
    train_set = FullResVFIDataset(train_out, 'train', flip_aug=flip_aug, **common_kwargs)
    val_set = FullResVFIDataset(val_out, 'val', flip_aug=False, **common_kwargs)

    return train_set, val_set