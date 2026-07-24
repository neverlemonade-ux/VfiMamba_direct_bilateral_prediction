"""
Fine-tune VFIMamba (large variant) on ONE phase of the curriculum built by
curriculum_builder.py. Single-GPU (one RTX 3090).

    python curriculum_builder.py --config config.yaml     # once, builds Phase1..4/ + Val/
    python train.py --config config.yaml --run run_config.yaml --phase 1
    # ... after Phase 1 finishes, manually start Phase 2 (optionally --restore_ckpt
    # pointed at Phase 1's best/last checkpoint) ...
    python train.py --config config.yaml --run run_config.yaml --phase 2 --restore_ckpt runs/phase1/checkpoints/..._best.pkl

config.yaml holds every pipeline-wide, curriculum-affecting setting
(the single global seed, dynamic_batch/train_scale tables, validation
mode). run_config.yaml holds only per-run training hyperparameters
(epochs, lr, run_name, ...) -- see run_config.yaml's comments. Only ONE
phase is ever loaded per run; which phase is picked with --phase (or
run_config.yaml's `phase:` key).

> **Revision note (this revision -- real batch loading):** train_loader
> previously forced batch_size=1 and implemented "dynamic batch_size"
> purely as gradient accumulation over single items. This revision
> switches to `dataset.BatchUidGroupedSampler` + `dataset.ragged_collate`
> (see make_train_loader below and dataset.py's module docstring), so a
> training step now processes a REAL minibatch -- every scene from one
> physical BatchNNN_<res>/ folder at once, stacked into (B, C, H, W)
> tensors -- instead of exactly one scene. This gives a real GPU-
> utilization win at low-resolution phases (where config.yaml's
> dynamic_batch table assigns a large batch_size and a single native-res
> forward pass often doesn't saturate the GPU) with ZERO special-casing
> for high-resolution phases: if that table resolves batch_size=1 for a
> given resolution, its batch_uid groups naturally contain exactly one
> scene, and this file's loop behaves exactly as it did before this
> revision for those steps.
>
> Four things changed as a direct consequence, all confined to the
> training loop below (the accumulation-window semantics -- flush on
> window_id change, divide by window_num_items -- are UNCHANGED; see the
> "ACCUMULATION WINDOWS" section further down, which still applies
> verbatim):
>
>   1. `steps_since_update` now counts ITEMS (scenes), incremented by
>      this step's actual batch size B, not by a fixed 1 per step. It
>      needs to reach `cur_effective_batch` (== window_num_items, an item
>      count) exactly as before -- only the increment granularity
>      changed, matching how many items a "step" now actually contains.
>
>   2. gt/timesteps arrive already padded to this minibatch's own
>      T_max = max(num_interior) with an accompanying `valid_mask`
>      (B, T_max) bool tensor (see dataset.ragged_collate). The loss is
>      computed per-item (masking out padded interior-frame slots, so a
>      shorter-scene item's contribution stays independent of how many
>      interior frames it had), THEN averaged across the B items in this
>      step -- exactly mirroring the pre-batching `frame_losses.mean()`
>      invariant, just with an extra averaging level for the new batch
>      dimension. See the "MASKED, PER-ITEM LOSS" section below for the
>      exact reasoning and why this loop is NOT vectorized further.
>
>   3. Because that per-step loss is now a MEAN over B items (not a
>      single item's loss), the backward pass needs
>          scaled_loss = mean_item_loss * B / cur_effective_batch
>      instead of the old `loss / cur_effective_batch`. Multiplying back
>      by B before dividing is what keeps this step's gradient
>      contribution equal to the SUM of what B individual
>      gradient-accumulation steps would have contributed (each
>      normalized by cur_effective_batch) -- omitting the `* B` would
>      silently shrink this step's gradient magnitude by a factor of B
>      relative to one-item-at-a-time accumulation, changing the
>      effective learning rate any time B > 1 without changing anything
>      else about the config. This is the single easiest thing to get
>      wrong when adding batching on top of an existing gradient-
>      accumulation scheme -- see the loop below for exactly where it's
>      applied.
>
>   4. `forward_multi_t` (MultiScaleFlow / AccelFlow, in
>      model_oldRepo/flow_estimation.py and model_oldRepo/accel_flow.py)
>      now receives `timesteps` as a (B, T_max) tensor instead of a
>      python list of scalars, and internally builds a per-SAMPLE t
>      column per interior-frame slot instead of one t shared by the
>      whole call -- see those two files' revision notes. Their list-of-
>      python-floats calling convention (used by Trainer.py's
>      inference_multi_t, which is unaffected by any of this) still
>      works unchanged; the tensor path is purely additive.
>
> `batch_uid`, `window_id`, and `window_num_items` are no longer wrapped
> in a length-1 batch dimension by the (now-removed) implicit batch_size=1
> collate -- dataset.ragged_collate returns them as plain scalars/strings
> directly, so the `[0]` indexing that used to unwrap them is gone (see
> the loop below).

=== ACCUMULATION WINDOWS: FLUSH ON window_id, DIVIDE BY window_num_items ===
Each training step now yields one native-resolution MINIBATCH (B scenes,
same resolution, per the revision note above) per physical step, instead
of exactly one scene per step -- but the accumulation-window design
itself is completely unchanged, because curriculum_builder.py pre-groups
the whole curriculum into accumulation windows (group_batches_into_accum_windows)
BEFORE this file ever sees it: consecutive same-resolution, non-extra
batch folders are chunked into windows of exactly accum_steps(resolution)
batches ("full" windows), and any resolution run that ends before
reaching accum_steps becomes a shorter "partial" window instead of being
discarded -- partial windows are collected and appended to the end of
each phase's stream (after every full window), rather than interleaved
throughout it. Every batch in a window -- full or partial -- is stamped
with a shared window_id and the window's TRUE total item count
(window_num_items), both written into batch_metadata.yaml and surfaced
per minibatch by dataset.py (identically across every scene in one
batch_uid group, since they're stamped per BATCH FOLDER, not per scene).

This means train.py does NOT need to infer window boundaries from
anything (a resolution change, a step count, or batch_uid) -- window_id
is the direct, authoritative signal:

  - FLUSH the optimizer exactly when window_id changes (not when
    resolution changes, not when batch_uid changes, not on an item count
    reaching some target). window_id is guaranteed by
    curriculum_builder.py to change only at genuine accumulation-window
    boundaries, whether that's own->replay, replay->replay, or
    full-window->trailing-partial-window. A window can and often does
    span MULTIPLE steps of this loop, since one batch_uid group is one
    physical batch folder and a window is usually several consecutive
    batch folders (see curriculum_builder.py's docstring).
  - DIVIDE the accumulated loss by window_num_items, the window's true
    item count -- read directly off disk, not looked up from the
    dynamic_batch resolution table. This is exact for both full windows
    (where it equals batch_size(resolution) * accum_steps(resolution))
    and partial windows (where it's whatever smaller count actually
    landed in that window), so partial windows are trained with a
    correctly-scaled average loss instead of either being skipped or
    silently under/over-weighted.

batch_uid is still read per step and logged for traceability (which
physical BatchNNN_<res>/ folder a minibatch came from), but it never
gates anything -- see `window_id` / `cur_window_id` in the training loop
below. The dynamic_batch resolution lookup (resolution.resolve_dynamic_batch)
is still used for the per-step train_scale/logging context, but no
longer supplies the accumulation target or loss divisor.

=== MASKED, PER-ITEM LOSS ===
A minibatch's gt/timesteps are padded to T_max = max(num_interior) across
its B items, with a `valid_mask` (B, T_max) marking real (non-padding)
slots -- see dataset.ragged_collate. Given the model's per-sample-t
support (flow_from_bi / flow_from_bi_accel already broadcast against a
(B,1,H,W)-shaped t), forward_multi_t computes ALL B items' predictions
for ALL T_max slots in one batched call regardless of masking -- the
expensive part (feature backbone + BiHead/BiIFBlock stack, or
+ accel_head) still runs exactly once per (img0, img1) pair per step, as
intended by that architecture. Only the loss REDUCTION needs to respect
per-item validity, and it's done with an explicit python-level loop
over B (and, within that, over each item's own valid T) rather than a
single vectorized masked-mean, because LapLoss's own reduction behavior
(mean over an entire tensor including any batch dimension you hand it)
is opaque to this file -- looping per (b, i) and calling criterion on a
single (1, C, H, W) pair at a time is the only way to guarantee each
item's own per-frame losses are averaged before being averaged again
across the batch, exactly matching the pre-batching per-item invariant
(loss weight independent of how many interior frames an item has). This
loop is cheap: it's B*T_max scalar reductions over already-computed
tensors, not B*T_max forward passes through the network.

=== VALIDATION ===
val.validation_native_mode=true (default): native resolution, batch
size 1, no bucketing -- appropriate for a val set of mixed resolutions.
val is NOT affected by any part of this revision -- make_val_loader below
is unchanged, still a plain DataLoader over val_set with no batch_uid
grouping or ragged collation.
val.validation_native_mode=false: val is treated like train's dynamic
path is NOT used; instead a flat val.validation_batch_size is used with
default collate, which requires every val item to already share a common
shape (i.e. a val set pre-normalized to one resolution).

=== PER-EPOCH SEEDED TIMESTEP SELECTION ===
train_set.set_epoch(epoch) is called at the start of every epoch below.
dataset.py derives each item's interior-frame/timestep pick from a
dedicated RNG keyed on (seed, epoch, sequence folder) -- see dataset.py's
module docstring. train_loader is built with persistent_workers=False
(the default, made explicit here) specifically so that DataLoader worker
processes are re-spawned -- and therefore re-pickle the Dataset object
with its just-updated self.epoch -- at the start of every epoch's
iteration. If persistent_workers were ever turned on, already-running
workers would keep the Dataset snapshot from the FIRST epoch and never
see later set_epoch() calls, silently breaking per-epoch variation. This
is unaffected by BatchUidGroupedSampler -- it only decides which indices
get grouped into a step, not how/when Dataset attributes propagate to
workers.

=== ACCEL LOSS AND `local` CONSISTENCY ===
When model.net has an accel_head (AccelFlow), accel_distillation_loss is
called with local=local (the SAME local-refinement flag used for this
step's main forward pass), not a hardcoded value, for every (b, i) pair
in the masked loop described above. accel_head is a single shared
module: it's called once inside the main forward (on a D that went
through local refinement whenever `local` is truthy) and once more inside
accel_distillation_loss (on its own separately-computed D). If those two
calls used different `local` settings, accel_head would be trained
against a different input distribution than the one it actually sees at
inference -- passing the same `local` through keeps them consistent. The
accel loss's TEACHER pass (real flow to the real middle frame) always
uses full local refinement internally regardless, since it's a no_grad
target computation and a better-refined target is strictly better
supervision.

NOTE ON ACCEL-LOSS COST: accel_distillation_loss recomputes
estimate_bi_flow (student pass) AND a second no_grad estimate_bi_flow
(teacher pass) from scratch for EVERY (b, i) pair in the masked loop,
since it's called once per valid interior frame of every item in the
batch. Unlike the photometric branch -- which gets its one-backbone-pass
amortization via forward_multi_t across the WHOLE minibatch -- the accel
branch pays for estimate_bi_flow 2x per (b, i) pair, now on
single-item (1, C, H, W) slices rather than the full batch, since the
teacher pass can't be shared across interior frames anyway (it depends
on gt[b, i], a different privileged frame per (b, i)) and this file has
no way to batch a variable-length, per-item set of teacher targets
without either padding structure the teacher call doesn't need or
another custom collate step. This is a real inefficiency, not a
correctness bug, exactly as it was before this revision -- left as-is
here since it's a throughput concern, not a correctness one.
"""
import argparse
import math
import os
import shutil
import sys
import time
from collections import Counter

import torch
import yaml

import config_loader
import resolution
from Trainer import Model, convert  # noqa: E402
from model_oldRepo import warplayer
from model_oldRepo.loss import LapLoss
from model_oldRepo.accel_flow import accel_distillation_loss  # used below when model.net has accel_head
import configCustom
sys.modules['config'] = configCustom  # Trainer.py does `from config import *`
from configCustom import MODEL_CONFIG  # noqa: E402

from dataset import prepare_datasets, worker_init_fn, BatchUidGroupedSampler, ragged_collate
from seeding import seed_everything


def lr_schedule(optimizer, global_step, steps_per_epoch, total_epochs,
                 warmup_epochs, base_lr, min_lr):
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    total_steps = int(total_epochs * steps_per_epoch)
    if global_step < warmup_steps:
        lr = base_lr * (global_step + 1) / warmup_steps
    else:
        e = global_step - warmup_steps
        total = max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * e / total))
        lr = min_lr + (base_lr - min_lr) * cos
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def format_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f'{h}h{m:02d}m{s:02d}s'
    if m > 0:
        return f'{m}m{s:02d}s'
    return f'{s}s'


def gpu_peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def load_pretrained_with_shape_check(net, ckpt_path, log):
    state = torch.load(ckpt_path, map_location='cuda')
    is_ddp_ckpt = any('module.' in k for k in state.keys())
    state = convert(state) if is_ddp_ckpt else state

    model_state = net.state_dict()
    dropped, filtered = [], {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape != v.shape:
            dropped.append(f'{k}: ckpt{tuple(v.shape)} vs model{tuple(model_state[k].shape)}')
        else:
            filtered[k] = v

    missing, unexpected = net.load_state_dict(filtered, strict=False)
    total_keys = len(model_state)
    log(f'  shape-mismatched keys (dropped): {len(dropped)}')
    for d in dropped[:10]:
        log(f'    {d}')
    if len(dropped) > 10:
        log(f'    ... and {len(dropped) - 10} more')
    log(f'  missing keys: {len(missing)} / {total_keys} | unexpected keys: {len(unexpected)}')

    bad = len(dropped) + len(missing)
    if bad > 0.2 * total_keys or len(unexpected) > 0.2 * total_keys:
        log('  WARNING: a large fraction of keys did not line up -- check that '
            'configCustom.MODEL_ARCH matches the architecture this checkpoint was trained with.')
    return dropped, missing, unexpected


def make_train_loader(train_set):
    """Real-minibatch train loader: groups by physical batch_uid (one
    BatchNNN_<res>/ folder -> one single-resolution minibatch of size
    equal to that folder's own scene count) and ragged-collates each
    group's variable interior-frame counts into a padded
    (B, T_max, ...) tensor + valid_mask. See dataset.py's module
    docstring and BatchUidGroupedSampler/ragged_collate for the full
    rationale. shuffle is intentionally absent -- BatchUidGroupedSampler
    itself is the batch_sampler and already yields groups in curriculum
    order; see that class's docstring for why shuffling here would be
    wrong.
    """
    sampler = BatchUidGroupedSampler(train_set)
    return torch.utils.data.DataLoader(
        train_set, batch_sampler=sampler, collate_fn=ragged_collate,
        num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
        multiprocessing_context='fork', persistent_workers=False)
        # persistent_workers=False (explicit, matches the default) so that workers
        # are re-spawned -- and re-pickle train_set, picking up the just-updated
        # self.epoch -- at the start of every epoch. See dataset.py's set_epoch()
        # for why this matters. Unaffected by the switch to batch_sampler.


def make_val_loader(cfg, val_set):
    val_cfg = cfg['val']
    native = val_cfg.get('validation_native_mode', True)
    batch_size = 1 if native else val_cfg.get('validation_batch_size', 1)
    return torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, worker_init_fn=worker_init_fn), native


def main(cfg, run_cfg, phase, restore_ckpt=None):
    torch.backends.cudnn.benchmark = False
    # NOTE: kept False, exactly as before this revision. Even though every
    # minibatch is now single-resolution (see BatchUidGroupedSampler),
    # consecutive STEPS can still change resolution/B (replay interleaving),
    # so cudnn still can't usefully autotune across the whole run. If you
    # want cudnn.benchmark=True's speedup, it would need to be scoped to
    # "steps sharing this exact (B, H, W)", not left as a single global flag
    # -- not attempted here.

    seed_everything(cfg['seed'])  # THE single global seed for this whole run

    r_cfg = run_cfg['run']
    d_cfg = run_cfg.get('data', {})
    t_cfg = run_cfg['train']
    e_cfg = run_cfg.get('eval', {})

    PRETRAINED = restore_ckpt or d_cfg.get('pretrained') or ''
    RUN_NAME = r_cfg['run_name']
    RUNS_DIR = r_cfg['runs_dir']

    EPOCHS = t_cfg['epochs']
    LR = float(t_cfg['lr'])
    MIN_LR = float(t_cfg.get('min_lr', 1e-6))
    WARMUP_EPOCHS = t_cfg.get('warmup_epochs', 1)
    WEIGHT_DECAY = float(t_cfg.get('weight_decay', 1e-4))
    GRAD_CLIP = t_cfg.get('grad_clip', 1.0)
    AMP = t_cfg.get('amp', True)
    W_ACCEL = float(t_cfg.get('w_accel', 1.0))

    PAD_MULTIPLE = cfg['pad_multiple']
    DYNAMIC_BATCH_THRESHOLDS = cfg['dynamic_batch']['thresholds']
    TRAIN_SCALE_ANCHORS = cfg['train_scale']['anchors']

    EVAL_FULL_RES_EVERY = e_cfg.get('eval_full_res_every', 5)
    EVAL_FULL_RES_MAX_SAMPLES = e_cfg.get('eval_full_res_max_samples', None)
    SAVE_EVERY_EPOCH = e_cfg.get('save_every_epoch', True)
    LOG_EVERY_STEPS = e_cfg.get('log_every_steps', 10)

    run_dir = os.path.join(RUNS_DIR, RUN_NAME)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'train.log')
    log_file = open(log_path, 'a')

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line)
        log_file.write(line + '\n')
        log_file.flush()

    log(f'===== starting run "{RUN_NAME}" (Phase{phase}) =====')
    log(f'settings: epochs={EPOCHS} seed={cfg["seed"]} lr={LR} min_lr={MIN_LR} '
        f'warmup_epochs={WARMUP_EPOCHS} amp={AMP} pretrained={PRETRAINED}')
    log(f'  cuda available: {torch.cuda.is_available()}'
        + (f' | device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''))

    # ---- datasets: read-only over the curriculum built by curriculum_builder.py ----
    train_set, val_set = prepare_datasets(cfg, phase, log=log)
    log(f'train items: {len(train_set)} | val items: {len(val_set)}')

    train_loader = make_train_loader(train_set)
    log(f'  train batches (batch_uid groups) per epoch: {len(train_loader)}')
    val_loader, val_native = make_val_loader(cfg, val_set)
    log(f'  validation_native_mode={val_native}')

    model = Model(-1)  # local_rank=-1 -> single GPU, no DDP wrapper

    if PRETRAINED and os.path.exists(PRETRAINED):
        log(f'warm-starting from {PRETRAINED}')
        load_pretrained_with_shape_check(model.net, PRETRAINED, log)
    else:
        log('no pretrained checkpoint given/found -- training from scratch')

    optimizer = torch.optim.AdamW(model.net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    local = model.local
    criterion = LapLoss(max_levels=5, channels=3)

    best_val = float('inf')
    steps_per_epoch = len(train_loader)
    total_steps_all = EPOCHS * steps_per_epoch
    train_start = time.time()
    global_step = 0
    history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(EPOCHS):
        warplayer.backwarp_tenGrid.clear()
        train_set.set_epoch(epoch)  # re-seeds this epoch's interior-frame/timestep picks -- see dataset.py
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        steps_since_update = 0    # counts ITEMS (scenes) now, not steps -- see revision note above
        cur_effective_batch = None  # locked in at the start of each accumulation window (see below)
        cur_window_id = None        # the accumulation window the CURRENT run of items belongs to
        n_steps = steps_per_epoch
        cur_lr = optimizer.param_groups[0]['lr']
        epoch_res_counts = Counter()
        epoch_partial_windows_seen = set()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        log(f'-- epoch {epoch+1}/{EPOCHS} starting | lr {cur_lr:.2e} | {n_steps} steps/epoch')

        for step, (img0, gt, img1, timesteps, valid_mask, h, w,
                   batch_uid, window_id, window_num_items) in enumerate(train_loader):
            # batch_uid / window_id / window_num_items are plain scalars now
            # (ragged_collate returns them directly) -- no [0] unwrapping
            # needed, unlike the pre-batching batch_size=1 default collate.
            window_num_items = int(window_num_items)

            B = img0.shape[0]
            img0 = img0.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)              # (B, T_max, C, H, W)
            img1 = img1.cuda(non_blocking=True)
            timesteps = timesteps.cuda(non_blocking=True)  # (B, T_max)
            valid_mask_cpu = valid_mask                     # keep a CPU copy for cheap python-side counting
            imgs = torch.cat((img0, img1), 1)

            # Resolution is identical across every item in this minibatch by
            # construction (single batch_uid == single resolution) -- take
            # item 0's h/w as representative for logging/scale lookups.
            h0, w0 = int(h[0].item()), int(w[0].item())
            long_edge = max(h0, w0)
            epoch_res_counts[(w0, h0)] += B

            batch_size_here, accum_here = resolution.resolve_dynamic_batch(
                w0, h0, DYNAMIC_BATCH_THRESHOLDS, PAD_MULTIPLE)
            item_effective_batch = batch_size_here * accum_here

            if steps_since_update > 0 and window_id != cur_window_id:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0
                cur_effective_batch = None
                cur_window_id = None

            if steps_since_update == 0:
                cur_effective_batch = window_num_items
                cur_window_id = window_id
                if window_num_items < item_effective_batch:
                    epoch_partial_windows_seen.add(window_id)

            step_scale = resolution.resolve_train_scale(w0, h0, TRAIN_SCALE_ANCHORS, PAD_MULTIPLE)
            cur_lr = lr_schedule(optimizer, global_step, steps_per_epoch, EPOCHS, WARMUP_EPOCHS, LR, MIN_LR)

            with torch.cuda.amp.autocast(enabled=AMP):
                # Backbone (+ scale downsample/upsample bookkeeping) runs ONCE
                # for this minibatch's img0/img1 pairs, regardless of B or
                # T_max -- this is the whole point of the doc2/doc4
                # architecture, now extended across the batch dimension too.
                # Requires model.net to be a MultiScaleFlow/AccelFlow-style
                # model (has forward_multi_t); this training loop is no
                # longer compatible with the original per-timestep-regression
                # architecture. `timesteps` is (B, T_max) -- see this
                # module's revision note and flow_estimation.py /
                # accel_flow.py's own revision notes for how forward_multi_t
                # now builds a per-sample t column per slot.
                preds = model.net.forward_multi_t(imgs, timesteps, local=local, scale=step_scale)
                # preds: list of T_max tensors, each (B, C, H, W)

                # ---- masked, per-item photometric loss ----
                # See this module's "MASKED, PER-ITEM LOSS" docstring section
                # for why this loop is not further vectorized: LapLoss's own
                # batch-reduction behavior is opaque here, so each item's own
                # valid interior frames are averaged FIRST (matching the
                # pre-batching per-item invariant), and only THEN averaged
                # across the B items in this step.
                item_losses = []
                for b in range(B):
                    n_valid = int(valid_mask_cpu[b].sum().item())
                    if n_valid == 0:
                        continue  # defensive; every real item has >=1 interior frame by construction
                    frame_losses_b = [criterion(preds[i][b:b + 1], gt[b:b + 1, i]) for i in range(n_valid)]
                    item_losses.append(torch.stack(frame_losses_b).mean())
                mean_item_loss = torch.stack(item_losses).mean()

                if hasattr(model.net, 'accel_head'):
                    accel_item_losses = []
                    for b in range(B):
                        n_valid = int(valid_mask_cpu[b].sum().item())
                        if n_valid == 0:
                            continue
                        accel_losses_b = [
                            accel_distillation_loss(
                                model.net, img0[b:b + 1], gt[b:b + 1, i], img1[b:b + 1],
                                float(timesteps[b, i].item()), local=local)
                            for i in range(n_valid)
                        ]
                        accel_item_losses.append(torch.stack(accel_losses_b).mean())
                    mean_item_loss = mean_item_loss + W_ACCEL * torch.stack(accel_item_losses).mean()

                # `mean_item_loss` is a MEAN over this step's B items. Scale
                # back up by B before dividing by cur_effective_batch so this
                # step's gradient contribution equals the SUM of what B
                # individual gradient-accumulation steps would have
                # contributed (each divided by cur_effective_batch) -- see
                # this module's revision note, point 3, for why omitting
                # `* B` would silently shrink the effective learning rate
                # any time B > 1.
                scaled_loss = mean_item_loss * B / cur_effective_batch

            scaler.scale(scaled_loss).backward()
            steps_since_update += B

            is_last_step_of_epoch = (step + 1) == n_steps
            # window_num_items is exact by construction (curriculum_builder.py
            # guarantees every window is fully self-contained), so this count
            # check and the window_id check above should always agree in
            # practice -- kept as a defensive fallback for is_last_step_of_epoch
            # and for robustness against any unexpected data irregularity.
            triggered_update = steps_since_update >= cur_effective_batch or is_last_step_of_epoch
            if triggered_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # step_loss: this step's mean per-item loss, BEFORE the *B/
            # cur_effective_batch rescaling used for backward -- i.e. the
            # batched analogue of the pre-batching single-item `loss.item()
            # * cur_effective_batch` (which recovered that single item's
            # unscaled loss). running_loss/n_steps below is therefore an
            # average PER STEP (per batch_uid group), not per scene --
            # a mild semantic shift from before this revision, since a step
            # can now contain more than one scene; still directly comparable
            # across epochs of the SAME run/config.
            step_loss = mean_item_loss.item()
            running_loss += step_loss
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0 or is_last_step_of_epoch:
                elapsed_total = time.time() - train_start
                avg_step_time = elapsed_total / global_step
                eta = avg_step_time * (total_steps_all - global_step)
                running_avg_loss = running_loss / (step + 1)
                valid_ts = timesteps[valid_mask.to(timesteps.device)]
                if valid_ts.numel() > 0:
                    t_lo, t_hi = valid_ts.min().item(), valid_ts.max().item()
                else:
                    t_lo, t_hi = 0.0, 0.0
                log(f'  epoch {epoch+1}/{EPOCHS} | step {step+1}/{n_steps} '
                    f'(global {global_step}/{total_steps_all}) | res {w0}x{h0} (long edge {long_edge}px) | '
                    f'B={B} | loss {step_loss:.4f} | avg_loss {running_avg_loss:.4f} | lr {cur_lr:.2e} | '
                    f't_range [{t_lo:.3f},{t_hi:.3f}] | scale {step_scale:.3f} | '
                    f'window={window_id} window_items={cur_effective_batch}/{item_effective_batch} '
                    f'({100*cur_effective_batch/item_effective_batch:.0f}% full) | '
                    f'src_batch={os.path.basename(batch_uid)} | '
                    f'gpu_peak {gpu_peak_gb():.2f}GB | elapsed {format_time(elapsed_total)} | ETA {format_time(eta)}')

            # Reset the window AFTER logging, so this step's log line still
            # shows the window target that was actually used for this step.
            if triggered_update:
                steps_since_update = 0
                cur_effective_batch = None
                cur_window_id = None

        train_loss = running_loss / max(1, n_steps)
        train_epoch_gpu_peak = gpu_peak_gb()

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        want_psnr = (epoch + 1) % EVAL_FULL_RES_EVERY == 0 or (epoch + 1) == EPOCHS
        psnr_total, psnr_count = 0.0, 0
        with torch.no_grad():
            for i, (img0, gt, img1, timestep, h, w) in enumerate(val_loader):
                img0 = img0.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                img1 = img1.cuda(non_blocking=True)
                imgs = torch.cat((img0, img1), 1)

                # native mode: batch_size=1, so h/w/timestep are per-sample scalars.
                # non-native (bucketed) mode: batch_size can be > 1 -- use the first
                # sample's h/w/timestep for the (assumed shared) scale/timestep, and
                # crop/PSNR per-sample below regardless of batch size.
                b = imgs.shape[0]
                step_timestep = float(timestep[0].item()) if b > 1 else float(timestep.item())
                h0 = int(h[0].item()) if b > 1 else int(h.item())
                w0 = int(w.item()) if b == 1 else int(w[0].item())
                step_scale = resolution.resolve_train_scale(w0, h0, TRAIN_SCALE_ANCHORS, PAD_MULTIPLE)

                _, _, _, pred = model.net(imgs, timestep=step_timestep, scale=step_scale, local=local)
                val_loss += criterion(pred, gt).item()

                do_psnr_this_sample = want_psnr and (
                    EVAL_FULL_RES_MAX_SAMPLES is None or i < EVAL_FULL_RES_MAX_SAMPLES)
                if do_psnr_this_sample:
                    for bi in range(b):
                        hi = int(h[bi].item()) if b > 1 else h0
                        wi = int(w[bi].item()) if b > 1 else w0
                        pred_cropped = pred[bi:bi + 1, :, :hi, :wi]
                        gt_cropped = gt[bi:bi + 1, :, :hi, :wi]
                        psnr_total += psnr(pred_cropped, gt_cropped)
                        psnr_count += 1

        val_loss /= max(1, len(val_loader))
        full_psnr_this_epoch = (psnr_total / psnr_count) if psnr_count else None
        epoch_gpu_peak = gpu_peak_gb()

        elapsed = time.time() - t0
        msg = (f'epoch {epoch+1:03d}/{EPOCHS} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | '
               f'lr {cur_lr:.2e} | {elapsed:.1f}s | gpu_peak {epoch_gpu_peak:.2f}GB '
               f'(train-only: {train_epoch_gpu_peak:.2f}GB)')
        if full_psnr_this_epoch is not None:
            msg += f' | full-res PSNR {full_psnr_this_epoch:.2f}dB (n={psnr_count})'
        log(msg)

        train_res_summary = ', '.join(
            f'{w}x{h}:{n}' for (w, h), n in sorted(epoch_res_counts.items(), key=lambda kv: -kv[1]))
        log(f'  epoch {epoch+1} train resolutions seen (scene count): {train_res_summary}')
        log(f'  epoch {epoch+1} partial (undersized) accumulation windows trained: '
            f'{len(epoch_partial_windows_seen)}')

        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                         'lr': cur_lr, 'full_psnr': full_psnr_this_epoch, 'gpu_peak_gb': epoch_gpu_peak})

        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_epoch{epoch+1:03d}_valloss{val_loss:.4f}.pkl')
            torch.save(model.net.state_dict(), ckpt_path)
            log(f'  saved checkpoint: {ckpt_path}')

        last_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_last.pkl')
        torch.save(model.net.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_best.pkl')
            torch.save(model.net.state_dict(), best_path)
            log(f'  -> new best (val_loss {best_val:.4f}), saved to {best_path}')

    # ---- final summary ----
    best_ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_best.pkl')
    last_ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_last.pkl')
    summary_path = os.path.join(run_dir, 'summary.txt')

    if history:
        total_elapsed = time.time() - train_start
        best_entry = min(history, key=lambda hh: hh['val_loss'])
        final_entry = history[-1]
        avg_train_loss = sum(hh['train_loss'] for hh in history) / len(history)
        avg_val_loss = sum(hh['val_loss'] for hh in history) / len(history)
        peak_gpu_overall = max(hh['gpu_peak_gb'] for hh in history)
        first_val_loss = history[0]['val_loss']
        change_pct = ((best_entry['val_loss'] - first_val_loss) / first_val_loss * 100
                      if first_val_loss > 0 else 0.0)

        lines = [
            '=' * 62,
            f' TRAINING RUN SUMMARY: {RUN_NAME} (Phase{phase})',
            '=' * 62,
            f' Epochs completed:      {len(history)} / {EPOCHS}',
            f' Total wall time:       {format_time(total_elapsed)}',
            f' Pretrained warm-start: {PRETRAINED if PRETRAINED else "(trained from scratch)"}',
            f' Peak GPU memory:       {peak_gpu_overall:.2f} GB',
            f' Train / val items:     {len(train_set)} / {len(val_set)}',
            '',
            f' BEST EPOCH: {best_entry["epoch"]}',
            f'   val_loss:            {best_entry["val_loss"]:.4f}',
            f'   train_loss:          {best_entry["train_loss"]:.4f}',
            f'   checkpoint:          {best_ckpt_path}',
            '',
            f' FINAL EPOCH ({final_entry["epoch"]}):',
            f'   train_loss:          {final_entry["train_loss"]:.4f}',
            f'   val_loss:            {final_entry["val_loss"]:.4f}',
            '',
            ' AVERAGES ACROSS RUN:',
            f'   avg train_loss:      {avg_train_loss:.4f}',
            f'   avg val_loss:        {avg_val_loss:.4f}',
            '',
            ' IMPROVEMENT (val_loss, epoch 1 -> best):',
            f'   {first_val_loss:.4f} -> {best_entry["val_loss"]:.4f}  ({change_pct:+.1f}%)',
            '=' * 62,
            f' Full log:        {log_path}',
            f' Best checkpoint: {best_ckpt_path}',
            f' Last checkpoint: {last_ckpt_path}',
            '=' * 62,
        ]
        summary_text = '\n'.join(lines)
        with open(summary_path, 'w') as f:
            f.write(summary_text + '\n')
        print('\n' + summary_text + '\n')
        log_file.write('\n' + summary_text + '\n')
        log_file.flush()
    else:
        log('No epochs were completed -- skipping final summary.')

    log(f'===== run "{RUN_NAME}" finished =====')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune VFIMamba on one phase of the curriculum')
    parser.add_argument('--config', type=str, default=None, help='path to config.yaml')
    parser.add_argument('--run', required=True, type=str, help='path to this run\'s yaml (hyperparams, run_name)')
    parser.add_argument('--phase', type=int, default=None, choices=[1, 2, 3, 4],
                         help='which curriculum phase to train on (overrides run yaml\'s run.phase)')
    parser.add_argument('--restore_ckpt', type=str, default=None,
                         help='optional override for data.pretrained in the run yaml')
    args = parser.parse_args()

    cfg = config_loader.load_config(args.config)
    with open(args.run) as f:
        run_cfg = yaml.safe_load(f)

    phase = args.phase or run_cfg['run'].get('phase')
    if phase is None:
        raise SystemExit('phase must be given via --phase or run.phase in the run yaml')

    main(cfg, run_cfg, phase, restore_ckpt=args.restore_ckpt)