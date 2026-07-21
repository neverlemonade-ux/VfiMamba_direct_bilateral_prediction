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

=== ACCUMULATION WINDOWS: FLUSH ON window_id, DIVIDE BY window_num_items ===
DataLoader yields one native-resolution sequence per physical step (see
dataset.py's module docstring for why items aren't stacked into batch >
1). curriculum_builder.py pre-groups the whole curriculum into
accumulation windows (group_batches_into_accum_windows) BEFORE this file
ever sees it: consecutive same-resolution, non-extra batch folders are
chunked into windows of exactly accum_steps(resolution) batches ("full"
windows), and any resolution run that ends before reaching accum_steps
becomes a shorter "partial" window instead of being discarded -- partial
windows are collected and appended to the end of each phase's stream
(after every full window), rather than interleaved throughout it. Every
batch in a window -- full or partial -- is stamped with a shared
window_id and the window's TRUE total item count (window_num_items), both
written into batch_metadata.yaml and surfaced per item by dataset.py.

This means train.py does NOT need to infer window boundaries from
anything (a resolution change, a step count, or batch_uid) -- window_id
is the direct, authoritative signal:

  - FLUSH the optimizer exactly when window_id changes (not when
    resolution changes, not when batch_uid changes, not on a step count
    reaching some target). window_id is guaranteed by
    curriculum_builder.py to change only at genuine accumulation-window
    boundaries, whether that's own->replay, replay->replay, or
    full-window->trailing-partial-window.
  - DIVIDE the accumulated loss by window_num_items, the window's true
    item count -- read directly off disk, not looked up from the
    dynamic_batch resolution table. This is exact for both full windows
    (where it equals batch_size(resolution) * accum_steps(resolution))
    and partial windows (where it's whatever smaller count actually
    landed in that window), so partial windows are trained with a
    correctly-scaled average loss instead of either being skipped or
    silently under/over-weighted.

batch_uid is still read per item and logged for traceability (which
physical BatchNNN_<res>/ folder an item came from), but it never gates
anything -- see `window_id` / `cur_window_id` in the training loop below.
The dynamic_batch resolution lookup (resolution.resolve_dynamic_batch) is
still used for the per-item train_scale/logging context, but no longer
supplies the accumulation target or loss divisor.

=== VALIDATION ===
val.validation_native_mode=true (default): native resolution, batch
size 1, no bucketing -- appropriate for a val set of mixed resolutions.
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
see later set_epoch() calls, silently breaking per-epoch variation.

=== ACCEL LOSS AND `local` CONSISTENCY ===
When model.net has an accel_head (AccelFlow), accel_distillation_loss is
called with local=local (the SAME local-refinement flag used for this
step's main forward pass), not a hardcoded value. accel_head is a single
shared module: it's called once inside the main forward (on a D that went
through local refinement whenever `local` is truthy) and once more inside
accel_distillation_loss (on its own separately-computed D). If those two
calls used different `local` settings, accel_head would be trained
against a different input distribution than the one it actually sees at
inference -- passing the same `local` through keeps them consistent. The
accel loss's TEACHER pass (real flow to the real middle frame) always
uses full local refinement internally regardless, since it's a no_grad
target computation and a better-refined target is strictly better
supervision.
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
import configCustom
sys.modules['config'] = configCustom  # Trainer.py does `from config import *`
from configCustom import MODEL_CONFIG  # noqa: E402

from dataset import prepare_datasets, worker_init_fn
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


def make_val_loader(cfg, val_set):
    val_cfg = cfg['val']
    native = val_cfg.get('validation_native_mode', True)
    batch_size = 1 if native else val_cfg.get('validation_batch_size', 1)
    return torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True, worker_init_fn=worker_init_fn), native


def main(cfg, run_cfg, phase, restore_ckpt=None):
    torch.backends.cudnn.benchmark = False

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

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn, multiprocessing_context='fork',
        persistent_workers=False)
        # shuffle=False is deliberate: curriculum_builder.py's on-disk order IS the
        # intended training order (own progression + interleaved replay, with
        # trailing partial accumulation windows appended at the end); shuffling
        # here would undo that.
        # persistent_workers=False (explicit, matches the default) so that workers
        # are re-spawned -- and re-pickle train_set, picking up the just-updated
        # self.epoch -- at the start of every epoch. See this module's docstring
        # and dataset.py's set_epoch() for why this matters.
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
        steps_since_update = 0
        cur_effective_batch = None  # locked in at the start of each accumulation window (see below)
        cur_window_id = None        # the accumulation window the CURRENT run of items belongs to
        n_steps = steps_per_epoch
        cur_lr = optimizer.param_groups[0]['lr']
        epoch_res_counts = Counter()
        epoch_partial_windows_seen = set()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        log(f'-- epoch {epoch+1}/{EPOCHS} starting | lr {cur_lr:.2e} | {n_steps} steps/epoch')

        for step, (img0, gt, img1, timestep, h, w, batch_uid, window_id, window_num_items) in enumerate(train_loader):
            batch_uid = batch_uid[0]  # DataLoader wraps single strings/ints in a list at batch_size=1
            window_id = window_id[0]
            window_num_items = int(window_num_items.item())

            img0 = img0.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            img1 = img1.cuda(non_blocking=True)
            imgs = torch.cat((img0, img1), 1)

            step_timestep = float(timestep.item())
            h, w = int(h.item()), int(w.item())
            long_edge = max(h, w)
            epoch_res_counts[(w, h)] += 1

            # Still resolved per-item -- used for train_scale and for the
            # log line's "what a full window here would look like"
            # context, but no longer the source of the accumulation
            # target or the loss divisor (window_num_items, read straight
            # off disk via dataset.py, is exact for both full and
            # partial windows -- see this module's docstring).
            batch_size_here, accum_here = resolution.resolve_dynamic_batch(
                w, h, DYNAMIC_BATCH_THRESHOLDS, PAD_MULTIPLE)
            item_effective_batch = batch_size_here * accum_here

            # window_id is the direct, authoritative boundary signal --
            # curriculum_builder.py guarantees it changes exactly at
            # accumulation-window boundaries (own->replay, replay->replay,
            # full-window->trailing-partial-window) and never inside one.
            # Flush FIRST, using whatever accumulated so far.
            if steps_since_update > 0 and window_id != cur_window_id:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0
                cur_effective_batch = None
                cur_window_id = None

            # Lock the window's divisor to its TRUE item count on the
            # first item of a new window -- exact for both full windows
            # (equals batch_size(resolution)*accum_steps(resolution)) and
            # partial windows (whatever smaller count actually landed
            # there), so no separate "ratio" calculation is needed.
            if steps_since_update == 0:
                cur_effective_batch = window_num_items
                cur_window_id = window_id
                if window_num_items < item_effective_batch:
                    epoch_partial_windows_seen.add(window_id)

            step_scale = resolution.resolve_train_scale(w, h, TRAIN_SCALE_ANCHORS, PAD_MULTIPLE)

            cur_lr = lr_schedule(optimizer, global_step, steps_per_epoch,
                                  EPOCHS, WARMUP_EPOCHS, LR, MIN_LR)

            with torch.cuda.amp.autocast(enabled=AMP):
                _, _, _, pred = model.net(imgs, timestep=step_timestep, scale=step_scale, local=local)
                loss = criterion(pred, gt)
                if hasattr(model.net, 'accel_head'):  # AccelFlow only; no-op for plain flow_estimation.py
                    from model_oldRepo.accel_flow import accel_distillation_loss
                    loss = loss + W_ACCEL * accel_distillation_loss(
                        model.net, img0, gt, img1, step_timestep, local=local)
                    # local=local (NOT hardcoded): accel_head is a shared module
                    # also invoked inside model.net's main forward above using this
                    # same `local` setting -- passing it through here keeps
                    # accel_head's student-side D consistent between the
                    # photometric branch and the accel-supervision branch, instead
                    # of silently training it against a different (unrefined)
                    # input distribution than it will see at inference.
                loss = loss / cur_effective_batch

            scaler.scale(loss).backward()
            steps_since_update += 1

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

            step_loss = loss.item() * cur_effective_batch
            running_loss += step_loss
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0 or is_last_step_of_epoch:
                elapsed_total = time.time() - train_start
                avg_step_time = elapsed_total / global_step
                eta = avg_step_time * (total_steps_all - global_step)
                running_avg_loss = running_loss / (step + 1)
                log(f'  epoch {epoch+1}/{EPOCHS} | step {step+1}/{n_steps} '
                    f'(global {global_step}/{total_steps_all}) | res {w}x{h} (long edge {long_edge}px) | '
                    f'loss {step_loss:.4f} | avg_loss {running_avg_loss:.4f} | lr {cur_lr:.2e} | '
                    f't {step_timestep:.3f} | scale {step_scale:.3f} | '
                    f'window={window_id} window_items={cur_effective_batch}/{item_effective_batch} '
                    f'({100*cur_effective_batch/item_effective_batch:.0f}% full) | '
                    f'src_batch={os.path.basename(batch_uid)} | '
                    f'gpu_peak {gpu_peak_gb():.2f}GB | elapsed {format_time(elapsed_total)} | ETA {format_time(eta)}')

            # Reset the window AFTER logging, so this step's log line still
            # shows the window target that was actually used for this item.
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
        log(f'  epoch {epoch+1} train resolutions seen: {train_res_summary}')
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