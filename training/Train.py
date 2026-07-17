"""
Fine-tune VFIMamba (large variant) on a custom multi-frame dataset,
single-GPU (one RTX 3090).

All settings live in a YAML config file. Usage:
    python train.py --config trainingSetting.yaml
    python train.py --config trainingSetting.yaml --restore_ckpt ckpt/VFIMamba.pkl

=== DATA HANDLING HAS MOVED TO dataset.py ===
Everything about how sequences are discovered, split into train/val,
augmented, given a dynamic per-item timestep, given a resolution-aware
flow-estimation scale, and padded to a multiple of 32 now lives in
dataset.py. This file calls dataset.prepare_datasets() once and receives
two Datasets that already yield fully-prepared
(img0, gt, img1, timestep, scale, orig_h, orig_w) tuples -- it never reads
a frame off disk, touches a random seed, or pads a tensor itself. If you
want to change how data is prepared, edit dataset.py; if you want to
change how the model is trained on that data, edit this file.

Everything else (per-step warmup+cosine LR, checkpoint loading with shape
checks, batch size pinned to 1 physical / N accumulated since full-res
sequences can't be stacked, best/last checkpoint tracking, full-res PSNR
folded into the regular validation loop) is unchanged from before -- see
inline comments below.
"""
import argparse
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
import yaml

import configCustom
sys.modules['config'] = configCustom  # Trainer.py does `from config import *` -- point it here

from Trainer import Model, convert  # noqa: E402  (must come after the sys.modules patch above)
from model_oldRepo import warplayer
from configCustom import MODEL_CONFIG
from model_oldRepo.loss import LapLoss  # the repo's real loss

from dataset import prepare_datasets, worker_init_fn


def load_config(path):
    with open(path) as f:
        C = yaml.safe_load(f)
    C['_source_path'] = path
    return C


def lr_schedule(optimizer, global_step, steps_per_epoch, total_epochs,
                 warmup_epochs, base_lr, min_lr):
    """
    Linear warmup + cosine decay, computed PER OPTIMIZER STEP -- warmup
    ramps smoothly from ~base_lr/warmup_steps up to base_lr across the
    first `warmup_epochs * steps_per_epoch` steps, then cosine-decays to
    min_lr. Must be called once per training step (passing the running
    global_step), not once per epoch, or a single warmup_epoch reaches
    peak LR on the very first step with no ramp at all.
    """
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


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def load_pretrained_with_shape_check(net, ckpt_path, log):
    """
    Warm-start `net` from `ckpt_path`, dropping any key whose checkpoint
    tensor shape doesn't match the current model's shape instead of
    letting load_state_dict raise. strict=False alone only tolerates
    missing/unexpected KEY NAMES, not a shape mismatch on a key present in
    both -- that still raises regardless of strict. This filters out
    shape-mismatched keys manually first, so the rest of the checkpoint
    still applies, and logs exactly what was dropped/missing/unexpected.
    """
    state = torch.load(ckpt_path, map_location='cuda')
    is_ddp_ckpt = any('module.' in k for k in state.keys())
    state = convert(state) if is_ddp_ckpt else state

    model_state = net.state_dict()
    dropped = []
    filtered = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape != v.shape:
            dropped.append(f'{k}: ckpt{tuple(v.shape)} vs model{tuple(model_state[k].shape)}')
        else:
            filtered[k] = v

    missing, unexpected = net.load_state_dict(filtered, strict=False)
    total_keys = len(model_state)

    log(f'  shape-mismatched keys (dropped, left at random/pretrained init): {len(dropped)}')
    for d in dropped[:10]:
        log(f'    {d}')
    if len(dropped) > 10:
        log(f'    ... and {len(dropped) - 10} more')
    log(f'  missing keys (absent from checkpoint entirely): {len(missing)} / {total_keys}')
    log(f'  unexpected keys (in checkpoint, not in model):  {len(unexpected)}')

    bad = len(dropped) + len(missing)
    if bad > 0.2 * total_keys or len(unexpected) > 0.2 * total_keys:
        log('  WARNING: a large fraction of keys did not line up (shape '
            'mismatch + missing + unexpected). If configCustom.MODEL_ARCH '
            f'(F / depth / W) does not match the architecture {ckpt_path} '
            'was actually trained with, this warm start is doing far less '
            'than you think -- much of the network may still be at random '
            'init.')

    return dropped, missing, unexpected


def main(C):
    torch.backends.cudnn.benchmark = False

    d_cfg = C['data']
    r_cfg = C['run']
    t_cfg = C['train']
    e_cfg = C['eval']

    PRETRAINED         = d_cfg.get('pretrained') or ''    # '' / null -> train from scratch
    RUN_NAME           = r_cfg['run_name']                # everything (log + checkpoints) goes under runs/<run_name>/
    RUNS_DIR           = r_cfg['runs_dir']
    SEED               = r_cfg.get('seed', 42)

    EPOCHS             = t_cfg['epochs']
    BATCH_SIZE         = t_cfg.get('batch_size', 1)       # physical batch; see ACCUM_STEPS below
    if BATCH_SIZE != 1:
        # full-res sequences can be different native sizes and can't be
        # stacked into a batch > 1 -- see dataset.FullResVFIDataset.
        raise ValueError(
            f'train.batch_size must be 1 for full-res training (got {BATCH_SIZE}); '
            f'use train.accum_steps in the yaml to get a larger effective batch instead')
    ACCUM_STEPS        = t_cfg.get('accum_steps', 8)      # effective batch = BATCH_SIZE * ACCUM_STEPS

    LR                 = float(t_cfg['lr'])                # peak LR -- fine-tuning wants far less than the
                                                            # ~2e-4 you'd use to train this from scratch
    MIN_LR             = float(t_cfg.get('min_lr', 1e-6))
    WARMUP_EPOCHS      = t_cfg.get('warmup_epochs', 1)     # applied PER STEP -- see lr_schedule() docstring
    WEIGHT_DECAY       = float(t_cfg.get('weight_decay', 1e-4))
    GRAD_CLIP          = t_cfg.get('grad_clip', 1.0)
    AMP                = t_cfg.get('amp', True)

    NUM_WORKERS        = d_cfg.get('num_workers', 4)

    EVAL_FULL_RES_EVERY       = e_cfg.get('eval_full_res_every', 5)
    EVAL_FULL_RES_MAX_SAMPLES = e_cfg.get('eval_full_res_max_samples', None)
    SAVE_EVERY_EPOCH          = e_cfg.get('save_every_epoch', True)
    LOG_EVERY_STEPS           = e_cfg.get('log_every_steps', 10)

    run_dir = os.path.join(RUNS_DIR, RUN_NAME)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'train.log')
    log_file = open(log_path, 'a')  # append -- re-running the same RUN_NAME keeps history

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line)
        log_file.write(line + '\n')
        log_file.flush()

    log(f'===== starting run "{RUN_NAME}" ==========================================')
    log(f'settings: epochs={EPOCHS} batch={BATCH_SIZE} accum={ACCUM_STEPS} '
        f'seed={SEED} lr={LR} min_lr={MIN_LR} warmup_epochs={WARMUP_EPOCHS} '
        f'amp={AMP} pretrained={PRETRAINED}')
    log(f'full log for this run: {log_path}')

    # Archive the exact config used for this run next to its checkpoints,
    # so a run's settings are never ambiguous later.
    if C.get('_source_path'):
        try:
            shutil.copy(C['_source_path'], os.path.join(run_dir, 'config.yaml'))
        except OSError as ex:
            log(f'  (could not archive config.yaml: {ex})')

    # ---- ALL data handling (split, seeding, dynamic timestep, padding) ----
    train_set, val_set = prepare_datasets(C, run_dir, log=log)
    log(f'train items: {len(train_set)} | val items: {len(val_set)}')

    # multiprocessing_context='fork' is pinned explicitly here because Python
    # 3.14 changed the POSIX multiprocessing default from 'fork' to
    # 'forkserver'. FullResVFIDataset.__getitem__ never touches CUDA (only
    # cv2.imread + CPU tensor conversion), so 'fork' is safe here even though
    # the model will already be on the GPU in the main process by the time
    # these workers spawn -- only the main process ever calls into CUDA.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn, multiprocessing_context='fork')
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn, multiprocessing_context='fork')

    model = Model(-1)  # local_rank=-1 -> single GPU, no DDP wrapper

    # ---- warm start (shape-checked -- see load_pretrained_with_shape_check) ----
    if PRETRAINED and os.path.exists(PRETRAINED):
        log(f'warm-starting from {PRETRAINED}')
        load_pretrained_with_shape_check(model.net, PRETRAINED, log)
    else:
        log('no pretrained checkpoint given/found -- training from scratch')

    optimizer = torch.optim.AdamW(model.net.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    local = model.local  # from config.LOCAL, set in Model.__init__
    criterion = LapLoss(max_levels=5, channels=3)  # matches the repo's own training objective

    best_val = float('inf')
    steps_per_epoch = len(train_loader)
    total_steps_all = EPOCHS * steps_per_epoch
    train_start = time.time()
    global_step = 0
    history = []
    logged_resolutions = set()  # (w, h) already logged with their resolved scale, avoids log spam

    for epoch in range(EPOCHS):
        warplayer.backwarp_tenGrid.clear()
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        n_steps = steps_per_epoch
        cur_lr = optimizer.param_groups[0]['lr']

        # img0/gt/img1 arrive already padded to a multiple of 32; h/w are
        # the ORIGINAL pre-padding dimensions (for cropping predictions
        # back before PSNR later); timestep/scale are already resolved
        # per-sample by dataset.py.
        for step, (img0, gt, img1, timestep, scale, h, w) in enumerate(train_loader):
            img0 = img0.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            img1 = img1.cuda(non_blocking=True)
            imgs = torch.cat((img0, img1), 1)

            step_timestep = float(timestep.item())  # batch_size is pinned to 1, see above
            step_scale = float(scale.item())
            h, w = int(h.item()), int(w.item())

            if (w, h) not in logged_resolutions:
                log(f'  detected resolution {w}x{h} (long edge {max(h, w)}px) '
                    f'-> train_scale {step_scale}')
                logged_resolutions.add((w, h))

            cur_lr = lr_schedule(optimizer, global_step, steps_per_epoch,
                                  EPOCHS, WARMUP_EPOCHS, LR, MIN_LR)

            with torch.cuda.amp.autocast(enabled=AMP):
                _, _, _, pred = model.net(imgs, timestep=step_timestep, scale=step_scale, local=local)
                loss = criterion(pred, gt) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == n_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * ACCUM_STEPS
            running_loss += step_loss
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0 or (step + 1) == n_steps:
                elapsed_total = time.time() - train_start
                avg_step_time = elapsed_total / global_step
                eta = avg_step_time * (total_steps_all - global_step)
                running_avg_loss = running_loss / (step + 1)
                log(f'  epoch {epoch+1}/{EPOCHS} | step {step+1}/{n_steps} '
                    f'(global {global_step}/{total_steps_all}) | '
                    f'loss {step_loss:.4f} | avg_loss {running_avg_loss:.4f} | '
                    f'lr {cur_lr:.2e} | t {step_timestep:.3f} | scale {step_scale} | '
                    f'elapsed {format_time(elapsed_total)} | '
                    f'ETA {format_time(eta)}')

        train_loss = running_loss / max(1, n_steps)

        # ---- full-resolution validation: loss every epoch, PSNR on a cadence ----
        model.eval()
        val_loss = 0.0
        want_psnr = (epoch + 1) % EVAL_FULL_RES_EVERY == 0 or (epoch + 1) == EPOCHS
        psnr_total, psnr_count = 0.0, 0
        with torch.no_grad():
            for i, (img0, gt, img1, timestep, scale, h, w) in enumerate(val_loader):
                img0 = img0.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                img1 = img1.cuda(non_blocking=True)
                imgs = torch.cat((img0, img1), 1)
                step_timestep = float(timestep.item())
                step_scale = float(scale.item())
                h, w = int(h.item()), int(w.item())

                _, _, _, pred = model.net(imgs, timestep=step_timestep, scale=step_scale, local=local)
                val_loss += criterion(pred, gt).item()

                do_psnr_this_sample = want_psnr and (
                    EVAL_FULL_RES_MAX_SAMPLES is None or i < EVAL_FULL_RES_MAX_SAMPLES)
                if do_psnr_this_sample:
                    # Padding was appended (bottom/right only), so cropping
                    # the first h rows / w cols out of both the padded
                    # prediction and the padded gt exactly recovers the
                    # original unpadded content for a fair PSNR.
                    pred_cropped = pred[:, :, :h, :w]
                    gt_cropped = gt[:, :, :h, :w]
                    psnr_total += psnr(pred_cropped, gt_cropped)
                    psnr_count += 1
        val_loss /= max(1, len(val_loader))
        full_psnr_this_epoch = (psnr_total / psnr_count) if psnr_count else None

        elapsed = time.time() - t0
        msg = (f'epoch {epoch+1:03d}/{EPOCHS} | train_loss {train_loss:.4f} | '
               f'val_loss {val_loss:.4f} | lr {cur_lr:.2e} | {elapsed:.1f}s')
        if full_psnr_this_epoch is not None:
            msg += f' | full-res PSNR {full_psnr_this_epoch:.2f}dB'
        log(msg)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': cur_lr,
            'full_psnr': full_psnr_this_epoch,
        })

        if SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(
                ckpt_dir,
                f'{MODEL_CONFIG["LOGNAME"]}_epoch{epoch+1:03d}_valloss{val_loss:.4f}.pkl'
            )
            torch.save(model.net.state_dict(), ckpt_path)

        last_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_last.pkl')
        torch.save(model.net.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_best.pkl')
            torch.save(model.net.state_dict(), best_path)
            log(f'  -> new best (val_loss {best_val:.4f}), saved to {best_path}')

    # ---- final run summary: the thing to actually look at tomorrow morning ----
    best_ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_best.pkl')
    last_ckpt_path = os.path.join(ckpt_dir, f'{MODEL_CONFIG["LOGNAME"]}_last.pkl')
    summary_path = os.path.join(run_dir, 'summary.txt')

    if history:
        total_elapsed = time.time() - train_start
        best_entry = min(history, key=lambda h: h['val_loss'])
        final_entry = history[-1]
        avg_train_loss = sum(h['train_loss'] for h in history) / len(history)
        avg_val_loss = sum(h['val_loss'] for h in history) / len(history)

        psnr_entries = [h for h in history if h['full_psnr'] is not None]
        best_psnr_entry = max(psnr_entries, key=lambda h: h['full_psnr']) if psnr_entries else None

        first_val_loss = history[0]['val_loss']
        change_pct = ((best_entry['val_loss'] - first_val_loss) / first_val_loss * 100
                      if first_val_loss > 0 else 0.0)

        w = 62
        lines = []
        lines.append('=' * w)
        lines.append(f' TRAINING RUN SUMMARY: {RUN_NAME}')
        lines.append('=' * w)
        lines.append(f' Epochs completed:      {len(history)} / {EPOCHS}')
        lines.append(f' Total wall time:       {format_time(total_elapsed)}')
        lines.append(f' Pretrained warm-start: {PRETRAINED if PRETRAINED else "(trained from scratch)"}')
        lines.append('')
        lines.append(f' BEST EPOCH: {best_entry["epoch"]}')
        lines.append(f'   val_loss:            {best_entry["val_loss"]:.4f}')
        lines.append(f'   train_loss:          {best_entry["train_loss"]:.4f}')
        if best_psnr_entry is not None:
            lines.append(f'   best full-res PSNR:  {best_psnr_entry["full_psnr"]:.2f} dB '
                          f'(epoch {best_psnr_entry["epoch"]})')
        lines.append(f'   checkpoint:          {best_ckpt_path}')
        lines.append('')
        lines.append(f' FINAL EPOCH ({final_entry["epoch"]}):')
        lines.append(f'   train_loss:          {final_entry["train_loss"]:.4f}')
        lines.append(f'   val_loss:            {final_entry["val_loss"]:.4f}')
        lines.append(f'   lr:                  {final_entry["lr"]:.2e}')
        lines.append('')
        lines.append(' AVERAGES ACROSS RUN:')
        lines.append(f'   avg train_loss:      {avg_train_loss:.4f}')
        lines.append(f'   avg val_loss:        {avg_val_loss:.4f}')
        lines.append('')
        lines.append(' IMPROVEMENT (val_loss, epoch 1 -> best):')
        lines.append(f'   {first_val_loss:.4f} -> {best_entry["val_loss"]:.4f}  ({change_pct:+.1f}%)')
        lines.append('=' * w)
        lines.append(f' Full log:        {log_path}')
        lines.append(f' Summary file:    {summary_path}')
        lines.append(f' Best checkpoint: {best_ckpt_path}')
        lines.append(f' Last checkpoint: {last_ckpt_path}')
        lines.append('=' * w)
        summary_text = '\n'.join(lines)

        with open(summary_path, 'w') as f:
            f.write(summary_text + '\n')

        print('\n' + summary_text + '\n')
        log_file.write('\n' + summary_text + '\n')
        log_file.flush()
    else:
        log('No epochs were completed -- skipping final summary.')

    log(f'===== run "{RUN_NAME}" finished =========================================')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-tune VFIMamba on a full-resolution multi-frame dataset with dynamic timestep')
    parser.add_argument('--config', required=True, type=str,
                         help='path to YAML config file (all settings live here)')
    parser.add_argument('--restore_ckpt', type=str, default=None,
                         help='optional override for data.pretrained in the yaml')
    args = parser.parse_args()

    C = load_config(args.config)
    if args.restore_ckpt:
        C['data']['pretrained'] = args.restore_ckpt

    main(C)