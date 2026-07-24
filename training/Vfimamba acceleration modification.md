# VFIMamba Fine-Tuning: Data Pipeline and Bidirectional-Flow Architecture

This document traces the whole system end to end, in the order things actually
happen: raw frames on disk → curriculum construction → dataset loading →
training loop → the model itself (the bidirectional-flow + learned-acceleration
architecture that replaces VFIMamba's original timestep-conditioned flow
regression). A changelog of bugs found and fixed while wiring these pieces
together is at the end.

```
AllTrainingData/                     (raw: <Dataset>/<Scene>/frame_NNNN.png)
        │
        ▼  curriculum_builder.py   — run ONCE
TrainingData/Phase1..4/, ValidationData/, curriculum_metadata.yaml
        │
        ▼  dataset.py              — read-only, one phase at a time
FullResVFIDataset + BatchUidGroupedSampler + ragged_collate
        │
        ▼  train.py                — one phase per run
DataLoader → training loop → model.net (MultiScaleFlow / AccelFlow)
        │
        ▼  model_oldRepo/{flow_estimation,accel_flow}.py
bidirectional flow → per-t derivation → synthesis → prediction
```

---

## 0. Quick recap: does the pipeline actually do this?

A compact checklist against the behaviors this pipeline is expected to have,
each with the section that covers it and any nuance worth knowing:

1. **Batches are loaded together because everything is padded up to the
   nearest 32.** ✓ `pad_multiple: 32` rounds every sample's `(w,h)` up once
   (§1.3) *before* phase/batch assignment, so every scene inside one
   `BatchNNN_<res>/` folder already shares one exact padded resolution by
   construction — that shared resolution is exactly what makes
   `BatchUidGroupedSampler` + `ragged_collate` (§2.5) safe to stack into a
   real `(B,C,H,W)` tensor with zero cropping or cross-resolution padding.
2. **Phases 2/3/4 contain a mix of resolutions**, because each phase retains
   ~80% of its own (native-resolution) data and exports the other ~20% evenly
   to every *later* phase as replay (§1.4 step 3). So phase 2's stream mixes
   its own resolution with replay from phase 1; phase 3 mixes its own with
   replay from phases 1 and 2; phase 4 similarly, from 1–3. Phase 1 has
   nothing earlier to import, so it only sees its own native resolution(s).
3. **Optimizer updates happen per accumulation window, at the full effective
   batch size** — true for full windows (`window_num_items ==
   batch_size × accum_steps`, §3.2). One caveat: trailing *partial* windows
   (a resolution/bucket run that ends before reaching a full `accum_steps`)
   still flush, just at their own smaller true item count, rather than being
   dropped or silently mis-weighted.
4. **Lower-resolution (replay) accumulation windows sit between the
   higher-resolution (own) windows**, so the optimizer processes one full
   window — one full update — at a time, alternating own/replay rather than
   front- or back-loading replay (§1.4 step 7, §3.2).
5. **Bilateral flow (`F_{0→1}`, `F_{1→0}`) is estimated once, and combined
   with a learned acceleration term to derive flow at any `t`** — this is the
   whole point of §4: flow between the two *real* frames is computed once,
   then `p(t) = t·D + a·t·(t-1)` derives the per-t flow cheaply (§4.2–4.3).
6. **That reused-computation saving is what lets every middle frame of a
   training scene be used**, not just one sampled frame per scene per epoch —
   `dataset.py` now returns *all* interior frames per item (§2.2), and
   `forward_multi_t` (§4.2) evaluates all of them in one batched call, so the
   expensive backbone still runs exactly once per `(img0,img1)` pair
   regardless of how many middle frames that scene has (§3.3).
7. **Scenes with thousands of frames get a hard frame-difference cap** per
   phase (`data.max_frame_span_by_phase`), so `img0`↔`img1` can never span an
   ill-posed, multi-second gap even in Phase 1 (§2.4).
8. **When a scene exceeds that cap, a random sub-clip is drawn for that
   phase** — not a fixed max-length clip every time. The drawn length is
   random in `[3, max_frame_span]`, with a random start position, re-drawn
   every epoch from the same per-epoch-seeded RNG as the timestep pick — so
   over the course of training a long scene's full range gets sampled, not
   just the widest allowed gap (§2.4).
9. **Differing resolutions get their own batch_size/accum_steps cutoff in
   config**, specifically to speed up training at lower-resolution phases:
   `dynamic_batch.thresholds` assigns large `batch_size`/`accum_steps=1` at
   small resolutions (real batching pays off there) and `batch_size=1`/large
   `accum_steps` at large resolutions (VRAM-bounded) — read by both
   `curriculum_builder.py` (to build batch folders and bucket-group
   accumulation windows) and `train.py`/`dataset.py` (to actually batch and
   flush) from the same table, so the two never disagree (§1.4 steps 2 & 5,
   §2.5).
10. **Training also directly supervises the acceleration prediction**,
    which is the fix for bilateral flow's core weakness (constant-velocity-
    only motion). The real middle frame gives a closed-form target `a_true`
    every step; the blind predictor `a_student` (same inputs as inference) is
    trained against it via L1 loss, alongside the photometric loss (§4.3,
    §3.4).

---

## 1. Stage 1 — `curriculum_builder.py`: building the curriculum

Run once, before any training: `python curriculum_builder.py --config config.yaml`.
It is **not incremental** — every run rebuilds `TrainingData/` and
`ValidationData/` from scratch from the current `--src` listing, so that the
same seed always produces the same on-disk layout regardless of run history.

### 1.1 Extract
`discover_sequences()` recursively scans `<Dataset>/<Scene>/` (any number of
datasets, any number of scenes, any number of frames — nothing assumes
uniformity) and records full metadata per usable scene: resolution, frame
count, on-disk path, which frame names are actually present (each frame name
is resolved to whichever extension exists, independently per scene, so
different datasets can mix formats).

### 1.2 Split
`split_train_val()` does a single seeded shuffle-and-slice over **all**
extracted samples, once, before any phase/curriculum logic runs. Held-out
validation samples never re-enter phase assignment, replay export, or batch
folder creation. (A manual, pre-separated validation set is also supported via
`val.use_manual_val`.)

### 1.3 Bucket
`annotate_resolution_buckets()` rounds every training sample's `(w, h)` up to
the nearest `pad_multiple` **once**, storing `padded_w`/`padded_h` on each
sample. Every downstream lookup (phase assignment, batch grouping, dynamic
batch/accum/scale resolution) reads this precomputed value instead of
re-deriving it.

### 1.4 Curriculum: phases, batches, replay, accumulation windows
This is the core of the file. In order:

1. **`assign_phases`** buckets each sample into its own phase by padded
   resolution (`config.yaml`'s `curriculum.phase_buckets`), then sorts each
   phase's own data ascending by resolution — the "progress low → high"
   requirement.
2. **`group_into_batches`** immediately collapses each phase's own ascending
   run into whole **batch objects**: consecutive same-padded-resolution items
   are chunked into pieces of size `batch_size(resolution)` (from
   `config.yaml`'s `dynamic_batch.thresholds` table). A batch is the unit of
   everything from here on — never an individual scene. A batch always holds
   exactly one exact padded resolution. A trailing chunk shorter than the
   configured `batch_size` is flagged `is_extra`.
3. **`split_retain_export`**: each phase retains ~80% (configurable via
   `retain_ratio`) of its own **non-extra** batches (measured by total item
   count) and exports the rest as replay, seeded-shuffled, to every later
   phase. `is_extra` batches are never candidates for export — an incomplete
   batch shipped to a different phase would just be dead weight, since
   `dataset.py` never trains on `Extra/` content anyway.
4. **`distribute_export_evenly`** round-robins each phase's exported batches
   across all later phases, as close to an equal share (by batch count) as
   deterministically possible. A batch always moves as one indivisible unit.
5. **`group_batches_into_accum_windows`** chunks each stream (a phase's own
   retained batches, and each replay source, separately) into **accumulation
   windows**: consecutive non-extra batches that resolve to the same
   `(batch_size, gradient_accumulation)` **bucket** — not necessarily the same
   exact resolution — taken `accum_steps` at a time. This bucket-based (rather
   than exact-resolution) grouping matters most for replay: exported batches
   are shuffled before reaching this step, so a shuffled stream almost never
   has long runs of *identical* exact resolution back-to-back. Grouping by
   bucket instead lets shuffled replay reach full-sized accumulation windows
   far more often, instead of degrading to a series of single-batch partial
   windows. A same-bucket run that ends before reaching `accum_steps` becomes
   a **partial window** instead of being discarded.
6. **`_tag_windows`** stamps every batch in a window with a shared
   `window_id` (globally unique — one `itertools.count()` shared across every
   phase and stream) plus the window's **true total item count**
   (`window_num_items`) and, for reference, the theoretical full-window target
   (`window_target_items`). This is what `train.py` actually divides the loss
   by and flushes the optimizer on — read directly off disk, not looked up
   from a table, so it's exact for both full and partial windows.
7. **`interleave_grouped_streams`** spreads each replay stream's full windows
   evenly across the *gaps* between a phase's own full windows (never inside
   one), so replay is distributed throughout the phase rather than sitting in
   one block.
8. **`build_curriculum`** assembles each phase's final order as:
   **interleaved full windows → trailing partial windows (own, then each
   replay source, in that order) → trailing `is_extra` batches (same
   ordering)**. Partial windows and extra batches are appended at the end
   rather than interleaved, since they're not part of the main progression.
   `is_extra` batches never pass through `_tag_windows` — they carry no
   `window_id` at all.

### 1.5 Batch folders
`write_phase_folder()` does no grouping of its own — it just numbers and
writes, in order, the batch objects Step 4 already built. Any batch flagged
`is_extra` is written under `Phase<N>/Extra/BatchNNN_<res>_origP<phase>_n<count>/`
instead of a normal `Phase<N>/BatchNNN_<res>/` slot; its `origin_phase` (the
phase whose own data produced it, not necessarily the phase it currently sits
in) is recorded in both its own `batch_metadata.yaml` and the phase's
`extra_metadata.yaml`. Non-extra batches additionally carry
`window_id`/`window_num_items`/`window_target_items`/`window_is_partial` in
their metadata.

```
TrainingData/
  Phase1/
    phase_metadata.yaml
    Batch001_960x544/
      batch_metadata.yaml
      00000_own__DatasetA__Scene003/        (symlink to original scene dir)
      00001_own__DatasetA__Scene007/
    Batch002_1024x576/ ...
    Extra/
      extra_metadata.yaml
      Batch001_960x544_origP1_n2/
      Batch002_1920x1088_origP2_n1/          (undersized replay from Phase2)
  Phase2/, Phase3/, Phase4/  (same pattern, more replay sources)
ValidationData/
  <Dataset>/<Scene>/                         (symlink; Dataset/Scene hierarchy preserved)
curriculum_metadata.yaml                     (seed, split info, per-phase summary)
```

The zero-padded leading index on each scene folder name is the training order
within its batch folder, monotonically increasing across the whole phase
(including `Extra/`) — this lets `dataset.py` reproduce the exact curriculum
order with a plain nested `sorted()` walk, no randomness re-derivation needed.
`window_id` is an independent tag layered on top of that ordering; it doesn't
affect folder naming, only which items `train.py` treats as one accumulation
window.

---

## 2. Stage 2 — `dataset.py`: loading one phase

`dataset.py` is read-only over whatever `curriculum_builder.py` already wrote.
It never splits, shuffles, or re-derives curriculum order itself.

### 2.1 Directory walk
`_list_train_seq_dirs` walks `Phase<N>/BatchNNN_<res>/<scene>/` in curriculum
order (batch folder name, then scene name, both `sorted()`), skipping `Extra/`
and `phase_metadata.yaml` entirely — undersized batches were only ever
scaffolding, never meant to be trained on. For every scene it returns
`(scene_dir, sample_id, batch_uid, window_id, window_num_items)`, with
`window_id`/`window_num_items` read straight out of that batch's
`batch_metadata.yaml`. `_list_val_seq_dirs` walks `ValidationData/<Dataset>/<Scene>/`
alphabetically — order carries no curriculum meaning for validation.

### 2.2 `FullResVFIDataset`
One `Dataset` instance per phase (train) or the whole validation set (val).

- **Train mode**: one item per scene. `__getitem__` picks interior
  frames/timestep(s) fresh **every epoch** (see 2.3), optionally from a
  sub-clip of a long scene (see 2.4). Returns `(img0, gt_stack, img1,
  timesteps, h, w, batch_uid, window_id, window_num_items)` — `gt_stack` is
  `(num_interior, C, H, W)`, since **all** interior frames of the (possibly
  clipped) scene are used per item, not one random pick.
- **Val mode**: every interior frame of every sequence is enumerated once at
  construction time (capped by `val_max_interior_per_seq`), fixed order,
  deterministic across epochs, never affected by max-frame-span clipping.

Both modes resolve `frame_names` (extension-less, from config) to actual
on-disk filenames the same way `curriculum_builder.py`'s `find_frame_file` did,
so the two files agree on what counts as "present."

### 2.3 Per-epoch seeded timestep selection
Each item's interior-frame/timestep pick (and sub-clip draw, if applicable) is
drawn from `seeding.rng_for(seed, 'timestep', epoch, sample_id)` — a dedicated
RNG keyed on the sample's own id and the *current epoch*, not Python's global
`random` module. This makes picks reproducible per `(seed, epoch, scene)` and
different from one epoch to the next. `train.py` **must** call
`train_set.set_epoch(epoch)` at the top of every epoch for this to work, and
the train `DataLoader` is built with `persistent_workers=False` specifically so
that worker processes are re-spawned (and re-pickle the just-updated
`self.epoch`) every epoch — persistent workers would freeze on the first
epoch's snapshot and silently stop seeing later `set_epoch()` calls.

### 2.4 Max frame span (long-scene clipping)
Some source scenes run to hundreds or thousands of frames. Using a scene's
literal first/last frame as `img0`/`img1` regardless of length would let the
`img0`↔`img1` gap span many seconds of real motion, in any phase including
Phase 1 — far outside what a two-frame flow-based model has a basis to
interpolate. `config.yaml`'s `data.max_frame_span_by_phase` caps this per
phase (missing entry or explicit `null` = uncapped). When a scene's frame
count exceeds the cap, `__getitem__` draws a random contiguous sub-clip of
length `[3, max_frame_span]` — using the *same* per-epoch-seeded RNG as the
timestep pick, so it's reproducible per epoch and varies across epochs, and a
long scene's full range still gets sampled over the course of training, just
never all at once. `img0`/`img1` become the clip's first/last frame; interior
selection then runs on the clip. This is a per-item **clip**, deliberately
distinct terminology from `window_id`/`window_num_items` (train.py's
gradient-accumulation concept) — the two are unrelated. Validation is never
clipped.

### 2.5 Real batching: `BatchUidGroupedSampler` + `ragged_collate`
Two things make real (not accumulation-simulated) batching safe:

1. Every scene inside one `BatchNNN_<res>/` folder shares **exactly one**
   padded resolution by construction (`group_into_batches`'s invariant) — so a
   batch_uid-grouped minibatch never needs cross-resolution cropping/padding,
   only stacking.
2. Scenes within one batch_uid group can still have **differing** interior-
   frame counts (frame count depends on the per-scene max-frame-span draw) —
   so collation still needs padding + a mask.

`BatchUidGroupedSampler` groups the dataset's flat item list by `batch_uid`
(one physical batch folder), in curriculum order both across and within
groups, and is used as a `batch_sampler` (not shuffled — the curriculum order
*is* the intended training order). `ragged_collate` then:
- stacks `img0`/`img1` trivially (single resolution per group),
- pads `gt`/`timesteps` to this minibatch's own `T_max = max(num_interior)`,
- returns a `valid_mask` `(B, T_max)` bool tensor marking real vs. padded
  slots,
- passes `batch_uid`/`window_id`/`window_num_items` through unchanged (taken
  from item 0 — identical across the whole group by construction, since
  they're stamped per batch folder, not per scene).

Because `config.yaml`'s `dynamic_batch` table assigns large `batch_size` at
small resolutions and `batch_size=1` at large resolutions (to keep VRAM
roughly constant across native-resolution training), this sampler naturally
produces large real minibatches at low-resolution phases — where GPU
utilization benefits most — and single-item batches at high-resolution
phases, with zero phase-conditional code anywhere: what changes phase to
phase is purely the config table.

---

## 3. Stage 3 — `train.py`: the training loop

`python train.py --config config.yaml --run run_config.yaml --phase N
[--restore_ckpt ...]`. `config.yaml` holds pipeline-wide, curriculum-affecting
settings (seed, dynamic_batch/train_scale tables, validation mode);
`run_config.yaml` holds only per-run hyperparameters. Only one phase is loaded
per run — phases are trained sequentially, by hand, optionally warm-starting
each phase from the previous phase's best checkpoint.

### 3.1 Loaders
`make_train_loader` wraps `train_set` in
`DataLoader(batch_sampler=BatchUidGroupedSampler(train_set),
collate_fn=ragged_collate, ..., persistent_workers=False)`. `make_val_loader`
is a plain `DataLoader` over `val_set` — native mode (batch_size=1, no
bucketing) by default, appropriate for a val set of mixed resolutions;
non-native mode uses a flat batch size and requires a val set pre-normalized
to one resolution.

### 3.2 Accumulation windows: flush on `window_id`, divide by `window_num_items`
Each step now yields one native-resolution **minibatch** (B scenes) instead of
exactly one scene — but the window design itself, built entirely by
`curriculum_builder.py` before `train.py` ever sees it, is unchanged:

- **Flush** the optimizer exactly when `window_id` changes — not on a
  resolution change, not on `batch_uid` changing, not on an item-count target.
  A window can (and usually does) span multiple physical steps, since one
  `batch_uid` group is one batch folder and a window is usually several
  consecutive folders.
- **Divide** the accumulated loss by `window_num_items`, read straight off
  disk — exact for both full windows (`batch_size × accum_steps`) and partial
  windows (whatever smaller count actually landed there).
- `steps_since_update` counts **items** (incremented by this step's actual `B`,
  not by a fixed 1 per step), and must reach `cur_effective_batch ==
  window_num_items` exactly.
- `batch_uid` is still read and logged per step for traceability, but never
  gates anything.

### 3.3 Masked, per-item loss
`gt`/`timesteps` arrive padded to `T_max` with `valid_mask`. Given the model's
per-sample-`t` support, `forward_multi_t` computes **all** `B × T_max`
predictions in one batched call regardless of masking — the expensive
backbone + flow-refinement stack still runs exactly once per `(img0, img1)`
pair per step, which is the whole point of the architecture in §4. Only the
loss **reduction** needs per-item validity: for each item `b`, its own valid
interior frames' losses are averaged first (`LapLoss`'s own batch-reduction
behavior is opaque to this file, so a single vectorized masked-mean can't be
trusted to preserve the per-item invariant), and only *then* averaged across
the `B` items in the step — mirroring the original single-item
`frame_losses.mean()` invariant, now with one extra averaging level for the
batch dimension.

Because this per-step loss is a **mean** over `B` items, the backward pass
needs
```python
scaled_loss = mean_item_loss * B / cur_effective_batch
```
not a bare `loss / cur_effective_batch` — multiplying back by `B` first keeps
this step's gradient contribution equal to the sum of what `B` individual
gradient-accumulation steps would have contributed. Omitting the `* B` would
silently shrink the effective learning rate any time `B > 1`, without
changing anything else about the config — this is the single easiest mistake
when layering real batching on top of an existing accumulation scheme.

### 3.4 Acceleration loss integration
When `model.net` has an `accel_head` (i.e. it's an `AccelFlow`),
`accel_distillation_loss` is called with `local=local` — the **same**
local-refinement flag used for this step's main forward pass — for every
`(b, i)` valid pair in the masked loop (see §4.3 and §9.4 for why this flag
must match). Its cost is real: it recomputes `estimate_bi_flow` twice
(student + teacher) per `(b, i)` pair, on single-item slices, since the
teacher pass can't be batched across interior frames (each depends on a
different privileged frame). This is a throughput cost, not a correctness
bug — flagged in §6.

### 3.5 Everything else
Cosine LR schedule with linear warmup; `torch.cuda.amp` mixed precision with
gradient clipping; per-epoch `warplayer.backwarp_tenGrid.clear()` (needed
because native/mixed-resolution training means the warp grid cache would
otherwise grow unbounded); periodic logging including resolution, batch size,
timestep range, window fill fraction, and GPU peak memory; full-resolution
PSNR every `eval_full_res_every` epochs; per-epoch and best/last checkpoint
saving; a final run summary written to `summary.txt`.

---

## 4. The architecture change: bidirectional flow + learned acceleration

### 4.1 Original vs. new paradigm
**Original VFIMamba**: the Head/IFBlock stack directly regresses the
*bilateral, timestep-conditioned* flow, `F_{t→0}`/`F_{t→1}`, at every
refinement stage — `timestep` is concatenated as an input channel throughout,
so the whole flow-refinement stack must rerun from scratch for every
requested `t`.

**This architecture**: the refinement stack (`BiHead`/`BiIFBlock`) instead
regresses a single, **t-independent** bidirectional flow pair, `F_{0→1}` and
`F_{1→0}`, between the two real input frames, plus a t-independent
occlusion/visibility field. `timestep` is removed from every conv input
entirely. A cheap, non-learned step then derives `F_{t→0}`/`F_{t→1}` (and a
t-dependent blend mask) from that single pair for as many `t` values as
needed, using the classical linear-combination-of-bidirectional-flows formula
(Super SloMo / Jiang et al.). The expensive backbone + refinement work now
runs **once** per `(img0, img1)` pair, regardless of how many timestamps are
requested — this is exactly what stage 3's `forward_multi_t` batched call
depends on.

**Trade-off**: this reintroduces an explicit constant-velocity assumption
between frame 0 and frame 1 — a strictly weaker motion model than direct
per-t regression, which at least has an implicit (if two-frame-limited) prior
for nonlinear motion. The synthesis UNet is the only place left that can
partially correct for curved motion, since it still sees the actual warped
images. §4.2 addresses this trade-off directly with a learned correction term.

### 4.2 `flow_estimation.py` — the t-independent backbone
- **`BiHead`** / **`BiIFBlock`** (renamed from `Head`/`IFBlock`): identical
  coarse-to-fine structure, but no `timestep` channel anywhere. Output is
  still 5 channels per stage, reinterpreted: `[:2] = F_{0→1}`, `[2:4] =
  F_{1→0}`, `[4:5] = visibility logit`.
- **`estimate_bi_flow(img0, img1, local, af)`**: the coarse-to-fine refinement
  loop across `flow_num_stage` stages, plus optional local refinement stages —
  this is the piece that used to rerun per-`t` in the original design and now
  runs exactly once.
- **`flow_from_bi(bi_flow, t)`** (no learned parameters):
  ```
  F_{t→0} = -(1-t)·t·F_{0→1} + t²·F_{1→0}
  F_{t→1} =  (1-t)²·F_{0→1} - t·(1-t)·F_{1→0}
  ```
  `t` may be a python float or a tensor broadcastable to `bi_flow`'s
  `(B,1,H,W)` spatial dims, so per-sample-variable `t` works with no
  special-casing.
- **`mask_from_visibility(visibility, t)`**: `mask_t = (1-t)·V0 / ((1-t)·V0 +
  t·(1-V0))`, `V0 = sigmoid(visibility)` — a t-dependent blend weight derived
  from the single t-independent occlusion field.
- **`synthesize(img0, img1, af, flow_t, mask_t)`**: unchanged in spirit from
  the original — warp both frames by the derived `flow_t`, feed
  img/warp/mask/flow/features into the UNet, blend + residual — just now
  consuming a *derived* `flow_t`/`mask_t` instead of a directly regressed one.
- **`forward_multi_t(x, timesteps, local, scale)`**: the actual payoff.
  `estimate_bi_flow` runs once; `flow_from_bi` + `mask_from_visibility` +
  `synthesize` repeat per requested `t`. Accepts either a `(B, T_max)` tensor
  (train.py's batched path — see §3.3) or a plain iterable of floats shared
  across the whole call (the legacy path used by `Trainer.py`'s
  `inference_multi_t`).
- **Backward-compat shims** `estimate_flow_and_mask` / `calculate_flow` /
  `coraseWarp_and_Refine` preserve the original method names/signatures so
  `Trainer.py`'s `hr_inference` (which calls flow estimation and synthesis as
  two separate steps, e.g. to run flow at a downscaled resolution and
  synthesis at full resolution) needs no changes.

### 4.3 `accel_flow.py` — learned, directly-supervised acceleration
Two frames alone cannot carry a second-derivative (acceleration) signal — any
two-frame model, original or bidirectional, can only produce a *learned
prior* standing in for real acceleration. This file makes that prior explicit
and directly supervised instead of an implicit, unverifiable side-effect of
photometric loss, by using the real (available-at-train-time) middle frame to
derive a closed-form target.

**Motion model**:
```
D = F_{0→1}                       (from estimate_bi_flow, unprivileged)
a = learned acceleration field     (NEW — AccelHead, unprivileged)
p(t) = t·D + a·t·(t-1)             (vanishes at t=0 and t=1)
F_{t→0} = -p(t)
F_{t→1} =  p(t) - D
```
`a = 0` reduces exactly to the linear baseline in §4.2. Neither `D` nor `a`
depends on `t` — `t` only enters through the cheap `p(t)` evaluation, which is
what lets one `a` guess be reused for any number of requested timesteps.

**`AccelHead`**: a small extra head run once after `estimate_bi_flow`
converges. Takes `img0`, `img1`, and `D` (via `warp(img0, D)` for context),
outputs a `tanh`-bounded correction field `a` — bounded so it can only ever
correct the linear path, not dominate it, keeping early training stable.

**Getting a real supervision target** — the core mechanism. At train time, a
real triplet `(img0, img_mid_gt, img1, t_gt)` is available, `t_gt` unrestricted
in `(0,1)` (not fixed to 0.5):
1. Run the same bidirectional estimator on the **real** pair `(img0,
   img_mid_gt)` to get a measured (not guessed) `p_true = F_{0→mid}`.
2. Since `p_true = t_gt·D + a_true·t_gt·(t_gt-1)`, solve directly:
   `a_true = (p_true - t_gt·D) / (t_gt·(t_gt-1))`.
3. Train the blind predictor (`a_student`, which only ever sees `img0`/`img1`,
   same as at inference) against this target under `torch.no_grad()`:
   `loss_accel = L1(a_student, a_true.detach())`.

**`training_step`** wires this together end to end (student forward → photo
loss → privileged target under `no_grad` → accel loss). **`accel_distillation_loss`**
is the standalone version `train.py` actually calls, for scripts that already
compute `pred` themselves and have their own loss-accumulation logic.

**`forward_multi_t` override**: required, not optional. The base class's
`forward_multi_t` (inherited by default) would call `self.flow_from_bi` — the
linear-only formula — per timestep, silently ignoring `accel_head` for every
requested `t`. This override computes `D`/`a` once, then loops over
timesteps using `flow_from_bi_accel(D, a, t)` + `synthesize`, preserving the
"expensive stuff runs once" payoff while actually using the acceleration term.

**`forward()` scale handling**: mirrors the base class's downsample-then-
upsample pattern for `scale > 0` — `estimate_bi_flow` at the downsampled
resolution, then `feature_bone`/`accel_head` re-run at full resolution before
deriving `flow_t`/`mask_t`.

**What this does and doesn't solve**: the acceleration guess now has an
explicit, closed-form, geometrically-derived target instead of being an
unverifiable side-effect of pixel loss — but at inference there is still no
ground-truth middle frame, so `a_student` remains an extrapolation from two
frames, just a better-calibrated one. Needs real triplets with known `t_gt`
(fixed `t=0.5` datasets like Vimeo90k work; a variable-`t`, higher-fps dataset
like X4K1000FPS lets the head generalize across timestamps rather than
overfitting to one).

---

## 5. Wiring: `configCustom.py`, `Trainer.py`

`model_oldRepo/__init__.py` re-exports `MultiScaleFlow` (the new bidirectional
architecture, *not* stale — it already points at §4.2, not the original
direct-regression module) as `mamba_estimation`. That alias has no
`accel_head`, so if `configCustom.py`'s `MODEL_TYPE` goes through it,
`train.py`'s `hasattr(model.net, 'accel_head')` gate never fires and §4.3
silently never trains. `configCustom.py` must import `AccelFlow` directly:

```python
from model_oldRepo import mamba_extractor
from model_oldRepo.accel_flow import AccelFlow

MODEL_CONFIG = {
    'LOGNAME': LOG,
    'MODEL_TYPE': (mamba_extractor, AccelFlow),
    'MODEL_ARCH': init_model_config(..., accel_scale=1.0, accel_hidden=48),
}
```
`init_model_config` threads `accel_scale`/`accel_hidden` through to the
returned config dict, since `AccelFlow.__init__` reads them via `kargs`.

`Trainer.py` gains **`inference_multi_t`**, the public entry point for
interpolating at an arbitrary list of timestamps in one pass:
```python
model.inference_multi_t(img0, img1, local, timesteps=[0.1, 0.35, 0.5, 0.72, 0.9], scale=0)
```
which calls `self.net.forward_multi_t(imgs, timesteps, local=local,
scale=scale)` — for `AccelFlow`, correctly reusing a single acceleration guess
across every requested timestep.

`feature_extractor.py` (the Mamba backbone) needs **no changes**: its 5
per-stage feature maps at `embed_dims=[32,64,128,256,512]` line up exactly
with what `BiHead`/`estimate_bi_flow` expect, and none of the architecture
changes touch how the backbone is consumed.

---

## 6. Known limitations and flagged (not fixed) issues

- **`solve_accel_target`'s eps clamp has a sign bug near `t≈0` or `t≈1`.**
  `t_gt·(t_gt-1)` is always negative for `t_gt` in `(0,1)`, but the clamp
  substitutes a bare `+eps` (positive) whenever `|denom| < eps` — flipping the
  sign of `a_true` for items landing within `eps=1e-4` of an endpoint, silently
  rather than as a crash. Doesn't matter for fixed-`t=0.5` datasets (Vimeo90k);
  may matter for variable-`t` datasets (X4K1000FPS) depending on how interior
  frames are sampled. If it matters, replace the bare `eps` with a
  sign-preserving clamp (e.g. `-eps`, since `denom` is always negative
  in-range).
- **`accel_distillation_loss` cost.** It recomputes `estimate_bi_flow` at
  **native resolution** regardless of `step_scale`, twice (student + teacher)
  per valid interior frame of every batch item — a training step with
  `accel_head` active runs the expensive backbone up to 3× the main path's
  cost, further multiplied by `num_interior`. On a single RTX 3090 this could
  OOM at higher-resolution curriculum phases even though the main path fits.
  Not fixed — a real throughput concern, not a correctness bug. If VRAM is
  tight, consider adding a downsample/upsample step to the accel loss's two
  `estimate_bi_flow` calls.
- **Extra keys in `backbonecfg`** (`motion_dims`, `num_heads`, `mlp_ratios`,
  `qkv_bias`, `window_sizes`) are silently absorbed and unused by
  `feature_extractor` — pre-existing, unrelated to any of the above.

---

## 7. Changelog — bugs found and fixed during integration

1. **`configCustom.py`**: `MODEL_TYPE` must import `AccelFlow` directly from
   `accel_flow.py`, not via the `mamba_estimation` alias (which has no
   `accel_head`) — see §5.
2. **`local`-refinement consistency** (`accel_flow.py` / `train.py`):
   `accel_distillation_loss`/`training_step` now take an explicit `local`
   argument for the **student**'s `D`, matching the main forward pass's flag
   for that step; the **teacher** pass always uses `local=True` regardless
   (runs under `no_grad`, costs nothing, and a better-refined target is
   strictly better supervision). Previously the student's `D` was always
   computed with a hardcoded `local=False`, training `accel_head` against a
   different input distribution than it sees at inference.
3. **`AccelFlow.forward_multi_t` missing override**: without it, Nx/multi-t
   inference on an `AccelFlow` instance would silently use the linear-only
   `flow_from_bi` and ignore `accel_head` entirely — see §4.3.
4. **`AccelFlow.forward()` ignored `scale`**: previously accepted but unused,
   silently disabling the downsample-then-upsample speed path used by
   `Model.inference(..., scale=...)`. Now mirrors the base class.
5. **`dataset.py`**: added `max_frame_span_by_phase` clipping for long scenes
   — see §2.4. Not a bugfix, a new safeguard against multi-second, poorly-
   defined `img0`↔`img1` gaps in early curriculum phases.
6. **`curriculum_builder.py`, `Extra/` routing**: `group_batches_into_accum_windows`
   used to skip `is_extra` batches when flushing a run-in-progress, but never
   returned them anywhere — so they were silently dropped instead of reaching
   `Extra/` as the docstrings always claimed. Fixed by having that function
   return a third list (`extra_batches`, in original stream order) and having
   `build_curriculum` collect and append them (own stream + every replay
   source) to the end of each phase's order, after trailing partial windows.
7. **`curriculum_builder.py`, bucket-based accumulation windows**: windowing
   used to require exact `(padded_w, padded_h)` equality, which fragmented
   shuffled replay streams into mostly single-batch partial windows (shuffled
   exports rarely have consecutive identical exact resolutions). Fixed by
   grouping on `(batch_size, gradient_accumulation)` **bucket** equivalence
   instead — see §1.4 step 5.
8. **`train.py` / `dataset.py`, real minibatch loading**: replaced the
   forced-`batch_size=1`, gradient-accumulation-only design with
   `BatchUidGroupedSampler` + `ragged_collate`, giving a real GPU-utilization
   win at low-resolution phases with zero special-casing for high-resolution
   phases (which naturally degrade to `B=1` when the config table says so).
   Required, as a consequence: `steps_since_update` now counts items, not
   steps; the masked per-item loss described in §3.3; the `* B /
   cur_effective_batch` gradient rescaling in §3.3; and `forward_multi_t`
   accepting a `(B, T_max)` tensor (§4.2) instead of a flat list of scalars.
9. **`train.py`, missing `accel_distillation_loss` import**: the
   `hasattr(model.net, 'accel_head')` branch called this function without
   ever importing it — a guaranteed `NameError` on the first step of any
   `AccelFlow` run. Fixed with `from model_oldRepo.accel_flow import
   accel_distillation_loss`.
10. **`train.py`, undefined `step_timestep` in the training log line**: a
    per-step log line referenced a variable that only ever existed in the
    validation loop, left over from before the multi-interior-frame design —
    guaranteed `NameError` at the first `LOG_EVERY_STEPS` boundary. Fixed by
    logging the item's actual timestep range (`t_range [t_lo,t_hi]`) instead.
11. **`train.py`, `val_loss` scored on padded pixels**: `_to_padded_tensor`
    replicate-pads `gt` in both train and val, and the PSNR branch already
    cropped to each sample's true `(h, w)` before scoring — but `val_loss`
    was computed on the full padded tensors *before* that crop, so `val_loss`
    and PSNR were silently measured over different pixel counts on the same
    batch. Fixed by cropping before computing the loss too, sharing the same
    per-sample loop PSNR already needed for bucketed (`b > 1`) validation
    batches. **`val_loss` values from before and after this fix are not
    directly comparable** — expect a one-time discontinuity in logged
    `val_loss` at the point a run switches versions, not a new problem.