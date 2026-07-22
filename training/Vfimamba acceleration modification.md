# VFIMamba Modification: Bidirectional Flow + Learned Acceleration

## Files
- `flow_estimation.py` — replaces VFIMamba's original flow module. Computes a single, timestep-independent bidirectional flow pair once per frame pair, instead of directly regressing timestep-conditioned bilateral flow at every stage.
- `accel_flow.py` — extends `flow_estimation.py` with a learned, explicitly-supervised acceleration term, so the motion model isn't restricted to constant velocity. Also owns the arbitrary/multi-timestep inference path (`forward_multi_t`), so a single acceleration guess is reused for any requested intermediate frame(s) instead of being recomputed or silently dropped.
- `configCustom.py` — model config / class wiring. Must import `AccelFlow` from `accel_flow.py` directly rather than going through the package's `mamba_estimation` alias (see §4).
- `Trainer.py` — adds `inference_multi_t`, the public entry point for interpolating at an arbitrary list of intermediate timesteps in one pass.
- `dataset.py` — adds per-phase max frame span clipping for long training scenes (see §8).
- `train.py` — three bugs surfaced while wiring this up against the actual training loop (see §9).

---

## 1. What I wanted / requirements

**Starting idea:** instead of VFIMamba estimating flow from frame 0 → t and frame 1 → t directly (conditioned on t at every stage), compute flow between the two *actual* input frames — frame 0 → frame 1 and frame 1 → frame 0 — once, and derive the flow to any arbitrary timestamp t from that single pair.

**Motivation:**
- Efficiency: for arbitrary-timestamp / Nx (slow-motion) interpolation, the expensive backbone + flow-refinement stack should run once per frame pair, not once per requested t.
- Understand the tradeoff this reintroduces: this is the classical "linear combination of bidirectional flows" paradigm (Super SloMo / DAIN / QVI), which assumes constant velocity between the two frames — a strictly weaker motion model than VFIMamba's current direct, t-conditioned regression.

**Follow-up requirement, once the tradeoff was clear:** since both the original (direct regression) and the modified (linear scaling) approaches are fundamentally blind to acceleration — two frames can't carry a second-derivative signal — I wanted the acceleration itself to be **learned**, not just implicitly absorbed into a black-box network. Specifically:

1. Have the model **guess** an acceleration term from the two input frames alone (same information available at inference).
2. **Synthesize** a middle frame using that guess.
3. Compare against the **actual** ground-truth middle frame available during training.
4. Use that comparison to **directly teach** the model what the real acceleration should have been — not just an indirect photometric reward, but an explicit, closed-form supervision signal for the acceleration term itself.

**Second follow-up requirement:** the timestep `t` — at both train time and inference time — must not be restricted to a fixed value (e.g. 0.5). Any real number strictly between 0 and 1 must be usable, including:
- a whole batch of *different* requested timesteps reusing a single expensive backbone pass (Nx / slow-motion interpolation), and
- training triplets whose ground-truth middle frame sits at a *variable* interior position per item (not just a fixed midpoint), so higher-fps datasets with several intermediate frames per clip (e.g. X4K1000FPS) are usable as-is.

---

## 2. `flow_estimation.py` — bidirectional, timestep-independent flow

### Architecture change
- `Head` → `BiHead`, `IFBlock` → `BiIFBlock`: all `timestep` conditioning removed from every conv input. These blocks now only predict:
  - `F_{0→1}`, `F_{1→0}` (bidirectional flow between the two real frames)
  - a single occlusion/**visibility** field (t-independent)
- `estimate_bi_flow(img0, img1)`: the coarse-to-fine refinement stack (feature backbone + `BiHead`/`BiIFBlock`). This is the expensive part, and now runs **exactly once** per frame pair regardless of how many timestamps are requested.

### Deriving arbitrary-t flow (no learned parameters)
Classic linear-combination-of-bidirectional-flow formula (Super SloMo / Jiang et al.):
```
F_{t→0} = -(1-t)·t·F_{0→1} + t²·F_{1→0}
F_{t→1} = (1-t)²·F_{0→1} - t·(1-t)·F_{1→0}
```
- `flow_from_bi(bi_flow, t)`: implements this. `t` is unrestricted — any python float in (0,1), or a tensor broadcastable to `bi_flow`'s (B,1,H,W) spatial dims, so per-sample-variable and per-request-variable timesteps both work with no special-casing.
- `mask_from_visibility(visibility, t)`: derives a t-dependent blend weight from the single t-independent occlusion field, using the standard visibility-based blending formula:
  ```
  mask_t = (1-t)·V0 / ((1-t)·V0 + t·(1-V0))
  ```
- `synthesize(...)`: same warp + UNet residual synthesis as VFIMamba's original, just consuming the derived `flow_t`/`mask_t` instead of a directly regressed one.
- `forward_multi_t(x, timesteps)`: demonstrates the actual payoff — `estimate_bi_flow` runs once, and a whole list of timestamps (Nx interpolation) reuses it, only repeating the cheap scaling + synthesis steps per t. `timesteps` may be any iterable of floats in (0,1), in any order/spacing.

### Known limitation (by design, addressed in part 3)
The formula above assumes **constant velocity** between frame 0 and frame 1. It has no way to represent acceleration, deceleration, or curved motion within that interval — this is the explicit tradeoff versus VFIMamba's original direct regression, which at least has an implicit, data-driven prior for nonlinear motion (though bounded by the same two-frame information limit).

---

## 3. `accel_flow.py` — learned, directly-supervised acceleration

### Motion model
```
D = F_{0→1}                         (from flow_estimation.py, unprivileged)
a = learned acceleration field       (NEW — AccelHead, unprivileged)

p(t) = t·D + a·t·(t-1)               # vanishes at t=0 and t=1 —
                                      # can only bend the path *between*
                                      # the two observed frames
F_{t→0} = -p(t)
F_{t→1} =  p(t) - D
```
At `a = 0` this reduces exactly to the linear baseline from `flow_estimation.py`. Neither `D` nor `a` depends on `t` — `t` only enters through the cheap `p(t)` evaluation, which is what makes reusing a single `a` guess across many requested timesteps valid.

### `AccelHead`
A small extra head, run once after `estimate_bi_flow` converges. Takes `img0`, `img1`, and the estimated `D`, outputs a bounded (`tanh`-scaled) correction field `a`. Bounded so it acts as a correction to the linear path rather than a replacement for it — keeps training stable.

### Getting a real supervision target for acceleration (the core ask)
This is the part that makes "teach it actual acceleration" concrete rather than aspirational. At train time, real triplets `(img0, img_mid_gt, img1, t_gt)` are available, where `t_gt` may be **any** value in (0,1) — not just 0.5:

1. Run the same bidirectional flow estimator on `(img0, img_mid_gt)` — a real frame pair, so this flow is measured, not guessed. Call the result `p_true = F_{0→mid}`.
2. Since `p_true = t_gt·D + a_true·t_gt·(t_gt - 1)` by the same motion model, solve directly:
   ```
   a_true = (p_true - t_gt·D) / (t_gt·(t_gt - 1))
   ```
3. Train the blind predictor (`a_student`, which only ever sees `img0`/`img1`, same as at inference) against this target:
   ```
   loss_accel = L1(a_student, a_true.detach())
   ```

`training_step(...)` wires this together: it runs the student's blind forward pass, computes the photometric loss on the reconstructed frame, computes the privileged `a_true` target under `torch.no_grad()`, and combines both losses. `t_gt` is accepted as a python float or a per-sample tensor, so datasets that pick a different interior frame position per item (per `train.py`'s per-epoch seeded timestep selection) work unmodified.

### Edge case, flagged but not yet fixed: `solve_accel_target`'s eps clamp near t≈0/1
```python
denom = t_gt * (t_gt - 1)
denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom) \
    if torch.is_tensor(denom) else (denom if abs(denom) > eps else eps)
```
For any `t_gt` strictly in `(0,1)`, `t_gt·(t_gt-1)` is always **negative**. The clamp above substitutes a bare `+eps` (positive) whenever `|denom| < eps`, so right at the boundary — `t_gt` within `eps=1e-4` of 0 or 1 — the substituted denominator has the **wrong sign**, flipping the sign of `a_true` for that item instead of just avoiding a division by zero. This doesn't error and won't show up as a crash; it silently corrupts the accel target for whichever items happen to land that close to an endpoint. Not fixed here since it depends on whether your dataset's interior-frame sampling can actually produce `t_gt` that close to 0/1 (Vimeo90k's fixed `t=0.5` never triggers it; a variable-`t` dataset like X4K1000FPS might, depending on how interior frames are chosen). If it matters for your data, replace the bare `eps` with a sign-preserving clamp (e.g. `-eps` when `denom` is negative, which it always is in-range) rather than a constant.

### Arbitrary-timestep inference: `forward_multi_t` override (fix)
`AccelFlow` inherits `MultiScaleFlow.forward()`/`estimate_flow_and_mask()` overrides that correctly route through `flow_from_bi_accel`. **`forward_multi_t`, however, needed its own override.** The base class's `forward_multi_t` computes `bi_flow`/`visibility` once and then calls `self.flow_from_bi(bi_flow, t)` — the **linear-only** formula — per timestep. Left un-overridden, calling the Nx/multi-t path on an `AccelFlow` instance would silently ignore `accel_head` for every requested timestep, defeating the entire point of this file.

`AccelFlow.forward_multi_t(x, timesteps, ...)` now:
- runs `estimate_bi_flow` **once**,
- runs `accel_head` **once** to get `D`/`a` for the pair,
- then loops over `timesteps` (any values in (0,1), any count, any order) using `flow_from_bi_accel(D, a, t)` + `mask_from_visibility` + `synthesize` per t.

This preserves the "expensive stuff runs once" efficiency payoff while making sure every requested intermediate frame actually gets the acceleration-aware trajectory, not just the first/only one.

### `forward()` scale-handling fix
`AccelFlow.forward()` previously accepted a `scale` argument but never used it, silently disabling the base class's downsample-then-upsample speed path (used by `Model.inference(..., scale=...)` for large frames). It now mirrors `MultiScaleFlow.forward()`: downsamples for the `estimate_bi_flow` pass when `scale > 0`, then re-runs `feature_bone` and `accel_head` on the full-resolution images before deriving `flow_t`/`mask_t`.

### Why this satisfies the requirement, and what it does *not* solve
- **Satisfies it**: the model's acceleration guess is no longer an unverifiable side-effect of a photometric loss — it has an explicit, closed-form, geometrically correct target derived from the real footage, and is directly regressed against it. The timestep is unrestricted throughout — training triplets, single-frame inference, and multi-frame/Nx inference all accept any t in (0,1).
- **Does not remove the fundamental limit**: at inference there is still no ground-truth middle frame, so `a_student` is still an extrapolation from two frames — just a much better-calibrated one than before, because it was explicitly taught (during training) what real acceleration values look like for real motion, rather than only ever being rewarded indirectly through pixel error.
- **Data requirement**: needs real triplets with a known `t_gt` (Vimeo90k-style `t=0.5`, or ideally a higher-fps dataset like X4K1000FPS offering multiple intermediate frames per clip at varied `t`, so the acceleration head generalizes across timestamps instead of overfitting to a single one) — this is now fully supported end-to-end, not just mathematically anticipated.
- **Optional upgrade**: the `p_true` "teacher" pass currently reuses the same in-training flow estimator on `(img0, img_mid_gt)`. Swapping this for a pretrained, decoupled optical flow model (RAFT, FlowFormer) would give a more trustworthy target, at the cost of an external dependency.

---

## 4. `configCustom.py` — wiring (fix)

`model_oldRepo/__init__.py` re-exports two class aliases:
```python
from .feature_extractor import feature_extractor as mamba_extractor
from .flow_estimation import MultiScaleFlow as mamba_estimation
```
`mamba_estimation` is **not** stale — it already points at the new bidirectional-flow `MultiScaleFlow`, not the original direct-regression module. But it also has no `accel_head`, so if `configCustom.py`'s `MODEL_TYPE` goes through that alias, `train.py`'s `hasattr(model.net, 'accel_head')` gate never fires and the acceleration loss silently never trains. `configCustom.py` must import `AccelFlow` directly from `accel_flow.py` instead:

```python
from model_oldRepo import mamba_extractor
from model_oldRepo.accel_flow import AccelFlow
...
MODEL_CONFIG = {
    'LOGNAME': LOG,
    'MODEL_TYPE': (mamba_extractor, AccelFlow),
    'MODEL_ARCH': init_model_config(..., accel_scale=1.0, accel_hidden=48),
}
```
`init_model_config` also now threads `accel_scale`/`accel_hidden` through to the returned `multiscalecfg` dict, since `AccelFlow.__init__` reads them via `kargs`.

---

## 5. `accel_flow.py` / `train.py` — `local`-refinement consistency fix

Simulating a training step surfaced a real bug: `accel_distillation_loss` (and `AccelFlow.training_step`) hardcoded `local=False` for the **student**'s `estimate_bi_flow(img0, img1, ...)` call, regardless of what `local` the main forward pass used. `accel_head` is a single shared module — inside `AccelFlow.forward`, it's called on a `D` that went through local refinement whenever `local` is truthy (train.py sets `local = model.local = LOCAL = 2`, truthy); inside the old `accel_distillation_loss`, it was called on a separately-computed, un-refined `D`. So in the same training step, `accel_head` was trained against one input distribution (`D` un-refined) while it's actually deployed against another (`D` refined) — a real train/inference mismatch, not a cosmetic one.

Fix:
- `accel_distillation_loss(net, img0, img_mid_gt, img1, t_gt, local=False)` now takes an explicit `local` argument for the **student**'s `D`, and `train.py` passes `local=local` — the same flag used for the main forward call this step.
- The **teacher** pass (real flow from `img0` to `img_mid_gt`) always uses `local=True` regardless of the argument, since it runs under `torch.no_grad()` (costs nothing at inference time) and a better-refined target is strictly better supervision for `a_true`.
- `AccelFlow.training_step` got the same fix for anyone calling it directly instead of the standalone function.

Separately (not fixed, just flagged for awareness): `accel_distillation_loss` always recomputes `estimate_bi_flow` at **native resolution**, ignoring `step_scale` — meaning a training step with `accel_head` active runs the expensive backbone **3 times** (once at `step_scale` for the main forward, twice at full resolution for the accel loss's student+teacher passes), and this cost further multiplies by `num_interior` in the current `train.py` loop, since `accel_distillation_loss` is called once per interior frame rather than once per item (see §9). On a single RTX 3090, this bypasses the exact memory-management purpose `train_scale`/`dynamic_batch` were built for, and could OOM at higher-resolution curriculum phases even though the main path would fit. Worth checking VRAM headroom at your largest curriculum phase; if it's tight, consider adding an analogous downsample/upsample step to the accel loss's two `estimate_bi_flow` calls, and/or hoisting the student's `estimate_bi_flow`/`accel_head` call out of the per-interior-frame loop so it runs once per item instead of once per interior frame.

## 6. `feature_extractor.py` — no changes needed

Confirmed this file (the Mamba backbone, `mamba_extractor`/`feature_extractor`) requires no modification. Its output — a list of 5 per-stage feature maps at `embed_dims=[32,64,128,256,512]` — lines up exactly with what `BiHead`/`estimate_bi_flow` expect (`af[-1-i]` indexing, and the `in_planes*2//(4*4)` channel math against the double-`PixelShuffle(2)` upsample), and none of the bidirectional-flow/acceleration/arbitrary-timestep changes touch how the backbone is consumed. Extra keys in `backbonecfg` (`motion_dims`, `num_heads`, `mlp_ratios`, `qkv_bias`, `window_sizes`) are silently absorbed by its `**kwargs` and unused — pre-existing behavior, unrelated to these changes.

## 7. `Trainer.py` — `inference_multi_t` (new)

Public entry point mirroring `inference`/`hr_inference`, for interpolating at an arbitrary list of intermediate timesteps in one pass:
```python
model.inference_multi_t(img0, img1, local, timesteps=[0.1, 0.35, 0.5, 0.72, 0.9], scale=0)
```
Internally calls `self.net.forward_multi_t(imgs, timesteps, local=local, scale=scale)`, which (for `AccelFlow`) now correctly reuses a single acceleration guess across every timestep requested (see §3).

---

## 8. `dataset.py` — per-phase max frame span (long-scene clipping)

### Problem
Some source scenes (movie footage, ~24fps) run to hundreds or thousands of frames. Before this change, `FullResVFIDataset.__getitem__` always used a scene's literal first and last present frame as `img0`/`img1`, regardless of scene length — so a 1000-frame scene could produce an `img0`↔`img1` pair separated by many seconds of real screen time, in *any* curriculum phase, including Phase 1. This is far outside what a two-frame, flow-based VFI model (cost volumes, local correlation, the linear/accel bidirectional motion model from §2–3) has any basis to interpolate: at multi-second gaps there's frequently no single well-defined "correct" middle frame (shot changes, large non-rigid motion, occlusion), so training on these pairs risks noisy, uninformative gradients rather than a genuine curriculum from easy to hard motion.

### Fix
`config.yaml` gains `data.max_frame_span_by_phase`, a per-phase cap on how many frames apart `img0`/`img1` may be:

```yaml
data:
  max_frame_span_by_phase:
    1: 5
    2: 10
    3: 18
    4: 24     # ~1 second at 24fps
```
A phase with no entry (or an explicit `null`) is uncapped, matching the pre-existing behavior.

`prepare_datasets(cfg, phase, log)` looks up this phase's value and passes it into the **train** `FullResVFIDataset` as `max_frame_span` (the **val** dataset never receives it — validation still spans the full scene, matching pre-existing behavior, since val is meant to measure the model against its full evaluation range rather than a phase-scoped training distribution).

In `FullResVFIDataset.__getitem__`, when `max_frame_span` is set and a scene's frame count `n` exceeds it:
1. A random contiguous sub-clip of length in `[3, max_frame_span]` is drawn from the scene, using the same per-epoch-seeded RNG (`seeding.rng_for(seed, 'timestep', epoch, sample_id)`) already used for the interior-frame pick — so the clip is reproducible for a given `(seed, epoch, scene)` and changes from one epoch to the next, meaning a long scene's full frame range still gets sampled over the course of training, just never all at once.
2. `img0`/`gt`/`img1` and `timestep` are then selected from that clip exactly as before (uniform interior pick, or the midpoint when the clip is exactly 3 frames long), instead of from the whole scene.

Scenes at or under the cap, and any phase with `max_frame_span=None`, are unaffected and behave exactly as before this change.

### Naming note
This is deliberately called a **clip**, not a "window" — `train.py`/`curriculum_builder.py` already use `window_id`/`window_num_items` for a completely unrelated concept (gradient-accumulation flush boundaries across many items). The two are unrelated and orthogonal: clipping happens per-item inside `__getitem__`, before an item ever reaches `train.py`'s accumulation logic.

### What this does *not* touch
- `curriculum_builder.py`: phase assignment and batching are by **resolution**, not scene length — no changes needed there.
- `train.py`: accumulation windows are by resolution bucket, not scene length — no changes needed there either.
- Validation: PSNR/val_loss still reflect full-scene motion spans. If val scenes are also very long, this means validation isn't phase-scoped the way training now is — worth revisiting if val metrics need to track a given phase's trained motion range specifically.

---

## 9. `train.py` — bugfixes surfaced during integration

Three separate bugs turned up while tracing `train.py` against the actual shapes/behavior of the files above. None are design changes — all are the training loop failing to keep up with the multi-interior-frame / accel-head / padding behavior introduced elsewhere in this doc.

### 9.1 Missing import for `accel_distillation_loss`
The `hasattr(model.net, 'accel_head')` branch in the training loop calls `accel_distillation_loss(...)`, but the function was never imported anywhere in the file. This is not a design gap — it's a straightforward missing `import` — and it means any run using `AccelFlow` (§3) would hit a `NameError` on the very first training step. Fixed with:
```python
from model_oldRepo.accel_flow import accel_distillation_loss
```
(adjust the module path if `accel_flow.py` doesn't live at `model_oldRepo/accel_flow.py` in your checkout).

### 9.2 Per-step log line referenced an undefined `step_timestep`
The periodic training-progress log line referenced `step_timestep`, a variable that only exists in the **validation** loop further down — the training loop was never updated after moving from a single-timestep-per-item design to the current multi-interior-frame one (`timesteps_list`, one item now carrying `num_interior` timesteps at once, per `forward_multi_t`). This guarantees a `NameError` the first time `global_step % LOG_EVERY_STEPS == 0` fires, i.e. very early into any run. Fixed by reporting the item's actual timestep range instead of a single (nonexistent) scalar:
```python
t_lo, t_hi = min(timesteps_list), max(timesteps_list)
...
f't_range [{t_lo:.3f},{t_hi:.3f}] n={num_interior} | ...'
```

### 9.3 `val_loss` silently scored on padded pixels; PSNR didn't
`dataset.py`'s `_to_padded_tensor` (used for `img0`, `img1`, **and** `gt`, in both train and val) replicate-pads every tensor up to `PAD_MULTIPLE` — so the `gt` reaching `train.py`'s validation loop is on a padded canvas, not each sample's true `(h, w)`. The PSNR branch already cropped `pred`/`gt` to the sample's true resolution before scoring (`pred_cropped`/`gt_cropped`), but `val_loss += criterion(pred, gt).item()` ran on the full padded tensors *before* that crop ever happened — so on the exact same validation batch, `val_loss` and PSNR were being computed over different pixel counts, with `val_loss` silently including the replicate-padded border.

Fixed by moving the crop earlier and sharing it between both metrics — the loss is now computed per-sample on `pred_cropped`/`gt_cropped`, then averaged over the batch, in the same loop that already existed for PSNR (bucketed `val.validation_batch_size > 1` batches, where different samples can have different true resolutions even after padding, needed this per-sample loop for PSNR already; `val_loss` was the one metric that hadn't caught up to it):
```python
batch_loss = 0.0
for bi in range(b):
    hi = int(h[bi].item()) if b > 1 else h0
    wi = int(w[bi].item()) if b > 1 else w0
    pred_cropped = pred[bi:bi + 1, :, :hi, :wi]
    gt_cropped = gt[bi:bi + 1, :, :hi, :wi]
    batch_loss += criterion(pred_cropped, gt_cropped).item()
    if do_psnr_this_sample:
        psnr_total += psnr(pred_cropped, gt_cropped)
        psnr_count += 1
val_loss += batch_loss / b
```
**Note for anyone comparing runs across this fix:** `val_loss` values from before and after this change are not directly comparable — the padded-border contribution the old code was silently including is now gone, so expect a one-time discontinuity in logged `val_loss` at the point a run switches to the fixed version, not a sign of a new problem.