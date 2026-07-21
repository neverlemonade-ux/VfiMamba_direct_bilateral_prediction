# VFIMamba Curriculum Pipeline — How Everything Fits Together

This explains the whole pipeline end to end: what each file does, what the
output folders look like, and exactly how resolution, train_scale, dynamic
batch size / gradient accumulation, and timestep are each computed. This
revision expands every section with the *actual* config numbers your
tables use (rather than round hypothetical ones) so each worked example is
independently verifiable against `config.yaml`, and adds a full
function-by-function walkthrough of `resolution.py`.

> **Revision note (data-integrity fixes, earlier revision):**
> 1. `Extra/` batches are explicitly, by design, **excluded from training**
>    (Section 7).
> 2. `group_into_batches()` **chunks each resolution run into
>    `batch_size`-sized pieces** (Sections 6–7, 9) instead of building one
>    giant batch object per run.
> 3. The per-epoch timestep RNG is keyed on each scene's stable
>    `sample_id` (read back from `batch_metadata.yaml`), not on its full
>    on-disk path — so the timestep draw depends only on
>    `(seed, epoch, dataset, scene)`, never on curriculum *layout*
>    (Section 10).
> 4. `is_extra` (incomplete/leftover) batches are **never exported as
>    replay** — `split_retain_export()` filters them out before the ~20%
>    export target is even computed, so an incomplete batch always stays
>    in the phase whose own data produced it (Sections 6, 7).
>
> **Revision note (accumulation windows):** `curriculum_builder.py` builds
> **explicit, self-contained accumulation windows** *before* writing
> anything to disk (`group_batches_into_accum_windows()`), instead of
> leaving window boundaries implicit in `train.py`'s step counter. Every
> batch in a window — full or partial — is stamped (`_tag_windows()`)
> with a shared `window_id` and the window's **true total item count**
> (`window_num_items`), written into `batch_metadata.yaml`. `train.py`
> flushes the optimizer exactly when `window_id` changes and divides the
> accumulated loss by `window_num_items`, read straight off disk. Full
> details in Section 9.
>
> **Revision note (Extra/ routing bugfix, `curriculum_builder.py`):** a
> prior revision of that file computed `is_extra` batches correctly but
> `group_batches_into_accum_windows()` only ever returned
> `(full_groups, partial_groups)` — the `is_extra` batches it skipped
> during window grouping were flushed out of the in-progress run and
> silently dropped, never reaching `build_curriculum()`'s `final_order`,
> so `write_phase_folder()`'s (already-correct) `is_extra` branch never
> received anything and `Extra/` was always empty. Fixed by having
> `group_batches_into_accum_windows()` return a **third** list —
> `extra_batches`, in original stream order — which `build_curriculum()`
> now collects (own stream + every replay source) and appends to the end
> of each phase's final order, after the trailing partial windows. See
> Sections 6–7 for the corrected flow.
>
> **Revision note (documentation expansion, earlier revision):** no
> pipeline code changed. `resolution.py`'s own **module-level** docstring
> was separately corrected (it previously described the threshold lookup
> as "largest-below-or-equal-wins," which contradicted both
> `lookup_threshold_table()`'s own docstring and its actual `bisect_left`
> implementation — both of which are, and always were, a **ceiling**
> lookup: smallest table key that is `>= value`). That was a comment-only
> fix inside `resolution.py` — this document never asserted the wrong
> semantics, so nothing here needed to change as a result. See Section 11
> for the full behavioral explanation, including boundary/tie cases.
>
> **Revision note (this revision — manual validation dataset support):**
> `config.yaml` gained two new keys under `val:` — `use_manual_val`
> (bool, default `false`) and `manual_val_src` (path, only read when
> `use_manual_val` is `true`). When enabled, `curriculum_builder.py`'s
> `run()` **skips the seeded `split_train_val()` step entirely**: every
> sample discovered under `paths.src_dir` goes to training, and
> validation instead comes from a **second, independent**
> `discover_sequences()` call over `manual_val_src` (same `Dataset/Scene`
> layout, same frame-resolution logic as `src_dir`). The resulting
> `val_seqs` list is handed to the same, unmodified `write_val_folder()`
> — so `ValidationData/` ends up in exactly the same on-disk shape either
> way, and `dataset.py` / `train.py` require **zero changes**: neither
> file knows or cares whether `ValidationData/` came from a held-out
> split or a manually-provided folder. See Section 4 for the full
> behavior, including the train/val sample_id overlap warning.

---

## 1. The pipeline in one picture

```
config.yaml  (single source of truth: seed, paths, curriculum rules,
              dynamic_batch table, train_scale anchors, val mode --
              including whether val is a seeded split or a manual folder)
     │
     ▼
curriculum_builder.py   (run ONCE per dataset version)
     │
     │  1. discover_sequences()        -- recursively scan raw data
     │  2. split_train_val()  OR       -- seeded split, before anything else
     │     manual val branch           -- discover_sequences() on
     │                                    manual_val_src instead, all of
     │                                    src_dir goes to training
     │  3. annotate_resolution_buckets() -- round every train sample's (w,h)
     │  4. build_curriculum()          -- phase assignment, batching,
     │                                    replay, WINDOW TAGGING, interleave,
     │                                    trailing-partial-window placement,
     │                                    trailing-extra-batch placement
     │  5. write_phase_folder()        -- write out already-built batch
     │     write_val_folder()             folders (each carrying its
     │     write_top_level_metadata()      window_id/window_num_items)
     ▼
TrainingData/Phase1..4/ + ValidationData/ + *.yaml metadata   (on disk)
     │
     ▼
dataset.py   (read-only: turns the folders above into a torch Dataset,
              EXCLUDING Extra/ -- see Section 7 -- and surfacing each
              item's window_id / window_num_items -- see Section 9)
     │
     ▼
train.py + run_config.yaml   (run per phase: actual fine-tuning loop;
                               flushes on window_id, divides by
                               window_num_items -- see Section 9)
     │
     ▼
Trainer.py / model_oldRepo / configCustom.py   (the model itself)
```

Each arrow is a hard boundary in what's trusted:

- `config.yaml → curriculum_builder.py`: the *only* place curriculum-shaping
  numbers live — including whether validation comes from a seeded split
  or a manually-provided folder. `curriculum_builder.py` never hardcodes
  a batch size, phase threshold, retain ratio, or val source — everything
  comes from `cfg`.
- `curriculum_builder.py → disk`: the *only* place any randomness or
  grouping decision is made. Once `TrainingData/` and `ValidationData/`
  are written, every downstream file treats them as ground truth and
  re-derives nothing — this holds regardless of which val mode produced
  `ValidationData/`.
- `disk → dataset.py`: read-only, no shuffling, no re-grouping — a plain
  nested `sorted()` walk reproduces `curriculum_builder.py`'s exact
  intended order (Section 10).
- `dataset.py → train.py`: `train.py` trusts every field `dataset.py`
  hands it (`window_id`, `window_num_items`, `batch_uid`) as authoritative
  and does no independent boundary inference (Section 9).

You run `curriculum_builder.py` once (it rebuilds `TrainingData/` and
`ValidationData/` from scratch every time — it is **not** incremental, by
design: an incremental/append-in-place design would make the on-disk
layout depend on run history rather than purely on `(seed, src content)`
or `(src content, manual_val_src content)`, breaking "same inputs, same
layout"). Then you run `train.py` once per phase (1 → 2 → 3 → 4), each
time pointing `--restore_ckpt` at the previous phase's best/last
checkpoint.

---

## 2. What each file does

| File | Role |
|---|---|
| **`config.yaml`** | The *only* place you edit to retune anything that affects the curriculum itself: the global `seed`, `paths.src_dir`/`paths.dataset_root`, `pad_multiple`, phase bucket thresholds, retain/export ratio, the `dynamic_batch` table (batch_size + accum per resolution — this table is what determines a *full* window's target size), the `train_scale` anchors, and validation source (`val.use_manual_val` / `val.manual_val_src`, or the default seeded `val.val_split`). Shared by all three scripts so they can never disagree with each other. |
| **`config_loader.py`** | Tiny helper that loads `config.yaml` and fills in empty-dict defaults for missing sections. |
| **`seeding.py`** | The *one* place randomness is seeded. `seed_everything(seed)` is called once at the start of `curriculum_builder.py` and `train.py`. `rng_for(seed, *salt)` hands out an independent, reproducible `random.Random()` per pipeline stage (train/val split, replay export selection, replay destination assignment, per-item timestep) so different stages' randomness can't bleed into each other — without needing a second seed anywhere. Uses `hashlib` instead of Python's `hash()` specifically so it's reproducible across processes/machines. Note: `window_id` assignment (Section 6) is **not** randomness — it's a plain sequential counter (`itertools.count()`), since window identity only needs to be unique, not seeded. Note also: when `val.use_manual_val` is `true`, `'trainval_split'`'s `rng_for` salt is never drawn at all — see Section 4. |
| **`resolution.py`** | All the "given a resolution, look something up" math, shared by every other file: `padded_dims()` (round up to nearest `pad_multiple`), `long_edge()`, the generic `lookup_threshold_table()` (ceiling lookup — see Section 11), `resolve_dynamic_batch()` / `resolve_dynamic_batch_metadata()`, and `resolve_train_scale()` (piecewise-linear interpolation, a distinct mechanism from the table lookups). `resolve_phase()` also exists here but is currently **unused** — `curriculum_builder.py`'s `assign_phases()` reimplements the same two-line lookup inline rather than calling it (Section 11 flags this). One implementation of each *used* piece of math, no duplicated logic across files. |
| **`curriculum_builder.py`** | The dataset-organizing script (details in the rest of this doc). Reads raw data once, splits it (or, if `val.use_manual_val` is set, discovers a separate manually-provided validation folder instead of splitting — Section 4), buckets it, builds the 4-phase curriculum with replay, chunks each stream into explicit accumulation windows and tags them with `window_id`/`window_num_items` (`group_batches_into_accum_windows()` + `_tag_windows()`), interleaves full windows, appends trailing partial windows and then trailing `is_extra` batches, and writes `TrainingData/`, `ValidationData/`, and all the metadata files. `interleave_grouped_streams()` operates on complete, already-tagged window *groups*, not raw batches, and only ever handles full windows (partial windows and extras bypass it — see Section 6). |
| **`dataset.py`** | The `torch.utils.data.Dataset` that reads the *already-built* folders from `curriculum_builder.py`. Does no splitting/shuffling of its own — it trusts the on-disk order completely, applies no augmentation of its own (Section 10), deliberately **skips `Extra/`** (Section 7), and reads each batch folder's `window_id`/`window_num_items` back out of `batch_metadata.yaml` to hand to `train.py` per item (Section 9). Handles frame loading, padding, and interior-frame/timestep selection. Reads `ValidationData/` identically regardless of whether it was populated by a seeded split or a manual folder — it has no visibility into which mode produced it. |
| **`train.py`** | The actual training loop for **one phase**. Reads `config.yaml` (curriculum-wide settings) + `run_config.yaml` (per-run hyperparameters: epochs, lr, run name). Flushes the optimizer exactly on a `window_id` change and divides the accumulated loss by `window_num_items` (Section 9) — no longer infers accumulation boundaries from resolution or a step count. Computes train_scale per item, runs the forward/backward/optimizer-step loop, validates, checkpoints, and writes a run summary. |
| **`run_config.yaml`** | Per-run knobs only: `run_name`, `epochs`, `lr`, `pretrained`/`--restore_ckpt`, eval frequency, etc. Never anything that affects the curriculum itself — that's `config.yaml`'s job, so two runs of the same phase can't silently use different curriculum data. |
| **`configCustom.py`** | Defines `MODEL_CONFIG` (backbone/architecture hyperparameters, checkpoint log-name). Injected as the `config` module `Trainer.py` imports (`from config import *`). |
| **`Trainer.py`** | Thin wrapper around the VFIMamba network: builds it from `MODEL_CONFIG`, loads/saves state dicts, exposes `.net()` forward calls used by `train.py`. Not modified by this pipeline. |

---

## 3. Raw input layout (what you provide)

```
AllTrainingData/
├── Dataset1/
│   ├── Scene1/   <frames...>
│   ├── Scene2/   <frames...>
│   └── ...
├── Dataset2/
│   ├── Scene1/   <frames...>
│   └── ...
└── ...
```

`discover_sequences(src_dir, frame_names, extensions, min_frames=3)`
recursively scans **exactly two folder levels** below `src_dir`:
`<Dataset>/<Scene>/`. It iterates `sorted(src_dir.iterdir())` for datasets
and `sorted(dataset_dir.iterdir())` for scenes — so folder discovery
order is filesystem-independent and reproducible before any seeded logic
even runs. Any number of datasets, any number of scenes, any number of
frames per scene — nothing is hardcoded. A dataset folder with zero scene
subfolders is logged as a warning and skipped, not treated as fatal.

This exact function — same signature, same two-level walk, same
per-frame extension resolution — is also what reads your **manual
validation folder** when `val.use_manual_val` is enabled (Section 4).
There is no separate "validation discovery" code path; a manually
provided validation set is scanned with identically strict rules to
`src_dir`.

**Per-scene usability check, precisely:** for each `name` in
`data.frame_names`, `find_frame_file(scene_dir, name, extensions)` tries
`scene_dir / f'{name}{ext}'` for every `ext` in `image_extensions`, in
the order given, and returns the first match. A scene is usable only if
at least `min_frames` (default 3) of the configured `frame_names`
resolve this way — **not** however many image files happen to be
sitting in the folder. Two implications worth being explicit about:

- If a scene has, say, 5 physical frame files but your `data.frame_names`
  list only names 3 of them, that scene's `frame_count` is **3**, not 5
  — the other 2 files are simply invisible to the pipeline.
- Each frame name is resolved to an extension **independently, per
  scene** — `DatasetA/Scene1/frame1.png` and `DatasetB/Scene9/frame1.jpg`
  are both valid matches for the same `frame_names` entry `"frame1"`, so
  datasets can mix image formats freely.

If a scene passes the frame-count check, `get_resolution()` opens the
**first** image file in that folder (alphabetically, via
`find_first_image()`) to read `(w, h)` — this is a real image open
(`PIL.Image.open`), not a dimension read from any metadata file, and a
read failure (corrupt file, unsupported format) causes that scene to be
skipped with a warning, not to crash the whole run.

Each discovered scene becomes one **sample dict**:

```python
{
    'sample_id': f'{dataset_name}__{scene_name}',
    'dataset': dataset_name,
    'scene': scene_name,
    'name': sample_id,          # duplicate of sample_id, kept for readability
    'path': scene_dir,          # Path object, absolute
    'w': w, 'h': h,              # NATIVE resolution, read from the first frame
    'frame_count': len(present_frames),
    'present_frames': present_frames,   # actual resolved filenames incl. extension
}
```

This dict is the unit that flows through the **entire** rest of the
pipeline — split, bucketing, phase assignment, batching, export, window
tagging — all of it is just filtering/grouping/annotating this same dict,
never re-deriving `w`/`h`/`frame_count` from disk again. Samples
discovered from `manual_val_src` (Section 4) are the exact same dict
shape, produced by the exact same function — they just never enter
phase assignment/batching/replay, since they go straight to
`write_val_folder()`.

---

## 4. Train/validation split

There are **two mutually exclusive ways** validation data is produced,
selected by `val.use_manual_val` in `config.yaml`. Both ultimately feed
the same `write_val_folder()` call, so `ValidationData/`'s on-disk shape
is identical either way — `dataset.py` and `train.py` never know or care
which mode built it.

### 4a. Default: seeded split off of `src_dir`

`split_train_val(seqs, seed, val_split)` does one seeded shuffle-and-slice
over **every** discovered sample — training and validation both come from
`seqs`, `discover_sequences()`'s full return value, **before** any
resolution bucketing, phase assignment, or replay logic:

```python
rng = rng_for(seed, 'trainval_split')
shuffled = seqs[:]
rng.shuffle(shuffled)
n_val = max(1, int(len(shuffled) * val_split)) if shuffled else 0
val = shuffled[:n_val]
train = shuffled[n_val:]
```

Two details worth calling out precisely:

- `n_val = max(1, ...)` means **at least one** validation sample is always
  carved out (as long as any sample exists at all) — even with a tiny
  dataset and a small `val_split`, you can never accidentally end up with
  zero validation data from a nonzero input.
- The split happens on the **whole shuffled list**, so `val` is exactly
  the first `n_val` samples of one seeded permutation and `train` is the
  remainder — there's no independent randomness for which items land in
  which split beyond that single shuffle.

### 4b. Manual: a separately-provided validation dataset

```yaml
val:
  val_split: 0.2                  # ignored when use_manual_val is true
  validation_native_mode: true
  validation_batch_size: 1
  use_manual_val: false           # true -> skip the seeded split
  manual_val_src: ""              # only read when use_manual_val is true
```

When `val.use_manual_val` is `true`, `curriculum_builder.py`'s `run()`
**does not call `split_train_val()` at all**, and the `'trainval_split'`
`rng_for` salt is never drawn:

```python
train_seqs = seqs   # nothing held out of src_dir -- everything discovered
                     # under paths.src_dir goes to training
val_seqs = discover_sequences(manual_val_src, frame_filenames, extensions)
```

- `manual_val_src` is scanned with the **identical** `discover_sequences()`
  call used for `src_dir` — same `Dataset/Scene` two-level layout, same
  `frame_names`/`image_extensions` resolution rules, same
  `get_resolution()` check, same `min_frames=3` usability gate. It is
  **not** a different, looser code path.
- A missing or empty `manual_val_src`, or a `manual_val_src` that yields
  zero usable scenes, is fatal (`SystemExit`) rather than silently
  falling back to a seeded split — a misconfigured manual-val run should
  fail loudly, not quietly train with no validation set.
- **Overlap check:** since `train_seqs` is now *all* of `src_dir` with
  nothing held out, if the same `sample_id` (`<dataset>__<scene>`)
  happens to exist in both `src_dir` and `manual_val_src`, that scene
  would be trained on **and** validated on. This is logged as a warning
  (listing up to 5 overlapping `sample_id`s) rather than blocking the
  run, since you may intentionally be pointing `manual_val_src` at a
  copy or a curated subset — but it's worth checking if you see it
  unexpectedly.
- `val_split` itself is left in `config.yaml` but is **ignored** in this
  mode — it's not read anywhere in the manual-val branch.

Either way, validation samples are written to:

```
ValidationData/
└── <Dataset>/
    └── <Scene>/     (symlink to the original scene folder)
```

Dataset/Scene hierarchy is **preserved** here (unlike training batch
folders below), so you can always trace a val sample back to its source
— including, in manual-val mode, back to `manual_val_src` rather than
`src_dir`. `dataset.py` reads this in plain alphabetical order (dataset,
then scene) via `_list_val_seq_dirs()` — validation order carries no
curriculum meaning, so it doesn't need to, and this holds regardless of
which mode populated the folder.

**Note on `write_val_folder()` and re-running the pipeline:** this
function unconditionally does `shutil.rmtree(val_dir)` then rebuilds
`ValidationData/` from whatever `val_seqs` it's given, every run. This
means `ValidationData/` is never a safe place to manually drop files
yourself — anything placed there directly, outside of a
`curriculum_builder.py` run, will be deleted the next time the script
runs. If you want a custom validation set, it must live in its own
folder referenced via `val.manual_val_src`, not inside `dataset_root`'s
`ValidationData/` itself.

Validation mode is controlled by `val.validation_native_mode`
(`make_val_loader()` in `train.py`):
- `true` (default): `batch_size = 1`, native resolution, no bucketing.
- `false`: `batch_size = val_cfg.get('validation_batch_size', 1)` with
  default collate — only valid if your val set has already been
  normalized to one resolution outside this pipeline, since PyTorch's
  default collate requires every tensor in a batch to share a shape.
  This applies identically whether `ValidationData/` came from a seeded
  split or a manual folder.

Validation has no accumulation windows at all (Section 9 doesn't apply to
it) — every val item is forwarded independently, one at a time, in
`train.py`'s validation loop.

---

## 5. Resolution bucketing

For every **training** sample (never validation samples — they're
already set aside by Section 4, regardless of which mode produced them),
`annotate_resolution_buckets(train_seqs, multiple)` computes
`resolution.padded_dims(s['w'], s['h'], multiple)` once and stores the
result directly on the sample dict:

```python
s['padded_w'], s['padded_h'] = pw, ph
```

Worked example at `pad_multiple = 32`: a native `1917×1079` frame becomes
`padded_w, padded_h = 1920, 1088` (`round_up(1917, 32) = 1920`,
`round_up(1079, 32) = 1088` — see Section 11 for the exact arithmetic).

Everything downstream — phase assignment (`assign_phases`), batch-folder
grouping (`group_into_batches`), accumulation-window sizing
(`group_batches_into_accum_windows`), and every dynamic batch/accum/scale
lookup — reads `s['padded_w']`/`s['padded_h']` instead of recomputing
`padded_dims()` from `s['w']`/`s['h']` again. This matters for more than
just avoiding duplicate work: it guarantees a single sample's padded
resolution is computed **exactly once**, so it can never silently drift
between two call sites even if `pad_multiple` were (incorrectly) passed
differently somewhere — there's only one call site that matters.

---

## 6. Curriculum phases, replay, and accumulation windows

The key design rule here: **replay is marked and moved at the *batch*
level, and accumulation windows are built and tagged explicitly, before
anything is written to disk.** Batching a phase's own data *first*,
chunking each resulting stream into complete accumulation windows
*second*, and only ever exporting/importing whole, already-window-tagged
units, means training never has to guess where one gradient-accumulation
step ends and the next begins.

`build_curriculum(train_seqs, cfg, seed)` runs these steps, per phase
`p` in `(1, 2, 3, 4)`. Note that `train_seqs` here is unaffected by
which validation mode was used (Section 4) — in manual-val mode it's
simply all of `src_dir`'s discovered samples rather than
seeded-split-minus-val; everything from Section 5 onward behaves
identically either way.

### 6.1 `assign_phases()` — bucket each sample's own data by resolution

```python
phase_buckets = cfg['curriculum']['phase_buckets']
```

| max_long_side | phase |
|---|---|
| 1280 | 1 |
| 1920 | 2 |
| 2560 | 3 |
| ∞ | 4 |

For each training sample, `assign_phases()` computes
`resolution.long_edge(s['padded_w'], s['padded_h'])` and looks that up in
`phase_buckets` via `lookup_threshold_table` (ceiling semantics — Section
11), reading the `'phase'` field of the matching row. **Worked example:**
a sample padded to `1920×1088` has long edge `1920` → the smallest
`max_long_side` that is `>= 1920` is the row itself (`1920`) → **Phase 2**.
A sample padded to `1900×1068` (long edge `1900`) also lands in the
`1920` row → **Phase 2** as well, since `1900 <= 1920`. A sample at
`1921×...` (long edge `1921`) would skip past the `1920` row entirely and
land in the `2560` row → **Phase 3**.

Each phase's own data is then sorted **ascending by resolution** (the
"progress low→high" requirement), using a compound sort key:
`(long_edge, padded_w, padded_h, name)` — the trailing `name` (i.e.
`sample_id`) is a tie-breaker that makes the sort itself fully
deterministic even when two samples share an identical padded resolution
(Python's `sort()` is stable, but relying on insertion order alone would
make the result depend on `discover_sequences()`'s filesystem iteration
order rather than being independently reproducible from the data alone).

### 6.2 `group_into_batches()` — chunk each phase's own data into batches

Runs on each phase's own (already sorted) data, **before** any
retain/export decision is made. Two steps:

1. **Collapse into runs.** Walk the sorted list and start a new run every
   time `(padded_w, padded_h)` changes from the previous item. Because
   the list is already sorted ascending by resolution, a "run" here is
   exactly one contiguous resolution value's worth of samples.
2. **Chunk each run into `batch_size`-sized pieces**, where `batch_size`
   is looked up **per-run** via
   `resolution.resolve_dynamic_batch_metadata(pw, ph, dynamic_batch_thresholds, multiple)['batch_size']`
   — i.e. every run can have a different chunk size, because different
   resolutions have different configured `batch_size`.

**Worked example**, using the real `dynamic_batch` table below (Section
9.1), for a Phase-1 run of samples all padded to `960×544` (long edge
`960` → `batch_size = 7`, `accum_steps = 2`): if that run has **47**
items, `range(0, 47, 7)` produces chunks `[0:7], [7:14], ..., [42:47]` —
**six full 7-item batches**, plus **one 5-item leftover batch**. Only
that trailing 5-item chunk gets `is_extra=True` (`len(chunk) < bs` →
`5 < 7` → `True`); the six 7-item chunks all have `is_extra=False`. Each
batch dict looks like:

```python
{'resolution': (960, 544), 'items': [...7 or 5 seq dicts...],
 'origin_phase': 1, 'is_extra': False,  # or True for the leftover chunk
 'configured_batch_size': 7}
```

`origin_phase` is stamped here, at creation time, and **never changes**
even if the batch later travels to a different phase as replay — it
always names the phase whose own data produced the batch (Section 7).

### 6.3 `split_retain_export()` — which whole batches stay vs. get replayed

```python
retain_ratio = cfg['curriculum']['retain_ratio']   # e.g. 0.8
```

For phases 1–3 (phase 4 has no later phases, so it always retains
everything — `has_later_phases=False` short-circuits the function to
`return batches, []`), roughly `retain_ratio` (by **item** count, not
batch count) of a phase's own complete batches stay; the rest are
exported. Precisely:

1. `exportable = [b for b in batches if not b['is_extra']]` — `is_extra`
   batches are filtered out **before** anything else; they're never
   candidates for export at all, regardless of retain_ratio.
2. `target_export_items = total_items * (1 - retain_ratio)` — computed
   over `exportable` batches only. At `retain_ratio = 0.8`, that's 20% of
   the phase's own *exportable* item count.
3. `rng = rng_for(seed, 'export_select', phase_num)` shuffles a copy of
   `exportable`, then batches are pulled off the shuffled pool one at a
   time, accumulating `exported_items`, **until** `exported_items >=
   target_export_items`. Because whole batches are pulled (not partial
   items), the actual exported fraction will usually **overshoot**
   `target_export_items` slightly — by at most one batch's worth of
   items — never undershoot it.

**Worked example:** suppose Phase 1's exportable batches total 500
items across various resolutions, `retain_ratio = 0.8` →
`target_export_items = 100`. The shuffled pool might, say, pull a
23-item batch, then a 41-item batch (running total 64, still `< 100`),
then a 38-item batch (running total 102, now `>= 100`) — export stops
there, with exactly those 3 batches exported (102 items, not exactly
100) and every other batch retained.

`retained` preserves the **original ascending-resolution order** of
`batches` (including `is_extra` batches, which by construction always
land in `retained` since they were filtered out of the export pool
entirely) — computed as `[b for b in batches if id(b) not in export_ids]`,
which iterates `batches` (not the shuffled pool), so ordering survives.
`exported` is returned in the pool's shuffled order — its internal order
doesn't carry curriculum meaning, only reproducibility does.

### 6.4 `distribute_export_evenly()` — round-robin to later phases

```python
buckets = distribute_export_evenly(own_exported, seed, phase_num, later)
```

Re-shuffles the already-shuffled `export_batches` with a **different**
salted RNG (`rng_for(seed, 'export_distribute', phase_num)` — distinct
from `'export_select'` in 6.3, so the two shuffles are independent), then
assigns batches to destination phases with plain index modulo:
`buckets[dest_phases[i % len(dest_phases)]].append(batch)`. For Phase 1
(`dest_phases = [2, 3, 4]`) exporting 3 batches, that's one batch to each
of Phase 2, 3, 4 — for 7 batches, phases get 3/2/2 respectively (index
`0,3,6 → dest[0]`; `1,4 → dest[1]`; `2,5 → dest[2]`), i.e. as close to
even as integer division allows, deterministically.

Each batch moves as a **single indivisible unit** — `distribute_export_evenly`
never splits a batch's `items` list across two destination phases.

### 6.5 `group_batches_into_accum_windows()` — chunk into windows, per stream

This runs **separately** on each stream: a phase's own `retained`
batches, and — independently — each individual replay source it
received (keyed `'P1'`, `'P2'`, ... in `replay_pool[p]`), never mixing
streams together in one grouping pass.

Walking one stream's batches in order:

- An `is_extra` batch immediately flushes whatever same-resolution run
  was in progress (breaking it exactly like a resolution change would),
  is appended to `extra_batches` as an individual batch dict (not a
  window group), and window state resets to empty — **`is_extra` batches
  never participate in window grouping at all.**
- A non-extra batch either continues the current same-resolution run or,
  if its resolution differs from `cur_res`, flushes the current run
  first. `cur_target` — the number of batches a **full** window needs at
  this resolution — is looked up fresh at the start of each run via
  `resolve_dynamic_batch_metadata(...)['gradient_accumulation']`. Once
  `len(cur_group) == cur_target`, that group is immediately closed out
  into `full_groups` and a new group starts.
- At the very end, `_flush()` closes out whatever's left: if it happens
  to equal `cur_target` it's still a full group; otherwise (a run that
  ended before reaching a full window) it becomes a `partial_groups`
  entry.

**Worked partial-window example:** a replay stream contains 5 consecutive
batches, all at padded resolution `1920×1088` (long edge `1920` →
`accum_steps = 4` from the table in Section 9.1). Grouping produces **one
full window** of 4 batches, then a **trailing partial window** of the
remaining 1 batch (`window_is_partial=True` once tagged). If that same
stream instead had exactly 8 batches at that resolution, it would produce
**two full windows** of 4 and **no partial window** at all — a run whose
batch count is an exact multiple of `accum_steps` never produces a
partial group.

Returns `(full_groups, partial_groups, extra_batches)` — the third
element is the bugfix described in the revision notes at the top of this
document: it used to not exist, and `is_extra` batches encountered during
grouping were simply discarded rather than returned to the caller.

### 6.6 `_tag_windows()` — stamp `window_id` / item counts onto full and partial groups

```python
window_counter = itertools.count(1)   # ONE counter, shared across every
                                       # phase and every stream in the run
```

For every group (own-full, own-partial, replay-full, replay-partial —
**never** `extra_batches`, which skip this step entirely), `_tag_windows`
assigns:

- `window_id = f'W{next(counter):06d}'` — e.g. `W000001`, `W000002`,
  ... — globally unique because `window_counter` is created once in
  `build_curriculum()` and threaded through every call, across all 4
  phases and every stream within each phase. IDs are **never reused and
  never collide** between an own-stream and a replay-stream, or between
  phases.
- `window_num_items = sum(len(b['items']) for b in g)` — the window's
  **true, exact** total item count. For a full window this equals
  `batch_size × accum_steps` for that resolution by construction; for a
  partial window it's whatever smaller count actually landed there.
- `window_target_items = dyn['batch_size'] * dyn['gradient_accumulation']`
  — the theoretical full-window size, for logging/diagnostics only. On a
  full window this equals `window_num_items`; on a partial window it's
  strictly larger.
- `window_is_partial` — `True`/`False`, set from the caller's
  `is_partial` argument (own-partial and replay-partial groups both pass
  `True`; full groups pass `False`).

Every batch **within** a group receives the **same** `window_id` /
`window_num_items` / `window_target_items` / `window_is_partial` — the
tag describes the window, not the individual batch, and every batch in
one window shares one resolution by construction (a window is built from
one same-resolution run).

### 6.7 Assembling the final per-phase order

Inside `build_curriculum()`'s per-phase loop:

```python
ordered = interleave_grouped_streams(own_full, replay_full_groups)
for origin_stream, g in trailing_partials:
    ordered += [(origin_stream, b) for b in g]
for origin_stream, b in trailing_extras:
    ordered.append((origin_stream, b))
```

`trailing_partials` collects **every** partial group for this phase, in
a fixed order: this phase's own partial groups first
(`('__primary__', g)` for each `g` in `own_partial`), then each replay
source's partial groups, in the same order `replay_pool[p]` was
populated (i.e. the order phases were processed, `P1` before `P2` before
`P3` for a phase receiving replay from all three). `trailing_extras`
follows the identical pattern, one step later — own extras first, then
each replay source's extras in that same order — and is appended **after
every partial window**, last of all.

So the final on-disk order for one phase is, in this exact sequence:
**(1)** interleaved full windows (own + every replay source, spread via
`interleave_grouped_streams` — Section 6.8), **(2)** every trailing
partial window (own's, then each replay source's), **(3)** every
trailing `is_extra` batch (own's, then each replay source's) — routed to
`Extra/` by `write_phase_folder` regardless of where in this list they
sit (Section 7).

### 6.8 `interleave_grouped_streams()` — spreading replay through the phase

```python
own_seq = [('__primary__', b) for g in own_groups for b in g]
if not replay_pool_groups:
    return own_seq
```

If a phase received no replay at all (e.g. Phase 1 receives none, since
there's no earlier phase to replay from), this is a no-op flatten of the
phase's own full windows — no interleaving logic runs.

Otherwise: `n_gaps = len(own_groups) + 1` — the number of "slots" between
and around the phase's own window groups (one more slot than groups,
since replay can also land before the first own group or after the
last). For each replay source's list of full-window groups, each group
`i` (of `n` total from that source) is assigned to gap index
`min(int((i + 0.5) * n_gaps / n), n_gaps - 1)` — this spreads that
source's `n` groups as evenly as possible across the `n_gaps` slots
using each group's *midpoint* position, so a source with few groups still
gets spread across the phase rather than clumping at the start. Multiple
replay sources landing in the same gap are simply concatenated there (in
whichever order `replay_pool_groups.items()` iterates, which follows
insertion order — `P1` before `P2` before `P3`).

The final flatten walks gap 0, then own group 0 (if it exists), then gap
1, then own group 1, ... ending with the last gap — **never splicing
into the middle of a group**, so an accumulation window's batches always
stay contiguous on disk regardless of how much replay surrounds it.

All of this is fully deterministic given `seed` — same seed, same raw
data ⇒ byte-identical output every time, including every `window_id`
assignment (the counter's iteration order is itself a deterministic
function of the steps above, since phase order `1,2,3,4` and stream
order `own, then P1, then P2, ...` are both fixed). This is unaffected by
which validation mode (Section 4) is in use, since validation samples
never enter `build_curriculum()` at all.

---

## 7. Batch folders (Step 5), output format, `Extra/`, and windows

By the time a phase reaches this step, its content is already a merged
*sequence of whole, window-tagged batches* (Section 6) — own batches and
imported replay batches, interleaved (full windows), then appended
(partial windows), then appended again (extra batches) — but never
decomposed back into loose items. `write_phase_folder(phase_num,
ordered_batches, training_root, multiple, dynamic_batch_thresholds,
train_scale_anchors, seed)` just walks that sequence and writes each
batch out in order, assigning `BatchNNN` numbers and monotonically
increasing `global_index` values as it goes — **it does not group or
re-derive batches itself**, it only numbers and writes what Step 4
already built.

```
TrainingData/
├── Phase1/
│   ├── phase_metadata.yaml
│   ├── Batch001_960x544/                       (window W000001, full)
│   │   ├── batch_metadata.yaml
│   │   ├── 00000_own__DatasetA__Scene003/     (symlink)
│   │   ├── 00001_own__DatasetA__Scene007/
│   │   └── ...                                 (up to batch_size items)
│   ├── Batch002_960x544/                       (window W000001, full --
│   │   └── ...                                  same window_id as Batch001:
│   │                                            accum_steps(960x544) = 2
│   │                                            spans these 2 folders)
│   ├── Batch003_1024x576/                      (window W000002, full)
│   │   └── ...
│   ├── ...
│   └── Batch017_1280x704/                      (window W000038, PARTIAL --
│       └── ...                                  trailing remainder of a
│                                                run, appended near the end
│                                                of Phase1's stream; still
│                                                a normal numbered batch,
│                                                NOT under Extra/)
│   └── Extra/
│       ├── extra_metadata.yaml
│       ├── Batch001_960x544_origP1_n5/        (5-item remainder BATCH,
│       │   └── 00047_own__DatasetC__Scene011/   Section 6.2's example --
│       │                                        a different, batch-level
│       │                                        concept from a partial
│       │                                        WINDOW above; never has
│       │                                        window_id at all)
│       └── Batch002_1920x1088_origP2_n1/      (arrived as replay from
│           └── ...                              Phase 2, is_extra there
│                                                too, sits in Phase1's
│                                                Extra/ per Phase1's own
│                                                write_phase_folder call)
├── Phase2/
├── Phase3/
└── Phase4/
```

- **Batch folder name**: `BatchNNN_<paddedW>x<paddedH>` — every item
  inside one batch folder shares that exact resolution, and (outside
  `Extra/`) has exactly `batch_size` items for that resolution, by
  construction. `dyn['resolution']` (a `f'{pw}x{ph}'` string from
  `resolve_dynamic_batch_metadata`) supplies the resolution part of the
  name.
- **Scene folder name**: `<global_idx:05d>_<origin_label>__<dataset>__<scene>`
  — `origin_label` is `'own'` when `origin_stream == '__primary__'`, else
  `f'replay-{origin_stream}'` (e.g. `replay-P1`). `global_idx` is a
  zero-padded, monotonically increasing counter across the **whole
  phase** — it is **not** reset per batch, and **not** reset when writing
  moves from normal `BatchNNN/` folders into `Extra/` either, since both
  `normal_idx` (for `BatchNNN/` numbering) and `extra_idx` (for
  `Extra/BatchNNN/` numbering) are separate counters from `global_idx`
  (item ordering).
- Scene identity is **never flattened** — `_place()` symlinks
  (`os.symlink(..., target_is_directory=True)`, falling back to
  `shutil.copytree` if symlinks aren't supported — e.g. on Windows
  without Developer Mode) the *original* scene directory in as a whole
  folder, so each scene keeps every one of its original frames. This is
  the exact same `_place()` used by `write_val_folder()` (Section 4), so
  a manually-provided validation scene is symlinked in exactly the same
  way a training scene is.
- **`window_id` spans batch-folder boundaries, not `Extra/` boundaries.**
  A window made of `accum_steps > 1` batches is *several consecutive
  `BatchNNN/` folders sharing the same `window_id`* — the folder
  boundary and the window boundary are different things: the folder
  boundary is where `write_phase_folder` happened to split
  `batch['items']` into a separate directory, the window boundary is an
  explicit, written-to-disk id that `train.py` treats as authoritative.

### `Extra/` — undersized *batches*, excluded from training by design

Each phase folder has one `Extra/` subfolder, a sibling of `Batch001`,
`Batch002`, etc. Because batching chunks by `batch_size` (Section 6.2)
*before* any replay marking or window grouping, the only way a batch can
end up undersized is if a resolution run's final leftover chunk doesn't
fill a full `batch_size`.

**`dataset.py` does not train on `Extra/` content.**
`_list_train_seq_dirs()` explicitly skips any directory entry named
`Extra` or `phase_metadata.yaml` (`_PHASE_SKIP_ENTRIES = {'Extra',
'phase_metadata.yaml'}`) when walking a phase folder with
`sorted(os.listdir(phase_dir))`. `is_extra` batches are also **never**
included in `group_batches_into_accum_windows()` (Section 6.5) — they
are skipped there the same way a resolution change is, so they can never
accidentally end up with a `window_id`. This is why `batch_metadata.yaml`
for an `Extra/` batch has no `window_id` / `window_num_items` /
`window_target_items` / `window_is_partial` fields at all — those keys
are only ever written for non-extra batches (see the metadata section
below).

**An `Extra/` batch also never leaves the phase whose own data produced
it** — `origin_phase` inside `batch_metadata.yaml` / the folder name
(`Batch{idx:03d}_{res}_origP{origin_phase}_n{scene_count}`) will always
equal the phase's own data, even when the batch arrived here **as
replay**: `split_retain_export()` filters `is_extra` batches out of the
export pool entirely (Section 6.3), so an undersized batch is *never* a
candidate for `distribute_export_evenly()` in the first place. The only
way an `Extra/` folder's name shows `origP2` while sitting inside
`Phase1/Extra/` is if a **full-size, non-extra** batch was exported from
Phase 2 as replay into Phase 1, and *that* batch's stream, once inside
Phase 1, produced a trailing remainder shorter than `accum_steps`... no —
that's a partial *window*, which stays as a normal `BatchNNN/`, never
`Extra/`. In practice, cross-phase `origP{n}` entries under `Extra/`
would only arise if a replay-imported batch were itself somehow
undersized at its *origin* phase — which `split_retain_export()`
prevents by construction. So in this implementation, every batch you'll
actually find under any phase's `Extra/` will show `origP` equal to that
same phase number.

**Do not confuse an undersized *batch* (`is_extra=True`, routed to
`Extra/`, no `window_id`) with a *partial window* (`window_is_partial:
true`, still a normal numbered `BatchNNN/` folder, still has a
`window_id`, still trained on).** They are two independent concepts at
two different granularities:

| | granularity | still trained on? | folder location | has `window_id`? |
|---|---|---|---|---|
| `is_extra` batch | one `batch_size`-chunk | **no** | `Extra/BatchNNN_..._origP<n>_n<count>/` | no |
| partial window | one accumulation window (1+ batches) | **yes** | ordinary `BatchNNN/`, appended near end of phase | yes |

### Metadata files

- **`batch_metadata.yaml`** (one per batch folder, including those under
  `Extra/`): `batch_index`, `resolution`, `batch_size`,
  `gradient_accumulation`, `effective_batch_size`, `train_scale`,
  `num_scenes`, and a `scenes` list (each entry: `global_index`,
  `sample_id`, `dataset`, `scene`, `origin`, `folder`,
  `source_resolution`, `frame_count`). **Non-`Extra/` batches
  additionally carry `window_id`, `window_num_items`,
  `window_target_items`, `window_is_partial`** — the fields `train.py`'s
  accumulation logic reads (via `dataset.py`). `Extra/` batches instead
  carry `origin_phase` and `scene_count`. `dataset.py` reads this file
  back to recover each scene's `sample_id` (Section 10) and, for
  non-`Extra/` batches, its `window_id`/`window_num_items` (Section 9) —
  so it must stay in sync with the folders it describes; nothing in this
  pipeline regenerates it independently of `write_phase_folder()`.
- **`phase_metadata.yaml`** (one per phase): `phase`, `seed`,
  `num_items` (everything written, including `Extra/`),
  `num_trainable_items` (everything **outside** `Extra/`),
  `num_batches` (normal batches only), `num_extra_batches`,
  `num_windows` (count of **distinct** `window_id`s among non-extra
  batches in this phase), `num_partial_windows` (how many of those are
  partial), and the full list of non-extra `batches`' metadata blocks.
- **`extra_metadata.yaml`** (only written if `Extra/` ends up non-empty
  for that phase — `if extra_idx:` guards the write): `phase` plus a
  `batches` list mirroring every batch currently sitting in `Extra/`.
- **`curriculum_metadata.yaml`** (top-level, one per build, written by
  `write_top_level_metadata`): `seed`, `val_split`,
  `num_total_sequences`/`num_train_sequences`/`num_val_sequences`, the
  full list of validation samples (sorted by `sample_id`), and a
  per-phase summary (`num_items`, `num_batches`, `num_extra_batches`,
  `num_windows`, `num_partial_windows`) keyed by phase number. In
  manual-val mode, `val_split`'s value here reflects whatever was in
  `config.yaml` but was not actually used to select validation samples —
  check `val.use_manual_val`/`val.manual_val_src` in the `config.yaml`
  that produced a given `dataset_root` if you need to know which mode
  built it.

`curriculum_builder.py`'s final console/log summary (`log_final_summary`)
reports, per phase: batch/extra-batch counts, trainable/extra scene
counts, and window/partial-window counts — then totals across all 4
phases, plus the validation count (from either mode, indistinguishably).

---

## 8. Train scale — how it's actually computed

`resolution.resolve_train_scale(w, h, anchors, multiple)`:

1. Pads `(w, h)` to the nearest `multiple` via `padded_dims()`, takes the
   long edge via `long_edge()`.
2. Sorts `anchors` by their x-coordinate (long edge), then
   piecewise-linearly interpolates:
   ```
   anchors:
     - [1024, 1.0]
     - [2048, 0.5]
     - [4096, 0.25]
   ```
   Long edge `<= 1024` → scale **1.0** (clamped, `pts[0][1]`). Long edge
   `>= 4096` → scale **0.25** (clamped, `pts[-1][1]`). Between two
   anchors, e.g. long edge `1536` (strictly between `1024` and `2048`):
   `t = (1536 - 1024) / (2048 - 1024) = 0.5` →
   `scale = 1.0 + 0.5 * (0.5 - 1.0) = 0.75`.
3. `train.py` looks this up **fresh, per item**, from that item's own
   native `(w, h)` (not the batch's nominal resolution — though for a
   sample from `TrainingData/`, padded `(w,h)` is fixed per batch anyway
   since a batch is one resolution by construction), and passes it
   straight into that step's forward call:
   `model.net(imgs, timestep=step_timestep, scale=step_scale, local=local)`.

Because it's computed per item at forward time (not accumulated across
steps), there's no interaction with gradient accumulation or with
accumulation-window boundaries — train_scale and windowing are fully
independent axes; a single accumulation window can, in principle, contain
items that resolve to different train_scale values if `w`/`h` varied
within it (in practice they won't, since a window is built from one
same-resolution run — but the *mechanism* doesn't assume that).

---

## 9. Accumulation windows, in full — construction, tagging, and consumption

### 9.1 The `dynamic_batch` table — still defines *target* window size

`resolution.resolve_dynamic_batch_metadata(w, h, thresholds, multiple)`
looks up `(batch_size, accum_steps)` from your `dynamic_batch.thresholds`
table:

| max_long_side | batch_size | accum_steps | effective (target) batch |
|---|---|---|---|
| 640 | 10 | 1 | 10 |
| 960 | 7 | 2 | 14 |
| 1280 | 5 | 2 | 10 |
| 1920 | 3 | 4 | 12 |
| 2560 | 2 | 5 | 10 |
| 3840 | 1 | 10 | 10 |
| ∞ | 1 | 10 | 10 |

This table is read in exactly two places now:

- **`curriculum_builder.py`**, at build time, in
  `group_into_batches()` (to chunk each run into `batch_size`-sized
  pieces) and in `group_batches_into_accum_windows()` /
  `_tag_windows()` (to decide how many consecutive same-resolution batch
  folders make up one **full** window — `accum_steps` — and to compute
  `window_target_items` for logging).
- **`train.py`**, at train time, purely for **logging context** (the
  "would-be-full" denominator in the training log — see 9.3) and for
  `train_scale`. It no longer supplies the accumulation target or the
  loss divisor — those come from `window_num_items`, read off disk.

### 9.2 What curriculum_builder.py hands off: `window_id` + `window_num_items`

As described in Section 6, every batch folder (outside `Extra/`) is
written with `window_id` and `window_num_items` in its
`batch_metadata.yaml`. `dataset.py` reads both straight out of that file
(`_list_train_seq_dirs()`) and returns them per item alongside the
existing `batch_uid` (the batch folder's own path — present for
traceability/logging only, never for accumulation logic; it can change
**multiple times within one window**, since a window can span several
consecutive `BatchNNN/` folders).

### 9.3 What train.py does with them

Since the physical DataLoader batch is always 1 (`batch_size=1` in
`train_loader`), `train.py` still accumulates gradient over a run of
consecutive items before calling `optimizer.step()` — but the window
boundary and the divisor are now both read directly off disk instead of
computed:

```python
for step, (img0, gt, img1, timestep, h, w,
           batch_uid, window_id, window_num_items) in enumerate(train_loader):
    ...
    # window_id is the direct, authoritative boundary signal --
    # guaranteed to change exactly at window boundaries.
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
        cur_effective_batch = window_num_items   # true count -- exact for
        cur_window_id = window_id                # both full and partial windows

    ...
    loss = criterion(pred, gt) / cur_effective_batch
    scaler.scale(loss).backward()
    steps_since_update += 1

    is_last_step_of_epoch = (step + 1) == n_steps
    # defensive fallback only -- with window_num_items exact by
    # construction, this should always agree with the window_id check above
    triggered_update = steps_since_update >= cur_effective_batch or is_last_step_of_epoch
    if triggered_update:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        steps_since_update = 0
        cur_effective_batch = None
        cur_window_id = None
```

Note the **two** flush sites: one at the *top* of the loop (fires when
the incoming item's `window_id` differs from the window currently being
accumulated — catches the boundary *between* windows) and one at the
*bottom* (fires once `steps_since_update` reaches the current window's
true size, or at the last step of the epoch — catches the *end* of the
very last window in the epoch, which has no "next different window_id"
item to trigger the top check). In ordinary operation these two checks
always agree, since `window_num_items` is exact by construction — the
bottom check is a defensive fallback, not the primary mechanism.

The per-item `resolution.resolve_dynamic_batch()` lookup is still made
every step, but purely to print
`window_items={cur_effective_batch}/{item_effective_batch} (NN% full)`
in the log line — a diagnostic showing how full a given window was
relative to its resolution's theoretical target (100% for a full window,
less for a partial one), nothing more; it never feeds back into
`cur_effective_batch` itself.

### 9.4 Why the boundary can never be inferred wrong

Before this design, a resolution run whose batch-folder count wasn't an
exact multiple of `accum_steps` would have left a partially-accumulated
window that could silently absorb items from whatever resolution came
next in curriculum order — a different resolution, with a different,
wrongly-sized effective batch target still in effect. That's structurally
impossible now:

- The boundary is decided **once, at build time**
  (`group_batches_into_accum_windows`), never inferred at train time.
- A trailing remainder that doesn't fill a full window is never silently
  absorbed into the next resolution's items — it becomes its own tagged,
  correctly-sized partial window, physically separated from other
  resolutions' windows in curriculum order (appended after all full
  windows for that phase), and trained with a loss divisor
  (`window_num_items`) that matches exactly what's actually in it.
- `train.py`'s step-count check is retained only as a defensive fallback
  for `is_last_step_of_epoch`; in ordinary operation it always agrees
  with the `window_id` check.

---

## 10. Timestep / interior-frame selection

In `dataset.py`, per training item (`FullResVFIDataset.__getitem__`,
`mode='train'`):

```python
sample_id = <read back from batch_metadata.yaml for this scene folder>
epoch_rng = seeding.rng_for(seed, 'timestep', epoch, sample_id)
n = len(present)                                        # frames actually matched from data.frame_names
k = 1 if n == 3 else epoch_rng.randint(1, n - 2)        # interior index, deterministic per (seed, epoch, sample)
img0, gt, img1 = present[0], present[k], present[-1]
timestep = k / (n - 1)
```

- `img0` = first present frame (t=0), `img1` = last present frame (t=1),
  `gt` = the frame at index `k` (interior), used as the supervision
  target at `timestep = k/(n-1)`.
- **`n == 3`** (the only case possible with a 3-name `frame_names` list):
  only one interior index exists (`k=1`), so `timestep` is always
  exactly **0.5** — no RNG draw happens at all in this case (the `if`
  short-circuits before `epoch_rng` would even be needed, though
  `rng_for` is cheap enough that this is more a correctness note than a
  performance one).
- **`n > 3`**: `k` is drawn from `[1, n-2]` (inclusive both ends, via
  `randint`) using `seeding.rng_for(seed, 'timestep', epoch, sample_id)`.

**How `sample_id` is obtained:** `_list_train_seq_dirs()` reads each
batch folder's `batch_metadata.yaml` — which `curriculum_builder.py`
already writes with an exact `sample_id` (`<dataset>__<scene>`) per scene
— and pairs every scene directory with that value (alongside `window_id`
/ `window_num_items`, read from the same file — Section 9). `__getitem__`
keys the timestep RNG on that `sample_id`, **not** on the scene's
on-disk path, and **not** on its `window_id` either.

This is a deliberate independence: `window_id` is a *layout* detail
(which accumulation window a scene's batch happens to belong to, which
can shift if `dynamic_batch.thresholds`, `retain_ratio`, or the
window-building logic itself ever changes) — exactly the kind of
incidental detail this `sample_id`-keying was designed to insulate the
timestep draw from in the first place. So: the accumulation-window design
in Sections 6–9 changes *how items are grouped for gradient accumulation*,
but has **zero effect** on *which interior frame gets picked* for any
given scene, on any given epoch, at any given seed.

Everything else about timestep selection:

- **`dataset.py` applies no augmentation at read time** — no flip, no
  crop, no color/exposure/lens adjustment; `_to_padded_tensor()` only
  converts to a float tensor in `[0,1]` and pads (replicate) up to
  `pad_multiple`. Any augmentation happens upstream, before
  `curriculum_builder.py` ever runs, baked into the frames on disk.
- **Validation is different on purpose**:
  `_pick_val_interior_indices(n, max_per_seq)` precomputes a fixed,
  deterministic list of interior indices per scene **once**, at dataset
  construction (`__init__`, not `__getitem__`) — if `n - 2 <=
  max_per_seq` every interior index is used; otherwise
  `np.linspace(0, len(interior)-1, max_per_seq)` picks an evenly-spaced
  subset. Every val interior frame enumerated this way is a **separate
  dataset item** (unlike train, where one scene = one item per epoch with
  a re-rolled `k`) — so `len(val_set)` can exceed the number of val
  scenes. Val doesn't use `rng_for` for timestep at all, and has no
  accumulation windows at all (Section 4). This behavior is identical
  whether `ValidationData/` was populated by a seeded split or a manual
  folder — `dataset.py` treats every scene under `ValidationData/`
  the same regardless of provenance.

**Plumbing note:** `train.py` must call `train_set.set_epoch(epoch)` at
the start of every epoch, and the `DataLoader` must keep
`persistent_workers=False` (the default, set explicitly in `train.py`),
so that workers are re-spawned — and re-pickle the just-updated
`self.epoch` — every epoch. If `persistent_workers` were ever turned on,
already-running workers would keep the Dataset snapshot from the *first*
epoch and silently stop seeing later `set_epoch()` calls. `window_id` /
`window_num_items` are static per-item metadata read from disk, not
stateful like `self.epoch`, so they need no equivalent re-pickling
safeguard.

---

## 11. `resolution.py` in detail

This file has no dependency on any other pipeline file — it's pure
resolution/threshold math, imported by `curriculum_builder.py`,
`dataset.py` (indirectly, via values `train.py` computes), and
`train.py`.

### `round_up(v, multiple)`

```python
return ((v + multiple - 1) // multiple) * multiple
```

Standard "round up to nearest multiple" via integer ceiling division.
`round_up(1917, 32)`: `(1917+31)//32 = 1948//32 = 60`, `60*32 = 1920`.
`round_up(1079, 32)`: `(1079+31)//32 = 1110//32 = 34`, `34*32 = 1088`.
Values already an exact multiple pass through unchanged:
`round_up(1920, 32) = 1920`.

### `padded_dims(w, h, multiple)`

Just `(round_up(w, multiple), round_up(h, multiple))`. This is the exact
shape `dataset.pad_to_multiple()` will produce for a frame of native size
`(w, h)` — the two functions are guaranteed to agree because they're
built from the same rounding rule (one at metadata-computation time in
`curriculum_builder.py`/`train.py`, one at tensor-padding time in
`dataset.py`), even though `pad_to_multiple()` itself is a torch-tensor
padding function and doesn't call `round_up()` directly — it derives the
same pad amount from `(multiple - h % multiple) % multiple`, which is
arithmetically equivalent.

Because `round_up` is idempotent on values that are already multiples,
`padded_dims()` is safe to call on **either** raw native `(w, h)` **or**
an already-rounded `(padded_w, padded_h)` pair — both
`resolve_dynamic_batch_metadata` (called with native `(w,h)` from
`train.py`, and with already-padded values from `curriculum_builder.py`,
which stores them as `s['padded_w']`/`s['padded_h']`) rely on this.

### `long_edge(w, h)`

`max(w, h)`. Used everywhere a single scalar "how big is this frame"
number is needed for a threshold-table lookup — phase assignment,
dynamic batch/accum, and (inside `resolve_train_scale`) train_scale
interpolation.

### `lookup_threshold_table(value, rows, key='max_long_side')` — the ceiling lookup

```python
ordered = sorted(rows, key=lambda r: r[key])
keys = [r[key] for r in ordered]
idx = bisect.bisect_left(keys, value)
if idx >= len(ordered):
    idx = len(ordered) - 1
return ordered[idx]
```

This is a **ceiling** lookup: it returns the row with the **smallest**
`key` that is **`>= value`** — not the largest key that is `<= value`
(a "floor" lookup, which this function does *not* implement, despite an
earlier, since-corrected comment in this file's own module docstring
claiming otherwise — see the revision note at the top of this document).

Mechanically: `bisect.bisect_left(keys, value)` finds the leftmost index
at which `value` could be inserted into the already-sorted `keys` list
while keeping it sorted. Two cases matter:

- **`value` matches a key exactly** — e.g. `keys = [640, 960, 1280, ...]`,
  `value = 960`. `bisect_left` returns the index of that existing `960`
  entry (index `1`), **not** the index after it — so an exact match
  selects **that row itself**, not the next one up. A sample whose long
  edge is precisely `960` gets the `960` row's `batch_size`/`accum_steps`,
  not the `1280` row's.
- **`value` falls strictly between two keys** — e.g. `value = 700`
  (between `640` and `960`). `bisect_left` returns the index of `960`
  (the first key `> 700`), so the `960` row is selected — this is the
  "round up to the next threshold" behavior the phase-bucket and
  dynamic-batch tables are designed around (both tables' `max_long_side`
  values are meant to be read as "the largest input this row's settings
  apply to," which only makes sense under ceiling semantics).
- **`value` exceeds every key** — `idx` comes back `>= len(ordered)`, so
  it's clamped to `len(ordered) - 1`, returning the **last** row. This is
  exactly why every table in `config.yaml` is expected to end with a
  `.inf`-keyed catch-all row (`max_long_side: .inf`): without it, any
  value larger than the largest finite threshold would silently fall
  back to whatever the last *finite* row happens to be, rather than an
  explicit "anything huge lands here" bucket.

`rows` does **not** need to be pre-sorted by the caller — the function
sorts a copy (`sorted(rows, ...)`) internally every call, so callers can
pass `cfg['curriculum']['phase_buckets']` or
`cfg['dynamic_batch']['thresholds']` in whatever order they appear in
`config.yaml`.

### `resolve_phase(w, h, phase_buckets, multiple)` — currently unused

```python
pw, ph = padded_dims(w, h, multiple)
return lookup_threshold_table(long_edge(pw, ph), phase_buckets)['phase']
```

This is a complete, correct convenience wrapper around exactly the
two-line pattern `assign_phases()` uses — but nothing in
`curriculum_builder.py`, `dataset.py`, or `train.py` actually calls it;
`assign_phases()` reimplements the same lookup inline instead (Section
6.1). It's dead code as of this revision. Not a bug — it produces the
right answer if called — just an unused duplicate of logic that already
exists elsewhere, which is worth knowing about if you're looking for
"the one place phase assignment happens" and don't want to be misled into
thinking this function is part of the live call path.

### `resolve_dynamic_batch(w, h, thresholds, multiple)`

```python
pw, ph = padded_dims(w, h, multiple)
row = lookup_threshold_table(long_edge(pw, ph), thresholds)
return row['batch_size'], row['accum_steps']
```

A 2-tuple convenience wrapper used only by `train.py`, purely for the
per-step logging denominator (Section 9.3) — it is **not** used for the
accumulation target or loss divisor, both of which come from
`window_num_items` instead.

### `resolve_dynamic_batch_metadata(w, h, thresholds, multiple)`

Same lookup, richer return value — everything a batch folder's metadata
needs:

```python
{
    'resolution': f'{pw}x{ph}',
    'padded_w': pw, 'padded_h': ph,
    'batch_size': batch_size,
    'gradient_accumulation': accum,
    'effective_batch_size': batch_size * accum,
}
```

Called throughout `curriculum_builder.py` (batching, window sizing,
metadata writing) with either raw native or already-padded `(w, h)` —
safe either way, per `padded_dims()`'s idempotency (see above).

### `resolve_train_scale(w, h, anchors, multiple)`

The one function in this file that is **not** a threshold-table lookup —
it's piecewise-linear interpolation over `(long_edge, scale)` anchor
points, clamped at both ends. See Section 8 for the full worked example
(`1536 → 0.75`). Implementation detail worth noting: the final `return
pts[-1][1]` after the loop is unreachable in practice, since the two
clamp checks above it (`le <= pts[0][0]` and `le >= pts[-1][0]`) already
cover every value outside the anchor range, and every value inside it is
caught by the loop's `x0 <= le <= x1` check against consecutive anchor
pairs — it exists purely as a defensive fallback.

---

## 12. Quick-reference: invariants worth remembering

- **`is_extra` (batch-level) and `window_is_partial` (window-level) are
  independent.** An `is_extra` batch never has a `window_id` at all; a
  partial window is a completely normal, trained-on, numbered batch
  folder that just happens to be smaller than a full window. See the
  comparison table in Section 7.
- **`window_id` changes exactly at accumulation-window boundaries** —
  never mid-window, even though the *folder* (`batch_uid`) it's attached
  to can change multiple times within one window (Section 9.2).
- **Ceiling, not floor**: every `max_long_side`-keyed table lookup in
  this pipeline (`phase_buckets`, `dynamic_batch.thresholds`) resolves to
  the smallest configured threshold that is `>=` the sample's long edge
  — an exact match selects that row, not the next one up (Section 11).
- **`origin_phase` never changes once a batch is created** — it always
  names the phase whose own data produced the batch, regardless of which
  phase's folder it's ultimately written into as replay (Sections 6.2,
  7).
- **`sample_id` (not on-disk path, not `window_id`) is the only input to
  the per-item timestep RNG** — curriculum layout changes (different
  `retain_ratio`, different window sizes, etc.) never change which
  interior frame a given scene draws on a given epoch (Section 10).
- **Nothing downstream of `curriculum_builder.py` re-derives randomness.**
  `dataset.py` and `train.py` trust the on-disk order and the metadata
  files completely — the seeded shuffle-and-slice steps all happen once,
  inside `curriculum_builder.py`, before anything is written.
- **`ValidationData/` is always fully rebuilt, never hand-editable.**
  `write_val_folder()` deletes and rewrites it on every run regardless of
  which validation mode produced it — never place files into it
  directly; use `val.manual_val_src` instead (Section 4).
- **Manual val (`val.use_manual_val: true`) holds nothing back from
  `src_dir`.** All of `src_dir` becomes training data in this mode — it
  is not combined with a seeded split. If you want some of `src_dir`
  withheld *in addition to* a manual val set, that requires a further
  pipeline change beyond what's described here.