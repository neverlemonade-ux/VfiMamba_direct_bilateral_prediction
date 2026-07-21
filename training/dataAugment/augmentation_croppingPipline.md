# VFIMamba Fine-Tune Data Pipeline

This pipeline turns raw scene footage (folders of numbered frames) into a
large, varied set of training examples for a video frame interpolation
model. It runs in **two separate stages**, each its own script and config:

```
Stage 1: cropping.py       Stage 2: augmentation.py
  (WHICH pixels/frames)  ->   (HOW those pixels look)
```

Run cropping first. Point augmentation at cropping's output. You can also
train directly on cropping's output if you don't want pixel augmentation.

---

## Why two stages instead of one script?

The original pipeline was a single combined script. It's now split so each
concern is isolated and independently rerunnable:

- **cropping.py** decides *where* in each scene to sample from, *whether*
  that sample has real motion worth training on, and *how long* each
  training clip is.
- **augmentation.py** decides *what* happens to the pixels of an
  already-cropped clip — lighting, blur, noise, compression, lens effects,
  reversing playback, and optional duplication.

This means you can generate one cropped dataset and run several different
augmentation passes over it without re-doing the (expensive) cropping and
motion-checking work each time.

---

## Stage 1 — `cropping.py`

**Input:** `data.input_root`, a folder of scene subfolders, each full of
contiguously-numbered frames (`000000.png`, `000001.png`, ...).

**Output:** `data.output_root` (defaults to `<input_root>_cropped_<size>`,
e.g. `train_10k_cropped_128`), a folder of output scene folders, plus a
`manifest.jsonl` recording exactly what was written.

### What it does, step by step

1. **Discover frames per scene.** Probes indices starting at
   `frame_index_start`, stopping at the first missing index. Tries each
   configured extension (`png`, `jpg`, ...) and locks in whichever one
   matched first for speed. Scenes with too few frames are skipped.

2. **Decide how many crops to take from this scene.**
   ```
   num_crops = min(width // crop_w, height // crop_h)
   ```
   optionally capped by `max_crops_per_scene`. The scene is divided into a
   grid with at least that many cells (shuffled, seeded) so no two crops
   in the same scene can land on the same spot.

3. **Motion-gate each crop box.** A crop that lands on a static sky or
   locked-off background teaches the model nothing. Before writing
   anything, the script samples a handful of frames inside that crop box,
   downsamples them, and measures frame-to-frame grayscale difference. If
   the score is too low (`min_motion_score`), the box is **rerolled** to a
   fresh random position (up to `max_reroll_attempts` times). If nothing
   ever passes, the crop slot is either skipped (default) or kept anyway
   with a logged warning, depending on `skip_on_failure`.

4. **Crop every frame through the passing box, once.** One crop pass per
   box — not one per output variant.

5. **Slice that cropped sequence into progressively shorter variants.**
   All variants keep the *front* of the sequence and trim off the *end*.
   Each new length is ~90% of the *previous* variant's length (not 90% of
   the original), so a 100-frame scene yields:
   ```
   100 → 90 → 81 → 72 → 64 → ...
   ```
   This keeps going as long as the last variant is still ≥
   `min_frames_to_continue`, and never produces anything shorter than
   `min_output_frames`. This gives the model training examples across a
   range of effective clip durations, without re-picking the crop
   position or re-running the motion gate for each length.

6. **Write output folders + manifest.** Each folder name encodes the
   scene, crop size, crop index (if more than one crop), and frame count,
   e.g. `scene003_128_00_f090`. `manifest.jsonl` records, per folder: the
   source scene, crop box, source resolution, full-scene frame count,
   this variant's frame count, motion score, and reroll attempts.

### Key config knobs (`cropping_config.yaml`)

| Section | Key | What it controls |
|---|---|---|
| `data` | `input_root` / `output_root` | Where scenes come from / go |
| `data` | `frame_index_digits` | Zero-padding width of frame filenames (`000000` = 6 digits) |
| `crop` | `crop_size` | One size per run — int (square) or `[w, h]` |
| `crop` | `max_crops_per_scene` | Cap on dynamically-computed crop count |
| `motion_check` | `min_motion_score` | How much motion a crop must show to be kept |
| `sequence` | `reduction_pct` | How much shorter each successive variant is |
| `sequence` | `min_output_frames` | Floor on how short a variant can get |

To build a **size ladder** (e.g. 128, 256, 320×240...), rerun this script
once per size against the same `input_root` — output names disambiguate
by size automatically.

---

## Stage 2 — `augmentation.py`

**Input:** `data.data_root` — this is cropping.py's `output_root`. Also
reads cropping's `manifest.jsonl` (via `data.crop_manifest`) to recover
each folder's true crop position and original scene resolution.

**Output:** `data.output_root` (defaults to `<data_root>_augmented`), one
output folder per input folder (plus extras if duplication triggers),
with its own `manifest.jsonl` recording what augmentations were applied.

### The three augmentation tiers

Every augmentation type falls into exactly one tier:

1. **Scene-fixed lens/optics** — `lens_distortion`, `chromatic_aberration`,
   `vignetting`. Drawn **once per original source scene** (grouped via the
   crop manifest's `source_scene` field) and reused unchanged across every
   crop index and every sequence-length variant of that scene. A real
   lens doesn't change between crops of the same shot, so this tier never
   varies within a scene.

2. **Mutually-exclusive tone/exposure group** — `exposure_adjustment`,
   `white_balance_shift`, `gamma_correction`, `color_jitter`. A group-level
   coin flip (`groups.tone_exposure.prob`) decides if *any* of these fire
   on a folder; if so, exactly **one** is picked, weighted by that type's
   own `prob` relative to the others.

3. **Independent, per-folder** — everything else: flips/rotation, blurs
   (gaussian/motion/defocus), noise, random erasing, jpeg/H.264
   compression. Each is its own independent coin flip per output folder.

Within one folder, a chosen augmentation is applied **identically across
every frame** (to keep inter-frame motion physically consistent) — except
`gaussian_noise` and `jpeg_compression`, which vary their exact
per-frame realization while sharing one severity parameter for the whole
folder.

### Other things it does

- **`reverse` (temporal):** the only surviving temporal augmentation —
  plays a folder's frames backwards. (The old `variable_interval`
  augmentation was removed: cropping.py's sequence-length variants now
  cover "vary the effective duration" directly.)
- **Duplication:** a folder can be duplicated into 2+ output copies, each
  independently drawing its own tier-2/tier-3 augmentation combo (checked
  for uniqueness against its siblings). All duplicates of one folder still
  share that folder's scene-level lens/optics draw.
- **Lens/optics math needs the *true* pre-crop position** — that's why it
  reads cropping's manifest. If the manifest is missing, or a folder has
  no entry, lens/optics effects just treat that folder as if it *is* the
  full original frame (still work, just centered differently).

### Key config knobs (`augmentation_config.yaml`)

| Section | Key | What it controls |
|---|---|---|
| `data` | `data_root` | Must point at cropping.py's `output_root` |
| `data` | `crop_manifest` | Defaults to `<data_root>/manifest.jsonl` |
| `augmentation.types` | per-type `enabled` / `prob` | Turn each augmentation on/off and its firing chance |
| `augmentation.groups.tone_exposure` | `prob` | Chance *any* tone/exposure effect fires at all |
| `augmentation.temporal.reverse` | `enabled` / `prob` | Chance a folder's frames play backwards |
| `augmentation.duplication` | `enabled` / `prob` / `max_total_instances` | Extra augmented copies per folder |

---

## Running it end to end

```bash
# Stage 1: crop + motion-filter + generate length variants
python cropping.py --config cropping_config.yaml

# Stage 2: point at Stage 1's output, apply pixel augmentation
python augmentation.py --config augmentation_config.yaml
```

Both stages are **idempotent by default** (`overwrite: false` skips any
output folder that already exists) and **fully reproducible** given the
same `seed` — reruns against the same input produce byte-identical output.

Both also write a `manifest.jsonl` to their own `output_root`, giving a
durable, browsable record of exactly what was generated at each stage —
useful for debugging, auditing, or re-deriving stage-2's per-scene
grouping from stage-1's crop boxes.