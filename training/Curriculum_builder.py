#!/usr/bin/env python3
"""
curriculum_builder.py

Builds the deterministic 4-phase progressive-resolution curriculum with
replay, from a RAW DATASET ROOT organized up to 3 levels deep:

    AllTrainingData/
        <Dataset>/
            <Scene>/
                frame_0001.png, frame_0002.png, ...

Any number of datasets, any number of scenes per dataset, any number of
frames per scene are supported. Nothing below assumes a single dataset, a
single scene, uniform frame counts, or a fixed folder count -- the whole
tree is scanned recursively and every usable scene becomes one sample.

> **Revision note (this revision -- Extra/ routing bugfix):** a previous
> revision of this file silently DROPPED every `is_extra` batch instead of
> routing it to `Extra/` as this docstring (and write_phase_folder's own
> docstring) always claimed it would. The bug: `group_batches_into_accum_windows()`
> flushed and skipped `is_extra` batches when it encountered them (correct
> -- they must never enter a training window), but never returned them
> anywhere -- it only returned `(full_groups, partial_groups)`. Since
> `build_curriculum()`'s `final_order[p]` was built exclusively from those
> two return values (plus replay versions of the same), `is_extra` batches
> never made it into `ordered_batches`, so `write_phase_folder()` -- which
> HAS a fully correct `is_extra` branch that writes to `Extra/`, tags
> `origin_phase`, and updates `extra_metadata.yaml` -- never once saw one.
> Net effect: `Extra/` was always empty, `extra_metadata.yaml` was never
> written, and every `num_extra_batches` / extra-scene count in every
> metadata file and in `log_final_summary()` silently read 0 even on runs
> where undersized leftover batches genuinely existed and were computed.
>
> Fixed by making `group_batches_into_accum_windows()` return a THIRD list
> -- the `is_extra` batches it skipped, in their original stream order --
> and having `build_curriculum()` collect those (own stream + every replay
> source, same pattern already used for trailing partial windows) and
> append them to the end of each phase's final order, after the trailing
> partial windows. They still never go through `_tag_windows()` (so they
> still correctly carry no `window_id`/`window_num_items`), and
> `write_phase_folder()` requires no changes at all -- it already knew
> exactly what to do with an `is_extra` batch, it just never used to
> receive one.

=== PIPELINE, IN ORDER ===

  Step 1 -- EXTRACT: recursively discover every <Dataset>/<Scene> folder
      with enough frames present, and record full metadata for each
      (dataset name, scene name, sample id, frame filenames, source
      resolution, frame count, on-disk path) *before* anything else
      happens. See discover_sequences().

  Step 2 -- SPLIT: a single seeded shuffle-and-slice over ALL extracted
      samples, done ONCE, before any curriculum/phase logic. Held-out
      validation samples never re-enter phase assignment, replay export,
      or batch-folder creation. See split_train_val().

  Step 3 -- BUCKET: round every training sample's dimensions up to the
      nearest `pad_multiple` and record that as its padded resolution.
      Everything downstream (phase assignment, batch grouping, dynamic
      batch/accum/scale lookups) reads this precomputed value instead of
      re-deriving it. See annotate_resolution_buckets().

  Step 4 -- CURRICULUM: assign each training sample's OWN data to a phase
      by padded resolution (assign_phases), then IMMEDIATELY collapse
      each phase's own (ascending-resolution) data into whole BATCH
      OBJECTS (group_into_batches) -- batches, not individual scenes, are
      the unit of everything that follows. A phase then retains ~80%
      (configurable) of its own batches (by total item count) and
      exports the rest EVENLY to every later phase as replay
      (split_retain_export / distribute_export_evenly), and finally each
      phase's own retained batch-stream is interleaved with whichever
      replay batch-streams it imported from earlier phases
      (interleave_grouped_streams), so replay is spread throughout the
      phase rather than sitting in one contiguous block. Retain/export/
      replay/interleave all operate on whole batches now, never
      individual scenes -- a batch that was built once (as a resolution
      run of a phase's own data) travels as a single unit through
      export, round-robin distribution, and interleaving, and is written
      out as a single folder wherever it lands. See build_curriculum().

      ACCUMULATION WINDOWS: within each stream (a phase's own retained
      batches, and each replay source separately), consecutive
      same-resolution, non-extra batches are chunked into accumulation
      windows of exactly accum_steps(resolution) batches
      (group_batches_into_accum_windows). Every batch in one window is
      stamped with a shared `window_id` (_tag_windows) plus the window's
      TRUE total item count (`window_num_items`) -- this is what
      train.py divides the loss by, and what it flushes the optimizer
      on, instead of inferring window boundaries from a resolution
      change or a step count. A resolution run that doesn't have enough
      batches left to fill a full window becomes a PARTIAL window
      (fewer than accum_steps batches, `window_is_partial=True`) rather
      than being discarded -- these partial windows are still fully
      valid, trainable accumulation windows (train.py just divides by
      their true, smaller item count), so instead of interleaving them
      throughout the phase like full windows, they're collected and
      appended to the END of the phase's stream, after every full
      (own + replay) window has been emitted. `is_extra` batches (see
      Step 5) are excluded from window grouping entirely -- they are
      also collected (as individual batches, not windowed groups) and
      appended after the trailing partial windows, so they still reach
      write_phase_folder() and land in Extra/, without ever being tagged
      with a window_id. See build_curriculum().

  Step 5 -- BATCH FOLDERS: write_phase_folder() no longer derives any
      grouping itself -- it just numbers and writes out, in order, the
      batch objects Step 4 already built and merged. Any batch whose item
      count is smaller than the configured batch_size for its resolution
      (see config.yaml's dynamic_batch table) is too small to reach a
      full effective batch AT THE BATCH level (independent of the
      window-level partial/full distinction above), so it's written
      under Extra/ instead of a normal numbered BatchNNN/ slot. This can
      happen to an "own" batch (a short leftover run of a phase's own
      data) or to a replay batch that lands in a phase it wasn't built
      in -- either way its origin_phase (the phase whose own data
      originally produced it, NOT the phase it currently sits in) is
      recorded in its batch_metadata.yaml and in that phase's
      extra_metadata.yaml. Extra/ batches are excluded from window
      grouping entirely (see group_batches_into_accum_windows) and are
      never trained on by dataset.py -- see that file's module
      docstring. See write_phase_folder() / group_into_batches().

Output layout under paths.dataset_root:

    TrainingData/
      Phase1/
        phase_metadata.yaml
        Batch001_960x544/
          batch_metadata.yaml
          00000_own__DatasetA__Scene003/     (symlink to original scene dir)
          00001_own__DatasetA__Scene007/
          ...
        Batch002_1024x576/
          ...
        Extra/
          extra_metadata.yaml
          Batch001_960x544_origP1_n2/        (undersized batch, built here)
            batch_metadata.yaml
            00047_own__DatasetA__Scene011/
          Batch002_1920x1088_origP2_n1/      (undersized batch, arrived as
            batch_metadata.yaml                replay from Phase2)
            00048_replay-P2__DatasetB__Scene004/
      Phase2/
        Batch001_1920x1088/
          00000_own__DatasetB__Scene002/
          00001_replay-P1_960x544__DatasetA__Scene003/   (only when a whole
        Batch002_960x544/                                  imported replay
          ...                                              batch lands here)
        Extra/
          ...
      Phase3/, Phase4/  (same pattern, more replay sources)

    ValidationData/
      <Dataset>/
        <Scene>/                          (symlink to original sequence dir)
        ...

    curriculum_metadata.yaml    (seed, split info, per-phase/batch summary)

The zero-padded leading index on each scene folder is the training order
within its batch folder (monotonically increasing across the whole
phase, including Extra/); batch folders themselves are numbered
separately within their own numbering stream (normal BatchNNN/ vs
Extra/BatchNNN/). Together they let dataset.py reproduce the exact
curriculum order with a plain nested sorted() walk -- no re-derivation of
any randomness needed. window_id is an independent, explicit tag on top
of that ordering -- it does not affect folder naming or numbering, only
which items dataset.py/train.py treat as belonging to one accumulation
window. is_extra batches never receive a window_id at all (see the
revision note above and group_batches_into_accum_windows).

This script is NOT incremental (no manifest / no append-in-place) --
every run rebuilds TrainingData/ and ValidationData/ from scratch from the
current --src listing. That's required for "running the pipeline twice
with the same seed produces the same folder layout": an incremental
design would make the layout depend on the history of previous runs, not
just on (seed, src content).
"""
import argparse
import itertools
import logging
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image

import config_loader
import resolution
from seeding import rng_for, seed_everything


def find_first_image(folder, extensions):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in extensions:
            return p
    return None


def get_resolution(folder, extensions):
    img_path = find_first_image(folder, extensions)
    if img_path is None:
        return None
    try:
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception as ex:  # noqa: BLE001 - keep scanning past unreadable files
        logging.warning('could not read %s: %s', img_path, ex)
        return None


# =============================================================================
# Step 1 -- EXTRACT: recursive Dataset -> Scene discovery, full metadata
# =============================================================================
def find_frame_file(scene_dir, frame_name, extensions):
    """Try frame_name + each extension, in the order given by
    image_extensions, and return the first match. Returns None if no
    extension exists for this frame_name in this scene."""
    for ext in extensions:
        candidate = scene_dir / f'{frame_name}{ext}'
        if candidate.exists():
            return candidate
    return None


def discover_sequences(src_dir, frame_names, extensions, min_frames=3):
    """Recursively scan src_dir/<Dataset>/<Scene>/ (exactly two levels of
    folders below src_dir) and return one metadata dict per usable scene.
    A scene is usable if >= min_frames of frame_names resolve to an
    actual file (any extension in `extensions`).

    Each dataset/scene can use a totally different image format -- the
    extension is resolved independently per frame_name, per scene.
    """
    src_dir = Path(src_dir)
    seqs = []
    n_datasets = 0
    for dataset_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        n_datasets += 1
        scene_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
        if not scene_dirs:
            logging.warning('dataset %s has no scene subfolders, skipping', dataset_dir)
            continue
        for scene_dir in scene_dirs:
            present_frames = []
            for name in frame_names:
                found = find_frame_file(scene_dir, name, extensions)
                if found is not None:
                    present_frames.append(found.name)  # actual resolved filename incl. extension

            if len(present_frames) < min_frames:
                logging.debug('skipping %s: only %d/%d required frames present',
                               scene_dir, len(present_frames), min_frames)
                continue
            res = get_resolution(scene_dir, extensions)
            if res is None:
                logging.warning('skipping %s: no readable frames', scene_dir)
                continue
            w, h = res
            dataset_name = dataset_dir.name
            scene_name = scene_dir.name
            sample_id = f'{dataset_name}__{scene_name}'
            seqs.append({
                'sample_id': sample_id,
                'dataset': dataset_name,
                'scene': scene_name,
                'name': sample_id,
                'path': scene_dir,
                'w': w,
                'h': h,
                'frame_count': len(present_frames),
                'present_frames': present_frames,   # now real filenames, e.g. "frame1.png"
            })
    logging.info('Scanned %d dataset folder(s) under %s', n_datasets, src_dir)
    return seqs


# =============================================================================
# Step 2 -- SPLIT: seeded, by-sample, before any curriculum logic
# =============================================================================

def split_train_val(seqs, seed, val_split):
    """Deterministic seeded shuffle-and-slice, by SAMPLE (dataset+scene),
    done ONCE before any curriculum logic. Val samples are set aside
    completely and never touched by phase assignment, replay, or ordering
    below."""
    rng = rng_for(seed, 'trainval_split')
    shuffled = seqs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if shuffled else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


# =============================================================================
# Step 3 -- BUCKET: precompute each training sample's padded resolution
# =============================================================================

def annotate_resolution_buckets(train_seqs, multiple):
    """Round every training sample's native (w, h) up to the nearest
    `multiple` ONCE, and store it on the sample dict as padded_w/padded_h.
    Phase assignment, batch grouping, and the dynamic batch/accum/scale
    metadata lookups all read this precomputed value rather than
    re-deriving it independently."""
    for s in train_seqs:
        pw, ph = resolution.padded_dims(s['w'], s['h'], multiple)
        s['padded_w'], s['padded_h'] = pw, ph
    return train_seqs


# =============================================================================
# Step 4 -- CURRICULUM: phase assignment, batching, retain/export, interleave
# =============================================================================

def assign_phases(train_seqs, phase_buckets, multiple):
    """Bucket each train sequence into its OWN phase by (precomputed)
    padded resolution, then sort each phase's own data ascending by
    resolution (the "progress low->high" requirement)."""
    by_phase = {1: [], 2: [], 3: [], 4: []}
    for s in train_seqs:
        le = resolution.long_edge(s['padded_w'], s['padded_h'])
        p = resolution.lookup_threshold_table(le, phase_buckets)['phase']
        by_phase[p].append(s)
    for p in by_phase:
        by_phase[p].sort(key=lambda s: (
            resolution.long_edge(s['padded_w'], s['padded_h']),
            s['padded_w'], s['padded_h'], s['name']))
    return by_phase


def group_into_batches(seqs, phase_num, multiple, dynamic_batch_thresholds):
    """seqs: one phase's OWN sequences, already sorted ascending by
    padded resolution (see assign_phases). First collapses seqs into
    maximal RUNS of consecutive same-padded-resolution items, then
    chunks EACH run into batch_size-sized pieces (batch_size looked up
    per-resolution from config.yaml's dynamic_batch table) -- this runs
    BEFORE any retain/export/replay logic, so batches are always built
    from a single phase's own data and then travel as whole units
    through everything downstream (see this module's docstring).

    A run's final leftover chunk, if shorter than that resolution's
    configured batch_size, is flagged is_extra=True -- too small to
    reach a full effective batch, so write_phase_folder() routes it to
    Extra/ instead of a numbered BatchNNN/ folder, wherever it
    ultimately ends up being written. dataset.py deliberately does NOT
    train on Extra/ content -- see that file's module docstring -- so
    is_extra batches are excluded from training by design, not just
    mis-filed. is_extra batches are also excluded from accumulation-
    window grouping entirely -- see group_batches_into_accum_windows --
    but they ARE still carried through build_curriculum() and handed to
    write_phase_folder() (see that function and the revision note at the
    top of this file) so they actually land in Extra/ instead of being
    dropped.

    Returns: list of batch dicts:
        {'resolution': (padded_w, padded_h), 'items': [seq, ...],
         'origin_phase': phase_num, 'is_extra': bool,
         'configured_batch_size': int}
    """
    runs = []
    cur_res = None
    cur_items = []
    for seq in seqs:
        res = (seq['padded_w'], seq['padded_h'])
        if cur_res is None or res != cur_res:
            if cur_items:
                runs.append((cur_res, cur_items))
            cur_res = res
            cur_items = []
        cur_items.append(seq)
    if cur_items:
        runs.append((cur_res, cur_items))

    batches = []
    for res, items in runs:
        pw, ph = res
        dyn = resolution.resolve_dynamic_batch_metadata(pw, ph, dynamic_batch_thresholds, multiple)
        bs = dyn['batch_size']
        # Chunk this resolution run into batch_size-sized pieces. A final
        # partial chunk (< bs items) is the only way is_extra can now be
        # True -- it's a real leftover remainder, not "the whole run
        # happened to be small."
        for start in range(0, len(items), bs):
            chunk = items[start:start + bs]
            batches.append({
                'resolution': res,
                'items': chunk,
                'origin_phase': phase_num,
                'is_extra': len(chunk) < bs,
                'configured_batch_size': bs,
            })
    return batches


def split_retain_export(batches, seed, phase_num, retain_ratio, has_later_phases):
    """Which of this phase's own COMPLETE batch objects (already built by
    group_into_batches, in ascending-resolution order) stay vs get
    exported as replay. is_extra batches (undersized leftover chunks) are
    never candidates for export -- they're excluded from both the
    ~20% target calculation and the shuffle/selection pool below, so they
    always fall through to 'retained' and stay in their origin phase's
    own Extra/ folder. This matters because dataset.py never trains on
    Extra/ content (see that file's module docstring) -- an incomplete
    batch that got shipped off to a different phase would just be dead
    weight sitting in that phase's Extra/ for no reason, and would also
    be a batch whose scene count doesn't match its resolution's
    configured batch_size/accum_steps/train_scale expectations, which is
    exactly the kind of ambiguity this pipeline is designed to avoid (see
    this module's docstring, Section 6).

    Selection happens at the WHOLE-BATCH level, but the ~20%
    (1 - retain_ratio) target is measured in total ITEM count across this
    phase's own EXPORTABLE (non-extra) batches only.

    RETAINED keeps the original ascending-resolution batch order intact
    (is_extra batches included, in their natural position). EXPORTED
    batches are seeded-shuffled (their internal order doesn't carry
    meaning, just needs to be reproducible) before being handed off for
    even round-robin assignment to later phases.
    """
    if not has_later_phases or retain_ratio >= 1.0:
        return batches, []

    exportable = [b for b in batches if not b['is_extra']]
    total_items = sum(len(b['items']) for b in exportable)
    target_export_items = total_items * (1 - retain_ratio)

    rng = rng_for(seed, 'export_select', phase_num)
    pool = exportable[:]
    rng.shuffle(pool)

    export_ids = set()
    exported_items = 0
    for b in pool:
        if exported_items >= target_export_items:
            break
        export_ids.add(id(b))
        exported_items += len(b['items'])

    retained = [b for b in batches if id(b) not in export_ids]  # preserves ascending order, includes is_extra batches
    exported = [b for b in pool if id(b) in export_ids]
    return retained, exported


def distribute_export_evenly(export_batches, seed, phase_num, dest_phases):
    """Round-robin the (already shuffled) exported BATCH objects across
    every later phase so each destination gets as close to an equal
    share (by batch count) as possible, deterministically. Each batch
    moves as a single indivisible unit -- its items are never split
    across destination phases."""
    rng = rng_for(seed, 'export_distribute', phase_num)
    pool = export_batches[:]
    rng.shuffle(pool)
    buckets = {d: [] for d in dest_phases}
    for i, batch in enumerate(pool):
        buckets[dest_phases[i % len(dest_phases)]].append(batch)
    return buckets


def group_batches_into_accum_windows(batches, dynamic_batch_thresholds, multiple):
    """Chunk an ordered list of batch objects into accumulation-window
    groups: consecutive same-resolution, NON-EXTRA batches, taken
    accum_steps at a time (accum_steps looked up per-resolution from
    dynamic_batch). is_extra batches are skipped entirely for window
    grouping purposes -- they always go to Extra/ via write_phase_folder
    regardless of window logic, and dataset.py never trains on Extra/
    content, so including them in a window would just corrupt that
    window's true item count for no benefit.

    Returns (full_groups, partial_groups, extra_batches):
      - full_groups: groups that reached exactly accum_steps(resolution)
        batches -- these are what interleave_grouped_streams places
        throughout the phase, same as before.
      - partial_groups: the trailing remainder of a same-resolution run
        that ended before reaching accum_steps (at most one per run).
        These are NOT discarded -- a partial window is still a fully
        valid, trainable accumulation window; train.py just divides its
        loss by the window's true (smaller) item count instead of the
        theoretical batch_size*accum_steps target. build_curriculum()
        collects every partial group (own + each replay source) for a
        phase and appends them to the END of that phase's stream, after
        all full windows -- see that function.
      - extra_batches: every is_extra batch encountered in `batches`,
        in their original relative order, returned as individual batch
        dicts (NOT grouped into windows -- they never get a window_id).
        BUGFIX (see the revision note at the top of this file): these
        used to be silently dropped here -- flushed out of the current
        run-in-progress and then discarded, never returned to the
        caller. build_curriculum() now collects this list (own stream +
        every replay source, mirroring how it already collects
        partial_groups) and appends the batches to the end of the
        phase's final order, so write_phase_folder() actually receives
        them and routes them to Extra/ as originally intended.
    """
    full_groups, partial_groups, extra_batches = [], [], []
    cur_group, cur_res, cur_target = [], None, None

    def _flush():
        if not cur_group:
            return
        if len(cur_group) == cur_target:
            full_groups.append(cur_group)
        else:
            partial_groups.append(cur_group)

    for b in batches:
        if b['is_extra']:
            # Extra batches never participate in window grouping -- flush
            # whatever run was in progress (an is_extra batch breaks a
            # same-resolution run just like a resolution change would,
            # since it's about to be pulled out to Extra/ regardless),
            # then record the batch itself so it isn't lost -- see the
            # BUGFIX note above and in this function's docstring.
            _flush()
            cur_group, cur_res, cur_target = [], None, None
            extra_batches.append(b)
            continue

        pw, ph = b['resolution']
        dyn = resolution.resolve_dynamic_batch_metadata(pw, ph, dynamic_batch_thresholds, multiple)
        accum_steps = dyn['gradient_accumulation']
        if cur_res is None or b['resolution'] != cur_res:
            _flush()
            cur_group, cur_res, cur_target = [], b['resolution'], accum_steps
        cur_group.append(b)
        if len(cur_group) == cur_target:
            full_groups.append(cur_group)
            cur_group, cur_res, cur_target = [], None, None
    _flush()
    return full_groups, partial_groups, extra_batches


def _tag_windows(groups, counter, dynamic_batch_thresholds, multiple, is_partial):
    """Stamp every batch in each group with a shared window_id (unique
    across the whole run -- `counter` is one itertools.count() shared
    across every phase and every stream, so ids never collide between
    phases or between an own-stream and a replay-stream), plus the
    window's TRUE total item count (window_num_items) and, for
    reference/logging only, the theoretical target item count a FULL
    window at this resolution would have (window_target_items).

    window_num_items is what train.py actually divides the loss by and
    flushes the optimizer on -- not a table lookup -- so it is exact by
    construction for both full and partial windows.

    NEVER called on extra_batches (see group_batches_into_accum_windows)
    -- is_extra batches must never carry a window_id.
    """
    for g in groups:
        wid = f'W{next(counter):06d}'
        num_items = sum(len(b['items']) for b in g)
        pw, ph = g[0]['resolution']
        dyn = resolution.resolve_dynamic_batch_metadata(pw, ph, dynamic_batch_thresholds, multiple)
        target_items = dyn['batch_size'] * dyn['gradient_accumulation']
        for b in g:
            b['window_id'] = wid
            b['window_num_items'] = num_items
            b['window_target_items'] = target_items
            b['window_is_partial'] = is_partial


def build_curriculum(train_seqs, cfg, seed):
    """Builds each phase's final on-disk batch order.

    For each phase p: own batches are split into retained/exported
    (split_retain_export), exported batches are round-robined to later
    phases (distribute_export_evenly), and then -- separately, per
    stream (own retained + each individual replay source) --
    group_batches_into_accum_windows() chunks non-extra batches into
    full/partial accumulation windows AND collects any is_extra batches
    encountered (see that function's docstring and the BUGFIX revision
    note at the top of this file).

    Full windows are interleaved (interleave_grouped_streams); partial
    windows are appended after that, in stream order (own's partial
    first, then each replay source's, in the order replay_pool was
    populated); is_extra batches are appended last of all, in that same
    stream order, as individual (non-windowed) batches -- so
    write_phase_folder() sees, in order: interleaved full windows,
    trailing partial windows, then trailing extra batches. Extra batches
    never receive a window_id (they never pass through _tag_windows),
    matching write_phase_folder()'s expectation that only non-extra
    batches carry window_id/window_num_items/window_target_items/
    window_is_partial.
    """
    multiple = cfg['pad_multiple']
    phase_buckets = cfg['curriculum']['phase_buckets']
    retain_ratio = cfg['curriculum']['retain_ratio']
    dynamic_batch_thresholds = cfg['dynamic_batch']['thresholds']

    own_by_phase = assign_phases(train_seqs, phase_buckets, multiple)
    own_batches_by_phase = {
        p: group_into_batches(own_by_phase[p], p, multiple, dynamic_batch_thresholds)
        for p in (1, 2, 3, 4)
    }

    retained = {}
    replay_pool = {1: {}, 2: {}, 3: {}, 4: {}}

    for p in (1, 2, 3, 4):
        later = [q for q in (2, 3, 4) if q > p]
        own_retained, own_exported = split_retain_export(
            own_batches_by_phase[p], seed, p, retain_ratio, has_later_phases=bool(later))
        retained[p] = own_retained
        if own_exported:
            buckets = distribute_export_evenly(own_exported, seed, p, later)
            for dest, items in buckets.items():
                if items:
                    replay_pool[dest][f'P{p}'] = items

    # One counter shared across every phase and every stream, so window_id
    # is globally unique -- never reused across phases or between an
    # own-stream and a replay-stream.
    window_counter = itertools.count(1)

    final_order = {}
    for p in (1, 2, 3, 4):
        own_full, own_partial, own_extra = group_batches_into_accum_windows(
            retained[p], dynamic_batch_thresholds, multiple)
        _tag_windows(own_full, window_counter, dynamic_batch_thresholds, multiple, is_partial=False)
        _tag_windows(own_partial, window_counter, dynamic_batch_thresholds, multiple, is_partial=True)

        replay_full_groups = {}
        # Collect every partial window for this phase (own + each replay
        # source) -- these all get appended after the interleaved full
        # stream, in stream order (own's partial first, then each replay
        # source's partial in the same order replay_pool was populated).
        trailing_partials = [('__primary__', g) for g in own_partial]
        # Same pattern for is_extra batches (BUGFIX -- see revision note
        # at the top of this file): collected per stream here, appended
        # to the very end of the phase's order, after trailing_partials,
        # further down. Individual batch dicts, not window groups.
        trailing_extras = [('__primary__', b) for b in own_extra]

        for src, batches in replay_pool[p].items():
            full_g, partial_g, extra_g = group_batches_into_accum_windows(
                batches, dynamic_batch_thresholds, multiple)
            _tag_windows(full_g, window_counter, dynamic_batch_thresholds, multiple, is_partial=False)
            _tag_windows(partial_g, window_counter, dynamic_batch_thresholds, multiple, is_partial=True)
            replay_full_groups[src] = full_g
            trailing_partials += [(src, g) for g in partial_g]
            trailing_extras += [(src, b) for b in extra_g]

        ordered = interleave_grouped_streams(own_full, replay_full_groups)
        for origin_stream, g in trailing_partials:
            ordered += [(origin_stream, b) for b in g]
        # Extras are appended last of all -- after every full window and
        # every partial window -- as individual batches (not groups), so
        # write_phase_folder() receives them and routes each one to
        # Extra/ via its existing is_extra branch. This is the fix: prior
        # to this revision, own_extra/extra_g were computed but never
        # reached `ordered` at all.
        for origin_stream, b in trailing_extras:
            ordered.append((origin_stream, b))

        final_order[p] = ordered
    return final_order


# =============================================================================
# Step 5 -- BATCH FOLDERS: write out the already-built, already-merged batches
# =============================================================================

# Non-batch entries that can appear directly under a Phase<N>/ folder and
# must not be walked as if they were a BatchNNN_<res>/ folder of scenes.
_PHASE_SKIP_ENTRIES = {'Extra', 'phase_metadata.yaml'}


def _place(name, src_path, dst_dir, log=logging.info):
    dst = dst_dir / name
    try:
        os.symlink(Path(src_path).resolve(), dst, target_is_directory=True)
    except OSError as e:
        log(f'symlink failed for {name} ({e}) -- copying instead '
            f'(enable Windows Developer Mode for symlinks under WSL/NTFS)')
        shutil.copytree(src_path, dst)


def write_phase_folder(phase_num, ordered_batches, training_root, multiple,
                        dynamic_batch_thresholds, train_scale_anchors, seed):
    """Writes TrainingData/PhaseN/ from the already-built batch objects in
    ordered_batches (list of (origin_label, batch_obj) tuples, in final
    curriculum order -- see build_curriculum). Each batch_obj was built by
    group_into_batches() on ITS OWN phase's data, before any
    retain/export/replay/interleave happened -- so this function's only
    job is to number the batches in the order it received them and write
    them to disk; it does not group or re-derive batches itself. As of
    the Extra/-routing bugfix (see the revision note at the top of this
    file), ordered_batches now genuinely includes is_extra batches (own
    and replay), appended by build_curriculum() after every full and
    partial accumulation window -- this function's is_extra branch below
    was always ready to handle them, it simply never used to receive any.

    A batch whose item count is below the configured batch_size for its
    resolution (batch_obj['is_extra'], set by group_into_batches) is
    written under Extra/ instead of getting a normal BatchNNN/ slot --
    see this module's docstring for why (too small to reach a full
    effective batch). Non-extra batches additionally carry the
    accumulation-window fields stamped by _tag_windows (window_id,
    window_num_items, window_target_items, window_is_partial) -- is_extra
    batches never go through window grouping, so they have none of these
    fields and dataset.py never reads them for Extra/ content anyway.

    Writes batch_metadata.yaml per batch, extra_metadata.yaml (only if
    this phase's Extra/ ends up non-empty) summarizing every batch
    currently sitting in Extra/ -- including ones that arrived here via
    replay from another phase (origin_phase always names the phase that
    originally BUILT the batch, not the phase it currently sits in) --
    and phase_metadata.yaml summarizing the whole phase.
    """
    phase_dir = Path(training_root) / f'Phase{phase_num}'
    if phase_dir.exists():
        shutil.rmtree(phase_dir)  # full rebuild every run -- see module docstring
    phase_dir.mkdir(parents=True)
    extra_dir = phase_dir / 'Extra'

    phase_metadata = {
        'phase': phase_num,
        'seed': seed,
        'num_items': sum(len(b['items']) for _, b in ordered_batches),
        'num_trainable_items': sum(len(b['items']) for _, b in ordered_batches if not b['is_extra']),
        'num_batches': 0,        # normal (non-extra) batches only -- filled in below
        'num_extra_batches': 0,
        'num_windows': 0,        # distinct window_id count among non-extra batches
        'num_partial_windows': 0,
        'batches': [],
    }
    extra_metadata = {
        'phase': phase_num,
        'batches': [],
    }

    global_idx = 0
    normal_idx = 0
    extra_idx = 0
    seen_window_ids = set()
    seen_partial_window_ids = set()

    for origin_stream, batch in ordered_batches:
        pw, ph = batch['resolution']
        dyn = resolution.resolve_dynamic_batch_metadata(pw, ph, dynamic_batch_thresholds, multiple)
        train_scale = resolution.resolve_train_scale(pw, ph, train_scale_anchors, multiple)
        scene_count = len(batch['items'])
        is_extra = batch['is_extra']

        if is_extra:
            extra_idx += 1
            batch_index = extra_idx
            batch_name = f'Batch{batch_index:03d}_{dyn["resolution"]}_origP{batch["origin_phase"]}_n{scene_count}'
            batch_dir = extra_dir / batch_name
        else:
            normal_idx += 1
            batch_index = normal_idx
            batch_name = f'Batch{batch_index:03d}_{dyn["resolution"]}'
            batch_dir = phase_dir / batch_name

        batch_dir.mkdir(parents=True)

        batch_meta = {
            'batch_index': batch_index,
            'resolution': dyn['resolution'],
            'batch_size': dyn['batch_size'],
            'gradient_accumulation': dyn['gradient_accumulation'],
            'effective_batch_size': dyn['effective_batch_size'],
            'train_scale': train_scale,
            'num_scenes': scene_count,
            'scenes': [],
        }
        if is_extra:
            batch_meta['origin_phase'] = batch['origin_phase']
            batch_meta['scene_count'] = scene_count
        else:
            # accumulation-window fields -- always present on non-extra
            # batches, stamped by _tag_windows in build_curriculum().
            batch_meta['window_id'] = batch['window_id']
            batch_meta['window_num_items'] = batch['window_num_items']
            batch_meta['window_target_items'] = batch['window_target_items']
            batch_meta['window_is_partial'] = batch['window_is_partial']
            seen_window_ids.add(batch['window_id'])
            if batch['window_is_partial']:
                seen_partial_window_ids.add(batch['window_id'])

        for seq in batch['items']:
            origin_label = 'own' if origin_stream == '__primary__' else f'replay-{origin_stream}'
            scene_folder_name = f'{global_idx:05d}_{origin_label}__{seq["dataset"]}__{seq["scene"]}'
            _place(scene_folder_name, seq['path'], batch_dir)
            batch_meta['scenes'].append({
                'global_index': global_idx,
                'sample_id': seq['sample_id'],
                'dataset': seq['dataset'],
                'scene': seq['scene'],
                'origin': origin_label,
                'folder': scene_folder_name,
                'source_resolution': f'{seq["w"]}x{seq["h"]}',
                'frame_count': seq['frame_count'],
            })
            global_idx += 1

        with open(batch_dir / 'batch_metadata.yaml', 'w') as f:
            yaml.safe_dump(batch_meta, f, sort_keys=False)

        if is_extra:
            extra_metadata['batches'].append(batch_meta)
        else:
            phase_metadata['batches'].append(batch_meta)

    phase_metadata['num_batches'] = normal_idx
    phase_metadata['num_extra_batches'] = extra_idx
    phase_metadata['num_windows'] = len(seen_window_ids)
    phase_metadata['num_partial_windows'] = len(seen_partial_window_ids)

    with open(phase_dir / 'phase_metadata.yaml', 'w') as f:
        yaml.safe_dump(phase_metadata, f, sort_keys=False)

    if extra_idx:
        with open(extra_dir / 'extra_metadata.yaml', 'w') as f:
            yaml.safe_dump(extra_metadata, f, sort_keys=False)

    logging.info(
        'Phase%d: %d item(s) in %d batch folder(s) (+%d in Extra/) written to %s | '
        '%d accumulation window(s) (%d partial)',
        phase_num, phase_metadata['num_items'], normal_idx, extra_idx, phase_dir,
        phase_metadata['num_windows'], phase_metadata['num_partial_windows'])
    return phase_metadata


def write_val_folder(val_seqs, dataset_root):
    """ValidationData/<dataset>/<scene>/ -- preserves the raw dataset's
    Dataset/Scene hierarchy (unlike training batch folders, which flatten
    scenes directly under each batch) so validation samples stay easy to
    trace back to their source for debugging."""
    val_dir = Path(dataset_root) / 'ValidationData'
    if val_dir.exists():
        shutil.rmtree(val_dir)
    val_dir.mkdir(parents=True)
    for seq in sorted(val_seqs, key=lambda s: (s['dataset'], s['scene'])):
        ds_dir = val_dir / seq['dataset']
        ds_dir.mkdir(parents=True, exist_ok=True)
        _place(seq['scene'], seq['path'], ds_dir)
    logging.info('ValidationData: %d sequence(s) written to %s', len(val_seqs), val_dir)


def write_top_level_metadata(dataset_root, seed, val_split_ratio, train_seqs, val_seqs, phase_metadatas):
    meta = {
        'seed': seed,
        'val_split': val_split_ratio,
        'num_total_sequences': len(train_seqs) + len(val_seqs),
        'num_train_sequences': len(train_seqs),
        'num_val_sequences': len(val_seqs),
        'validation_samples': [
            {
                'sample_id': s['sample_id'],
                'dataset': s['dataset'],
                'scene': s['scene'],
                'source_resolution': f'{s["w"]}x{s["h"]}',
                'frame_count': s['frame_count'],
            }
            for s in sorted(val_seqs, key=lambda s: s['sample_id'])
        ],
        'phases': {
            p: {
                'num_items': pm['num_items'],
                'num_batches': pm['num_batches'],
                'num_extra_batches': pm.get('num_extra_batches', 0),
                'num_windows': pm.get('num_windows', 0),
                'num_partial_windows': pm.get('num_partial_windows', 0),
            }
            for p, pm in phase_metadatas.items()
        },
    }
    with open(Path(dataset_root) / 'curriculum_metadata.yaml', 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    logging.info('curriculum_metadata.yaml written to %s', dataset_root)


def interleave_grouped_streams(own_groups, replay_pool_groups):
    """own_groups: this phase's own retained FULL batches, already
    chunked into accumulation-window groups (ascending resolution order
    preserved, group order unchanged).
    replay_pool_groups: {src_label: [group, group, ...]} -- each
    replay source's exported FULL batches, ALSO pre-chunked into
    accumulation-window groups the same way.

    Only FULL windows are interleaved here -- partial windows AND
    is_extra batches are collected separately by build_curriculum() and
    appended after this function's output (partials, then extras), so
    neither ever gets spread into the gaps below.

    Spreads each replay stream's groups evenly across the GAPS between
    own groups (never inside one), so every accumulation window --
    whether own or replay -- is built from batches of exactly one
    resolution. Returns a flat list of (origin_label, batch) tuples,
    ready for write_phase_folder -- group membership is preserved
    because each group's batches are always emitted together,
    back-to-back, uninterrupted by anything else.
    """
    own_seq = [('__primary__', b) for g in own_groups for b in g]
    if not replay_pool_groups:
        return own_seq

    n_gaps = len(own_groups) + 1
    slots = [[] for _ in range(n_gaps)]
    for src_label, groups in replay_pool_groups.items():
        n = len(groups)
        if n == 0:
            continue
        for i, g in enumerate(groups):
            gap_idx = min(int((i + 0.5) * n_gaps / n), n_gaps - 1)
            slots[gap_idx].append((src_label, g))

    ordered = []
    for i in range(n_gaps):
        for src_label, g in slots[i]:
            for b in g:
                ordered.append((src_label, b))
        if i < len(own_groups):
            for b in own_groups[i]:
                ordered.append(('__primary__', b))
    return ordered


def run(cfg, src_dir, dataset_root):
    seed = cfg['seed']
    seed_everything(seed)  # for anything downstream that reads torch/numpy global RNG state too

    extensions = set(cfg['image_extensions'])
    frame_filenames = cfg['data']['frame_names']
    multiple = cfg['pad_multiple']

    # ---- Step 1: extract everything first, before any splitting/phasing ----
    seqs = discover_sequences(src_dir, frame_filenames, extensions)
    if not seqs:
        raise SystemExit(f'No usable sequences found under {src_dir}')
    logging.info('Discovered %d usable sequence(s) total', len(seqs))

    # ---- Step 2: train/val split -- either seeded split-off-of-src_dir, or a
    # manually-provided, already-separate validation dataset ----
    val_cfg = cfg['val']
    if val_cfg.get('use_manual_val'):
        manual_val_src = val_cfg.get('manual_val_src')
        if not manual_val_src:
            raise SystemExit(
                'val.use_manual_val is true but val.manual_val_src is empty -- '
                'set it to a Dataset/Scene-structured folder, same layout as paths.src_dir.')
        manual_val_src = Path(manual_val_src)
        if not manual_val_src.is_dir():
            raise SystemExit(f'val.manual_val_src does not exist or is not a directory: {manual_val_src}')

        train_seqs = seqs  # nothing held out of src_dir -- all of it trains
        val_seqs = discover_sequences(manual_val_src, frame_filenames, extensions)
        if not val_seqs:
            raise SystemExit(f'No usable sequences found under manual_val_src {manual_val_src}')

        # Guard against silently sharing sample_ids with the training set --
        # symlink placement and metadata assume val samples are their own thing.
        train_ids = {s['sample_id'] for s in train_seqs}
        overlap = train_ids & {s['sample_id'] for s in val_seqs}
        if overlap:
            logging.warning(
                '%d sample_id(s) appear in BOTH src_dir (train) and manual_val_src (val): %s%s',
                len(overlap), sorted(overlap)[:5], ' ...' if len(overlap) > 5 else '')

        logging.info('Manual val: %d train (all of src_dir) / %d val (from %s)',
                     len(train_seqs), len(val_seqs), manual_val_src)
    else:
        train_seqs, val_seqs = split_train_val(seqs, seed, val_cfg['val_split'])
        logging.info('Split: %d train / %d val (seed=%s, val_split=%s)',
                     len(train_seqs), len(val_seqs), seed, val_cfg['val_split'])

    write_val_folder(val_seqs, dataset_root)

    # ---- Step 3: precompute each training sample's rounded resolution ----
    annotate_resolution_buckets(train_seqs, multiple)

    # ---- Step 4: phase assignment + batching + retain/export + interleave ----
    final_order = build_curriculum(train_seqs, cfg, seed)

    # ---- Step 5: write out the already-built, already-merged batches ----
    training_root = Path(dataset_root) / 'TrainingData'
    dynamic_batch_thresholds = cfg['dynamic_batch']['thresholds']
    train_scale_anchors = cfg['train_scale']['anchors']

    phase_metadatas = {}
    for p in (1, 2, 3, 4):
        phase_metadatas[p] = write_phase_folder(
            p, final_order[p], training_root, multiple,
            dynamic_batch_thresholds, train_scale_anchors, seed)

    write_top_level_metadata(dataset_root, seed, cfg['val']['val_split'],
                              train_seqs, val_seqs, phase_metadatas)

    log_final_summary(phase_metadatas, val_seqs)

    logging.info('Done. Curriculum written to %s', dataset_root)


def log_final_summary(phase_metadatas, val_seqs):
    total_trainable = 0
    total_extra_scenes = 0
    total_windows = 0
    total_partial_windows = 0
    logging.info('==================== SUMMARY ====================')
    for p in (1, 2, 3, 4):
        pm = phase_metadatas[p]
        extra_scenes = pm['num_items'] - pm['num_trainable_items']
        total_trainable += pm['num_trainable_items']
        total_extra_scenes += extra_scenes
        total_windows += pm.get('num_windows', 0)
        total_partial_windows += pm.get('num_partial_windows', 0)
        logging.info(
            'Phase%d: %d batch(es), %d extra batch(es) | %d trainable scene(s), %d extra scene(s) | '
            '%d window(s) (%d partial)',
            p, pm['num_batches'], pm['num_extra_batches'], pm['num_trainable_items'], extra_scenes,
            pm.get('num_windows', 0), pm.get('num_partial_windows', 0))
    logging.info('---------------------------------------------------')
    logging.info('Total training examples (excl. Extra/): %d', total_trainable)
    logging.info('Total scenes sitting in Extra/ (all phases): %d', total_extra_scenes)
    logging.info('Total accumulation windows (all phases): %d (%d partial)',
                 total_windows, total_partial_windows)
    logging.info('Validation examples: %d', len(val_seqs))
    logging.info('====================================================')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    cfg = config_loader.load_config(args.config)
    src_dir = Path(args.src or cfg['paths']['src_dir'])
    dataset_root = Path(args.dst or cfg['paths']['dataset_root'])
    dataset_root.mkdir(parents=True, exist_ok=True)

    run(cfg, src_dir, dataset_root)


if __name__ == '__main__':
    main()