"""
resolution.py -- nearest-multiple bucketing, a generic threshold-table
lookup (smallest table key that is >= the input value -- "ceiling" wins,
NOT "largest-below-or-equal"), and piecewise-linear anchor interpolation,
used for phase assignment and dynamic batch/accum/scale. One
implementation of each, shared by curriculum_builder.py, dataset.py, and
train.py, instead of the previous pipeline's several near-duplicate
copies.
"""
import bisect


def round_up(v, multiple):
    return ((v + multiple - 1) // multiple) * multiple


def padded_dims(w, h, multiple):
    """Nearest-`multiple` padded (w, h) -- the exact shape
    dataset.pad_to_multiple() will produce for a frame of this size."""
    return round_up(w, multiple), round_up(h, multiple)


def long_edge(w, h):
    return max(w, h)


def lookup_threshold_table(value, rows, key='max_long_side'):
    """
    rows: dicts each containing a numeric `key` (a .inf catch-all row is
    expected in every table in config.yaml) plus arbitrary payload
    fields. Returns the first row, in ascending-key order, whose key >=
    value.
    """
    ordered = sorted(rows, key=lambda r: r[key])
    keys = [r[key] for r in ordered]
    idx = bisect.bisect_left(keys, value)
    if idx >= len(ordered):
        idx = len(ordered) - 1
    return ordered[idx]


def resolve_phase(w, h, phase_buckets, multiple):
    pw, ph = padded_dims(w, h, multiple)
    return lookup_threshold_table(long_edge(pw, ph), phase_buckets)['phase']


def resolve_dynamic_batch(w, h, thresholds, multiple):
    """Returns (batch_size, accum_steps) for a sample of native size
    (w, h). Both are combined by the caller into a single micro-step
    counter -- see dataset.py's module docstring."""
    pw, ph = padded_dims(w, h, multiple)
    row = lookup_threshold_table(long_edge(pw, ph), thresholds)
    return row['batch_size'], row['accum_steps']


def resolve_dynamic_batch_metadata(w, h, thresholds, multiple):
    """Same lookup as resolve_dynamic_batch, but returns the full
    dict of everything a batch-folder's metadata file needs to record:
    the rounded resolution itself, the configured batch_size/accum, and
    the resulting effective_batch_size (batch_size * accum_steps).

    NOTE: batch_size/accum_steps are read verbatim from config.yaml's
    dynamic_batch.thresholds table -- this function does not invent or
    hardcode any resolution->batch_size mapping of its own, it only
    looks one up and reports the effective batch size that table
    actually produces (which may not be exactly 10 if the configured
    table doesn't hit that exactly -- see config.yaml's comments).

    Idempotent on already-padded input: padded_dims() of a value that's
    already a multiple of `multiple` returns it unchanged, so this is
    safe to call with either raw native (w, h) or an already-rounded
    (padded_w, padded_h) pair.
    """
    pw, ph = padded_dims(w, h, multiple)
    row = lookup_threshold_table(long_edge(pw, ph), thresholds)
    batch_size, accum = row['batch_size'], row['accum_steps']
    return {
        'resolution': f'{pw}x{ph}',
        'padded_w': pw,
        'padded_h': ph,
        'batch_size': batch_size,
        'gradient_accumulation': accum,
        'effective_batch_size': batch_size * accum,
    }


def resolve_train_scale(w, h, anchors, multiple):
    """Piecewise-linear interpolation over (long_edge, scale) anchors,
    clamped at both ends."""
    pw, ph = padded_dims(w, h, multiple)
    le = long_edge(pw, ph)
    pts = sorted(anchors, key=lambda p: p[0])
    if le <= pts[0][0]:
        return pts[0][1]
    if le >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= le <= x1:
            t = (le - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]  # unreachable given the clamps above