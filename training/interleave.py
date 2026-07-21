"""
interleave.py -- deterministically merge a primary ordered stream with
zero or more independent replay streams so that:
  - each stream's own internal (already-decided) relative order is
    preserved exactly (primary stays low->high resolution; each replay
    source keeps whatever order it arrived in),
  - every stream is spread as evenly as possible across the full merged
    length, so no source clumps into one contiguous block.

This step is purely deterministic given its inputs -- all the actual
randomness (which sequences got exported, which destination phase they
went to) already happened upstream via a seeded RNG. Re-running with the
same inputs always produces the same merged order.
"""


def interleave_streams(primary, replay_sources):
    """
    primary: list of items, already in the order they should appear in
        (relative order preserved).
    replay_sources: dict[name] -> list of items, each list already in the
        order it should appear in (relative order preserved per source).

    Returns: list of (origin_label, item) tuples, origin_label is
        '__primary__' for primary items or the replay_sources key for
        everything else.
    """
    sources = [('__primary__', primary)] + sorted(replay_sources.items())
    sources = [(name, items) for name, items in sources if items]
    if not sources:
        return []

    scored = []
    for src_idx, (name, items) in enumerate(sources):
        n = len(items)
        for i, item in enumerate(items):
            # Fractional position in [0, 1); the src_idx offset keeps
            # equal-length streams from landing on identical fractions
            # and colliding instead of interleaving.
            frac = (i + src_idx / len(sources)) / n
            scored.append((frac, src_idx, i, name, item))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [(name, item) for _, _, _, name, item in scored]