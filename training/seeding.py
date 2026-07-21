"""
seeding.py -- the ONE place random seeding happens in this pipeline.

config.yaml has exactly one `seed` value. Every stage that needs its own
independent-but-reproducible randomness (train/val split, which sequences
get exported as replay, which destination phase an exported sequence
lands in, ...) calls rng_for(seed, *salt) to get a private
random.Random() derived from the global seed plus a label -- so different
stages don't accidentally share/interfere with each other's random
streams, without needing a second manually-chosen seed anywhere.

IMPORTANT: this uses hashlib, not Python's built-in hash(), because
hash() of strings is randomized per-process (PYTHONHASHSEED) unless
explicitly disabled -- using it here would silently break
run-to-run reproducibility for anyone who hasn't set that env var.
"""
import hashlib
import random

import numpy as np
import torch


def seed_everything(seed):
    """Call once, at the very start of curriculum_builder.py / train.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """DataLoader workers re-derive their seed from torch's (already-
    seeded) RNG, so multi-worker loading is reproducible too."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_for(seed, *salt):
    """Deterministic, independent random.Random() for one pipeline stage.
    Same (seed, salt) always produces the same stream, on any machine,
    any Python process (unlike hash())."""
    key = f'{seed}:' + ':'.join(str(s) for s in salt)
    digest = hashlib.sha256(key.encode('utf-8')).digest()
    sub_seed = int.from_bytes(digest[:8], 'big')
    return random.Random(sub_seed)