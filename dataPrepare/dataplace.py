#!/usr/bin/env python3
"""
organize_interpolation_dataset.py

Purpose
-------
Video-interpolation training data is usually stored as one sub-folder per
clip/example, where each sub-folder contains a sequence of frame images
(e.g. frame_0001.png, frame_0002.png, ...). Because interpolation models are
usually trained with fixed-shape tensors per batch, it's convenient to
pre-group examples of the *same resolution* into batches ahead of time
("static" batches that a dynamic dataloader can then shuffle at the batch
level instead of the sample level).

This script:
 1. Scans a source directory of example sub-folders.
 2. Detects the frame resolution (width, height) of each sub-folder by
    reading its first image file.
 3. Buckets each resolution into a size class (see SETTINGS below), which
    determines both the batch size AND which "Phase" output folder it goes
    into. Only 4 Phase folders exist (Phase1..Phase4) even though there are
    5 size classes, because two classes (~3K and ~4K) share Phase4 -- they
    just get different batch sizes within it, since VRAM cost rises very
    steeply between 2K and 3K.
 4. Within each *exact* resolution, groups sub-folders together and chunks
    them into batches of the size dictated by the size class they belong to.
 5. Writes each full batch out as its own folder under <dst>/PhaseN/.
 6. Left-over examples that can't fill a complete batch are staged in
    <dst>/extra/ in a folder whose name records the current count, the
    intended batch size, phase, and resolution.
 7. THIS SCRIPT IS SAFE TO RE-RUN / RUN INCREMENTALLY. It keeps a small
    manifest file (<dst>/.organize_manifest.json>) so that:
      - source example folders that were already organized on a previous
        run are never copied/moved again (no duplicates), and
      - new examples matching an existing incomplete ("extra") batch are
        appended into that same pending batch instead of creating a second,
        parallel pending folder for the same resolution.
    When a previously-incomplete batch becomes full because of newly added
    source examples, it is automatically promoted (moved) from extra/ into
    the correct PhaseN/ folder.

Usage
-----
    python organize_interpolation_dataset.py \
        --src /path/to/raw_examples \
        --dst /path/to/organized_dataset \
        [--move] [--dry-run] [--verbose]

By default the script COPIES new example folders from --src (safe,
non-destructive). Pass --move if you'd rather move them out of --src.
Note: items that are only being *reshuffled inside --dst* (e.g. promoting an
extra/ batch into a Phase folder once it fills up) are always MOVED
regardless of --move, since they're already inside the managed dataset --
there is nothing to "preserve a copy of" there.
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

# Pillow is used only to read image dimensions (fast: it reads the file
# header, not the full pixel buffer).
from PIL import Image


# =============================================================================
# SETTINGS -- everything you're likely to want to tune lives in this block.
# =============================================================================

# Image extensions accepted when looking for "the first frame" of an example
# folder. Extend this if your dataset uses other formats.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# RESOLUTION_BUCKETS -- the single place that decides:
#   (a) which size class a frame resolution falls into,
#   (b) how many examples go into one training batch for that class, and
#   (c) which Phase output folder that class is written to.
#
# Classification is based on the LONGEST SIDE of the frame (max(width,
# height)), which is the usual convention for "1K/2K/3K/4K" naming. Change
# this if you'd rather bucket on width only, height only, or total pixel
# count (see classify_bucket() below).
#
# Tuned with a single RTX 5090 (32GB VRAM) as the reference GPU:
#   - Below 1K and ~1K frames are cheap enough to batch generously (8 / 4).
#   - ~2K is noticeably heavier, so batch size drops to 2.
#   - ~3K is NOT just "a bit more than 2K" -- pixel count (and activation
#     memory for optical-flow / feature pyramids used by most interpolation
#     architectures) roughly doubles again going from 2K to 3K, so it gets
#     its own entry with batch size 1 rather than being lumped in with 2K.
#   - ~4K also gets batch size 1. It shares Phase4 with ~3K (only 4 Phase
#     folders exist total), but is kept as a separate bucket entry so its
#     batch size can be tuned independently of ~3K if your GPU/model differs.
#
# Each row is: (label, max_long_side_exclusive, batch_size, phase_folder)
#   - "max_long_side_exclusive": a frame belongs to this bucket if its
#     longest side is STRICTLY LESS than this value. The last row uses
#     infinity so it always matches whatever falls through.
# Edit the numbers/batch sizes freely -- nothing else in the script needs
# to change.
RESOLUTION_BUCKETS = [
    # label        max_long_side   batch_size   phase_folder
    ("below_1k",   1024,           8,           "Phase1"),
    ("~1k",        2048,           4,           "Phase2"),
    ("~2k",        3072,           2,           "Phase3"),   # up to ~2.9K
    ("~3k",        4096,           1,           "Phase4"),   # ~3K - <4K
    ("~4k",        float("inf"),   1,           "Phase4"),   # 4K and above
]

# Name of the folder (directly under --dst) that holds incomplete batches.
EXTRA_DIR_NAME = "extra"

# Name of the manifest file (directly under --dst) used to make re-runs
# idempotent / incremental. It's plain JSON -- safe to inspect by hand.
MANIFEST_FILE_NAME = ".organize_manifest.json"


# =============================================================================
# Helper functions
# =============================================================================

def find_first_image(folder: Path) -> Path | None:
    """Return the first image file in `folder` (sorted for determinism)."""
    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return candidates[0] if candidates else None


def get_folder_resolution(folder: Path) -> tuple[int, int] | None:
    """Read the first image in `folder` and return its (width, height)."""
    first_image = find_first_image(folder)
    if first_image is None:
        return None
    try:
        with Image.open(first_image) as img:
            return img.width, img.height
    except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
        logging.warning("Could not read image %s: %s", first_image, exc)
        return None


def classify_bucket(width: int, height: int) -> tuple[str, int, str]:
    """
    Given a frame's (width, height), return (bucket_label, batch_size,
    phase_folder) from RESOLUTION_BUCKETS, based on the longest side.
    """
    long_side = max(width, height)
    for label, upper_bound, batch_size, phase_dir in RESOLUTION_BUCKETS:
        if long_side < upper_bound:
            return label, batch_size, phase_dir
    # Unreachable in practice since the last row's bound is infinity.
    label, _, batch_size, phase_dir = RESOLUTION_BUCKETS[-1]
    return label, batch_size, phase_dir


def chunk_list(items: list, size: int):
    """Split `items` into consecutive chunks of length `size` (last may be shorter)."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def res_key(width: int, height: int) -> str:
    """Canonical string key for a resolution, used in the manifest and folder names."""
    return f"{width}x{height}"


# ---------------------------------------------------------------------------
# Manifest handling (enables safe re-runs / incremental appends)
# ---------------------------------------------------------------------------

def load_manifest(dst_dir: Path) -> dict:
    """
    Load the JSON manifest that tracks: which source folders have already
    been organized, and the next free batch index per resolution (so batch
    numbering keeps incrementing across runs instead of restarting at 1).
    """
    manifest_path = dst_dir / MANIFEST_FILE_NAME
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not read manifest %s (%s); starting fresh.", manifest_path, exc)
    return {"processed_sources": [], "batch_counters": {}}


def save_manifest(dst_dir: Path, manifest: dict, dry_run: bool) -> None:
    """Persist the manifest back to disk (skipped in --dry-run mode)."""
    if dry_run:
        logging.info("[dry-run] Would save manifest with %d processed sources.",
                      len(manifest.get("processed_sources", [])))
        return
    manifest_path = dst_dir / MANIFEST_FILE_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Locating already-pending ("extra") batches so new examples can be
# appended to them instead of creating duplicate pending folders.
# ---------------------------------------------------------------------------

# Matches folder names like "3of8_intended_Phase1_960x540"
EXTRA_NAME_PATTERN = re.compile(
    r"^(?P<count>\d+)of(?P<batch_size>\d+)_intended_(?P<phase>Phase\d+)_(?P<width>\d+)x(?P<height>\d+)$"
)


def scan_pending_extras(dst_dir: Path) -> dict[str, dict]:
    """
    Look inside <dst>/extra for existing incomplete-batch folders and index
    them by resolution key ("WxH"). Returns a dict:
        { "1920x1080": {"folder": Path, "items": [Path, ...]}, ... }
    where "items" are the example sub-folders already physically staged
    inside that pending folder.
    """
    pending: dict[str, dict] = {}
    extra_dir = dst_dir / EXTRA_DIR_NAME
    if not extra_dir.is_dir():
        return pending

    for child in sorted(extra_dir.iterdir()):
        if not child.is_dir():
            continue
        match = EXTRA_NAME_PATTERN.match(child.name)
        if not match:
            logging.debug("Ignoring unrecognized folder in extra/: %s", child.name)
            continue
        key = f"{match.group('width')}x{match.group('height')}"
        items = sorted(p for p in child.iterdir() if p.is_dir())
        pending[key] = {"folder": child, "items": items}
    return pending


# ---------------------------------------------------------------------------
# Transferring folders
# ---------------------------------------------------------------------------

def transfer_folder(src_folder: Path, dst_folder: Path, move: bool, dry_run: bool) -> None:
    """
    Copy (default) or move `src_folder` to `dst_folder`. Used for BRAND-NEW
    examples coming from --src. Respects --dry-run.
    """
    if dst_folder.exists():
        logging.warning("Destination already exists, skipping to avoid duplicate: %s", dst_folder)
        return
    if dry_run:
        logging.info("[dry-run] %s %s -> %s", "MOVE" if move else "COPY", src_folder, dst_folder)
        return
    dst_folder.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src_folder), str(dst_folder))
    else:
        shutil.copytree(src_folder, dst_folder)


def relocate_staged_folder(src_folder: Path, dst_folder: Path, dry_run: bool) -> None:
    """
    Move a folder that is already INSIDE the managed --dst tree (e.g.
    promoting an item out of extra/ into a Phase folder as part of an
    incremental re-run). Always a move, never a copy -- there's no reason to
    duplicate data that's already been ingested once.
    """
    if dst_folder.exists():
        logging.warning("Destination already exists, skipping to avoid duplicate: %s", dst_folder)
        return
    if dry_run:
        logging.info("[dry-run] RELOCATE (within dst) %s -> %s", src_folder, dst_folder)
        return
    dst_folder.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_folder), str(dst_folder))


# =============================================================================
# Main organization logic
# =============================================================================

def organize_dataset(src_dir: Path, dst_dir: Path, move: bool, dry_run: bool):
    """
    Incrementally organize example sub-folders from `src_dir` into `dst_dir`.

    Safe to call repeatedly: already-processed source folders are skipped
    (tracked via the manifest), and new examples are appended onto any
    existing incomplete ("extra") batch for the same resolution rather than
    starting a duplicate pending folder.
    """
    manifest = load_manifest(dst_dir)
    processed: set[str] = set(manifest.get("processed_sources", []))
    batch_counters: dict[str, int] = manifest.get("batch_counters", {})

    # --- Step 1: discover source example folders, skipping ones we've
    # already organized on a previous run (this is what makes re-runs safe
    # / "always append but no duplicates"). ---------------------------------
    all_source_folders = sorted(p for p in src_dir.iterdir() if p.is_dir())
    new_source_folders = [p for p in all_source_folders if str(p.resolve()) not in processed]

    skipped_count = len(all_source_folders) - len(new_source_folders)
    logging.info(
        "Found %d example folders in source (%d already organized previously, %d new).",
        len(all_source_folders), skipped_count, len(new_source_folders),
    )

    # --- Step 2: detect resolution of each NEW example, group by EXACT
    # (width, height). ------------------------------------------------------
    new_groups: dict[str, list[Path]] = {}
    res_dims: dict[str, tuple[int, int]] = {}  # key -> (width, height)
    for folder in new_source_folders:
        resolution = get_folder_resolution(folder)
        if resolution is None:
            logging.warning("Skipping %s: no readable image frames found", folder)
            continue
        key = res_key(*resolution)
        new_groups.setdefault(key, []).append(folder)
        res_dims[key] = resolution

    # --- Step 3: load any existing incomplete batches from a previous run,
    # so their contents can be topped up instead of duplicated. -------------
    pending = scan_pending_extras(dst_dir)
    for key, entry in pending.items():
        if key not in res_dims:
            w_str, h_str = key.split("x")
            res_dims[key] = (int(w_str), int(h_str))

    all_keys = sorted(set(new_groups) | set(pending))
    if not all_keys:
        logging.info("Nothing new to organize.")
        save_manifest(dst_dir, manifest, dry_run)
        return

    # --- Step 4: for each resolution, combine (existing pending items) +
    # (new examples), classify its bucket, and chunk into batches. ---------
    for key in all_keys:
        width, height = res_dims[key]
        bucket_label, batch_size, phase_dir = classify_bucket(width, height)

        pending_entry = pending.get(key)
        pending_items = pending_entry["items"] if pending_entry else []
        pending_folder = pending_entry["folder"] if pending_entry else None
        new_items = new_groups.get(key, [])

        # Pending items are already staged inside --dst; new items still
        # live in --src. Order matters only for batch numbering
        # consistency, so pending items fill up first.
        combined_items = pending_items + new_items

        logging.info(
            "Resolution %s -> bucket '%s' -> %s (batch size %d): %d staged + %d new = %d total",
            key, bucket_label, phase_dir, batch_size,
            len(pending_items), len(new_items), len(combined_items),
        )

        chunks = list(chunk_list(combined_items, batch_size))

        for chunk in chunks:
            is_full_batch = len(chunk) == batch_size

            if is_full_batch:
                # Allocate the next batch index for this resolution and
                # persist it in the manifest so numbering never collides
                # or restarts across runs.
                next_index = batch_counters.get(key, 0) + 1
                batch_counters[key] = next_index
                batch_folder_name = f"batch_{key}_{next_index:04d}"
                dest_root = dst_dir / phase_dir / batch_folder_name
            else:
                # Incomplete batch -> (re)write the extra/ staging folder,
                # named with the CURRENT count so it's obvious at a glance
                # how many examples it still needs.
                batch_folder_name = f"{len(chunk)}of{batch_size}_intended_{phase_dir}_{key}"
                dest_root = dst_dir / EXTRA_DIR_NAME / batch_folder_name

            logging.info(
                "  -> %s: %d example(s) -> %s",
                "FULL batch" if is_full_batch else "incomplete batch (staged)",
                len(chunk), dest_root,
            )

            for item in chunk:
                dest_path = dest_root / item.name
                if pending_folder is not None and item.parent == pending_folder:
                    # Already living inside --dst/extra from a previous run:
                    # relocate (move) it rather than re-copying from source.
                    relocate_staged_folder(item, dest_path, dry_run)
                else:
                    # Brand-new example from --src.
                    transfer_folder(item, dest_path, move=move, dry_run=dry_run)
                    if not dry_run:
                        processed.add(str(item.resolve()))
                    else:
                        # Still record intent so a dry-run log reads sensibly;
                        # actual manifest is not written to disk either way.
                        processed.add(str(item.resolve()))

        # If the old pending folder is now fully drained and empty, remove
        # the leftover directory so extra/ doesn't accumulate empty husks.
        if pending_folder is not None and not dry_run and pending_folder.exists():
            remaining = list(pending_folder.iterdir())
            if not remaining:
                pending_folder.rmdir()

    # --- Step 5: persist manifest updates (processed sources + batch
    # counters) so the NEXT run knows what's already been done. ------------
    manifest["processed_sources"] = sorted(processed)
    manifest["batch_counters"] = batch_counters
    save_manifest(dst_dir, manifest, dry_run)


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Incrementally group video-interpolation example folders into resolution-based training batches."
    )
    parser.add_argument("--src", required=True, type=Path, help="Source directory containing example sub-folders")
    parser.add_argument("--dst", required=True, type=Path, help="Destination directory for the organized dataset")
    parser.add_argument("--move", action="store_true", help="Move NEW folders from --src instead of copying them")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without touching the filesystem")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.src.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)

    organize_dataset(src_dir=args.src, dst_dir=args.dst, move=args.move, dry_run=args.dry_run)

    logging.info("Done. Organized dataset written to %s", args.dst)


if __name__ == "__main__":
    main()