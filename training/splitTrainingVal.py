"""
Standalone convenience script to materialize the train/val split WITHOUT
running training -- useful if you just want to browse which sequences
would be trained on vs held out before committing to a run.

This is now a thin wrapper around dataset.materialize_split(), the same
function train.py calls automatically at the start of every run. There is
no split logic duplicated here anymore, so there's nothing that can drift
out of sync with what training actually uses -- point this at the same
yaml you'll train with and it reproduces exactly the same split.

Usage:
    python split_train_val.py --config train_config.yaml
"""
import argparse

import yaml

from dataset import materialize_split


def main():
    parser = argparse.ArgumentParser(description='Materialize the train/val split defined in a yaml config')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    d_cfg = C['data']
    r_cfg = C['run']

    data_root = d_cfg['data_root']
    val_split = d_cfg.get('val_split', 0.2)
    split_mode = d_cfg.get('split_mode', 'symlink')
    seed = r_cfg.get('seed', 42)

    run_dir = f"{r_cfg['runs_dir']}/{r_cfg['run_name']}"
    train_out = d_cfg.get('train_out') or f'{run_dir}/split/train'
    val_out = d_cfg.get('val_out') or f'{run_dir}/split/val'

    materialize_split(data_root, train_out, val_out, seed, val_split, mode=split_mode)


if __name__ == '__main__':
    main()