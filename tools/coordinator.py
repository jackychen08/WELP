"""Train only on completed WE iterations; propagate failures to the supervisor."""
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from committor import config
from committor.train import train, _init_wandb


def latest_complete(path):
    try:
        with h5py.File(path, 'r', locking=False) as handle:
            latest = 0
            for key in sorted(handle.get('iterations', {})):
                group = handle['iterations'][key]
                if 'seg_index' not in group:
                    break
                index = group['seg_index'][:]
                if not len(index) or not np.all(index['status'] == 2):
                    break
                if 'auxdata' not in group:
                    break
                latest = int(key.rsplit('_', 1)[1])
            return latest
    except (OSError, KeyError):
        return 0


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def main():
    toy = hasattr(config, 'ToyCommittorConfig')
    cfg = (config.ToyCommittorConfig if toy else config.CommittorConfig)()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    data = Path(cfg.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    _init_wandb(cfg)
    last = {src['label']: 0 for src in cfg.sim_sources}
    step = 0
    while True:
        latest = {src['label']: latest_complete(src['west_h5']) for src in cfg.sim_sources}
        done = (Path(cfg.shared_root)/'simulations.done').exists()
        warm = min(latest.values()) >= cfg.warmup_iters
        new = sum(latest[key] - last[key] for key in latest)
        progress = dict(last_trained=last, train_step=step, warmup_complete=(True if toy else warm))
        atomic_json(data/'training_state.json', progress)
        if warm and new and (new >= cfg.train_interval or done):
            step += 1
            print(f'Training call {step}, completed iterations {latest}', flush=True)
            if toy:
                train(cfg, step)
            else:
                train(cfg, step, westpa_iter=min(latest.values()))
            last = dict(latest)
            progress = dict(last_trained=last, train_step=step, warmup_complete=True)
            atomic_json(data/'training_state.json', progress)
            atomic_json(data/'coordinator_progress.json', progress)
        if done:
            if not step or not Path(cfg.checkpoint).exists():
                raise RuntimeError('No training update completed; increase the iteration budget')
            atomic_json(data/'completion.json', dict(last_trained=last, train_step=step))
            return
        time.sleep(min(cfg.coordinator_poll_seconds, 1.0))


if __name__ == '__main__':
    main()
