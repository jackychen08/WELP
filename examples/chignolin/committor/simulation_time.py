"""Cumulative completed sampling MD; excludes shared boundary preparation."""
from pathlib import Path
import json
import os
import numpy as np
import h5py

DT_NS=0.5e-6

def we_time(cfg):
    counts={}
    for src in cfg.sim_sources:
        count=0
        with h5py.File(src['west_h5'],'r',locking=False) as f:
            for group in f['iterations'].values():
                if 'seg_index' in group:
                    count+=int(np.count_nonzero(group['seg_index']['status']==2))
        counts[src['label']]=count*int(os.environ.get('ADP_TOTAL_STEPS', '4000'))
    steps=sum(counts.values())
    return {'compute/sim_steps_cumulative':steps,'compute/sim_time_ns':steps*DT_NS,
            **{f'compute/{k}_sim_time_ns':v*DT_NS for k,v in counts.items()}}

def grant_time(c):
    a=float(c.a_times[-1]) if len(c.a_times) else 0.
    b=float(c.b_times[-1]) if len(c.b_times) else 0.
    return {'compute/sim_steps_cumulative':int(a+b),'compute/sim_time_ns':(a+b)*DT_NS,
            'compute/A_sim_time_ns':a*DT_NS,'compute/B_sim_time_ns':b*DT_NS}
