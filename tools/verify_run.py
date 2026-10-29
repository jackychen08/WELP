"""Check completed iterations, probability conservation, training and checkpoint reloads."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch


def verify(work):
    settings = json.loads((work/'run_settings.json').read_text())
    completion = json.loads((work/'output_files/committor_data/completion.json').read_text())
    checkpoint = torch.load(work/'output_files/committor_data/committor_net.pt', map_location='cpu', weights_only=True)
    assert checkpoint['net'], 'empty network state'
    assert all(torch.isfinite(x).all() for x in checkpoint['net'].values()), 'nonfinite network state'
    report = dict(example=settings['example'], smoke=settings['smoke'],
                  training_calls=completion['train_step'], directions={})
    for sim in ('sim_AB','sim_BA'):
        counts=[]
        with h5py.File(work/sim/'output_files/west.h5','r') as h5:
            for key in sorted(h5['iterations']):
                group=h5['iterations'][key]
                index=group['seg_index'][:]
                if not np.all(index['status']==2):
                    continue
                iteration=int(key.rsplit('_',1)[1])
                weights=index['weight']
                assert np.isfinite(weights).all() and (weights>0).all(), (sim,iteration,'weights')
                assert np.isclose(weights.sum(),1,rtol=0,atol=1e-10), (sim,iteration,weights.sum())
                assert np.isfinite(group['pcoord'][:]).all(), (sim,iteration,'pcoord')
                for field in ('final_pos','xy'):
                    if field in group.get('auxdata',{}):
                        assert np.isfinite(group['auxdata'][field][:]).all(), (sim,iteration,field)
                counts.append(iteration)
        assert counts and counts==list(range(1,max(counts)+1)), (sim,counts)
        if settings['iterations'] is not None:
            assert counts[-1]==settings['iterations'], (sim,counts[-1])
        log=(work/f'{sim}.log').read_text()
        reloads=log.count('reloaded checkpoint')
        assert reloads>=1, (sim,'network never reloaded')
        assert completion['last_trained'][sim]>=counts[-1], (sim,'untrained final iteration')
        report['directions'][sim]=dict(completed_iterations=len(counts),checkpoint_reloads=reloads)
    log=(work/'coordinator.log').read_text()
    assert 'Traceback' not in log, 'training exception'
    losses=re.findall(r'(?:avg_loss=|\[train step \d+\] loss=)([-+\w.]+)',log)
    assert len(losses)>=2 and np.isfinite([float(x) for x in losses]).all(), 'missing/nonfinite losses'
    report['finite_loss_records']=len(losses)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workdir',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.workdir.resolve()),indent=2))
