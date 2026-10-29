#!/usr/bin/env python3
"""Create an isolated, fresh WELP run and supervise both WE directions and training."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent


def environment(work, smoke=False):
    env = os.environ.copy()
    env.pop('WEST_SIM_ROOT', None)
    env.update(COMMITTOR_SHARED_ROOT=str(work),
               COMMITTOR_SIM_ROOTS=f'{work}/sim_AB:{work}/sim_BA',
               PYTHONPATH=str(work), HDF5_USE_FILE_LOCKING='FALSE',
               WELP_SMOKE='1' if smoke else '0', PYTHONUNBUFFERED='1')
    env.setdefault('XDG_CACHE_HOME', str(work/'.cache'))
    env.setdefault('MPLCONFIGDIR', str(work/'.cache/matplotlib'))
    for key, value in {'SEED':'0', 'OMP_NUM_THREADS':'1', 'OPENMM_CPU_THREADS':'1',
                       'OPENMM_PLATFORM':'CPU', 'WANDB_MODE':'disabled',
                       'MPLBACKEND':'Agg'}.items():
        env.setdefault(key, value)
    env['ADP_GLOBAL_SEED'] = env['SEED']
    env['TOY_GLOBAL_SEED'] = env['SEED']
    if smoke:
        env.update(ADP_TOTAL_STEPS='10', TOY_N_STEPS='5')
    return env


def prepare(example, work, smoke, iterations):
    if work.exists():
        raise FileExistsError(f'Refusing to overwrite {work}; choose a new run directory')
    shutil.copytree(REPO/'examples'/example, work)
    env = environment(work, smoke)
    if example == 'toy':
        subprocess.run([sys.executable, str(work/'common_files/make_bstates.py'),
                        '--n', '1' if smoke else '10', '--root', str(work)],
                       env=env, cwd=work, check=True)
    else:
        (work/('common_files/lazzeri/generated/system.xml' if example == 'chignolin' else 'common_files/system.xml')).write_bytes(gzip.decompress((work/'assets/system.xml.gz').read_bytes()))
        for sim, basin in [('sim_AB', 'A'), ('sim_BA', 'B')]:
            basis = work/sim/'bstates'
            basis.mkdir()
            (basis/'initial.xml').write_bytes(gzip.decompress((work/f'assets/{basin}.xml.gz').read_bytes()))
            (basis/'bstates.txt').write_text('0 1.0 initial.xml\n')
    for sim in ('sim_AB', 'sim_BA'):
        cfg = work/sim/'west.cfg'
        text = cfg.read_text()
        if smoke:
            text = re.sub(r'bin_target_counts:\s*\d+', 'bin_target_counts: 2', text)
            text = re.sub(r'nbins:\s*\[\d+,\s*1,\s*1\]', 'nbins: [2, 1, 1]', text)
        if iterations is not None:
            text = re.sub(r'max_total_iterations:\s*\d+', f'max_total_iterations: {iterations}', text)
        cfg.write_text(text)
        (work/sim/'output_files').mkdir(exist_ok=True)
    (work/'run_settings.json').write_text(json.dumps(dict(example=example, smoke=smoke,
        iterations=iterations, seed=env['SEED'], platform=env['OPENMM_PLATFORM']), indent=2)+'\n')
    return env


def command(entry):
    return [sys.executable, '-c', f'from westpa.cli.core.{entry} import entry_point; entry_point()']


def sim_environment(env, work, sim):
    return dict(env, WEST_SIM_ROOT=str(work/sim), TARGET_BASIN='B' if sim == 'sim_AB' else 'A',
                SIM_NAME=sim, OUTPUT_DIR=str(work/sim/'output_files'))


def run(work, env, timeout):
    # Initialize before launching any training or simulation process.
    for sim in ('sim_AB', 'sim_BA'):
        with (work/f'{sim}.init.log').open('w') as log:
            subprocess.run(command('w_init') + ['--bstate-file', 'bstates/bstates.txt',
                '--tstate-file', 'tstate.file', '--segs-per-state', '2', '--work-manager=serial'],
                cwd=work/sim, env=sim_environment(env, work, sim), stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=timeout)
    # Enable the toy's existing drift gate before its fast simulations can run.
    # Its warmup and maximum-ahead defaults are both three iterations.
    data = work/'output_files/committor_data'
    data.mkdir(parents=True, exist_ok=True)
    (data/'training_state.json').write_text(json.dumps(dict(
        last_trained={'sim_AB': 0, 'sim_BA': 0}, train_step=0, warmup_complete=True)))
    processes = {}
    handles = []
    started = time.monotonic()
    try:
        launches = [('coordinator', [sys.executable, str(REPO/'tools/coordinator.py')], work, env)]
        launches += [(sim, command('w_run')+['--work-manager=serial'], work/sim,
                      sim_environment(env, work, sim)) for sim in ('sim_AB','sim_BA')]
        for label, cmd, cwd, child_env in launches:
            log = (work/f'{label}.log').open('w'); handles.append(log)
            processes[label] = subprocess.Popen(cmd, cwd=cwd, env=child_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            codes = {name: process.poll() for name, process in processes.items()}
            for name, code in codes.items():
                if code is not None and code != 0:
                    raise RuntimeError(f'{name} exited with {code}; see {work}/{name}.log')
            if all(codes[sim] == 0 for sim in ('sim_AB','sim_BA')):
                (work/'simulations.done').touch()
            if codes['coordinator'] == 0:
                if not (work/'simulations.done').exists():
                    raise RuntimeError('Coordinator exited before the simulations completed')
                break
            if time.monotonic() - started > timeout:
                raise TimeoutError(f'Run exceeded {timeout}s; logs are in {work}')
            time.sleep(0.2)
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()
    print(f'Completed: {work}', flush=True)


def main():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', choices=['toy','adp','chignolin'])
    parser.add_argument('--workdir', type=Path, required=True, help='new directory; never overwritten')
    parser.add_argument('--smoke', action='store_true', help='small pipeline check, not a scientific run')
    parser.add_argument('--iterations', type=int, help='override the production WE budget')
    parser.add_argument('--timeout', type=float, default=86400, help='per-initialization and supervised-run timeout (seconds)')
    args = parser.parse_args()
    if args.iterations is not None and args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    work = args.workdir.resolve()
    iterations = args.iterations if args.iterations is not None else (6 if args.smoke else None)
    env = prepare(args.example, work, args.smoke, iterations)
    run(work, env, args.timeout)


if __name__ == '__main__':
    main()
