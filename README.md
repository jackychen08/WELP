# WELP — Weighted Ensemble in the LooP

Compact, runnable examples of committor learning coupled to two steady-state
WESTPA simulations. A learned committor supplies the adaptive binning coordinate;
weighted-ensemble samples supply parent–children training data. Both directions
use q-only MAB binning and within-bin merge/re-split.

The molecular examples are extracted from the latest production WELP lineage
identified in the source workspace on October 29, 2026:

| Example | Source preset | Physical system |
| --- | --- | --- |
| `toy` | `toy_two_channel/per_qonly_wsm/s0` | ZJZ two-channel potential, overdamped dynamics |
| `adp` | `adp_seed_sweep/und_qonly_wsm_greedylike_a4500/s0` (August 31 preset) | 22-atom alanine dipeptide in vacuum, CHARMM27, 310 K, friction 0.3/ps, 1 fs steps, 100 steps/segment |
| `chignolin` | `we_chig_lazzeri_frequent_callplots_s4_20260923` | 6,468-atom solvated CHARMM27 system, 166 protein atoms, 340 K, friction 0.3/ps, 0.5 fs steps, 4,000 steps/segment |

Chignolin uses the **current native-contact definition**: 74 contacts, folded A
at structural Q >= 0.99, unfolded B at Q <= 0.01, with periodic distances computed
before whole-protein unwrapping for the neural network. It does not use the older
H-bond-sum/AMBER99SB example. Its physical inputs were shared with a SOC benchmark,
but the sampler, trainer and inference loop here are WELP; no SOC optimizer is
included. The October toy `semigroup_study` is a greedy-baseline study, not a newer
WELP implementation.

## Install

Tested with Python 3.11.5 on Linux. A CPU installation is sufficient for the smoke
checks; use a CUDA-capable OpenMM/PyTorch installation for substantial runs.
The dependencies are installed into your own environment, never from a lab path.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Alternatively, with `uv`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements.txt
```

OpenMM systems and one starting state per molecular basin are included in lossless
gzip form. There is no GROMACS installation, external checkpoint, private dataset,
remote download or W&B login needed to run these examples. W&B logging is disabled
by default. Its Python package is retained because the inherited trainers use its
metrics/image API.

## Run

From the repository root, choose a **new** output directory:

```bash
python run.py toy --workdir /tmp/welp-toy --smoke
python run.py adp --workdir /tmp/welp-adp --smoke
python run.py chignolin --workdir /tmp/welp-chignolin --smoke
```

The launcher copies the selected example into the output directory, expands its
inputs, initializes both directions, and supervises simulations and training.
It refuses to overwrite an existing directory. Child failures or timeouts terminate
the other processes and return an error. It does not resume an interrupted run.
Logs remain in the output directory for inspection.

`--smoke` uses six WE iterations per direction, two target walkers per bin, two
adaptive bins, two training epochs, batch size 16, and short segments (5 toy steps
or 10 molecular steps). The production training cadence is retained. These checks
exercise the pipeline, not convergence or rare-event rates.

For a scientific run, omit `--smoke`; optionally override the iteration budget:

```bash
OPENMM_PLATFORM=CUDA SEED=0 python run.py adp --workdir runs/adp-s0 --iterations 1000
OPENMM_PLATFORM=CUDA SEED=0 python run.py chignolin --workdir runs/chignolin-s0 --iterations 1000
```

Run sustained jobs in an appropriate scheduler allocation on a cluster.
`OPENMM_PLATFORM` defaults to `CPU`; `OPENMM_CPU_THREADS` defaults to 1.
PyTorch uses CUDA when available. Set `CUDA_VISIBLE_DEVICES` to select a GPU or
hide GPUs for an all-CPU run. `--timeout` defaults to 86,400 seconds for each
initialization and for the supervised run. Increase it for longer allocations.
`SEED` controls trainer and MD seeds; WESTPA resampling is not bitwise reproducible
from this seed alone.

Outputs include:

- `sim_AB/output_files/west.h5` and `sim_BA/output_files/west.h5`: weighted trajectories.
- `output_files/committor_data/committor_net.pt`: trained network and optimizer.
- `output_files/committor_data/completion.json`: final per-direction training progress.
- `coordinator.log`, `sim_AB.log`, `sim_BA.log`: losses and execution logs.

## What was kept and reduced

The latest molecular loss, replay, target-network and residual-network behavior is
retained. ADP and Chignolin both use warmup 1, training interval 1 (combined new
iterations), and maximum-ahead 1. ADP retains its full-batch preset and 100 epochs;
Chignolin retains batch size 1,024 and 120 epochs. They retain PER, semigroup and
boundary losses, and the scheduled BKE term (target 0.05, ramp over iterations
500–1,000). The toy retains its own PER/BKE/variance settings and three-iteration
training interval. The source configs are authoritative; these are not simplified
semigroup-only presets.

To keep the release small, each molecular example contains **one** full-system
starting state per basin, rather than the full boundary ensemble. Chignolin retains
the original fixed feature-normalization statistics. System/state XML is compressed
without numerical conversion. Unused network architectures, cluster scripts, seed
copies, trajectories, trained models, external test sets, rate-reference ensembles
and large diagnostic suites are omitted. The small input states and normalization
statistics are required runtime assets, not learned checkpoints.

The compact launcher/coordinator is shared across examples. It waits for completed
WE iterations, publishes progress atomically, flushes final training, and stops on
failure. The toy startup gate is enabled before launching the fast simulations so
a short run cannot finish before the first network update. This fresh-run interface
replaces cluster-specific supervision; the September resume/scratch orchestration
is intentionally not included.

These are standalone demonstrations of the current methods, **not reproduction
of all manuscript figures**: reduced starting ensembles change initial sampling;
optional accuracy/rate diagnostics are absent; only bounded CPU smoke behavior is
validated here. Production convergence and GPU performance are not established by
those checks.

## Validate

```bash
python tests/smoke.py --workdir /tmp/welp-validation
python tools/verify_run.py /tmp/welp-adp
```

The suite runs all three systems from fresh directories and checks completed
iterations in both directions, conserved probability weights, finite coordinates
and losses, checkpoint contents, network reloads, and final training progress.
See [VALIDATION.md](VALIDATION.md) for the recorded results.

The toy optionally includes a finite-difference reference generator. It is not
required for training, and its default grid solve can be expensive:

```bash
cd /tmp/welp-toy
python common_files/solve_true_committor.py --nx 101 --ny 101
```

See [SOURCE_AUDIT.json](SOURCE_AUDIT.json) and [PROVENANCE.json](PROVENANCE.json)
for source identities, preserved-method checks, and file hashes; see
[ATTRIBUTION.md](ATTRIBUTION.md) for method and physical-input attribution.
