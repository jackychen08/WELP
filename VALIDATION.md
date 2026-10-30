# Validation — October 29, 2026

All three examples passed `python tests/smoke.py --workdir <new directory>` in a
fresh, isolated Python 3.11.5 virtual environment installed from `requirements.txt`
using `uv pip install --torch-backend cpu`. No lab virtual environment or lab data
path was included in the test processes' PYTHONPATH. Tests ran on allocated SLURM
CPU compute with OpenMM CPU, one OpenMM thread per simulation, and W&B disabled.

| Example | Completed WE iterations per direction | Coordinator calls | Checkpoint reloads, AB / BA | Finite loss records |
| --- | ---: | ---: | ---: | ---: |
| Toy | 6 | 2 | 1 / 1 | 2 |
| ADP | 6 | 5 | 1 / 1 | 12 |
| Chignolin | 6 | 10 | 4 / 4 | 27 |

`tools/verify_run.py` checked positive finite weights and a total probability of
1 within absolute tolerance 1e-10 in every completed iteration; finite progress
coordinates and stored molecular/toy coordinates; finite checkpoint tensors and
reported losses; actual checkpoint reloads in both simulations; and final training
progress covering the completed simulation budget. Initial coordinator calls can
precede the availability of parent–children pairs, so call counts are not identical
to optimizer-update counts. Counts can vary with process scheduling.

The source audit passed 41 checks, including unchanged loss, replay and target
functions, inference/recycling drivers, selected neural-network classes, and the
molecular training function (apart from removed optional projection/rate logging).
The September 20 and September 23 Chignolin method files were identical. Source
identities and hashes are recorded in `SOURCE_AUDIT.json` and `PROVENANCE.json`.

A separate SIGTERM check confirmed that the launcher exits with status 143 and
reaps all three supervised child processes. Python syntax checks passed. The release was inspected for external symlinks,
personal absolute paths and common credential patterns, including decompressed
system/state assets. Only the source, small required inputs, documentation and
validation tools are included. Generated trajectories, checkpoints and test outputs
are excluded.

The smoke preset deliberately shortens segments and training. It validates the
closed loop and portability, not scientific convergence, rate estimates, full
manuscript reproduction, GPU behavior, or long-run restart behavior. The optional
large toy reference-grid calculation was not part of these checks.

## Paper Chignolin addition

The archived `C_both` preset (`chignolin_paper`) passed a separate clean-environment
CPU smoke run with its original ten-iteration warmup/cadence retained:

- 16 completed WE iterations in each direction, conserved probability weights,
  finite coordinates, finite checkpoint tensors and finite training losses.
- Three training calls and three finite Hill-rate evaluations from the original
  frozen boundary ensembles/fluxes; nine finite loss records.
- Two checkpoint reloads in each directional simulation; final training progress
  covered both complete simulation budgets.
- Twelve additional source checks: unchanged training function, target/collection
  functions, loss and rate estimator, drivers, logit mapper, basin definition,
  residual network, and production configuration fields (except logging name).
  The combined source audit now contains 53 passing checks.

The added example includes one full-solvent start per basin, rather than the
original 100 starts. Rate-evaluation ensembles are preserved in full. The result
validates the paper method implementation and estimator, not convergence to the
published curves. The other three examples were not rerun for this addition;
their prior isolated-environment results above remain applicable.
