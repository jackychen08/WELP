# Attribution and provenance

WELP combines weighted-ensemble exploration with learned committors. The inherited
code identifies Mitchell and collaborators' `sc_committor_rates` implementation
and the 2024 semigroup committor work as sources for network/training ideas and
adaptations. The toy uses the Zhang–Jasnow–Zuckerman two-channel potential.
Weighted-ensemble simulation and MAB binning are implemented by WESTPA; molecular
dynamics is implemented by OpenMM. Those installed dependencies retain their own
licenses.

The ADP system is serialized from the source run's `topol_vacuum.top`, which records
GROMACS generation by Andy Mitchell in 2020, using CHARMM27 parameters, NoCutoff,
and hydrogen-bond constraints. The release preserves the resulting OpenMM system
rather than shipping the GROMACS force-field installation.

The Chignolin runtime system, basis states, fixed feature statistics and native
contact definition come from the prepared WELP Lazzeri transfer. That transfer
pinned shared molecular inputs from `soc-committor` commit
`f40764daa6cd340c4a6253d1a103cecea40931cf` (`codex/chignolin-lazzeri-benchmark`).
Only the contact-definition helpers and required physical inputs are included;
there is no SOC training or control implementation here. The four pinned contact
files retain their original hashes, checked at runtime by `ArchiveQ`.

Chignolin physics: CHARMM27, PME, cutoff 0.95 nm, Ewald tolerance 1e-5, rigid water,
flexible protein, 340 K, friction 0.3/ps, LangevinMiddle integration at 0.5 fs,
constraint tolerance 1e-6. Native-contact Q is computed with the periodic box;
network distances use separately unwrapped and centered protein coordinates.

`PROVENANCE.json` records the workspace-relative origins and source/release hashes.
Those paths identify research run lineages; no local account paths are needed at
runtime. `SOURCE_AUDIT.json` records comparisons of core functions against the
latest production sources. Gzip files preserve the original serialized values.

No project-wide license has been assigned in this initial publication. Public
availability alone is not an additional license for redistribution; third-party
terms continue to apply. A project license can be added once the applicable
ownership and adapted-source terms are reconciled.
