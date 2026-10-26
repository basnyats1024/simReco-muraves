# simReco_muraves

Simulation-to-reconstruction bridge for the MURAVES detector.

This package adapts simulated detector data to the MURAVES reconstruction chain and provides a validated end-to-end workflow from detector hits to final golden-track output.

## Reconstruction Chain

```text
Simulated HitData
    ->
Bar-level aggregation
    ->
Poisson digitization
    ->
Plane construction
    ->
Official MURAVES clustering
    ->
ClusteredData ROOT output
    ->
Tracking reconstruction
    ->
TrackedData ROOT output
    ->
Golden-track selection
    ->
Tracks4ptree-compatible ROOT output
```

## Package Structure

```text
simReco_muraves/
├── cli/
├── clustering/
├── config/
├── golden_tracking/
├── io/
├── models/
└── tracking/

scripts/
├── validation/
└── manual_checks/

docs/
extras/
```

## Main Components

### Phase I: Detector Response and Clustering

Implemented in:

```text
simReco_muraves/clustering/
```

Responsibilities:

- Geant4 hit aggregation
- Strip mapping
- Poisson detector-response digitization
- Plane construction
- Official MURAVES clusterization
- ClusterRecord generation

### Phase II: Tracking Reconstruction

Implemented in:

```text
simReco_muraves/tracking/
```

Responsibilities:

- Cluster-to-event organization
- Track reconstruction
- Angle reconstruction
- TrackingResult generation
- TrackedData ROOT output support

Note:

```text
simReco_muraves/tracking/Tracking.py
```

contains adapted MURAVES reconstruction code and should be treated as the authoritative tracking implementation.

### Phase III: Golden Reconstruction

Implemented in:

```text
simReco_muraves/golden_tracking/
```

Responsibilities:

- Golden-track selection
- Event/result consistency checks
- Experimental-compatible output
- Tracks4ptree output generation
- Final ROOT writing

---

## Installation

```bash
pip install -e .
```

An editable install puts `simReco_muraves` on the import path and
registers the command-line entry points. Without it, the modules must
be run from the repository root with `PYTHONPATH=.` set.

To run the test suite as well, install the development extra:

```bash
pip install -e ".[dev]"
python -m pytest simReco_muraves/golden_tracking/tests -q
```

`pytest` is not a runtime dependency, so a plain install will not
provide it.

## Production Entry Points

Each stage can be run either as a console script or as a module:

Clusterization (simulated hits -> ClusteredData):

```bash
muraves-simCluster <hit_root_file> [-o OUTPUT_DIR]
python -m simReco_muraves.cli.run_clusterization <hit_root_file>
```

Tracking (ClusteredData -> TrackedData):

```bash
muraves-simTrk <clustered_root_file_or_directory>
python -m simReco_muraves.cli.run_tracking <clustered_root_file>
```

Golden reconstruction (ClusteredData or TrackedData -> Tracks4ptree):

```bash
muraves-simGTrk <root_file_or_directory>
python -m simReco_muraves.cli.run_gTrks <root_file>
```

`--help` on any of them lists the available options. Every stage
accepts `-o/--output` for a single input file and refuses to overwrite
an existing output unless `--force` is given.

### Running the chain end to end

```bash
muraves-simCluster  validation_data/raw_100.root
muraves-simTrk      validation_data/raw_100_clustered_part000.root
muraves-simGTrk     validation_data/raw_100_tracked_part000.root
```

The golden stage accepts either input type and detects which it has
been given. `--input-type {clustered,tracked}` asserts the expected
type and errors on a mismatch, rather than overriding detection.

Golden output derived from a clustered file and from the
corresponding tracked file resolve to the same default filename. That
is deliberate: the two routes are expected to produce identical
output, so the result does not depend on which one was taken, and the
route that was actually used is recorded in the file's provenance
rather than in its name.

The consequence is that producing both at once, in order to compare
them, needs `-o` or separate directories. An existing output is never
overwritten without `--force`.

```bash
muraves-simGTrk validation_data/raw_100_clustered_part000.root \
    -o validation_data/g_clustered.root
muraves-simGTrk validation_data/raw_100_tracked_part000.root \
    -o validation_data/g_tracked.root
```

`muraves-simCompare` does this for you, writing its intermediates to a
temporary directory.

### Digitization parameters

The clusterization stage exposes the detector-response constants that
set the simulated energy scale:

```bash
muraves-simCluster <hit_root_file> --pe-per-mev 8.5 --rng-seed 42
```

The default conversion factor is the one used for the Phase I-III
validation samples.

`--rng-seed` is a master seed. It drives both random processes in
clusterization, Poisson digitization and the experiment's random
smearing of single-strip cluster positions, through seeds derived per
plane and per event (`simReco_muraves/clustering/seeding.py`). The
same input, seed and code therefore give identical output; the result
does not depend on processing order or chunk boundaries; and changing
one plane's inputs changes only that plane's random numbers. The
scheme in use is recorded in each file's provenance.

Output produced before this scheme was introduced is not reproducible.
Single-strip smearing then drew from an unseeded system source, so
re-running gave a slightly different golden selection each time, with
about 0.6% of golden events changing between runs.

---

## Validation

Canonical regression suite:

```bash
python -m pytest \
  simReco_muraves/golden_tracking/tests \
  -q
```

The suite covers the golden-track layer, which both the clustered and
the tracked route share.

### Comparing the two golden routes

Golden output can be produced either straight from ClusteredData or via
persisted TrackedData. The two routes share a record builder but reach
it through different code, so they are checked against each other
directly:

```bash
muraves-simCompare <clustered_root_file>
```

This runs both routes from one input and compares every common branch,
entry by entry, exiting non-zero on any difference. Intermediate files
go to a temporary directory unless `--work-dir` is given.

Two golden files that already exist can be compared without re-running
anything:

```bash
muraves-simCompare --golden left.root right.root
```

Comparison is exact by default; `--rtol` and `--atol` relax it if
needed. NaN compares equal to NaN, since a branch that is NaN in both
routes agrees.

## Provenance

Every output file carries a record of what produced it, in a one-entry
`Metadata` tree alongside the physics tree. Each stage appends to the
lineage it inherits from its input, so a golden file records not only
its tracking parameters but the digitization constant that made the
clusters it descends from.

```bash
muraves-sim --provenance validation_data/raw_100_golden_part000.root
```

The record holds the stage name, a UTC timestamp, the git commit and
whether the working tree was dirty, the package and library versions,
and the parameters that stage actually used.

Recording is best-effort: if git is unavailable or the metadata cannot
be written, the stage warns and continues rather than failing a
production run. A file can therefore legitimately have no provenance,
which is also the case for anything produced before this was added.

Note that a golden file made from ClusteredData records three stages
and one made from TrackedData records four, so two golden files whose
physics branches are identical still differ in their metadata. That is
intended: the lineage describes how the file was made.

### Schema version

The golden output carries a schema version, bumped whenever branches
are added, removed, renamed, or change meaning. It is not bumped when
only values change: a different digitization constant produces
different numbers in the same schema, and the parameter record is what
distinguishes those files.

| Version | Change |
|---|---|
| 1 | First versioned layout |
| 2 | Adds `BestTrack4pIsTexpNull_xz` / `_xy` |

The flags record how the plane-4 candidate was chosen: `-1` when no
four-plane promotion happened, `1` when it was promoted but no
candidate had a valid Texp so the first was used, and `0` when a valid
Texp was found. This mirrors `best_track_4p_is_texp_null_*` in the
experimental reconstruction.

## Command overview

```bash
muraves-sim
```

lists the chain and every command in it.

## Reused reconstruction code

Clustering and tracking are the experiment's own code, taken from
`4_MainRecSof` and left untouched except where noted:

```text
simReco_muraves/clustering/ClusterLists.py
simReco_muraves/tracking/Tracking.py
simReco_muraves/tracking/Evaluate_Angular_Coordinates.py
```

Keeping these equivalent to the originals is deliberate. Cluster and
track definitions are then identical between simulation and experiment
by construction rather than by agreement, which is what makes the
comparisons in the Phase I-III validation meaningful.

### Adaptation history

Two errors were found by putting simulation and reconstructed data
into the same format and comparing them. Both were in this package,
not in the main reconstruction:

- tracking was run on plane indices (`X_POS = [0, 1, 2, 3]`) instead
  of the physical detector geometry, which made the three-plane polar
  angle wrong by roughly a factor of three: 10.9 degrees became 32.4
  degrees once the real geometry was propagated;
- the four-plane angles were unimplemented here and always returned
  -1. The four-plane angular reconstruction from `MURAVES_reco.py` was
  ported across to fix it.

### Observations on the reused code

The following are **not** changed here. They are recorded so that the
next person reading this code does not have to rediscover them, and
because the first may be of interest to whoever maintains the
experimental reconstruction.

**Texp gate in `MakeTracks`.** Best-track selection is gated on

```python
if Texp_cl1[i] > 0 and Texp_cl2[j] and Texp_cl3[k]:
```

Only the first term tests the sign. The other two are truthiness
tests, so a negative Texp on plane 2 or 3 passes the gate while a Texp
of exactly zero fails it. Simulation is unaffected, because simulated
hit times are always positive. On the experimental side the gate is
therefore more permissive than the first term suggests. This is
expected to be revisited once the simulation timing study gives a
clearer picture of how Texp should be used, so it is deliberately left
alone for now.

**Plane-4 offset.** `Z_add[0..2]` are added to the plane 1-3 cluster
positions, but `Z_add[3]` is never applied to plane 4. This has no
effect today, since `z_add[3]` is 0.0 for NERO, ROSSO and BLU in
`reco_parameters.json`, but the asymmetry would matter if that ever
changed.

**Compact four-plane indexing.** `Track_3p_to_4p_index` gains an entry
only when plane 4 holds at least one cluster, while the nested
four-plane arrays gain a row whenever the extrapolated position lands
in acceptance. These cannot fall out of step in practice, because
plane-4 clusters are a per-event property and the condition is
all-or-nothing within an event. It is worth knowing if that structure
is ever reworked.

### Angular convention

`TrackAngularCoordinates` defines the polar angle as

```text
theta = atan( |dz| / sqrt(dx^2 + dy^2) )
```

that is, the elevation above the xy-plane rather than the angle from
the telescope pointing axis. This is the collaboration's agreed
convention and matches the experimental reconstruction, so simulated
and reconstructed angles are directly comparable.

Because `|dz|` is used, theta never spans both hemispheres. Forward
and backward quantities must therefore be derived from phi, not from
theta.

### Geometry

The geometry constants in the tracking and golden entry points are the
**ROSSO** working point from `reco_parameters.json`:

```text
x_pos = [-0.265, 0.0, 0.262, 1.475]
z_add = [ 0.292, 0.251, 0.207, 0.0  ]
```

NERO and BLU differ, most visibly in the lever arm to plane 4 (1.492 m
against 1.475 m) and, for BLU, throughout `z_add`. Comparisons of
simulation against those two detectors carry that difference.

The proximity cuts are not independent constants: they are
`proximity_sigma_multiplier` (5.0) times the corresponding sigma, so
`0.004 -> 0.020` and `0.0035 -> 0.0175`. They are currently written
out literally, so a change to sigma will not propagate to them
automatically.

---

## Development Material

Historical studies, diagnostics, comparison plots, exploratory scripts, logs, and original reconstruction references have been intentionally separated from the production package and moved to:

```text
extras/
```

The production package itself is contained entirely inside:

```text
simReco_muraves/
```

---

## Status

Current state:

- Phase I implemented and validated
- Phase II implemented and validated
- Phase III implemented and validated
- Production code separated from diagnostics
- Regression suite passing
- Tracking implementation preserved
- Repository being prepared for publication and long-term maintenance
