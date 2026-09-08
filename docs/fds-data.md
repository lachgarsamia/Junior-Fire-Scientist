# FDS scenario data

## What the simulations are

The application visualizes a 24-scenario factorial study of a tabletop
candle experiment run in Fire Dynamics Simulator (FDS 6.7.1): a
1.0 m × 0.48 m domain containing a 0.73 m × 0.22 m enclosed room, with 1 or
2 candles as the fire source, a door, and a ventilation state. There is a
physical mockup of this box used alongside the app at demos/exhibitions.
The original 2019 experiment-design document for the outreach event this
study grew out of is kept at `protocol/2019_05_21.tex` (German, "Tag der
Neugier" / "Day of Curiosity") for provenance.

Factor levels (24 = 2 × 2 × 3 × 2):

| Factor | Levels |
|---|---|
| Candles (`c`) | 1 or 2 |
| Door width (`d`) | narrow or wide |
| Vertical opening / door vent (`vod`) | open / closed / HVAC fan |
| Vertical opening / ceiling vent (`voc`) | open / closed |

Each scenario folder's name begins with `c{candles}_d{door}_vod{vod}_voc{voc}`
(the current cluster-run output directories are
`c{...}_..._stage1_pleiades`) and contains FDS `.smv`/`.sf` slice output
plus `.s3d` volumetric soot output. `fds/sim/manifest.json` records, for
each of the 24 scenarios, its directory name and its position on the
candle/door/vod/voc factor axes; the application reads that file to map
the dataset and locates each scenario by directory name inside `fds/sim/`.

## What's tracked in this repository vs. generated

- **`fds/generate_sim.py`, `fds/template.fds`, `fds/template_hvac.fds`,
  `fds/start_job.batch`** — tracked. This is the actual pipeline that
  produced the 24 scenarios: `generate_sim.py` substitutes each factor
  combination into the FDS input-deck templates, and `start_job.batch` is
  the SLURM submission script used on the FZJ cluster. Kept so the
  simulation setup is fully documented and reproducible, even though
  running it requires FDS 6.7.1 and cluster access neither of which this
  repository can provide.
- **`fds/sim/`** — **not tracked** (~11 GB of raw simulation output; see
  `.gitignore`). The application reads this directory and will not start
  without it — it is handed over separately (see the README). To
  regenerate it instead, run `fds/generate_sim.py`
  to produce the 24 FDS input decks and then run FDS 6.7.1 on each
  (`fds/start_job.batch` is the SLURM submission script used on the FZJ
  cluster), and write a `manifest.json` alongside the output directories.
- **`tests/fixtures/c1_d0_vod0_voc0/`** — tracked. A trimmed copy of exactly
  one scenario's `.sf` files (only the quantity/direction/offset the test
  suite actually reads), so the parser and slice-loading tests run without
  the full dataset.

## Parser validation

`src/fds/slice/slice.py` (the `.smv`/`.sf` binary parser) was cross-validated
against the independent `fdsreader` library on a real scenario: timestamps
match exactly, and the per-frame maximum temperature — the statistic the
app's analytics actually consume — matches exactly across all 481 frames.
One narrow discrepancy at the single outermost boundary column was
investigated and resolved in favor of this project's own parser (see
`tests/test_slice_parser.py::TestOuterEdgeColumn`); `fdsreader`'s
global-mesh-stitching step was found to duplicate a boundary value rather
than this parser misreading it.

## Gated quantities

The quantity registry (`src/registry.py`) declares some quantities — full
CO-based FED, `VISIBILITY`, `HEAT FLUX`, `SOOT MASS FRACTION`, true
`U`/`W`-`VELOCITY` vector components — that the current FDS output does not
actually contain. Requesting one of these through `QuantityProvider` raises
`GatedQuantityError` with a clear reason instead of fabricating a value.
They're visible only in the read-only Quantities reference panel, never in
data-driven quantity pickers, so they can't silently break a feature.
Producing them for real requires re-running the simulations with richer
`&SLCF` output declared in the FDS input deck (a cluster job, not something
this app can do) — out of scope for this repository, but the seam is
already there in the code if that re-run ever happens.

## Forecasting model (optional, not required to run the app)

`ml/` trains a Fourier Neural Operator that forecasts future temperature
frames. It's fully independent of the app (`ml/` has its own dependencies —
torch, neuraloperator — never imported by `src/`) and is documented in
`ml/README.md`. If `predictions/` (its output, not tracked) is generated,
the app's Dataset page gains a "View model prediction" button; otherwise
that feature is simply absent.
