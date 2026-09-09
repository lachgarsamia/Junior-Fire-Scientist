# Junior Fire Scientist

An interactive educational application for exploring fire science through real
Fire Dynamics Simulator (FDS) simulations of a tabletop candle experiment.
Developed for Forschungszentrum Jülich (FZJ) and the University of Wuppertal.

## Requirements

- macOS
- Python 3 (3.9 or newer)
- The FDS simulation dataset, provided separately (see step 2)

`run.sh` handles the Python environment and dependencies for you — there is
nothing to `pip install` by hand.

## 1. Get the project

```
git clone https://github.com/lachgarsamia/Junior-Fire-Scientist.git
cd Junior-Fire-Scientist
```

## 2. Add the simulation data

The simulation data (~11 GB) is provided separately and is **not** on GitHub.
You will receive a folder named `sim`. It must end up at:

```
Junior-Fire-Scientist/fds/sim/
```

with `fds/sim/manifest.json` and 24 scenario directories inside:

```
Junior-Fire-Scientist/fds/sim/
├── manifest.json
├── c1_d0_vod0_voc0_stage1_pleiades/
├── ...
└── c2_d1_vod2_voc1_stage1_pleiades/
```

### If you only use Junior Fire Scientist

Copy the `sim` folder itself (not its contents) into the repository's `fds/`
directory. If you end up with `fds/sim/sim/`, you copied one level too deep.

### If you also use FireScope

Keep a single copy of the dataset. Put the real dataset in FireScope's
`fds/sim/`, then point Junior Fire Scientist at it with a symlink:

```bash
ln -s /path/to/FireScope/fds/sim \
      /path/to/Junior-Fire-Scientist/fds/sim
```

Both applications then read the exact same files, with no second 11 GB copy.
Junior Fire Scientist does not know or care that it is a symlink — it only
needs `fds/sim/manifest.json` to resolve.

### Verify

```
ls fds/sim/manifest.json
```

If that prints the path without an error, the dataset is in the right place.

## 3. Launch

```
chmod +x run.sh
./run.sh
```

The first launch creates a Python environment and installs dependencies, which
takes a few minutes. Every later launch is just:

```
./run.sh
```

This opens the welcome page with the **Kids** and **Grown-ups** buttons.

## If something goes wrong

**"FDS dataset not found"** — the dataset isn't where the app expects it.
Check that this file exists:

```
ls fds/sim/manifest.json
```

If it doesn't, redo step 2.

**"python3 not found"** — install Python 3 from <https://www.python.org/downloads/>
(or `brew install python` if you use Homebrew), then run `./run.sh` again.

**"permission denied" running `./run.sh`** — make it executable first:

```
chmod +x run.sh
./run.sh
```

**`fds/sim/sim/` by mistake** — you copied the `sim` folder one level too deep.
Move its contents up so `fds/sim/manifest.json` exists, or delete `fds/sim/`
and redo step 2.

## Dataset

- The ~11 GB Pleiades simulation dataset is provided separately and is
  intentionally not stored in Git.
- It contains 24 scenarios; the application expects it at `./fds/sim/`.
- Junior Fire Scientist does **not** depend on FireScope: it never looks for
  FireScope and contains no FireScope paths. The symlink in step 2 is only a
  convenience for colleagues who have both repositories and want to avoid a
  second copy — a plain copied `fds/sim/` works exactly the same.

## For developers

`./run.sh` opens the welcome page (passes `--welcome`). To boot straight into
the Public Fire Explorer kiosk instead:

```
./run.sh --public
```

The researcher UI is `src/main.py` with no flags — run that file directly if
you need it without the welcome page.

Tests (the environment must already be set up by `run.sh`):

```
~/.venvs/junior_fire_scientist/bin/pip install -e ".[dev]"
~/.venvs/junior_fire_scientist/bin/pytest
```

Project layout:

```
src/            application source
  public/       Public Fire Explorer
  pages/        researcher UI pages
  cinema/       rendering pipeline
  fds/          FDS .smv/.sf/.s3d parsers
tests/          pytest suite + a small tracked fixture dataset
fds/            simulation-generation pipeline (templates, SLURM script)
fds/sim/        the dataset (provided separately, not in git)
```

The loader reads `fds/sim/manifest.json` to map the 24 scenarios to their
candle/door/vent factor levels, locating each scenario by directory name inside
`fds/sim/`. See [`docs/architecture.md`](docs/architecture.md) and
[`docs/fds-data.md`](docs/fds-data.md) for more.

The "Grown-ups" button in the Public Fire Explorer optionally launches a
separate researcher application, FireScope, if it is installed alongside this
repository. It is not required for Junior Fire Scientist or its dataset.
