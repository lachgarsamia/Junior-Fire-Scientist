# Architecture overview

Junior Fire Scientist is a single PyQt5 application (package `fdsvis`) with
two distinct experiences sharing one data and rendering layer.

## Two experiences, one codebase

- **Public Fire Explorer** (`src/public/`) — the kids/museum-facing mode.
  Entered via the "Fire Explorer" button on the Home page or `python main.py
  --public` for unattended kiosk boot. It is a separate root widget (not a
  page in the researcher nav rail), so a child is never one click away from
  destructive researcher actions. `PublicExperience` (`public/experience.py`)
  is a state machine over 11 phases (attract, intro, observe, prediction,
  experiment, reveal, science, games, ...). The experience is deliberately
  split into three separate spaces: a calm **exploration playground**
  (real-time scene, thermometer, fan/candle toggles), an opt-in **games**
  area (Temp Hunt, Hot/Cold, Mystery, Compare, Map It), and **learning**
  content (science explanations, discovery notebook) that only surfaces
  after a discovery, never before it.
- **Researcher UI** — a nav rail (Home / Live Viewer / Compare / Dataset
  Explorer / Analysis / Export / About) built by `main_window.py` and
  `pages/`. This is the original scientific analysis environment the public
  mode is built on top of.

A related, separate repository ("FireScope") provides a full researcher
desktop app; the "Grown-ups" button on the Public Fire Explorer's welcome
screen launches it as an independent process if it's installed as a sibling
checkout (see `src/public/firescope_launcher.py`). It is optional — Junior
Fire Scientist runs standalone without it.

## Three layers

Every UI component depends only on Layer 2, never directly on another panel:

```
Layer 1 — Data          ScenarioStore · Descriptor/Signature engines · Derived Quantities
                                         │
Layer 2 — Selection     SelectionModel · SelectionContext · SelectionBus
                                         │
Layer 3 — Presentation  Panels · Plots · Dashboard · Public Fire Explorer
```

- **`ScenarioStore`** (`scenario_store.py`) is an LRU cache over a `.npy`
  disk cache keyed by `(scenario, SliceKey)`; a warm read is ~1–6 ms, which
  is what makes scenario switching in the Public Fire Explorer feel instant.
- **`QuantityProvider`** (`quantity_provider.py`) sits above the store and
  resolves both raw quantities (from FDS output) and derived quantities
  (computed client-side, e.g. temperature rise, dynamic pressure). Any
  quantity the current dataset doesn't support raises `GatedQuantityError`
  with a clear reason rather than fabricating a value — see
  [`fds-data.md`](fds-data.md) for what's gated and why.
- **`SelectionModel`/`SelectionBus`** (`selection.py`) is one immutable
  "what is being looked at" object (scenario, quantity, point, region,
  time, comparison, ...) broadcast to every subscribed panel, with an
  origin guard so a panel never reacts to its own update.

## Rendering

Two rendering technologies, both driven by real simulation data:

- **matplotlib embedded in Qt**, with explicit blitting (`widgets.py`'s
  `MplCanvas`) for the scientific heatmap views.
- **`cinema/`** — the cinematic effects pipeline (`cinema/pipeline.py`)
  turns a raw temperature array into a stylized frame via tonemap → flicker
  keyed to real heat-release rate → fire LUT → bloom → velocity-advected
  smoke → heat shimmer, plus sub-frame interpolation so 4 fps simulation
  data animates smoothly. This is what the Public Fire Explorer's scene
  always renders through; the researcher UI can toggle it on or off.

## Honesty constraints (load-bearing, not stylistic)

The application's design principle, enforced in code and tests, is that
every number and every claim on screen is computed from real FDS output —
nothing is invented for effect:

- `kid_language.py` / `experiments.py`'s `PUBLIC_METRICS` define explicit
  "noticeable delta" thresholds — a change too small to be real (e.g. a fan
  effect under 0.02 m/s) is never reported as a change.
- The Public Fire Explorer never claims a flow *direction* the coarse grid
  doesn't resolve (`TestFlowDirectionIsNeverClaimed`), and soot density is
  never presented as if it were a directly-measured temperature proxy
  (`TestSootVersusTemperatureProxy`).
- The Safe Assistant (`assistant.py`) is a bounded organizer of already-computed
  evidence; it never asserts a physical cause and refuses "why" questions
  outside its closed grammar.

Any change touching `story.py`, `kid_language.py`, or `experiments.py`'s
thresholds should re-run the corresponding test classes in
`tests/test_public_mode.py` and treat a failure as a correctness
regression, not a test to relax.
