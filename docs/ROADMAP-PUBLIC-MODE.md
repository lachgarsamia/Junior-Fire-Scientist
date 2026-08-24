# Public Fire Explorer — Architecture Review & Implementation Roadmap

Status: **proposal, no code written yet.** Based on a full read of the repository
at commit `011c98c` plus direct measurement of the simulation dataset.

---

## 0. Three findings that change the plan

These came out of reading the code and measuring the actual data, and they
should be settled before any UI work starts.

### 0.1 The door is the *weakest* effect in the dataset. The fan is the strongest.

The vision document proposes "Should we open the door?" as the headline
interactive experiment. Measured across all 24 scenarios, matched on every other
factor, the door (narrow `d0` vs. wide `d1`) produces:

| Effect of wide vs. narrow door | magnitude |
| --- | --- |
| Mean air speed | ±0.03 m/s, **and it flips sign** depending on vent state |
| End-state mean temperature | +0.0 to +1.7 °C |

That is a null-to-marginal result. A child who predicts "the fire will get much
bigger" and then sees nothing happen learns the wrong lesson, and a museum
visitor sees a broken-looking demo.

Meanwhile the ventilation factor `vod=2` (the HVAC/fan state) is dramatic and
unambiguous:

| Vent state (`vod`) | mean air speed | 99th-pct air speed | end-state room temp |
| --- | --- | --- | --- |
| 0 — open | 0.095 m/s | ~0.5 m/s | 26.5 / 32.5 °C |
| 1 — closed | 0.084 m/s | ~0.5 m/s | 26.5 / 32.8 °C |
| 2 — **HVAC/fan** | **0.623 m/s** | **~3.0 m/s** | **25.0 / 30.0 °C** |

(Two numbers in the last column are 1-candle / 2-candle cases.)

Turning the fan on multiplies airflow by ~7× and measurably cools the room.
Candle count is the second-strongest effect (+6 °C end-state mean, 26 → 32 °C).

**Recommendation: the MVP headline experiment is the fan, not the door.**
The door can stay as a later "and some things barely matter at all" experiment,
which is itself a good science lesson — but it must not be the first thing a
visitor touches.

Note this also means the existing Compare page preset `"door"`
([main_window.py:127](../src/main_window.py#L127)) is built on the dataset's
weakest signal. Its docstring says M2.3 found the effect shows in velocity —
that is true only in the sense that it is *slightly* less invisible there.

### 0.2 This is a tabletop candle experiment, not a building fire

Parsed geometry ([schematic.py:190-196](../src/schematic.py#L190-L196)): the
domain is **1.0 m × 0.48 m**, and the enclosed room is **0.73 m × 0.22 m**, with
one or two **candles** as the fire source. There is a physical mockup that sits
next to this app at demos — [controls/door_widget.py:5](../src/controls/door_widget.py#L5)
refers to "the physical door next to this app at the demo," and `banner/` and
`image_video_sim/` contain exhibition assets.

This is a gift, not a limitation. "That box on the table — this is what the air
inside it is really doing" is a far stronger museum experience than a cartoon
house, and it is honest. But it rules out the burning-house framing, the
"which room is safer?" experiment (there is one room), and the evacuation
narrative from the vision document.

It also constrains the temperature legend. A generic five-band
🟦cool→🔥dangerous scale would be actively misleading here: the flame core hits
~450 °C but the room air stays between 20 °C and ~35 °C for the whole run. The
honest and more interesting framing is *"the flame is blazing hot, yet the air
across the room only gets a few degrees warmer — and the fan changes even that."*

### 0.3 This repository reads the *other* project's data (fix before starting)

`fds/sim/manifest.json` in this "kids" copy contains absolute paths into the
sibling checkout:

```
entry0 path:      /Users/samialachgar/Desktop/fds_visualizer/fds/sim/c1_d0_vod0_voc0
store folder0:    /Users/samialachgar/Desktop/fds_visualizer/fds/sim/c1_d0_vod0_voc0
```

Verified by running `load_simulation_data()` here. It works today only because
that directory happens to exist. Any edit, move, or cleanup of the other project
silently breaks this one — and at an exhibition it would fail with "demo data"
in the title bar and nobody knowing why.

Fix is one line, because [manifest.py:175-184](../src/manifest.py#L175-L184)
regenerates a missing manifest automatically:

```sh
rm "fds/sim/manifest.json"   # regenerated with correct paths on next launch
```

Worth doing now, and worth a startup assertion in public mode that every
manifest path is inside `SIM_ROOT`.

---

## A. Current architecture

### A.1 Startup and ownership

```
src/main.py
  ├─ QApplication(style="Fusion")            main.py:36
  ├─ splash + progress                        main.py:39-51
  ├─ load_simulation_data()                   data_provider.py:127
  │    └─ ScenarioStore(folders, cache_dir)   scenario_store.py
  │       → SimulationData(store, data_matrix, manifest, is_factorial, is_demo)
  ├─ warms the default scenario's cache       main.py:58-59
  └─ MainWindow(sim_data)                     main_window.py:231
```

`MainWindow.__init__` ([main_window.py:232-378](../src/main_window.py#L232-L378))
is the single owner of nearly all app state: `controller` (SimulationController),
`time_controller`, theme/scale/colormap settings, `prediction_store`, HRR and
event caches, and — at the end — the `KioskController`. It is 4192 lines and
imports ~60 sibling modules. **This is the main architectural risk for the public
mode: anything added directly to `MainWindow` inherits that coupling.**

### A.2 Navigation shell

`_build_shell()` ([main_window.py:759](../src/main_window.py#L759)) builds a
`NavRail` + `QStackedWidget` of `Page` objects:

```python
nav_entries = [("home","Home"), ("live","Live Viewer"), ("compare","Compare"),
               ("dataset","Dataset Explorer"), ("analysis","Analysis"),
               ("export","Export"), ("about","About")]        # main_window.py:1038
```

`Page` ([pages/base.py:17](../src/pages/base.py#L17)) is a plain `QWidget` with
`on_enter()` / `on_leave()` lifecycle hooks, called by `_navigate_to()`
([main_window.py:1390](../src/main_window.py#L1390)). Adding a page is genuinely
a two-line change — one `nav_entries` tuple, one `self.pages` dict entry.

The `LivePage` is built eagerly; placeholder-style pages build lazily on first
`on_enter()`. `on_leave()` already pauses playback
([pages/live.py:42](../src/pages/live.py#L42)).

### A.3 Rendering — matplotlib, with a real effects pipeline

There is **no OpenGL, no QGraphicsView, no game engine.** Two rendering
technologies coexist, and both are usable for the public mode:

**1. matplotlib embedded in Qt** (the scientific field). `MplCanvas`
([widgets.py:79](../src/widgets.py#L79)) is a `FigureCanvasQTAgg` subclass with
explicit **blitting** (`capture_background()` / `blit_update(artists)`).
`SliceView` ([views.py:67](../src/views.py#L67)) owns one `imshow` heatmap plus a
stack of z-ordered overlay artists that are all blit-tracked together
([views.py:295-303](../src/views.py#L295-L303)):

| artist | zorder | purpose |
| --- | --- | --- |
| `heatmap` | 0 | the field itself |
| `ember_scatter` | 5 | ember/flame particles |
| `room_walls` / `room_door` / `room_vents` | 6 | room geometry, vents colored by state |
| `device_scatter` | 7 | virtual sensors |
| `streamline_collection` | 8 | true 3D streamlines |
| `hover_highlight` | 10 | linked-hover ring |

**2. Direct `QPainter` custom widgets** (the chrome). `SchematicWidget`
([schematic.py:238](../src/schematic.py#L238)) hand-paints the room, door, vents,
and layered candle flames (`draw_realistic_flame()`, glow + 3 body layers).
`DoorWidget` ([controls/door_widget.py:34](../src/controls/door_widget.py#L34))
animates a door arc with a 16 ms `QTimer` and ease-out-cubic easing.
`TourOverlay`, `EventMarkerBar`, and `_Sparkline` follow the same pattern.

**The `cinema/` package is the headline asset.** `EffectsPipeline.render()`
([cinema/pipeline.py:94](../src/cinema/pipeline.py#L94)) turns a raw temperature
array into an RGBA image via: auto-exposure → filmic tonemap → 1/f flicker keyed
to real HRR → fire LUT → bloom → smoke simulation (velocity-advected) → ambient
backdrop → heat shimmer. Plus `EmberParticles`, `velocity_arrows`, and
**sub-frame interpolation at 30 Hz** (`_INTERP_HZ`, [views.py:29](../src/views.py#L29))
decoupled from the 4 fps data rate.

This means **the "cartoon fire layer" the vision asks for already exists, is
already smooth, and is already driven by real physics.** It is gated behind
View → Cinematic mode. For the public experience it should simply be *on*.

### A.4 Playback

`TimeController` ([time_controller.py:26](../src/time_controller.py#L26)) is a
clean pull-based clock: `play/pause/seek/step/set_speed/set_loop/restart`,
emitting `time_changed(index)`. No scenario or store knowledge. Reusable as-is.

Practical number: scenarios are **481 frames at 4 fps = 120 s** of playback at
1×. Too long for a 90-second visitor. `set_speed(4)` gives a 30 s run,
`set_speed(8)` gives 15 s.

### A.5 Data pipeline

`ScenarioStore` — LRU in memory (`SCENARIO_CACHE_SIZE = 6`) over a `.npy` disk
cache keyed by `(scenario, SliceKey)`. A warm `get()` is ~1-6 ms, which is
exactly why the pull-based clock is safe. **All 24 TEMPERATURE and VELOCITY
fields are already cached on disk**, so scenario switching in public mode is
effectively instant. `QuantityProvider` sits above it with gating for quantities
that need the M-SIM re-run.

### A.6 Science-to-language layer (already built, heavily reusable)

| module | what it gives the public mode |
| --- | --- |
| [events.py:34](../src/events.py#L34) `detect_events()` | ignition, hazard crossings, fastest heating, peak, **smoke-layer descent**, stabilization — as time-stamped `Insight` objects. **This is a story spine, already computed.** |
| [auto_summary.py:147](../src/auto_summary.py#L147) `narrate_frame()` | one deterministic plain-language sentence per moment, already near kid-readable ("The smoke layer is forming under the ceiling.") |
| `summary_stats.build_summary_index()` | peak temp, time-to-300 °C, growth α, smoke-layer minimum |
| `layer_height.py`, `tenability.py`, `descriptors.py` | smoke layer height, hazard timing |

Everything here is template-filled from computed numbers — the codebase's own
stated rule is *"all numbers computed, none generated."* The public mode must
keep that rule.

### A.7 Exhibition infrastructure that already exists

- `KioskController` ([kiosk.py:31](../src/kiosk.py#L31)) — idle → Home after 3
  min, wake → Live, cursor auto-hide after 10 s. App-level event filter with a
  documented `shutdown()` contract.
- `TourOverlay` ([tour.py:28](../src/tour.py#L28)) — first-run guided tour.
- F11 fullscreen, demo bookmarks (`Ctrl+Shift+1-9`), an Esc-long-press
  "effects off" master switch, `docs/demo-script.md`.

The exhibition thinking is already in this codebase. Public mode extends it
rather than inventing it.

---

## B. Reusable infrastructure

Direct reuse, no modification needed:

| Need | Existing component |
| --- | --- |
| Animated fire + smoke + embers + shimmer | `cinema/` package via `SliceView.set_cinematic_mode()` |
| Smooth 30 Hz animation over 4 fps data | `SliceView` interpolation timer |
| Airflow arrows | `cinema/velocity_arrows.py`, `SliceView._update_velocity_arrows` |
| Room / door / vent geometry, vent state colors | `schematic.room_overlay_geometry()`, `SliceView.set_room_outline()` |
| Hand-drawn flames, door/vent icons | `schematic.draw_realistic_flame()`, `flame_icon()`, `door_icon()`, `vent_icon()` |
| Animated door swing | `controls/door_widget.py` |
| Playback clock, speed, loop | `TimeController` |
| Instant scenario switching | `ScenarioStore` (all fields pre-cached) |
| Story beats with real timestamps | `events.detect_events()` |
| Plain-language narration | `auto_summary.narrate_frame()` |
| Idle/attract, cursor hide | `KioskController` |
| Overlay + step-through UI pattern | `TourOverlay` |
| Theme tokens, focus rings, accessibility | `theme.py` (`Palette`, `build_qss`, `role=` properties) |
| Scale-up for readability | `MainWindow._set_ui_scale()`, `SliceView.set_ui_scale()` |

Reuse with care:

- `SliceView` — sound base for the public canvas, but its colorbar, ticks, and
  probe UI must be hidden. Add a `chrome=False`-style option rather than forking.
- `_apply_compare_preset()` — the right *mechanism* for A/B comparison, but it
  drives the research grid and sets `_compare_active`. Public comparison should
  use its own two `SliceView`s, not the research `ViewGrid`.

Do **not** reuse: `InspectorPanel`, `ViewGrid`/`GridCell` (combo boxes and
context menus are exactly the "dangerous researcher settings" to keep away from
a child), the analysis panels, `NavRail`.

---

## C. Required new architecture

### C.1 The mode boundary — recommendation

**Recommendation: Public mode is a separate root widget, not a nav-rail page.**

`MainWindow.setCentralWidget(shell)` becomes
`setCentralWidget(QStackedWidget[shell, public_root])`. Entering public mode
hides the menu bar, nav rail, and status bar; leaving restores them.

Rationale, from the actual repository:

1. A nav-rail page cannot hide the rail, and the rail is *always* one click from
   Analysis, Export, and "Open Study…" — the vision explicitly requires "no
   accidental access to dangerous researcher settings."
2. The menu bar carries destructive actions (`Open Study…`, session load).
   Only a root-level swap can hide it.
3. `KioskController` already assumes `on_idle → "home"`; in public mode it must
   instead reset the experience. A root-level mode makes that substitution clean.
4. Research code is untouched: `_build_shell()` still builds exactly what it
   builds today.

Exit must be deliberate and undiscoverable-by-accident: a 2-second long-press on
a corner target, or `Ctrl+Shift+R`. The existing Esc-long-press timer
([main_window.py:371](../src/main_window.py#L371)) is the precedent to copy.

Entry points: a "Fire Explorer" button on `HomePage`, and `python main.py
--public` for exhibitions (an unattended kiosk should never boot into the
research UI).

### C.2 New components

All under a new `src/public/` package, matching the existing `cinema/`,
`controls/`, `analytics/`, `pages/` convention. Add `"public"` to
`pyproject.toml`'s `packages` list.

| Component | File | Responsibility | Talks to | Reusable by research UI? |
| --- | --- | --- | --- | --- |
| `PublicExperience` | `public/experience.py` | Root widget; owns the public `QStackedWidget` of scenes; owns mode enter/exit; the *only* class that touches `MainWindow` | `MainWindow`, `TimeController`, `ScenarioStore` | No |
| `PublicScene` | `public/scene.py` | The big canvas: wraps one `SliceView` in forced-cinematic, chrome-free mode, plus a Qt overlay layer for cartoon elements | `SliceView`, `EffectsPipeline` | No (but pushes a `chrome` flag *into* `SliceView`, which research keeps) |
| `OverlayLayer` | `public/overlay.py` | Transparent `QWidget` sized to the scene; hosts mascot, speech bubble, arrows, badges; `paintEvent` + `QTimer` like `SchematicWidget` | `PublicScene` | No |
| `Mascot` + `SpeechBubble` | `public/mascot.py` | `QPainter`-drawn character with a small state machine (idle / pointing / surprised) and a typewriter speech bubble | `OverlayLayer` | No |
| `HeatMeter` | `public/meters.py` | Big thermometer + air-speed gauge, calibrated to *this* dataset's real ranges (§0.2) | `QuantityProvider` values | No |
| `ExperimentController` | `public/experiments.py` | Defines an experiment as data: prompt, choices, the two real `case_index` values, the reveal text; runs predict → play → reveal | `ScenarioStore`, `TimeController`, `manifest` | No |
| `StoryController` | `public/story.py` | Maps `events.detect_events()` output to narrated story beats at real frame indices | `events.py`, `auto_summary.py` | **Yes** — a research-side "Fire Story" narration could use it |
| `kid_language.py` | `public/kid_language.py` | Pure functions: value → kid phrase, honest to this dataset. No Qt | `registry.py` thresholds | **Yes** — pure, testable |
| `PublicKiosk` | `public/kiosk_public.py` | Thin adapter making `KioskController` reset the experience rather than navigate home | `KioskController` | Reuses, doesn't replace |

### C.3 Data flow

```
ScenarioStore ──get(case_index, SliceKey)──> np.ndarray (cached, ~1-6ms)
      │
TimeController ──time_changed(i)──> PublicExperience._on_tick(i)
      │                                    │
      │                                    ├─> PublicScene.show_frame(frame[i], velocity[i], frame[i+1], hrr)
      │                                    │        └─> SliceView (cinema pipeline, 30 Hz interpolation)
      │                                    ├─> HeatMeter.set_values(...)
      │                                    ├─> StoryController.beat_at(i) ─> Mascot.say(...)
      │                                    └─> ExperimentController.check_reveal(i)
      │
ExperimentController.apply_choice() ─> new case_index ─> PublicScene reload + TimeController.restart()
```

One rule, stated as a design constraint: **the public layer only ever reads.
It never computes physics.** Every number displayed comes from the store or from
`events.py` / `summary_stats.py`.

---

## D. UX roadmap

### Phase 1 — Public mode foundation
Mode switch, `PublicExperience` root, chrome-free full-bleed scene playing a
real scenario in cinematic mode at 4× speed, one huge Start/Replay button,
locked exit. **Deliverable: a visitor can walk up and watch a real FDS fire.**

### Phase 2 — Visual and language layer
Mascot + speech bubbles, thermometer and air-speed gauge calibrated to real
ranges, animated airflow arrows, kid-language layer, room/door/vent overlay
always on with vents color-coded by state.

### Phase 3 — Interactive experiments
`ExperimentController` with the prediction → choice → play → reveal loop.
Ship the **fan experiment** first (§0.1). Then candle count. Then the honest
"the door barely matters" experiment as a deliberate lesson.

### Phase 4 — Story mode
`StoryController` driving narrated beats off real `detect_events()` timestamps:
candle lit → air warms → plume rises → smoke gathers at the ceiling → smoke
layer descends → conditions settle. Every beat carries its real time in seconds.

### Phase 5 — Gamification
Lightweight and prediction-based: a "Fire Detective" badge for making any
prediction, "Airflow Expert" for the fan experiment, "Fire Scientist" for
completing all three. Score on *predictions made*, never on speed. Reset per
visitor.

### Phase 6 — Exhibition polish
Touch targets ≥ 64 px, readable at 2 m, `PublicKiosk` auto-reset after idle,
graceful degradation when a quantity is gated or data is missing, optional
sound, a performance pass, and an unattended-run soak test.

---

## E. Technical implementation plan

### Phase 1

**Create:** `src/public/__init__.py`, `public/experience.py`, `public/scene.py`,
`tests/test_public_mode.py`.

**Modify:**
- `main_window.py` — `_build_shell()` wraps the shell in a root `QStackedWidget`;
  add `enter_public_mode()` / `exit_public_mode()`; register the exit shortcut.
- `pages/home.py` — add the "Fire Explorer" entry button.
- `views.py` — add `SliceView.set_chrome_visible(bool)` to hide colorbar/ticks
  (small, additive, defaults to today's behavior so research is unchanged).
- `main.py` — `--public` flag.
- `pyproject.toml` — add `"public"` to `packages`.

**Risks.** `MainWindow` is 4192 lines and its `closeEvent` already unwinds
process-global Qt state (kiosk filter, busy-cursor stack); public mode must add
its own teardown to the same place or it will reproduce the documented
event-filter leak that once took the test suite from 27 s to 470 s
([kiosk.py:67-78](../src/kiosk.py#L67-L78)).

**Testing.** Follow `tests/test_pages.py`: `qapp` fixture, offscreen platform,
construct `MainWindow`, assert mode enter/exit, assert the research shell is
untouched afterward, assert no menu bar in public mode.

### Phase 2

**Create:** `public/overlay.py`, `public/mascot.py`, `public/meters.py`,
`public/kid_language.py`, `tests/test_kid_language.py`.

**Data flow note.** The overlay is a sibling `QWidget` stacked over the canvas,
repositioned in `resizeEvent` — *not* a matplotlib artist. Keeping cartoon
elements off the matplotlib figure protects the blitting fast path.

**Risk.** Overlay repaints fighting the 30 Hz interpolation timer. Mitigation:
overlay animates on its own ~20 Hz timer and only calls `update()` on its own
dirty rect.

**Testing.** `kid_language.py` is pure and gets real unit tests, including a
guard that ambient 20 °C never reads as "hot" and that 450 °C never reads as
"warm".

### Phase 3

**Create:** `public/experiments.py`, `tests/test_experiments.py`.

Experiments are declarative:

```python
FAN_EXPERIMENT = Experiment(
    key="fan",
    question="The fan can pull air through the room. Turn it on?",
    choices=[Choice("Fan ON", factors=dict(vod=2)),
             Choice("Fan OFF", factors=dict(vod=0))],
    held=dict(candles=0, door=1, voc=0),
    reveal_quantity="VELOCITY",
)
```

Case indices are resolved from `sim_data.manifest` at runtime (never hardcoded),
using the same matching logic as `_find_scenario()`
([main_window.py:2783](../src/main_window.py#L2783)).

**Risk — the important one.** An experiment whose reveal text overstates a weak
effect. Mitigation: a test that asserts each shipped experiment's measured effect
size exceeds a threshold, computed from the real store. This makes §0.1 a
permanent guardrail rather than a one-time observation.

### Phases 4-6

`StoryController` consumes `detect_events()` and needs no new physics. Badges
persist to `QSettings` under a `public/` key namespace, cleared on reset.
Phase 6 needs a real soak test — run the kiosk loop for an hour and watch RSS,
since `ScenarioStore`'s LRU is only 6 entries and public mode switches scenarios
frequently.

---

## F. MVP — "The Fan Experiment"

The smallest build that is genuinely impressive in 90 seconds:

```
Home → [ 🔥 FIRE EXPLORER ]
   ↓
Big room view, candle lit, real cinematic fire + smoke (4× speed, ~30s run)
   ↓
Mascot: "Watch the smoke. Where does it go?"   [smoke rises, gathers at ceiling]
   ↓
PAUSE.  "This room has a fan. What happens if we turn it on?"
        [ FAN ON ]   [ NOTHING CHANGES ]
   ↓
Runs the REAL scenario vod=2 — airflow arrows storm across the room,
air-speed gauge jumps 0.09 → 0.62 m/s, thermometer settles ~1.5 °C cooler
   ↓
Mascot: "The fan pushed the hot air out. The room stayed cooler."
   ↓
[ TRY AGAIN ]  [ WATCH BOTH SIDE BY SIDE ]
```

Scope: Phase 1 + the mascot/speech-bubble and gauge from Phase 2 + exactly one
experiment from Phase 3. Scenarios `c1_d1_vod0_voc0` and `c1_d1_vod2_voc0`,
both already disk-cached, resolved via manifest lookup.

Why this works: the payoff is a **7× measured change** that is impossible to
miss on screen, the child made a real prediction, and every pixel came from FDS.

---

## G. Future vision

- **Physical-mockup tie-in.** The box is on the table. "Open the real door, then
  watch what the simulation says happens inside" is an experience no purely
  digital exhibit can match.
- **Smoke-layer descent as the emotional core.** `layer_height.py` already
  computes it. Watching a ceiling smoke layer creep downward is the single most
  useful fire-safety intuition a child can leave with.
- **Progressive depth on one screen.** A "Show me the science" button that fades
  the cartoon layer down and the real colorbar and HRR curve up — the same data,
  the same instant. This is the honest version of the vision's four levels.
- **When the M-SIM re-run lands**, VISIBILITY and CO become available
  ([registry.py:105-121](../src/registry.py#L105-L121)) and unlock a genuine
  "can you still see the way out?" experience.
- **Multi-visitor prediction wall** — aggregate anonymous predictions and show
  "62% of visitors thought the fire would grow. Here's what really happened."

---

## Open questions for you

1. **Confirm the fan-first pivot** (§0.1). This is the biggest change from the
   vision document.
2. **Framing:** lean into "this is a real experiment box you can touch," or
   abstract toward a generic room?
3. **Target hardware** — touchscreen or mouse? It changes hit targets and
   whether hover is usable at all.
4. **Language** — English only, or German too, given FZ-Jülich and
   `banner_tag_der_neugier.pdf`?
