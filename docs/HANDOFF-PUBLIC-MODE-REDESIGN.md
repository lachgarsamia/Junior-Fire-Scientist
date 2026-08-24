# Public Fire Explorer — UX Redesign Handoff

Status: **the feature set described below exists and works.** `src/public/` is
~7,080 lines of uncommitted, tested code (352 test functions in
`tests/test_public_mode.py`). This is not a proposal document like
`docs/ROADMAP-PUBLIC-MODE.md` (which predates this code and describes a
6-phase plan that was mostly followed, then substantially exceeded — see §3).
This document is the opposite kind of handoff: **the thing got built,
individually every piece works, and the accumulated result needs a UX
redesign, not more features.**

Read this if you are picking up the Public Fire Explorer with no prior
context. It assumes you can read Python/PyQt5 and will look at the actual
files — every claim below is cross-referenced to `src/public/*.py`.

---

## 1. Project overview

**What it is.** The Public Fire Explorer is a museum/kiosk-facing mode bolted
onto the same PyQt5 application as the internal FDS research visualizer
(`fds_visualizer_kids`, which is itself a "kids" fork of a researcher tool,
`fds_visualizer` — a sibling checkout on this machine, see §7 warning). It is
not a separate app: `MainWindow` owns a two-entry `QStackedWidget`
(`root_stack`) holding the researcher `shell` and, lazily constructed, a
`PublicExperience` full-screen widget (`main_window.py:1065-1489`). Entering
public mode hides the menu bar and status bar, forces fullscreen, and repoints
the idle/kiosk callback so an idle visitor resets the experience instead of
navigating "home." Two entry points exist: a "🔥 Fire Explorer (public)"
button on the researcher Home page (`pages/home.py:21-103`), and a `--public`
CLI flag for unattended kiosk boot (`main.py:29-92`). Exit is deliberately
awkward — `Ctrl+Shift+R` only, no on-screen equivalent except a small ✕ button
— so a child cannot accidentally land in the researcher UI.

**Target users.** Children and general museum/exhibition visitors with zero
fire-science background, walk-up-and-use, likely touchscreen, unattended for
long stretches (kiosk idle/attract loop), and no assumption of English
fluency (English/German are both fully supported, see §6).

**Difference from the researcher visualizer.** The researcher app is a
nav-rail of pages (Home / Live Viewer / Compare / Dataset Explorer / Analysis
/ Export / About), an `InspectorPanel`, a `ViewGrid` of combo-box-configurable
cells, a menu bar with "Open Study…" and session load, and a full analysis
suite. None of that is reachable from public mode — it is architecturally a
different root widget, not a page in the same nav rail, specifically so a
child is never one accidental click from "Export" or "Open Study…"
(`docs/ROADMAP-PUBLIC-MODE.md` §C.1 explains why this separation was chosen;
the actual code follows that recommendation).

**The role of real FDS data.** This is not a cartoon house fire. The
underlying simulation is a **tabletop candle experiment**: a 1.0 m × 0.48 m
domain containing a 0.73 m × 0.22 m enclosed room, with 1 or 2 real candles as
the fire source, a door (narrow/wide), and a vent state (open / closed /
HVAC-fan). There is a physical mockup of this box that sits next to the
kiosk at demos. 24 precomputed scenarios (candle count × door × vent state)
are cached to disk as real FDS output; nothing displayed is invented. This
rule — "all numbers computed, none generated" — is enforced throughout
`src/public/`: `kid_language.py` and `experiments.py`'s `PUBLIC_METRICS` have
explicit `noticeable_delta` honesty thresholds (e.g. a fan effect below
0.02 m/s or 0.3 °C is not reported as a change at all), and the Mystery game's
narration cites the actual measured ceiling/floor temperatures rather than a
canned line. Any redesign must preserve this rule — it is the thing that
makes the exhibit honest and is load-bearing throughout the test suite
(`TestSootVersusTemperatureProxy`, `TestFlowDirectionIsNeverClaimed`,
`TestScientificCoherence`, etc.).

---

## 2. Current implemented experience

Architecturally, "screens" are not separate widgets. `PublicExperience`
(`experience.py:100`) is a `QStackedLayout` in `StackAll` mode holding exactly
two widgets stacked on top of each other for the entire session: `scene`
(the matplotlib/cinema-pipeline fire rendering) and `overlay` (a transparent
Qt widget carrying all chrome). "Screens" are **phases** of one state
machine — `Phase` enum in `state.py:22-33` — and `_render_phase()`
(`experience.py:865`) tears down and rebuilds the overlay's visible
buttons/cards on every transition. There are 11 phases: `ATTRACT`, `INTRO`,
`OBSERVE`, `PREDICTION`, `COUNTDOWN`, `EXPERIMENT`, `REVEAL`, `SCIENCE`,
`COMPLETE`, `GAMES_HOME`, `GAME_PLAY`.

**Scene visualization** (`scene.py`, `PublicScene`) — a chrome-stripped
wrapper around the same `SliceView` the researcher Live Viewer uses
(`_strip_chrome` hides colorbar/ticks/spines), cinematic mode always on: real
FDS temperature field, tonemap/bloom/flicker keyed to real heat-release rate,
smoke advected by real velocity, `set_flow_mode("activity")` for
direction-free flow dots (the team deliberately avoids claiming a flow
*direction* the coarse grid doesn't actually resolve — see
`TestFlowDirectionIsNeverClaimed`).

**Temperature interaction** — `PublicScene.probe_at()` (`scene.py:649`) maps a
tapped pixel to a real (x, z) location and returns the actual measured cell
value (never interpolated). `PublicOverlay` supports both a single tap and
drag-to-scan (holding and dragging sweeps the probe continuously).

**Thermometer** — `Thermometer` widget (`widgets.py:243`), 200 px wide,
≥360 px tall, docked along the overlay's right edge (anchored to the room's
real right wall via `room_wall_anchor` when available). Shows a caption, a
3-tier mood emoji (🥶/🙂/🥵), a painted bulb-and-tube graphic with 5 ticks
(0/100/200/300/400 °C, scaled to a 470 °C display ceiling), a numeric readout,
a phrase label, and a small trend sparkline. The fill color/height animate
toward the target (280 ms), but **the printed number is never interpolated**
— it always reflects a real sampled value. Three modes: `mean` (live scene
average), `point` (a single tap, static), `pinned` (resampled over time with a
trail) — this exists to serve both free exploration and the Map It / Compare
games.

**Candle/fire representation** — hand-painted candle body + wick + 3-layer
flicker flame (`scene.py`'s `_draw_candles`/`_draw_flame`), 1 or 2 candles
per the scenario's real `candles` factor, positioned at the actual burner x
coordinate from `schematic._CANDLE_X`.

**Fan and candle scenario changes** — `ExploreToggle` controls (`widgets.py`)
for Fan (OFF/ON) and Candles (1/2), always visible during `OBSERVE`. Flipping
one resets the other to its default — the app deliberately only ever varies
**one factor at a time** in free play. Door and vent-open/closed (`voc`) are
excluded from free play entirely, per the roadmap's §0.1 measurement that
those effects are near-null for this dataset; only the fan (~7× airflow
change) and candle count are exposed.

**Games/challenges** — a 6-tile hub (`GAMES_HOME`, `experience.py:1720`):
🔥 Temp Hunt (find the hottest **or** coolest point, target randomized per
visit), 🧊 Hot/Cold (find a hot spot, then a cool spot — two simultaneous
markers), 🔎 Mystery ("the fan cools the ceiling but warms the floor" — a real,
counter-intuitive, measured result), 🧪 Test an Idea (routes directly into the
guided Predict → Fan Experiment → Science chain, returning to the hub after),
📊 Compare (what-changed bar chart / same-point cross-scenario comparison),
🗺️ Map It (build a numbered trail of up to 6 measured points).

**Discovery systems** — `_record_discovery()` deliberately has **no score, no
badge, no achievement threshold** (explicit in its own docstring — the
roadmap's Phase 5 "gamification with 3 badges" was rejected during
implementation in favor of this). A discovery is just a real finding the
child made, recorded in plain language.

**Notebook/history** — "📓 My Discoveries" card: a live "🔬 Right now" board
(I changed / I measured / I discovered) plus a capped (4-item) reverse-
chronological history of past findings. Not in the original roadmap at all.

**Comparisons** — "What changed?" (bar-chart baseline vs. the explored
scenario), "Compare this place" (same (x,z) point sampled across two
scenarios, verdict-first: "much hotter" / "about the same" / etc.), and a
baseline **ghost overlay** (dashed contour of the baseline's settled frame
drawn under the live one).

**Animations and feedback** — `CelebrationOverlay` (`celebration.py`): a
70-particle confetti/star burst, ~1.65 s, fired on any Games "got it" moment,
paired with a synthesized 3-note success chime (`sound.py` — the only sound
effect in the app). `Mascot` (`mascot.py`): a QPainter character with 8 mood
states (idle/pointing/surprised/thinking/curious/watching/excited/explaining)
that bobs continuously and narrates via a speech bubble. `StoryController`
(`story.py`) turns `events.detect_events()` output into narrated beats
(ignition, fastest heating, peak, a bespoke ceiling-smoke-gathering detector,
stabilization) that fire once per run, spoken through the mascot, with a
banner "WHOOSH" flash when a metric crosses its own honesty threshold.

---

## 3. The main problem: feature accumulation

Every feature listed in §2 is individually real, individually tested (the
352-function test suite proves each one works correctly in isolation — e.g.
`TestThermometerAnimation`, `TestHottestPointDetection`,
`TestDiscoveryNotebook`, `TestBaselineGhost` all pass), and individually
justified by a real design rationale visible in the code's own comments. The
problem is not that any one piece is broken. The problem is what happens when
they are all live on the same surface at once.

**Concretely, on the `OBSERVE` screen alone** (the "free exploration" phase),
simultaneously visible or one tap away: 2 `BigButton`s (Help, Play/Pause), a
"🎮 Games" nav pill, 2 `ExploreToggle` groups (4 sub-buttons total: Fan OFF/ON,
Candles 1/2), an always-on thermometer, 2 meter chips, a mascot + speech
bubble, a `StageStrip` (WATCH → GUESS → TEST → DISCOVER), an exit button, two
language-toggle buttons, tap-to-probe, drag-to-scan, an idle-vent-hint timer
that can interrupt with a nudge, and — because `StoryController` beats fire
during `OBSERVE` too — an ambient narration track that can flash the banner
independent of anything the child just did. The code's own comments show the
team was already fighting this at the pixel level, not just conceptually: e.g.
a rejected third `BigButton` on this screen is annotated "measured 729 px
wide... a real overlap" (`experience.py` near line 1666-1671). That is a
symptom of too many things competing for one canvas, not a layout bug to be
patched.

**Why combining them creates overload, structurally:** the same one screen
is simultaneously trying to be (a) a calm ambient viewer, (b) a guided
scientific-method funnel (predict → test → reveal → science, with its own
`StageStrip` progress indicator), (c) an entry point into 5 distinct games,
and (d) a passive story/narration experience — and the "detour cards" (Help,
Compare, Same-place, Notebook) all layer on top of that same overlay rather
than living in a separable space. A child cannot easily tell, at any given
moment, "am I exploring, being taught, or playing a game?" — because all
three are visually present at once and the phase machine allows narration and
game-entry points to appear *during* what is supposed to be quiet exploration.

**The resulting UX confusion**, as reported by the person who commissioned
this document and consistent with what the architecture above predicts:
too many actions visible at once, too many competing visual elements (mascot
+ thermometer + toggles + stage strip + banner all fighting for a child's
eye), too many interruptions (idle hints, story beats, banner flashes
arriving unprompted), and exploration/games/explanation genuinely mixed
rather than separated. This document treats that report as ground truth; the
code inventory above is offered as the structural explanation for *why* it
happens, so a redesign attacks the actual cause (too many concerns sharing
one phase and one overlay) rather than re-skinning symptoms.

---

## 4. Current UX philosophy that should replace it

The child should be able to hold one sentence in their head at any moment:
**"I am exploring a fire experiment."** Everything else should be something
they walk *into* deliberately, not something ambient on top of that sentence.

Split the experience into three genuinely separate spaces, not three modes
layered on one screen:

1. **Exploration playground.** The default, calm state. Contains: the real
   scene, the thermometer, tap/drag-to-probe, the Fan/Candle toggles, and a
   single low-key way to go deeper (into Games or into "why"). Does **not**
   contain: story-beat interruptions arriving unprompted, the stage strip,
   prediction machinery, or celebration confetti. If a story beat is worth
   surfacing here, it should be something the child can glance at and ignore,
   not something that competes with the toggles for attention.

2. **Games/challenges.** A deliberately entered space (through the Games
   tile/nav from the playground). Contains: the 5 existing mini-games
   (Temp Hunt, Hot/Cold, Mystery, Compare, Map It) plus the guided
   Predict → Fan Experiment → Reveal flow ("Test an Idea"), each with its own
   celebration/feedback loop. Does **not** need to expose free-form toggles
   or the notebook mid-game — a game should feel like a bounded activity with
   a start and a clear "you got it" end, which the celebration system already
   provides.

3. **Learning/explanations.** The `SCIENCE` card, the Discovery Notebook, and
   the "What am I seeing?" Help content belong here — reachable *after* a
   child has touched, played, or discovered something, never shown ahead of
   or instead of the direct experience. This matches what's already
   half-true in the code (`REVEAL` happens after `EXPERIMENT`, `SCIENCE` is
   opt-in from `REVEAL`) — the redesign's job is to make this the rule
   everywhere, including inside free exploration, not just inside the guided
   experiment flow.

---

## 5. Redesign goals

### Main exploration screen
Should be calm, immediately interactive (tap/drag-to-probe works the instant
the scene appears, no card blocking it), and always show the two controls
that actually matter here (Fan, Candles) plus the thermometer. Avoid: modal
cards appearing unprompted, forced predictions before a child has touched
anything, constant prompts/nudges, and over-explaining (no story-beat text
competing with the visual itself). The existing `OBSERVE` phase and
`PublicScene`/`ExploreToggle`/`Thermometer` components are the right
foundation — the fix here is subtraction (remove the stage strip, ambient
story beats, and games-entry chrome from this screen's default state), not
addition.

### Games area
Should be optional and feel like *entering* a challenge — a clear
"you're in a game now" framing distinct from ambient exploration, matching
what `GAMES_HOME`/`GAME_PLAY` already structurally provide. Keep: Temp Hunt
(find hottest place), Hot/Cold, the Fan prediction game ("Test an Idea"),
Mystery. Keep the celebration feedback loop (confetti + chime + mascot
reaction via `CelebrationOverlay`/`play_success_chime()`/`Mascot`) exactly as
built — this is a genuinely good, already-tested piece and should not be
redesigned, only relocated so it triggers inside a bounded game rather than
bleeding into ambient exploration.

### Scientific explanations
Should surface after discovery, not before — this is already the `REVEAL` →
`SCIENCE` ordering for the guided experiment; extend the same rule to free
exploration (a "why" affordance the child can pull, never something pushed at
them).

---

## 6. Specific issues still needing attention

### Thermometer
The widget (`widgets.py:243`) is already docked to the room's real right wall
via `room_wall_anchor` and is 200×360 px — larger than a typical HUD widget —
so the reported problem ("looks like a UI widget, not an instrument, too
small, not visually connected") is likely about *rendering/visual language*
(flat overlay chip aesthetic) rather than position or size in code. Check
whether `room_wall_anchor` is actually resolving to the room edge in the
current build (it degrades to "a fixed offset" when it can't find the wall,
per the Explore inventory) before redesigning the visuals — the connection
might already be computed correctly and only need a more physical/instrument-
like paint job (the code already draws a bulb-and-tube graphic; consider
scaling that up and giving it more visual weight relative to the scene).

### Scenario switching
The roadmap (§A.5) states all 24 scenarios are pre-cached to disk and a warm
`ScenarioStore.get()` is ~1-6 ms — so switching data is not the bottleneck.
Any perceived delay when toggling Fan/Candles is more likely coming from the
reload/reset path around the toggle (`PublicScene` reload + `TimeController`
restart + whatever countdown/re-render sequencing `_apply_choice`-equivalent
logic in `experience.py` runs) rather than the data layer. Profile the actual
toggle-to-first-frame latency before assuming a data fix is needed; the fix
is probably in orchestration/sequencing, not caching.

### Language support
**This already exists and appears complete.** `i18n.py` is a ~230-key
`TRANSLATIONS` dict covering essentially every static string in the app,
`tr(key, **kwargs)` with an EN fallback, and the switch is instant because
every phase fully re-declares its text from live `tr()` calls on each render
— no restart needed. `lang_en_button`/`lang_de_button` (40×40 pills,
top-right) are always present. `kid_language.py` handles German word order
via template composition rather than substitution specifically to avoid
grammatical-gender bugs. Before doing any new i18n work, verify this against
the current build (spot-check a few screens in both languages) rather than
assuming the toggle needs to be built — the Explore inventory found no gaps
in the EN/DE key parity for the areas inspected, but a full audit against
every current UI string is worth doing once the redesign's screens are
finalized (new screens will need their own keys).

### Visual polish
The candle already has a 3-layer flicker flame keyed to real HRR; the fan
already has an activity glow that pulses with real per-frame velocity
magnitude, but **only when the fan is ON** — consider whether the fan should
read as a physical, present object even when off (so children register it as
"the fan" before they ever toggle it, rather than it appearing only as an
effect). Interactions (tap, drag-to-scan) already report a real value
immediately; the polish gap is about making the *objects* (candle, fan) read
as tangible physical things at rest, not just as effects that appear when
active.

---

## 7. Architecture considerations

**Do not keep adding features on top of existing UI.** The current state is
the result of exactly that pattern — the roadmap's own 6-phase plan was
followed and then exceeded (games hub, notebook, ghost overlay, trail/Map It,
full EN/DE i18n were all additions with no roadmap precedent; see the
Explore inventory's item 13 for the full diff against the roadmap). The
redesign's job is to **reorganize and prune what exists**, not add a 12th
concern to the overlay.

**Do not create duplicate systems.** `PublicScene`, `Thermometer`,
`ExploreToggle`, `CelebrationOverlay`, `Mascot`, `StoryController`, and the
single declared `FAN_EXPERIMENT` (`experiments.py:162`) are all validated,
tested, reusable components — reorganize which phase/screen calls them,
don't reimplement them. The `Phase` enum (`state.py`) plus `_render_phase()`
(`experience.py:865`) already give a clean seam: a "playground vs. games vs.
learning" split can likely be expressed as a reorganization of which phases
belong to which zone and what each phase's `_render_*` is allowed to show,
rather than a rewrite of the rendering primitives themselves.

**Do not sacrifice scientific correctness for visual effects.** The honesty
infrastructure — `noticeable_delta` thresholds in `PUBLIC_METRICS`
(`experiments.py:415`), `TestFlowDirectionIsNeverClaimed`,
`TestSootVersusTemperatureProxy`, `TestScientificCoherence`, the deliberate
rejection of the roadmap's "smoke layer descends" beat in favor of a
measured ceiling-gathering detector (`story.py`'s `_detect_ceiling_beat`) —
is load-bearing and represents real domain correction work already done.
Any redesign that touches `story.py`, `kid_language.py`, or
`experiments.py`'s thresholds should re-run the relevant test classes and
treat a failure as a correctness regression, not a test to loosen.

**Practical constraint:** `tests/test_public_mode.py` has 352 test functions
across 65 classes and functions as a fairly complete behavioral spec of the
current implementation. Before restructuring `_render_phase()`/`Phase`, skim
the class list (in the Explore inventory item 12, or just
`grep -n "^class Test" tests/test_public_mode.py`) to know what a
reorganization must not silently break — many tests assert on specific phase
transitions and screen contents that a redesign will legitimately need to
change, so expect to update tests deliberately alongside the redesign rather
than treating every failure as a regression.

**Also worth fixing while in this code:** `tests/isolation.py` /
`tests/test_repo_isolation.py` exist because this machine has a sibling
checkout of the app and an editable install can let its stale bytecode or
manifest paths silently contaminate this repo's test run (roadmap §0.3).
Run `pytest tests/test_repo_isolation.py -k TestThisCheckoutIsClean` first if
anything behaves inexplicably — it's a real, previously-encountered failure
mode on this machine, not theoretical.

---

## 8. Final objective

The goal is not to make the application contain more things. Every
individual feature already built — the thermometer, the fan/candle toggles,
the five games, the discovery notebook, the comparisons, the mascot, the
celebration animations, the bilingual support — is validated, tested, and
worth keeping. The goal is to **transform the accumulation of those features
into a polished, museum-quality interactive experiment** where a child can
walk up, touch the room, watch real fire and smoke respond, play a game if
they want one, and discover the science behind it — without ever needing
instructions, and without ever facing more than one clear thing to do at a
time.
