"""English/German UI text for the public exhibit -- the single source of
truth for every static string a child reads on screen, plus the current-
language state itself.

Deliberately a plain module-level dict + a couple of functions, not a
class or a Qt object: nothing here needs a QApplication to exist (the
translation table itself is just data), and every widget that shows text
already re-reads it fresh on every render (see PublicExperience.
_render_phase(), which fully re-declares each phase's chrome from live
calls every time) -- so switching languages needs no new "refresh"
plumbing anywhere else, just a call to set_language() followed by the
same _render_phase() every phase change already goes through.

Deliberately NOT used for the *composed*, data-driven sentences in
kid_language.py (e.g. "the air moved much faster") -- English and German
word order/case agreement can't be produced by substituting the same
words into the same slot, so those functions carry their own parallel
per-language templates internally (see kid_language.py's own language
parameter) rather than going through tr() at all.
"""

from __future__ import annotations

from typing import Callable

LANGUAGES = ("en", "de")

_current = "en"
_listeners: list = []


def get_language() -> str:
    return _current


def set_language(lang: str) -> None:
    global _current
    if lang not in LANGUAGES or lang == _current:
        return
    _current = lang
    for callback in list(_listeners):
        callback(lang)


def on_language_changed(callback: Callable[[str], None]) -> None:
    """Register a callback invoked with the new language every time
    set_language() actually changes it. PublicExperience subscribes once,
    at construction, to trigger a full re-render."""
    _listeners.append(callback)


def tr(key: str, **kwargs) -> str:
    """Look up `key` in the current language, `.format(**kwargs)`-ing any
    placeholders. Falls back to English, then to the key itself (loudly
    visible in the UI, easier to spot and fix than a silent KeyError) if
    a translation is ever missing."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    text = entry.get(_current, entry.get("en", key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text


# ---------------------------------------------------------------------
# Every static UI string, keyed by a short stable id. Grouped by where
# it's used so a translator (or a future audit) can find each one
# without hunting through the call sites.
# ---------------------------------------------------------------------
TRANSLATIONS = {
    # -- overlay.py: always-on chrome --------------------------------
    "meter_temperature_caption": {"en": "AIR TEMPERATURE", "de": "LUFTTEMPERATUR"},
    "meter_airflow_caption": {"en": "AIR MOVEMENT", "de": "LUFTBEWEGUNG"},
    "meter_airflow_not_measured": {"en": "not measured here", "de": "hier nicht gemessen"},
    "thermometer_default_caption": {"en": "🌡️ TEMPERATURE", "de": "🌡️ TEMPERATUR"},
    "thermometer_mean_caption": {"en": "🌡️ Whole room average", "de": "🌡️ Ganzer Raum, Durchschnitt"},
    "thermometer_at_flame": {"en": "🔥 At the flame", "de": "🔥 An der Flamme"},
    "thermometer_hot_spot": {"en": "🔥 Hot spot", "de": "🔥 Heiße Stelle"},
    "thermometer_cool_spot": {"en": "🧊 Cool spot", "de": "🧊 Kühle Stelle"},
    "help_button": {"en": "What am I seeing?", "de": "Was sehe ich hier?"},
    "pause_button": {"en": "Pause", "de": "Pause"},
    "play_button": {"en": "Play", "de": "Abspielen"},
    "games_button": {"en": "Games", "de": "Spiele"},
    "back_to_exploring": {"en": "Back to exploring", "de": "Zurück zum Entdecken"},
    "back_to_games": {"en": "Back to games", "de": "Zurück zu den Spielen"},
    "my_discoveries": {"en": "My Discoveries", "de": "Meine Entdeckungen"},
    "why_button": {"en": "Why?", "de": "Warum?"},
    "keep_exploring": {"en": "Keep exploring", "de": "Weiter entdecken"},
    "exit_tooltip": {"en": "Leave the Fire Explorer (Ctrl+Shift+R)",
                     "de": "Feuer-Entdecker verlassen (Strg+Umschalt+R)"},

    # -- welcome.py: landing page -----------------------------------
    "welcome_heading": {"en": "Welcome!", "de": "Willkommen!"},
    "welcome_subtitle": {"en": "Who's exploring today?", "de": "Wer erkundet heute?"},
    "welcome_kids_button": {"en": "Kids", "de": "Kinder"},
    "welcome_grownups_button": {"en": "Grown-ups", "de": "Erwachsene"},
    "welcome_coming_soon": {"en": "Coming soon", "de": "Bald verfügbar"},

    # -- Games hub tiles ------------------------------------------------
    "tile_temp_hunt": {"en": "Temp Hunt", "de": "Temperatur-Suche"},
    "tile_hot_cold": {"en": "Hot / Cold", "de": "Heiß / Kalt"},
    "tile_mystery": {"en": "Mystery", "de": "Rätsel"},
    "tile_test_idea": {"en": "Test an Idea", "de": "Idee testen"},
    "tile_compare": {"en": "Compare", "de": "Vergleichen"},
    "tile_map_it": {"en": "Map It", "de": "Karte erstellen"},
    "games_hub_prompt": {"en": "🎮  PICK A CHALLENGE", "de": "🎮  WÄHLE EINE HERAUSFORDERUNG"},
    "games_hub_say": {"en": "🎮 Pick a challenge!", "de": "🎮 Wähle eine Herausforderung!"},

    # -- Game screens: prompts and action pills -------------------------
    "prompt_temp_hunt": {"en": "🔥  TEMP HUNT", "de": "🔥  TEMPERATUR-SUCHE"},
    "prompt_hot_cold": {"en": "🧊  HOT OR COLD?", "de": "🧊  HEISS ODER KALT?"},
    "prompt_mystery": {"en": "🔎  CAN YOU FIGURE IT OUT?", "de": "🔎  KANNST DU ES HERAUSFINDEN?"},
    "prompt_compare": {"en": "📊  COMPARE", "de": "📊  VERGLEICHEN"},
    "prompt_map_it": {"en": "🗺️  MAP IT", "de": "🗺️  KARTE ERSTELLEN"},
    "hotcold_find_hot": {"en": "Find somewhere HOT 🔥", "de": "Finde eine HEISSE Stelle 🔥"},
    "hotcold_find_cool": {"en": "Now find somewhere COOL 🧊", "de": "Finde jetzt eine KÜHLE Stelle 🧊"},
    "hotcold_try_again": {"en": "Tap anywhere to try again!", "de": "Tippe irgendwo, um es nochmal zu versuchen!"},
    "temp_hunt_find_hottest": {
        "en": "🔥 Where do you think it's hottest? Touch the firebox to find out!",
        "de": "🔥 Wo ist es wohl am heißesten? Berühre die Feuerstelle, um es herauszufinden!"},
    "temp_hunt_find_coolest": {"en": "🧊 Can you find the COOLEST place?",
                              "de": "🧊 Findest du die KÜHLSTE Stelle?"},
    "mystery_hint": {
        "en": "💨 Turn the fan ON, then touch near the ceiling and near the floor. "
              "Where do you think it gets cooler?",
        "de": "💨 Schalte den Ventilator EIN und berühre dann die Stelle nahe der Decke "
              "und nahe dem Boden. Wo wird es wohl kühler?"},
    "compare_say": {"en": "🔧 Change the fan or candles, then compare!",
                    "de": "🔧 Ändere den Ventilator oder die Kerzen und vergleiche dann!"},
    "map_it_say": {"en": "👆 Tap around the room to build your own temperature map!",
                   "de": "👆 Tippe im Raum herum, um deine eigene Temperaturkarte zu erstellen!"},
    "action_what_changed": {"en": "What changed?", "de": "Was hat sich geändert?"},
    "action_compare_this_place": {"en": "Compare this place", "de": "Diese Stelle vergleichen"},
    "action_show_before": {"en": "Show before", "de": "Vorher zeigen"},
    "action_hide_before": {"en": "Hide before", "de": "Vorher ausblenden"},
    "action_clear_map": {"en": "Clear map", "de": "Karte löschen"},

    # -- Intro / Explore -------------------------------------------------
    "intro_title": {"en": "🔬 A real experiment", "de": "🔬 Ein echtes Experiment"},
    "intro_body": {
        "en": "A candle burns inside a small box. A supercomputer worked out "
              "what the air, heat and smoke did inside it.",
        "de": "Eine Kerze brennt in einer kleinen Box. Ein Supercomputer hat berechnet, "
              "was die Luft, die Hitze und der Rauch darin gemacht haben."},
    "intro_source": {"en": "Fire Dynamics Simulator (FDS) · Forschungszentrum Jülich",
                     "de": "Fire Dynamics Simulator (FDS) · Forschungszentrum Jülich"},
    "intro_say": {"en": "Everything you'll see really happened.",
                  "de": "Alles, was du gleich siehst, ist wirklich passiert."},
    "watch_the_fire": {"en": "Watch the fire", "de": "Feuer beobachten"},
    "explore_say": {"en": "👆 Tap or drag the fire to feel the heat — try the fan and candles too!",
                    "de": "👆 Tippe oder ziehe über das Feuer, um die Hitze zu spüren — probier "
                          "auch den Ventilator und die Kerzen aus!"},

    # -- Help card -------------------------------------------------------
    "help_title": {"en": "🔍 What am I seeing?", "de": "🔍 Was sehe ich hier?"},
    "help_say": {"en": "Everything here is a real fire simulation.",
                 "de": "Alles hier ist eine echte Feuersimulation."},
    "help_got_it": {"en": "Got it", "de": "Verstanden"},
    "seeing_smoke": {"en": "☁️ Smoke is collecting under the ceiling.",
                     "de": "☁️ Rauch sammelt sich unter der Decke."},
    "seeing_airflow_now": {"en": "💨 The air is really moving right now.",
                           "de": "💨 Die Luft bewegt sich gerade richtig stark."},
    "seeing_airflow_dots": {"en": "💨 Bigger dots mean the air is moving faster.",
                            "de": "💨 Größere Punkte bedeuten, dass sich die Luft schneller bewegt."},
    "seeing_colors": {"en": "🎨 Brighter colours mean hotter air.",
                      "de": "🎨 Hellere Farben bedeuten heißere Luft."},
    "seeing_flame": {"en": "🔥 The bright part is the flame and the hot air above it.",
                     "de": "🔥 Der helle Teil ist die Flamme und die heiße Luft darüber."},

    # -- Attract screen ---------------------------------------------------
    "attract_prompt": {"en": "🔥  FIRE EXPLORER", "de": "🔥  FEUER-ENTDECKER"},
    "attract_say": {"en": "Hi! I'm Ember. Come and explore a real fire!",
                    "de": "Hallo! Ich bin Ember. Komm und entdecke ein echtes Feuer!"},
    "attract_button": {"en": "EXPLORE THE FIRE", "de": "FEUER ENTDECKEN"},
    "attract_unavailable": {
        "en": "This experiment needs the full simulation study. The researcher app "
              "is still available.",
        "de": "Dieses Experiment braucht die vollständige Simulationsstudie. Die "
              "Forscher-App ist weiterhin verfügbar."},
    "back_to_app": {"en": "Back to the app", "de": "Zurück zur App"},

    # -- language toggle --------------------------------------------------
    "language_toggle_tooltip_en": {"en": "Switch to English", "de": "Auf Englisch umschalten"},
    "language_toggle_tooltip_de": {"en": "Auf Deutsch umschalten", "de": "Auf Deutsch umschalten"},

    # -- Discovery Notebook -----------------------------------------------
    "notebook_title": {"en": "📓 My Discoveries", "de": "📓 Meine Entdeckungen"},
    "notebook_say": {"en": "👀 Look what you've found so far!",
                     "de": "👀 Schau, was du bisher entdeckt hast!"},
    "board_i_changed": {"en": "🔧 I changed: {value}", "de": "🔧 Ich habe geändert: {value}"},
    "board_i_measured": {"en": "📍 I measured: {value}", "de": "📍 Ich habe gemessen: {value}"},
    "board_i_watched": {"en": "👀 I watched: {value}", "de": "👀 Ich habe beobachtet: {value}"},
    "board_i_discovered": {"en": "💡 I discovered: {value}", "de": "💡 Ich habe entdeckt: {value}"},

    # -- Map It (temperature trail) -----------------------------------------
    "trail_full": {"en": "🧹 Your map is full — clear it to measure more!",
                   "de": "🧹 Deine Karte ist voll — lösche sie, um mehr zu messen!"},
    "trail_first_reading": {"en": "🌡️ {value:.0f}°C — {where}", "de": "🌡️ {value:.0f}°C — {where}"},
    "trail_cleared": {"en": "🧹 Map cleared — measure some new spots!",
                      "de": "🧹 Karte gelöscht — miss ein paar neue Stellen!"},

    # -- Compare game: ghost, "what changed", same-place ----------------
    "ghost_shown_say": {"en": "👻 The dashed lines show how it looked before!",
                        "de": "👻 Die gestrichelten Linien zeigen, wie es vorher aussah!"},
    "airflow_reacted_say": {"en": "Whoa — look at the air move!",
                            "de": "Wow — schau, wie sich die Luft bewegt!"},
    "look_closely": {"en": "Look closely…", "de": "Schau genau hin…"},
    "candle_hottest_try_elsewhere": {"en": "🔥 That's the hottest part — try somewhere else!",
                                     "de": "🔥 Das ist die heißeste Stelle — versuch es woanders!"},
    "candle_burning_with_nudge": {
        "en": "🔥 That's the candle burning — the flame is where the heat starts! "
              "What happens if we change the fire?",
        "de": "🔥 Das ist die brennende Kerze — an der Flamme entsteht die Hitze! "
              "Was passiert, wenn wir das Feuer verändern?"},
    "candle_burning": {"en": "🔥 That's the candle burning — the flame is where the heat starts!",
                       "de": "🔥 Das ist die brennende Kerze — an der Flamme entsteht die Hitze!"},

    # -- experience.py: Dr. Funke's fact bank (general fire science,
    # never a claim about this run's own measurements) ------------------
    "scientist_name": {"en": "Dr. Frieda Funke", "de": "Dr. Frieda Funke"},
    # Kept short on purpose: Dr. Funke's own thought bubble lives in a
    # small gutter strip above the thermometer (see PublicOverlay.
    # _position_thought_bubble), not a full-width caption -- a first,
    # more sentence-length pass of these overflowed that space in a
    # real screenshot.
    "fact_hot_air_rises": {
        "en": "Hot air rises above cooler air.",
        "de": "Heiße Luft steigt über kühlerer Luft auf."},
    "fact_smoke_ceiling_first": {
        "en": "Smoke spreads along the ceiling first.",
        "de": "Rauch breitet sich zuerst an der Decke aus."},
    "fact_fire_triangle": {
        "en": "Fire needs fuel, oxygen, and heat.",
        "de": "Feuer braucht Brennstoff, Sauerstoff, Hitze."},
    "fact_flame_temperature": {
        "en": "A candle flame can top 1,000°C.",
        "de": "Eine Kerzenflamme wird über 1.000°C heiß."},
    "fact_moving_air_oxygen": {
        "en": "Moving air feeds a flame more oxygen.",
        "de": "Bewegte Luft gibt der Flamme mehr Sauerstoff."},
    "fact_cool_air_sinks": {
        "en": "Cool air sinks and pushes smoke away.",
        "de": "Kühle Luft sinkt und drängt Rauch weg."},
    "fact_blue_flame_hottest": {
        "en": "A flame's blue part is its hottest.",
        "de": "Der blaue Teil einer Flamme ist am heißesten."},
    "fact_closed_door_slows_fire": {
        "en": "A closed door can slow a fire down.",
        "de": "Eine geschlossene Tür verlangsamt ein Feuer."},
    "fact_smoke_more_dangerous": {
        "en": "Smoke is often more dangerous than flames.",
        "de": "Rauch ist oft gefährlicher als die Flammen."},
    "fact_firefighters_study_smoke": {
        "en": "Firefighters study how smoke moves.",
        "de": "Feuerwehrleute untersuchen, wie Rauch sich bewegt."},
    "celebration_line_1": {"en": "Great job!", "de": "Gut gemacht!"},
    "celebration_line_2": {"en": "You're a Fire Scientist!", "de": "Du bist ein Feuerwissenschaftler!"},
    "celebration_line_3": {"en": "Excellent!", "de": "Ausgezeichnet!"},
    "celebration_line_4": {"en": "Nailed it!", "de": "Perfekt gemacht!"},
    "this_scenario_fallback": {"en": "This scenario", "de": "Dieses Szenario"},
    "announce_whoosh": {"en": "{icon}  WHOOSH — watch the {label}!",
                        "de": "{icon}  WHOOSH — schau auf {label}!"},

    # -- story.py: narrated story beats ----------------------------------
    "beat_ignition_text": {"en": "The candle is lit. The air just above the flame is heating up.",
                           "de": "Die Kerze ist angezündet. Die Luft direkt über der Flamme wird heißer."},
    "beat_ignition_reaction": {"en": "There it goes!", "de": "Da geht's los!"},
    "beat_fastest_heating_text": {"en": "This is the moment the air is heating up fastest.",
                                  "de": "Das ist der Moment, in dem sich die Luft am schnellsten erwärmt."},
    "beat_fastest_heating_reaction": {"en": "It's heating up fast now!",
                                      "de": "Es wird jetzt schnell heißer!"},
    "beat_peak_text": {"en": "This is the hottest moment of the whole experiment.",
                       "de": "Das ist der heißeste Moment des ganzen Experiments."},
    "beat_peak_reaction": {"en": "That's as hot as it gets!", "de": "Heißer wird es nicht mehr!"},
    "beat_stabilize_text": {"en": "Things have settled down. The fire has found its rhythm.",
                            "de": "Es hat sich beruhigt. Das Feuer hat seinen Rhythmus gefunden."},
    "beat_stabilize_reaction": {"en": "It's settling down now.", "de": "Es beruhigt sich jetzt."},
    "beat_ceiling_text": {"en": "Smoke is gathering under the ceiling.",
                          "de": "Der Rauch sammelt sich unter der Decke."},
    "banner_whoosh": {"en": "💨  WHOOSH!", "de": "💨  WHOOSH!"},
    "banner_quiet_again": {"en": "💨  ...quiet again.", "de": "💨  ...wieder ruhig."},
    "explore_changed_fan": {"en": "🔧 You changed the experiment! Look what happened.",
                            "de": "🔧 Du hast das Experiment verändert! Schau, was passiert ist."},
    "explore_changed_other": {"en": "🔧 You changed the experiment! Did you notice?",
                              "de": "🔧 Du hast das Experiment verändert! Hast du es bemerkt?"},
    "compare_this_place_say": {"en": "📍 Touch a place in the fire to compare it!",
                               "de": "📍 Berühre eine Stelle im Feuer, um sie zu vergleichen!"},
    "what_changed_title": {"en": "🔎 What changed?", "de": "🔎 Was hat sich geändert?"},
    "what_changed_nothing": {
        "en": "Nothing measured here changed enough to notice yet — try a bigger change!",
        "de": "Hier hat sich noch nichts genug verändert, um es zu bemerken — probier "
              "eine größere Änderung!"},
    "hero_temp_at_spot": {"en": "🌡️ Temperature at this spot", "de": "🌡️ Temperatur an dieser Stelle"},
    "why_moving_air": {
        "en": "💨 Moving air carries heat from one place to another — it doesn't simply "
              "make everything colder.",
        "de": "💨 Bewegte Luft trägt Wärme von einem Ort zum anderen — sie macht nicht "
              "einfach alles kälter."},
    "why_hot_air_rises": {"en": "🌡️ Hot air rises and spreads unevenly through the room.",
                          "de": "🌡️ Heiße Luft steigt auf und verteilt sich ungleichmäßig im Raum."},
    "target_hottest": {"en": "hottest", "de": "heißeste"},
    "target_coolest": {"en": "coolest", "de": "kühlste"},
    "found_spot_say": {"en": "🎉 You found the {target} place! {icon} {value:.0f}°C",
                       "de": "🎉 Du hast die {target} Stelle gefunden! {icon} {value:.0f}°C"},
    "found_spot_title": {"en": "{target} SPOT FOUND", "de": "{target} STELLE GEFUNDEN"},
    "found_spot_discovery": {
        "en": "You found the {target} place in the room — {value:.0f}°C.",
        "de": "Du hast die {target} Stelle im Raum gefunden — {value:.0f}°C."},
    "hotcold_diff_big": {"en": "BIG difference!", "de": "GROSSER Unterschied!"},
    "hotcold_diff_small": {"en": "Almost the same!", "de": "Fast gleich!"},
    "hotcold_say": {"en": "🔥 {hot:.0f}°C vs 🧊 {cool:.0f}°C — {diff}",
                    "de": "🔥 {hot:.0f}°C vs 🧊 {cool:.0f}°C — {diff}"},
    "hotcold_discovery_title": {"en": "HOT VS COOL", "de": "HEISS VS KALT"},
    "hotcold_discovery_text": {
        "en": "One spot was {hot:.0f}°C, another was {cool:.0f}°C — a big difference!",
        "de": "Eine Stelle hatte {hot:.0f}°C, eine andere {cool:.0f}°C — ein großer Unterschied!"},
    "found_flame_say": {
        "en": "🎉 You found the hottest place! It's the flame itself — 🌡️ {value:.0f}°C",
        "de": "🎉 Du hast die heißeste Stelle gefunden! Es ist die Flamme selbst — 🌡️ {value:.0f}°C"},
    "found_flame_title": {"en": "HOTTEST SPOT FOUND", "de": "HEISSESTE STELLE GEFUNDEN"},
    "found_flame_discovery": {"en": "You found the flame itself — {value:.0f}°C.",
                              "de": "Du hast die Flamme selbst gefunden — {value:.0f}°C."},
    "mystery_solved_title": {"en": "🤯 SAME FAN. DIFFERENT PLACE.",
                             "de": "🤯 GLEICHER VENTILATOR. ANDERER ORT."},
    "mystery_solved_discovery_title": {"en": "SAME FAN. DIFFERENT PLACE.",
                                       "de": "GLEICHER VENTILATOR. ANDERER ORT."},
    "mystery_solved_discovery": {
        "en": "The same fan cooled the ceiling but warmed the floor!",
        "de": "Derselbe Ventilator hat die Decke gekühlt, aber den Boden gewärmt!"},
    "mystery_progress_ceiling": {
        "en": "❄️ Interesting! The air up here got cooler. What about down here?",
        "de": "❄️ Interessant! Die Luft hier oben ist kühler geworden. Wie ist es weiter unten?"},
    "mystery_progress_floor": {
        "en": "🔥 Wait… it got warmer down here! What about up there?",
        "de": "🔥 Warte… hier unten ist es wärmer geworden! Wie ist es weiter oben?"},
    "verdict_much_hotter": {"en": "🔥 Much hotter here", "de": "🔥 Hier viel heißer"},
    "verdict_much_cooler": {"en": "❄️ Much cooler here", "de": "❄️ Hier viel kühler"},
    "verdict_almost_same": {"en": "≈ Almost the same here", "de": "≈ Fast gleich hier"},
    "change_line_delta": {"en": "🌡️ {delta:.1f}°C {direction}", "de": "🌡️ {delta:.1f}°C {direction}"},
    "change_line_barely": {"en": "🌡️ Barely any change", "de": "🌡️ Kaum eine Veränderung"},
    "direction_warmer": {"en": "warmer", "de": "wärmer"},
    "direction_cooler": {"en": "cooler", "de": "kühler"},
    "last_watched_temp": {"en": "🌡️ Temperature {where}: {delta:.1f}°C {direction}",
                          "de": "🌡️ Temperatur {where}: {delta:.1f}°C {direction}"},

    # -- Reveal / Countdown / Experiment / Prediction / Complete --------
    "prediction_say": {"en": "Make a guess — every scientist starts with one!",
                       "de": "Mach eine Vorhersage — jede Wissenschaftlerin fängt so an!"},
    "prediction_prompt": {"en": "🤔  WHAT DO YOU THINK WILL HAPPEN?",
                          "de": "🤔  WAS, DENKST DU, WIRD PASSIEREN?"},
    "countdown_say": {"en": "Ready? Let's find out!", "de": "Bereit? Lass es uns herausfinden!"},
    "countdown_testing": {"en": "🧪  Testing your idea: {icon} {label}",
                          "de": "🧪  Deine Idee wird getestet: {icon} {label}"},
    "countdown_lets_test": {"en": "🧪  Let's test it!", "de": "🧪  Lass es uns testen!"},
    "experiment_say": {"en": "Here we go! Watch closely.", "de": "Los geht's! Schau genau hin."},
    "reveal_both_tried_say": {"en": "Now you know what {contrast} does!",
                              "de": "Jetzt weißt du, was {contrast} bewirkt!"},
    "reveal_baseline_say": {"en": "That was {baseline}. Shall we try {contrast}?",
                            "de": "Das war {baseline}. Sollen wir {contrast} ausprobieren?"},
    "reveal_spotted_it": {"en": "Did you spot it? {clause}!", "de": "Hast du es bemerkt? {clause}!"},
    "reveal_spot_what_changed": {"en": "Did you spot what changed?",
                                 "de": "Hast du bemerkt, was sich geändert hat?"},
    "science_fallback_say": {"en": "These are the real numbers from the simulation.",
                             "de": "Das sind die echten Zahlen aus der Simulation."},
    "complete_say": {"en": "Thanks for exploring! Want another go?",
                     "de": "Danke fürs Entdecken! Noch eine Runde?"},
    "reveal_headline_default": {"en": "Here's what happened", "de": "Das ist passiert"},
    "reveal_no_comparisons_line": {"en": "The simulation finished.",
                                   "de": "Die Simulation ist fertig."},
    "reveal_measured_dim": {
        "en": "Measured: {label} {baseline} → {contrast} {unit} (average over the whole run).",
        "de": "Gemessen: {label} {baseline} → {contrast} {unit} (Durchschnitt über den "
              "gesamten Lauf)."},
    "reveal_both_tested_headline": {"en": "🎉 You tested both!", "de": "🎉 Du hast beides getestet!"},
    "reveal_seen_both": {"en": "You've now seen {baseline} and {contrast}.",
                         "de": "Du hast jetzt {baseline} und {contrast} gesehen."},
    "reveal_almost_nothing": {
        "en": "Almost nothing changed that we can measure.",
        "de": "Es hat sich fast nichts verändert, was wir messen können."},
    "reveal_with_contrast": {"en": "With {contrast}, {clause}.", "de": "Mit {contrast}: {clause}."},
    "reveal_try_again_line": {"en": "Try again to see that for yourself.",
                              "de": "Versuch es nochmal, um das selbst zu sehen."},
    "reveal_headline_no_guess": {"en": "👀 Here's what really happened",
                                 "de": "👀 Das ist wirklich passiert"},
    "reveal_headline_matched": {"en": "🎉 Your prediction matched!",
                                "de": "🎉 Deine Vorhersage stimmte!"},
    "reveal_headline_great_guess": {"en": "🔎 Great prediction — let's see what happened",
                                    "de": "🔎 Tolle Vorhersage — schauen wir, was passiert ist"},
    "replay_start_again": {"en": "Start again", "de": "Nochmal von vorn"},
    "replay_try_again": {"en": "Try again", "de": "Nochmal versuchen"},
    "replay_try_other": {"en": "Try {label}", "de": "{label} ausprobieren"},
    "science_button": {"en": "Show me the science", "de": "Zeig mir die Wissenschaft"},
    "science_back_button": {"en": "Back", "de": "Zurück"},
    "science_title_both": {"en": "🔬 You tested both", "de": "🔬 Du hast beide getestet"},
    "science_title_default": {"en": "🔬 What did we measure?", "de": "🔬 Was haben wir gemessen?"},

    # -- PublicScene.location_phrase (where a tapped point physically is) -
    "location_at_flame": {"en": "at the flame", "de": "an der Flamme"},
    "location_near_ceiling": {"en": "near the ceiling", "de": "nahe der Decke"},
    "location_near_floor": {"en": "near the floor", "de": "nahe dem Boden"},
    "location_middle_of_room": {"en": "in the middle of the room", "de": "in der Mitte des Raums"},

    # -- kid_language.py: temperature/airflow bands ----------------------
    "temp_band_room": {"en": "Room temperature", "de": "Zimmertemperatur"},
    "temp_band_little_warm": {"en": "A little warm", "de": "Ein bisschen warm"},
    "temp_band_warm_air": {"en": "Warm air", "de": "Warme Luft"},
    "temp_band_hot": {"en": "Hot", "de": "Heiß"},
    "temp_band_very_hot": {"en": "Very hot", "de": "Sehr heiß"},
    "temp_band_flame": {"en": "Flame", "de": "Flamme"},
    "airflow_band_still": {"en": "Almost still", "de": "Fast still"},
    "airflow_band_gentle": {"en": "Gentle drift", "de": "Leichte Brise"},
    "airflow_band_moving": {"en": "Moving air", "de": "Bewegte Luft"},
    "airflow_band_strong": {"en": "Strong airflow", "de": "Starker Luftstrom"},
    "airflow_band_very_strong": {"en": "Very strong airflow", "de": "Sehr starker Luftstrom"},

    # -- kid_language.py: "find the hottest place" game reactions --------
    "heat_guess_cool": {"en": "Cool!", "de": "Kühl!"},
    "heat_guess_warmer": {"en": "Getting warmer!", "de": "Wird wärmer!"},
    "heat_guess_hot": {"en": "Getting hot!", "de": "Wird heiß!"},
    "heat_guess_thats_hot": {"en": "That's hot!", "de": "Das ist heiß!"},
    "heat_guess_wow": {"en": "WOW!", "de": "WOW!"},
    "heat_guess_wow_flame": {"en": "WOW! That's the flame!", "de": "WOW! Das ist die Flamme!"},

    # -- kid_language.py: warmth-above-ambient phrasing ------------------
    "warmth_same_as_start": {"en": "the same as when we started",
                             "de": "genauso wie am Anfang"},
    "warmth_one_degree": {"en": "about 1 degree warmer than when we started",
                          "de": "etwa 1 Grad wärmer als am Anfang"},
    "warmth_degrees_warmer": {"en": "about {delta:.0f} degrees warmer than when we started",
                              "de": "etwa {delta:.0f} Grad wärmer als am Anfang"},

    # -- kid_language.py: finding/mascot sentence templates --------------
    "intensifier_much": {"en": "much ", "de": "viel "},
    "intensifier_a_little": {"en": "a little ", "de": "ein bisschen "},
    "mascot_finding_both_tried": {
        "en": "🎉  You tested both! The biggest difference: {sentence}.",
        "de": "🎉  Du hast beides getestet! Der größte Unterschied: {sentence}."},
    "mascot_finding_default": {"en": "{icon}  The biggest change: {sentence}!",
                              "de": "{icon}  Die größte Veränderung: {sentence}!"},

    # -- kid_language.py: compare_phrase (noun-first to dodge German
    # grammatical gender, which a single fixed article can't get right
    # for every noun this is called with) ---------------------------------
    "compare_phrase_changed": {"en": "the {noun} changed", "de": "{noun} hat sich verändert"},
    "compare_phrase_times_stronger": {
        "en": "the {noun} became about {ratio:.0f} times stronger",
        "de": "{noun} wurde etwa {ratio:.0f}-mal stärker"},
    "compare_phrase_stronger": {"en": "the {noun} got noticeably stronger",
                               "de": "{noun} wurde merklich stärker"},
    "compare_phrase_half": {"en": "the {noun} dropped to about half",
                           "de": "{noun} fiel auf etwa die Hälfte"},
    "compare_phrase_weaker": {"en": "the {noun} got noticeably weaker",
                             "de": "{noun} wurde merklich schwächer"},
    "compare_phrase_barely": {"en": "the {noun} barely changed",
                             "de": "{noun} hat sich kaum verändert"},

    # -- experiments.py: FAN_EXPERIMENT -----------------------------------
    "experiment_fan_title": {"en": "The Fan Experiment", "de": "Das Ventilator-Experiment"},
    "experiment_fan_observe_prompt": {
        "en": "Watch the candle. Where do the hot air and smoke go?",
        "de": "Beobachte die Kerze. Wohin geht die heiße Luft und der Rauch?"},
    "experiment_fan_question": {
        "en": "This room has a fan in the ceiling. What happens if we switch it on?",
        "de": "Dieser Raum hat einen Ventilator an der Decke. Was passiert, wenn wir ihn einschalten?"},
    "prediction_bigger_fire": {"en": "Bigger fire", "de": "Größeres Feuer"},
    "prediction_cooler_room": {"en": "Cooler room", "de": "Kühlerer Raum"},
    "prediction_cooler_confirmation": {
        "en": "You guessed it — the air did get cooler.",
        "de": "Du hast es erraten — die Luft wurde tatsächlich kühler."},
    "prediction_nothing_changes": {"en": "Nothing changes", "de": "Nichts ändert sich"},
    "choice_fan_on": {"en": "Switch the fan ON", "de": "Ventilator EINSCHALTEN"},
    "choice_fan_on_short": {"en": "Fan ON", "de": "Ventilator AN"},
    "choice_fan_off": {"en": "Leave the fan OFF", "de": "Ventilator AUS lassen"},
    "choice_fan_off_short": {"en": "Fan OFF", "de": "Ventilator AUS"},

    # -- experiments.py: EXPLORE_CONTROLS ---------------------------------
    "control_fan_label": {"en": "Fan", "de": "Ventilator"},
    "control_candles_label": {"en": "Candles", "de": "Kerzen"},
    "control_vent2_label": {"en": "Vent", "de": "Lüftung"},
    "option_off": {"en": "OFF", "de": "AUS"},
    "option_on": {"en": "ON", "de": "AN"},
    "option_one_candle": {"en": "1", "de": "1"},
    "option_two_candles": {"en": "2", "de": "2"},
    "option_vent_open": {"en": "OPEN", "de": "OFFEN"},
    "option_vent_closed": {"en": "SHUT", "de": "ZU"},

    # -- experiments.py: PUBLIC_METRICS -----------------------------------
    "metric_stronger": {"en": "stronger", "de": "stärker"},
    "metric_weaker": {"en": "weaker", "de": "schwächer"},
    "metric_warmer": {"en": "warmer", "de": "wärmer"},
    "metric_cooler": {"en": "cooler", "de": "kühler"},
    "metric_hotter": {"en": "hotter", "de": "heißer"},
    "metric_about_the_same": {"en": "about the same", "de": "etwa gleich"},

    "metric_airspeed_label": {"en": "Air speed", "de": "Luftgeschwindigkeit"},
    "metric_airspeed_meaning": {"en": "Air speed is how fast the air is moving.",
                               "de": "Luftgeschwindigkeit ist, wie schnell sich die Luft bewegt."},
    "metric_airspeed_kid_subject": {"en": "the air moved", "de": "die Luft bewegte sich"},
    "metric_airspeed_kid_more": {"en": "faster", "de": "schneller"},
    "metric_airspeed_kid_less": {"en": "slower", "de": "langsamer"},
    "metric_airspeed_explanation": {
        "en": "Moving air carries heat and smoke along with it.",
        "de": "Bewegte Luft trägt Wärme und Rauch mit sich."},

    "metric_room_temp_label": {"en": "Air temp.", "de": "Lufttemp."},
    "metric_room_temp_meaning": {
        "en": "The average air temperature across the whole scene.",
        "de": "Die durchschnittliche Lufttemperatur über die ganze Szene."},
    "metric_room_temp_kid_subject": {"en": "the air got", "de": "die Luft wurde"},
    "metric_room_temp_kid_more": {"en": "warmer", "de": "wärmer"},
    "metric_room_temp_kid_less": {"en": "cooler", "de": "kühler"},
    "metric_room_temp_explanation": {
        "en": "This is the air around the fire, not the flame itself.",
        "de": "Das ist die Luft rund um das Feuer, nicht die Flamme selbst."},

    "metric_flame_temp_label": {"en": "Flame temp.", "de": "Flammentemp."},
    "metric_flame_temp_meaning": {
        "en": "Flame temperature is the hottest point anywhere in the scene.",
        "de": "Die Flammentemperatur ist der heißeste Punkt in der ganzen Szene."},
    "metric_flame_temp_kid_subject": {"en": "the flame burned", "de": "die Flamme brannte"},
    "metric_flame_temp_kid_more": {"en": "hotter", "de": "heißer"},
    "metric_flame_temp_kid_less": {"en": "cooler", "de": "kühler"},
    "metric_flame_temp_explanation": {
        "en": "A candle flame burns at much the same temperature whatever the room "
              "around it does.",
        "de": "Eine Kerzenflamme brennt bei fast der gleichen Temperatur, egal was im "
              "Raum um sie herum passiert."},
}
