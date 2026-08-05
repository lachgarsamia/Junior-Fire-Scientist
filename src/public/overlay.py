"""The public overlay: everything drawn *over* the fire.

A transparent QWidget that sits above PublicScene in a stacked layout,
holding Qt widgets only -- no matplotlib artists. That separation is
deliberate: SliceView's blitting path repaints the fire at 30 Hz, and
adding cartoon elements as matplotlib artists would drag them through
that same path. Here they are ordinary widgets that Qt composites, so
the science rendering stays exactly as fast as it is in the Live Viewer.

The overlay owns no state. PublicExperience drives it phase by phase.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from public import i18n
from public import kid_language as kid
from public.celebration import CelebrationOverlay
from public.mascot import Mascot, SpeechBubble, IDLE, POINTING, SURPRISED, THINKING
from public.widgets import (ACCENT, AIRFLOW, DELIGHT, INERT, PANEL_BORDER, TEXT, TEXT_DIM,
                            BigButton, Card, ExploreToggle, MeterChip, StageStrip, Thermometer,
                            TitleBanner)


class PublicOverlay(QtWidgets.QWidget):
    """Signals carry the visitor's intent up to PublicExperience, which
    owns the state machine -- the overlay never advances a phase itself."""

    start_requested = QtCore.pyqtSignal()
    prediction_made = QtCore.pyqtSignal(str)     # prediction key
    choice_made = QtCore.pyqtSignal(str)         # choice key
    science_toggled = QtCore.pyqtSignal()
    help_requested = QtCore.pyqtSignal()
    replay_requested = QtCore.pyqtSignal()
    exit_requested = QtCore.pyqtSignal()
    # A tap (or, while a button stays held, a drag) on the fire itself
    # (not on any button/card) -- position in this widget's own pixel
    # space, which PublicScene.probe_at() can use directly since the
    # scene fills the exact same rectangle.
    tapped = QtCore.pyqtSignal(QtCore.QPoint)
    # (control_key, value) from an explore toggle -- see
    # experiments.ExploreControl.
    explore_changed = QtCore.pyqtSignal(str, object)
    # "en" or "de" -- the language toggle is one of the only two always-
    # visible controls (with exit_button) not routed through button_row/
    # nav_row's per-phase redeclaration, so it stays a real signal rather
    # than a direct callback wired per-render.
    language_requested = QtCore.pyqtSignal(str)
    # Everything else (Help, Games, each Games-hub tile, each game's own
    # action buttons, Notebook) is wired directly from whichever
    # PublicExperience._render_* method adds that button via add_button()
    # -- those buttons don't exist for the life of the app the way these
    # do, so a persistent signal here would just be indirection.

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self._buttons: list = []
        self._probe_enabled = False
        self._dragging = False
        self._explore_toggles: dict = {}
        self._fan_marker_frac = None
        self._thermometer_anchor_frac = None

        root = QtWidgets.QVBoxLayout(self)
        # Bottom margin reserves the band the mascot and its speech
        # bubble occupy -- they are positioned absolutely (see
        # _position_mascot) rather than laid out, so that a tall card is
        # never squeezed to nothing to make room for decoration. The
        # right margin reserves the always-visible, absolutely-positioned
        # exit/language-toggle cluster in resizeEvent() (exit_button +
        # lang_de_button + lang_en_button, ~154 px wide from the window
        # edge) -- without it the airflow meter chip's laid-out width
        # extends underneath those floating buttons and gets visually
        # clipped by them, a real overlap at 800x600.
        root.setContentsMargins(40, 28, 170, 160)
        root.setSpacing(16)

        # --- top row: prompt banner + meters ---------------------------
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(16)
        self.banner = TitleBanner()
        top.addWidget(self.banner, 1)

        self.temperature_meter = MeterChip(i18n.tr("meter_temperature_caption"))
        self.airflow_meter = MeterChip(i18n.tr("meter_airflow_caption"))
        top.addWidget(self.temperature_meter)
        top.addWidget(self.airflow_meter)
        root.addLayout(top)

        # --- fan/vent marker row: a real, laid-out slot (not an
        # absolutely-positioned float) so it can never overlap the meters
        # above or the explore toggles/button row below regardless of
        # phase -- those overlaps were real and reproducible at 800x600
        # when this was purely coordinate math against the vent's exact
        # (and sometimes cramped) physical position. Horizontal position
        # within the row still comes from the real vent geometry (see
        # PublicScene.vent_marker_position) -- only the vertical slot is
        # fixed.
        self.fan_row = QtWidgets.QWidget()
        self.fan_row.setFixedHeight(34)
        self.fan_row.setStyleSheet("background: transparent;")
        self.fan_row.hide()
        root.addWidget(self.fan_row)

        # --- explore row: real, data-driven direct-manipulation controls
        # Wrapped in its own widget (not just a layout) so it can be
        # hidden as a unit outside the free-play window -- shown only
        # while OBSERVE is active (see PublicExperience._render_observe).
        self.explore_panel = QtWidgets.QWidget()
        # A plain QWidget paints its own opaque background by default,
        # which showed up as a hard black bar across the whole width --
        # everywhere else on this screen is the translucent overlay over
        # the fire, so this one row must not be a solid rectangle.
        self.explore_panel.setStyleSheet("background: transparent;")
        explore_layout = QtWidgets.QHBoxLayout(self.explore_panel)
        explore_layout.setContentsMargins(0, 0, 0, 0)
        # 20 -> 12: fit a third group (the vent2 control) at 800x600
        # without any group's buttons shrinking below their own text --
        # two groups had slack to spare, three did not (caught in an
        # 800x600 screenshot: "OPEN"/"SHUT" clipped to "OPE"/"SHU" at 20).
        explore_layout.setSpacing(12)
        explore_layout.addStretch(1)
        self._explore_layout = explore_layout
        # Toggles are inserted at this index (see set_explore_controls),
        # which starts at 0 -- *before* the stretch just added -- and
        # advances by one per toggle, so each new toggle lands to the
        # right of the previous one but always still left of the stretch,
        # keeping the row left-aligned regardless of insertion order.
        self._explore_insert_index = 0
        self.explore_panel.hide()
        root.addWidget(self.explore_panel)

        # --- nav row: slim rows of small pills for navigation-only
        # actions (Explore's "Games" entry, and each Games screen's own
        # "Back to..." plus any extra small actions like "Compare this
        # place"/"Clear map") -- kept OUT of button_row, whose BigButtons
        # only have room for two at 800x600 before overlapping the
        # thermometer's own reserved column (a real, measured overlap:
        # three BigButtons there measured 729 px wide, 81 px past the
        # thermometer's left edge). Small pills are narrower, but four of
        # them (the Compare game's own "Back to games"/"What changed?"/
        # "Compare this place"/"Show before") still don't fit one row in
        # the ~550 px this leaves once the thermometer's column is
        # cleared -- a real, measured overflow (Qt compresses each pill
        # below its own sizeHint rather than wrapping, so the *text*
        # silently clips) this wraps onto a second row for instead, the
        # same fix game_row's own two rows already used.
        self.nav_row = QtWidgets.QWidget()
        self.nav_row.setStyleSheet("background: transparent;")
        self._nav_rows_layout = QtWidgets.QVBoxLayout(self.nav_row)
        self._nav_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._nav_rows_layout.setSpacing(6)
        self._nav_buttons: list = []
        self._nav_sub_rows: list = []
        self.nav_row.hide()
        root.addWidget(self.nav_row)

        # --- Games hub: a grid of big tiles, one per challenge, shown
        # only while Phase.GAMES_HOME is on screen (see
        # PublicExperience._render_games_home). Reuses BigButton
        # unmodified -- no new button widget class -- laid out in a grid
        # instead of button_row's single horizontal line, since six tiles
        # don't fit one row at 800x600.
        self.games_home_panel = QtWidgets.QWidget()
        self.games_home_panel.setStyleSheet("background: transparent;")
        self._games_grid = QtWidgets.QGridLayout(self.games_home_panel)
        self._games_grid.setSpacing(14)
        self.games_home_panel.hide()
        root.addWidget(self.games_home_panel)

        root.addStretch(1)

        # --- centre: the card (question / reveal / science) ------------
        card_row = QtWidgets.QHBoxLayout()
        card_row.addStretch(1)
        self.card = Card()
        self.card.setMaximumWidth(920)
        self.card.hide()
        card_row.addWidget(self.card, 3)
        card_row.addStretch(1)
        root.addLayout(card_row)

        # --- the choice buttons, centred on their own row --------------
        # Deliberately not sharing a row with the mascot: two big buttons
        # plus a mascot and a speech bubble overflow the width on a
        # 1280-wide display, and the button is the thing a visitor must
        # be able to reach.
        button_holder = QtWidgets.QHBoxLayout()
        button_holder.addStretch(1)
        self.button_row = QtWidgets.QHBoxLayout()
        self.button_row.setSpacing(14)
        button_holder.addLayout(self.button_row)
        button_holder.addStretch(1)
        root.addSpacing(18)
        root.addLayout(button_holder)

        root.addStretch(1)

        # --- mascot + speech bubble: absolutely positioned, bottom-left
        # Deliberately outside the layout. As layout items they competed
        # with the card for vertical space, and on a short window the
        # card (whose wrapped labels can shrink to one line) lost and
        # vanished entirely.
        self.mascot = Mascot(self)
        self.bubble = SpeechBubble(self)
        self.bubble.hide()

        # Stage strip: a "where am I" cue, positioned absolutely at the
        # bottom of the reserved mascot band rather than laid out. As a
        # layout item it cost ~50 px of height and squeezed the
        # prediction buttons below the touch-target floor -- orientation
        # chrome must never take space from the thing being pressed.
        self.stages = StageStrip(self)
        self.stages.hide()

        # Countdown: one enormous glyph centred over the fire. Absolutely
        # positioned so it never disturbs the card layout underneath it.
        self.countdown = QtWidgets.QLabel("", self)
        self.countdown.setAlignment(QtCore.Qt.AlignCenter)
        self.countdown.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.countdown.setStyleSheet(
            "color: #FFD166; font-size: 96px; font-weight: 800;"
            "background: transparent;")
        self.countdown.hide()

        # The always-on thermometer (see Thermometer's own docstring for
        # its three reading modes). Positioned in resizeEvent, along the
        # right edge, clear of the card/button column.
        self.thermometer = Thermometer(self)
        self.thermometer.hide()

        # A small label anchored (left-right) to the real fan/HVAC vent's
        # physical x-position (see PublicScene.vent_marker_position) --
        # makes the fan a thing in the scene, not a settings toggle
        # floating in a control panel. Child of fan_row (not the overlay
        # directly), so its vertical slot is guaranteed clear by layout
        # rather than by coordinate math against whatever else the
        # current phase happens to show.
        self.fan_marker = QtWidgets.QLabel("", self.fan_row)
        self.fan_marker.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.fan_marker.setAlignment(QtCore.Qt.AlignCenter)
        self.fan_marker.hide()

        # Exit affordance: small, dim, and out of the way. Deliberately
        # not labelled "Quit" -- it returns to the researcher app, and a
        # visitor should have no reason to press it.
        self.exit_button = QtWidgets.QPushButton("✕", self)
        self.exit_button.setFixedSize(44, 44)
        self.exit_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.exit_button.setAccessibleName("Leave the Fire Explorer")
        self.exit_button.setToolTip(i18n.tr("exit_tooltip"))
        # padding/min-height/max-height pinned explicitly: the app-wide
        # QPushButton rule (theme.py) sets its own padding + min-height,
        # and Qt's stylesheet engine layers that in over setFixedSize()'s
        # constraint on any property this button's own QSS doesn't
        # re-declare -- without pinning all four here, the button was a
        # real, measured 44x48 oval instead of the intended 44x44 circle.
        self.exit_button.setStyleSheet(
            "QPushButton { background: rgba(20,26,36,180); color: #8894A6;"
            " border: 1px solid rgba(255,255,255,30); border-radius: 22px; font-size: 18px;"
            " padding: 0px; min-width: 44px; max-width: 44px;"
            " min-height: 44px; max-height: 44px; }"
            "QPushButton:hover { color: #F3F6FA; background: rgba(40,48,64,220); }")
        self.exit_button.clicked.connect(self.exit_requested)

        # Language toggle: two small pills, always visible regardless of
        # phase (same "never routed through the per-phase button rows"
        # treatment exit_button already gets), positioned just to its
        # left in resizeEvent. Emits language_requested; PublicExperience
        # owns actually switching public.i18n's current language and
        # re-rendering -- this widget only reflects it (see
        # set_language_active), never decides it.
        #
        # Plain "EN"/"DE" text, not flag emoji (🇬🇧/🇩🇪): those are
        # regional-indicator sequences that render as blank circles on
        # the exhibition build's font fallback -- the exact class of
        # missing-glyph problem CEILING_BEAT_ICON's own comment already
        # documents (story.py), just for a different glyph.
        self.lang_en_button = QtWidgets.QPushButton("EN", self)
        self.lang_de_button = QtWidgets.QPushButton("DE", self)
        for button, lang in ((self.lang_en_button, "en"), (self.lang_de_button, "de")):
            button.setFixedSize(40, 40)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, l=lang: self.language_requested.emit(l))
        self.set_language_active("en")

        # A brief confetti/star burst for "you got it!" moments -- raised
        # on top of everything else only for the ~1s it plays, otherwise
        # invisible and never blocking a tap (see celebration.py).
        self._celebration = CelebrationOverlay(self)

    def celebrate(self) -> None:
        self._celebration.burst()

    def set_language_active(self, lang: str) -> None:
        """Highlight whichever flag pill matches the current language --
        purely a reflection of PublicExperience/public.i18n's own state,
        never a decision made here.

        padding/min-height/max-height are pinned explicitly for the same
        reason exit_button's own QSS pins them (see its comment): the
        app-wide QPushButton rule in theme.py otherwise inflates these
        past their setFixedSize(40, 40), a real, measured 40x50 oval
        instead of a 40x40 circle."""
        active_qss = (f"QPushButton {{ background: {ACCENT}; color: #1A1005; "
                     "border: 2px solid #FFE29A; border-radius: 20px; "
                     "font-size: 14px; font-weight: 700; padding: 0px;"
                     " min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px; }")
        inactive_qss = ("QPushButton { background: rgba(20,26,36,180); color: #C7D0DC;"
                        " border: 1px solid rgba(255,255,255,30); border-radius: 20px; "
                        "font-size: 14px; font-weight: 700; padding: 0px;"
                        " min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px; }"
                        "QPushButton:hover { background: rgba(40,48,64,220); color: #F3F6FA; }")
        self.lang_en_button.setStyleSheet(active_qss if lang == "en" else inactive_qss)
        self.lang_de_button.setStyleSheet(active_qss if lang == "de" else inactive_qss)

    def retranslate(self) -> None:
        """Re-set every static label that PublicExperience._render_phase()
        does NOT already re-declare every render (the two meter chip
        captions, the exit button's tooltip) -- called once right after
        public.i18n.set_language(), alongside the usual _render_phase()
        that refreshes everything else."""
        self.temperature_meter.set_caption(i18n.tr("meter_temperature_caption"))
        self.airflow_meter.set_caption(i18n.tr("meter_airflow_caption"))
        self.exit_button.setToolTip(i18n.tr("exit_tooltip"))

    def mousePressEvent(self, event) -> None:
        """Reached only for a click that lands on empty translucent
        area -- Qt delivers the event to a child (a button, the card)
        instead whenever one covers that point, so this never fires for
        an ordinary button press. That is what makes it safe to treat as
        "the visitor tapped the fire itself"."""
        if self._probe_enabled and event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.tapped.emit(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Drag-to-scan: Qt keeps delivering moves to whichever widget
        got the initial press (an implicit grab) even once the pointer
        crosses over a child widget, so this keeps firing for the whole
        drag started by mousePressEvent above -- letting a child sweep
        the probe across the scene instead of tapping one point at a
        time. Never touches the loaded scenario; each move is just
        another PublicScene.probe_at() lookup on the frame already on
        screen."""
        if self._probe_enabled and self._dragging and (event.buttons() & QtCore.Qt.LeftButton):
            self.tapped.emit(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def set_probe_enabled(self, enabled: bool) -> None:
        self._probe_enabled = enabled
        if not enabled:
            self._dragging = False

    # -- thermometer ------------------------------------------------------
    def set_thermometer_visible(self, visible: bool) -> None:
        self.thermometer.setVisible(visible)
        if visible:
            # A phase change (e.g. meters hiding for REVEAL) doesn't fire
            # resizeEvent, so its position needs a manual refresh here to
            # stay collision-free -- see _position_thermometer.
            self._position_thermometer()
            self.thermometer.raise_()
        else:
            self.thermometer.clear()

    def update_thermometer(self, value_c: float, caption: str) -> None:
        self.thermometer.set_reading(value_c, caption)

    def update_thermometer_trail(self, values: list) -> None:
        self.thermometer.set_trail(values)

    # -- fan/vent marker ----------------------------------------------------
    def set_fan_marker(self, frac, on: bool) -> None:
        """Show the fan label in its dedicated row (fan_row), positioned
        left-right by `frac`'s x-fraction -- the real vent's own
        horizontal position -- or hide it when `frac` is None (this
        phase, this plane, or this study has nothing to anchor it to).
        Only the x-fraction is used: the row itself is laid out (see
        __init__), so the vertical position is already guaranteed clear
        of the meters/explore toggles/button row, rather than following
        the vent's exact (and sometimes cramped) physical height.
        """
        if frac is None:
            self._fan_marker_frac = None
            self.fan_marker.hide()
            self._update_fan_row_visibility()
            return
        self._fan_marker_frac = frac
        self.fan_marker.setText("🌬️  FAN ON" if on else "🚫  FAN OFF")
        self.fan_marker.setStyleSheet(
            f"background: {AIRFLOW if on else 'rgba(24, 30, 42, 220)'}; "
            f"color: {'#1A1005' if on else TEXT_DIM}; font-size: 13px; font-weight: 700;"
            "border-radius: 10px; padding: 4px 10px;")
        self.fan_marker.adjustSize()
        self._update_fan_row_visibility()
        self._position_fan_marker()
        self.fan_marker.show()
        self.fan_marker.raise_()

    def _position_fan_marker(self) -> None:
        """`frac`'s x is a fraction of the *scene's* full-bleed width
        (the overlay's own width -- see PublicScene.widget_fraction_for),
        but fan_marker is now a child of fan_row, which sits inset by the
        root layout's side margins -- converted back to fan_row's local
        coordinate space here rather than assuming the two line up."""
        if self._fan_marker_frac is None or self.fan_row.width() <= 0:
            return
        fx, _fy = self._fan_marker_frac
        x = int(fx * self.width() - self.fan_row.x() - self.fan_marker.width() / 2)
        y = (self.fan_row.height() - self.fan_marker.height()) // 2
        x = max(4, min(x, self.fan_row.width() - self.fan_marker.width() - 4))
        self.fan_marker.move(x, max(0, y))

    def _update_fan_row_visibility(self) -> None:
        """fan_row holds fan_marker alone now (the candle-count marker
        that used to share it was removed -- the count is already legible
        both in the scene itself, drawn candle-by-candle, and in the
        Candles toggle's own checked state, so a third copy of the same
        fact was crowding this corner of the screen for nothing new)."""
        self.fan_row.setVisible(self._fan_marker_frac is not None)

    def pulse_fan_marker(self) -> None:
        """A brief bounce on the fan/vent marker -- the vent's own "I
        felt that" acknowledgement for a direct tap in the scene (see
        PublicExperience._on_vent_tapped), the same idea as the candle's
        pulse_flame(). Captures the marker's geometry *now* (after the
        real toggle has already repositioned/relabelled it), not before,
        so the animation's own end state can never fight with
        _position_fan_marker's already-correct placement."""
        if self.fan_marker.isHidden():
            return
        original = self.fan_marker.geometry()
        grown = original.adjusted(-3, -2, 3, 2)
        anim = QtCore.QPropertyAnimation(self.fan_marker, b"geometry", self)
        anim.setDuration(260)
        anim.setKeyValueAt(0.0, original)
        anim.setKeyValueAt(0.4, grown)
        anim.setKeyValueAt(1.0, original)
        anim.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def pulse_explore_toggle(self, key: str) -> None:
        """See ExploreToggle.flash_highlight -- called right after a
        related tap in the scene itself (e.g. the candle) so the child's
        eye is drawn to the one real control that changes it, instead of
        a second in-scene picker duplicating it."""
        toggle = self._explore_toggles.get(key)
        if toggle is not None and not toggle.isHidden():
            toggle.flash_highlight()


    # -- explore controls (direct manipulation, see experiments.py) -----
    def set_explore_controls(self, controls: list) -> None:
        """Build one ExploreToggle per available control. Called once,
        at construction, with whatever `available_explore_controls()`
        found in this study's own manifest -- never a fixed UI list."""
        for toggle in self._explore_toggles.values():
            self._explore_layout.removeWidget(toggle)
            toggle.deleteLater()
        self._explore_toggles = {}
        self._explore_insert_index = 0
        for control in controls:
            # The fan is the one control where "on" has its own meaning
            # worth a color (air moving) distinct from generic "selected"
            # -- every other control keeps ExploreToggle's ACCENT default.
            checked_colors = [INERT, AIRFLOW] if control.key == "fan" else None
            toggle = ExploreToggle(control.label, control.icon, control.options,
                                    checked_colors=checked_colors)
            toggle.value_changed.connect(
                lambda value, key=control.key: self.explore_changed.emit(key, value))
            self._explore_layout.insertWidget(self._explore_insert_index, toggle)
            self._explore_insert_index += 1
            self._explore_toggles[control.key] = toggle
        self.explore_panel.setVisible(bool(controls))

    def set_explore_values(self, values: dict) -> None:
        """Snap each toggle's checked state to an explicit value per
        control key -- used to keep the toggle UI in sync with whatever
        scenario is actually loaded when OBSERVE re-renders without a
        fresh baseline load (e.g. closing the Help or Compare detour
        mid-exploration, where reset_explore_controls's blind defaults
        would otherwise desync the toggles from the scenario actually
        on screen)."""
        for key, value in values.items():
            toggle = self._explore_toggles.get(key)
            if toggle is not None:
                toggle.set_value_silently(value)

    def reset_explore_controls(self, controls: list, except_key: str = "") -> None:
        """Snap every control except `except_key` back to its default
        value -- keeps the one-factor-at-a-time rule honest in the UI,
        not just in the resolver (see ExploreControl's docstring)."""
        for control in controls:
            if control.key == except_key:
                continue
            toggle = self._explore_toggles.get(control.key)
            if toggle is not None:
                toggle.set_value_silently(control.default_value)

    def set_explore_visible(self, visible: bool) -> None:
        self.explore_panel.setVisible(visible and bool(self._explore_toggles))

    # -- nav row (small pills -- see nav_row's own construction comment) -
    # The Games entry's two "glow" stylesheets -- swapped on a slow repeat
    # timer (see _start_pulse) rather than a geometry/opacity animation,
    # since geometry isn't valid yet the instant a fresh button is built
    # (before the layout it just joined has had a pass), and this exact
    # stylesheet-swap idiom is already proven safe elsewhere in this file
    # (ExploreToggle.flash_highlight, TitleBanner.flash) -- just looped
    # here instead of reverted once.
    _PLAYFUL_QSS = f"""
        QPushButton {{
            background: {DELIGHT}; color: #2B1400;
            border: 3px solid #FFE9BE; border-radius: 16px;
            padding: 6px 14px; font-size: 14px; font-weight: 800;
        }}
        QPushButton:hover {{ background: #FFDD8A; }}
    """
    _PLAYFUL_GLOW_QSS = f"""
        QPushButton {{
            background: #FFDD8A; color: #2B1400;
            border: 3px solid {DELIGHT}; border-radius: 16px;
            padding: 6px 14px; font-size: 14px; font-weight: 800;
        }}
        QPushButton:hover {{ background: #FFDD8A; }}
    """

    def _small_pill(self, text: str, playful: bool = False) -> QtWidgets.QPushButton:
        """`playful=True` is reserved for the single "this is a fun
        activity, not a settings toggle" entry point (today: "Games") --
        a brighter, rounder style plus a slow idle glow, so it reads as an
        invitation rather than a standard UI control. Every other nav
        pill keeps the plain dark style."""
        button = QtWidgets.QPushButton(text, self.nav_row)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        if playful:
            button.setStyleSheet(self._PLAYFUL_QSS)
            self._start_pulse(button)
        else:
            button.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(24, 30, 42, 235); color: {TEXT};
                    border: 2px solid {PANEL_BORDER}; border-radius: 12px;
                    padding: 5px 10px; font-size: 13px; font-weight: 600;
                }}
                QPushButton:hover {{ background: rgba(44, 54, 72, 245); }}
            """)
        return button

    def _start_pulse(self, button: QtWidgets.QPushButton) -> None:
        """A slow, repeating "breathe" glow: alternate the button's own
        stylesheet between its normal and glow QSS every 700ms. A normal
        (repeating) QTimer parented to the button, so it's torn down
        automatically with it -- set_nav_buttons() rebuilds this button
        fresh on every render, it never outlives the button it's on."""
        timer = QtCore.QTimer(button)
        state = {"on": False}

        def _toggle():
            state["on"] = not state["on"]
            button.setStyleSheet(self._PLAYFUL_GLOW_QSS if state["on"] else self._PLAYFUL_QSS)
        timer.timeout.connect(_toggle)
        timer.start(700)

    # At most this many small pills per sub-row -- a real, measured
    # overflow (see nav_row's own construction comment): 4 pills'
    # natural width didn't fit the ~550 px available once the
    # thermometer's column is cleared, so Qt compressed each pill below
    # its own sizeHint and the text silently clipped. 3 comfortably does.
    _NAV_PILLS_PER_ROW = 3

    def set_nav_buttons(self, items: list) -> None:
        """Replace the nav rows' pills with one per (text, icon, callback)
        or (text, icon, callback, playful) -- called fresh on every
        render, the same convention button_row's add_button already
        follows. An empty list hides the row(s)."""
        for button in self._nav_buttons:
            self._discard(button)
        self._nav_buttons = []
        for row in self._nav_sub_rows:
            self._nav_rows_layout.removeWidget(row)
            self._discard(row)
        self._nav_sub_rows = []

        chunks = [items[i:i + self._NAV_PILLS_PER_ROW]
                 for i in range(0, len(items), self._NAV_PILLS_PER_ROW)]
        for chunk in chunks:
            row = QtWidgets.QWidget()
            row.setStyleSheet("background: transparent;")
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 222, 0)   # clear of the (now wider) thermometer column
            row_layout.setSpacing(10)
            for item in chunk:
                text, icon, callback = item[0], item[1], item[2]
                playful = item[3] if len(item) > 3 else False
                button = self._small_pill(f"{icon} {text}" if icon else text, playful=playful)
                button.clicked.connect(callback)
                row_layout.addWidget(button)
            row_layout.addStretch(1)
            self._nav_rows_layout.addWidget(row)
            self._nav_sub_rows.append(row)
            self._nav_buttons.extend(
                row_layout.itemAt(i).widget() for i in range(len(chunk)))
        self.nav_row.setVisible(bool(items))

    # -- Games hub (see PublicExperience._render_games_home) -------------
    def set_game_tiles(self, tiles: list) -> None:
        """Build one BigButton tile per (icon, label, callback), replacing
        whatever was there before -- called fresh each time the Games hub
        renders, the same "redeclare it every render" convention
        button_row's own add_button already follows. Three columns: six
        tiles in one row would run off the right edge at 800x600."""
        while self._games_grid.count():
            item = self._games_grid.takeAt(0)
            if item.widget() is not None:
                self._discard(item.widget())
        columns = 3
        for index, (icon, label, callback) in enumerate(tiles):
            tile = BigButton(label, icon, tall=True)
            tile.clicked.connect(callback)
            self._games_grid.addWidget(tile, index // columns, index % columns)
        self.games_home_panel.setVisible(bool(tiles))

    def set_games_home_visible(self, visible: bool) -> None:
        self.games_home_panel.setVisible(visible)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.exit_button.move(self.width() - self.exit_button.width() - 16, 16)
        self.lang_de_button.move(self.exit_button.x() - self.lang_de_button.width() - 8, 16)
        self.lang_en_button.move(
            self.lang_de_button.x() - self.lang_en_button.width() - 6, 16)
        self.countdown.setGeometry(0, 0, self.width(), self.height())
        strip_width = min(460, max(320, self.width() - 80))
        self.stages.resize(strip_width, self.stages.height())
        self.stages.move((self.width() - strip_width) // 2,
                         self.height() - self.stages.height() - 6)
        self._position_thermometer()
        self._position_fan_marker()
        self._position_mascot()

    # Reserved below the thermometer -- kept out of its own height budget
    # so there's always a clear gap above the stage strip/card, regardless
    # of how tall the thermometer itself ends up.
    _PROBE_DOCK_RESERVE = 104

    def set_thermometer_anchor(self, frac_x) -> None:
        """The room's own real right-wall x, as a fraction of this
        widget's width (see PublicScene.room_wall_anchor, fed through
        PublicExperience the same way the fan/candle markers already are)
        -- lets the thermometer dock beside the actual room geometry
        instead of a fixed pixel offset from the window edge. None (no
        manifest / wrong plane) falls back to the previous fixed-offset
        behaviour."""
        self._thermometer_anchor_frac = frac_x
        if self.thermometer.isVisible():
            self._position_thermometer()

    def _position_thermometer(self) -> None:
        """Below both meter chips when they're on screen, above the stage
        strip -- never overlapping either (a real, reproducible collision
        at 800x600 before this was computed from live sibling geometry
        instead of a fixed y). Falls back to a fixed top margin when the
        meters are hidden (REVEAL/SCIENCE): a hidden widget's geometry is
        stale, not "zero space", so trusting it there could place the
        thermometer anywhere -- harmless left-right (it sits at the right
        edge, clear of the centred card regardless of its y), but kept
        deliberate rather than accidental.
        """
        self.layout().activate()
        if self.temperature_meter.isVisible():
            top = max(self.temperature_meter.geometry().bottom(),
                      self.airflow_meter.geometry().bottom()) + 18
        else:
            # The exit button sits fixed at (width-60, 16, 44, 48)
            # regardless of the meters -- the thermometer's right edge
            # passes directly under it, so the fallback top must clear
            # its bottom too, not just an arbitrary small margin.
            top = self.exit_button.geometry().bottom() + 12
        bottom_limit = (self.stages.y() if self.stages.y() > 0 else self.height()) - 20
        height = max(220, min(380, bottom_limit - self._PROBE_DOCK_RESERVE - top))
        anchor = self._thermometer_anchor_frac
        if anchor is not None:
            # Docked just outside the room's real right wall, not
            # centred on it -- the wall itself stays visible, the
            # thermometer reads as "attached to the room" beside it.
            # In practice this offset rarely binds: the room's right
            # wall (ROOM_X[1]) is the domain's own right edge, i.e. the
            # scene's full-bleed right edge too, so there is no real
            # "outside the wall" space at 800x600 -- the clamp below
            # (screen width minus the thermometer's own width) is what
            # actually places it. Left in case a wider display ever
            # gives this room to matter.
            x = anchor * self.width() + 10
        else:
            x = self.width() - self.thermometer.width() - 14
        # A wide button (e.g. "Try Fan OFF" on the reveal/science cards)
        # can reach far enough right to clip the thermometer's left edge
        # by a couple of pixels -- pushed clear of the widest current
        # button rather than trusting a fixed inset to always be enough.
        button_rights = [b.geometry().right() for b in self._buttons if b.geometry().right() > 0]
        if button_rights:
            x = max(x, max(button_rights) + 12)
        self.thermometer.resize(self.thermometer.width(), height)
        # int(): `anchor` (and so `x`) can be a numpy.float64 -- extent
        # values come straight from the store's own array metadata --
        # and QWidget.move() rejects that type outright. Previously
        # masked by the fallback clamp below always winning with a plain
        # int and min() returning it unchanged; now that SCENE_WIDTH_FRAC
        # gives the anchor branch real room to be the smaller (and so
        # winning) operand, the unconverted numpy type reached move()
        # directly and raised.
        self.thermometer.move(int(min(x, self.width() - self.thermometer.width() - 4)), top)

    def _position_mascot(self) -> None:
        """Pin the mascot bottom-left and the bubble to its right, both
        sitting inside the band the layout's bottom margin reserves."""
        margin = 32
        mascot_y = self.height() - self.mascot.height() - margin
        self.mascot.move(margin, max(0, mascot_y))

        if not self.bubble.isVisible():
            return
        available = self.width() - self.mascot.width() - 3 * margin
        if self.thermometer.isVisible():
            # A long unwrapped line's natural sizeHint width can reach
            # past the thermometer's left edge (a real overlap at
            # 800x600 -- e.g. the observe-phase invitation sentence);
            # the bubble must wrap sooner when that column is occupied.
            available = min(available, self.thermometer.x() - self.mascot.x()
                            - self.mascot.width() - 12)
        hint = self.bubble.sizeHint()
        width = max(240, min(hint.width(), available))
        # sizeHint()'s height assumes its own preferred width; recompute
        # at the width the bubble will actually get, or long text clips.
        self.bubble.resize(width, self.bubble.heightForWidth(width))
        # A long wrapped line can grow the bubble tall enough that its
        # bottom edge dips into the stage strip's row (a real, if small,
        # overlap at 800x600) -- clamped above it rather than trusting
        # the fixed margin alone to always be enough clearance.
        y = self.height() - self.bubble.height() - margin - 6
        if self.stages.isVisible():
            y = min(y, self.stages.y() - self.bubble.height() - 6)
        self.bubble.move(self.mascot.x() + self.mascot.width() + 12, y)

    # -- helpers --------------------------------------------------------
    def say(self, text: str, mood: str = IDLE) -> None:
        self.bubble.set_text(text)
        self.mascot.set_mood(mood)
        self._position_mascot()

    def set_prompt(self, text: str) -> None:
        self.banner.set_prompt(text)

    def set_stage(self, index: int) -> None:
        self.stages.set_active(index)

    def show_countdown(self, text: str) -> None:
        self.countdown.setText(text)
        self.countdown.setGeometry(0, 0, self.width(), self.height())
        self.countdown.setVisible(bool(text))
        self.countdown.raise_()

    def hide_countdown(self) -> None:
        self.countdown.setText("")
        self.countdown.hide()

    def set_meters_visible(self, visible: bool) -> None:
        self.temperature_meter.setVisible(visible)
        self.airflow_meter.setVisible(visible)

    def update_meters(self, room_temp_c: float, mean_airspeed: float,
                      has_velocity: bool = True) -> None:
        """Live readings. Both values are direct reductions over the
        current FDS frame -- the phrase beside each is the only
        interpretation, and it comes from kid_language.

        Temperature is the scene-wide *average*, not the peak: the peak
        is the candle flame itself (~450 C in every scenario), which
        tells a child nothing about the surrounding air and reads as a
        danger number for a tabletop experiment that never becomes
        dangerous. Captioned "air temperature" rather than "room" because
        the mean covers the whole simulated area, not just the enclosed
        room -- see the room_temp Metric in experiments.py.
        """
        self.temperature_meter.set_reading(
            f"{room_temp_c:.0f} °C", kid.temperature_phrase(room_temp_c),
            kid.temperature_icon(room_temp_c))
        if has_velocity:
            self.airflow_meter.set_reading(
                f"{mean_airspeed:.2f} m/s", kid.airflow_phrase(mean_airspeed),
                kid.airflow_icon(mean_airspeed))
        else:
            self.airflow_meter.set_reading("—", i18n.tr("meter_airflow_not_measured"))

    @staticmethod
    def _discard(widget) -> None:
        """Remove a widget from the UI *now*, then free it.

        deleteLater() alone is not enough here: deferred deletions are
        only processed when the event loop unwinds to its main level, so
        a widget dropped during a phase change stays a visible child of
        the overlay until then -- which showed up as a previous phase's
        button still painted on top of the new one. Reparenting to None
        takes it out of the hierarchy immediately; deleteLater then frees
        it on Qt's own schedule.

        Any *repeating* QTimer this widget owns (e.g. the Games pill's
        idle-glow pulse, see _start_pulse) is stopped explicitly first --
        a real, reproduced segfault this closes: parenting the timer to
        the widget is not enough on its own, since the timer keeps firing
        in the gap between this call and Qt's own deferred deletion,
        and a fired callback that touches a widget already torn down by a
        more aggressive cleanup elsewhere (e.g. a test's window.close())
        crashes the whole process rather than raising a catchable error --
        the same class of bug the rest of this codebase's _defer()
        convention already guards against for scheduled callbacks.
        """
        for timer in widget.findChildren(QtCore.QTimer):
            timer.stop()
        widget.hide()
        widget.setParent(None)
        widget.deleteLater()

    def clear_buttons(self) -> None:
        for button in self._buttons:
            self.button_row.removeWidget(button)
            self._discard(button)
        self._buttons = []

    def add_button(self, text: str, icon: str, on_click, primary: bool = False,
                   tall: bool = False) -> BigButton:
        button = BigButton(text, icon, primary=primary, tall=tall)
        button.clicked.connect(on_click)
        self.button_row.addWidget(button)
        self._buttons.append(button)
        return button

    def clear_card(self) -> None:
        layout = self.card.body()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                self._discard(item.widget())
        self.card.hide()

    def show_card(self, title: str, lines: list, dim_lines: list = None,
                  block: str = "", hero=None) -> None:
        """Populate and reveal the central card. `hero` is an optional
        widget -- or list of widgets, most important first -- shown
        directly under the title (the science card's findings); `lines`
        are the main sentences; `block` is an optional pre-formatted
        table of measured values; `dim_lines` are secondary (sources,
        caveats)."""
        self.clear_card()
        layout = self.card.body()
        if title:
            layout.addWidget(Card.title_label(title))
        for widget in ([hero] if isinstance(hero, QtWidgets.QWidget) else (hero or [])):
            layout.addWidget(widget)
        if block:
            layout.addWidget(Card.block_label(block))
        for line in lines:
            layout.addWidget(Card.text_label(line))
        for line in dim_lines or []:
            layout.addWidget(Card.text_label(line, dim=True))
        self.card.show()

    def shutdown(self) -> None:
        self.mascot.stop()
