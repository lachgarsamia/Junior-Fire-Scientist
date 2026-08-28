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

import sys
import time
import traceback

from PyQt5 import QtCore, QtWidgets

from public import i18n
from public import kid_language as kid
from public.celebration import CelebrationOverlay
from public.mascot import (Mascot, SpeechBubble, Scientist,
                           IDLE, POINTING, SURPRISED, THINKING)
from public.widgets import (ACCENT, AIRFLOW, DELIGHT, INERT, PANEL_BG, PANEL_BORDER, TEXT,
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
        self._thermometer_reposition_timer = None
        self._mascot_reposition_timer = None
        # 0.0, not time.monotonic(): monotonic's own reference point is
        # already far in the past (often system boot), so this reads as
        # "long ago" from the very first seconds_since_say() call without
        # needing a sentinel.
        self._last_say_at = 0.0
        # TEMPORARY -- crash investigation instrumentation, see say()'s
        # own comment.
        self._say_depth = 0

        root = QtWidgets.QVBoxLayout(self)
        # Bottom margin reserves the band the mascot and its speech
        # bubble occupy -- they are positioned absolutely (see
        # _position_mascot) rather than laid out, so that a tall card is
        # never squeezed to nothing to make room for decoration. The
        # right margin reserves stat_panel (language toggle + both
        # meters + thermometer, see its own construction comment below)
        # -- also absolutely positioned, not laid out -- so every row
        # `root` actually manages (the banner, explore_panel, nav_row,
        # button_holder) stops short of it instead of running underneath
        # and getting visually clipped. 330, not a smaller value close
        # to the old 170: stat_panel is wider than the thermometer alone
        # now that the two meter chips sit beside it (see
        # _METER_CHIP_FLAT_WIDTH's own comment) -- a first version of
        # this panel used MeterChip's un-flat 210px-per-chip floor and
        # needed a 440px margin, which starved the explore-control row
        # below it to an illegible ~320px at 800x600, a real,
        # screenshotted regression the narrower flat chip width (plus
        # MeterChip's own caption-eliding resizeEvent) exists to fix.
        # 300 -> 330: even after the flat-chip fix, explore_panel's own
        # right edge (this margin's real boundary) and stat_panel's own
        # left edge (see _position_stat_group's fixed right-dock) landed
        # only ~3px apart -- technically not overlapping, but a real,
        # measured "about to collide" tightness at every window width,
        # not the visible breathing room asked for. The extra 30px is
        # spent entirely on that gap, not on stat_panel itself (whose
        # own width and right-edge inset are unchanged).
        root.setContentsMargins(40, 28, 330, 160)
        root.setSpacing(16)

        # --- top row: prompt banner -------------------------------------
        # The meter chips used to sit in this same row, to the banner's
        # right -- moved out (see stat_panel below) so they can join the
        # language toggle and the thermometer in one connected panel
        # instead of floating here as their own separate element.
        self.banner = TitleBanner()
        root.addWidget(self.banner)

        # --- stat panel: language toggle + both meters + thermometer,
        # grouped into one visually connected right-side panel instead of
        # three separately-floating elements (a real, repeated complaint
        # -- "these should read as one panel"). stat_panel itself is
        # nothing but the shared background/border behind them: the
        # three groups of widgets it backs are never reparented into it
        # (still direct children of this overlay, same as before), so
        # every existing reader of their own geometry in *this* widget's
        # coordinate space -- the scientist bubble's thermometer-column
        # clamp, the worker mascot's thermometer-bottom safety net --
        # keeps working unmodified; only _position_stat_group (replacing
        # the old _position_thermometer) changes, laying out all three
        # groups together and sizing this panel to wrap them. Same
        # PANEL_BG/PANEL_BORDER treatment explore_panel already uses, so
        # this reads as "the same kind of panel" as the rest of the
        # app's chrome. Constructed here, before the widgets it backs, so
        # it paints behind them by plain z-order (no explicit raise_()
        # needed on this side; each of the three still calls raise_() at
        # the points it always did, e.g. set_thermometer_visible).
        self.stat_panel = QtWidgets.QWidget(self)
        self.stat_panel.setObjectName("statPanel")
        self.stat_panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.stat_panel.setStyleSheet(f"""
            QWidget#statPanel {{
                background: {PANEL_BG};
                border: 1px solid {PANEL_BORDER};
                border-radius: 16px;
            }}
        """)

        self.temperature_meter = MeterChip(i18n.tr("meter_temperature_caption"), self, flat=True)
        self.airflow_meter = MeterChip(i18n.tr("meter_airflow_caption"), self, flat=True)

        # --- explore row: real, data-driven direct-manipulation controls
        # Wrapped in its own widget (not just a layout) so it can be
        # hidden as a unit outside the free-play window -- shown only
        # while OBSERVE is active (see PublicExperience._render_observe).
        self.explore_panel = QtWidgets.QWidget()
        self.explore_panel.setObjectName("explorePanel")
        # A bare QWidget needs WA_StyledBackground for its own QSS
        # background/border to actually paint (otherwise Qt falls back to
        # the app-wide QPushButton/QWidget defaults, which showed up as a
        # hard black bar across the whole width before this was set --
        # everywhere else on this screen is the translucent overlay over
        # the fire). Scoped by #explorePanel, not a bare `QWidget { ... }`
        # selector, so this background doesn't cascade onto the toggle
        # buttons/captions inside it -- Qt stylesheets apply an
        # unqualified QWidget rule to every descendant QWidget too.
        # One shared translucent panel behind all four toggle groups
        # (Vent 1/Candles/Vent 2/Door) so they read as sections of one
        # control bar rather than four separately-floating clusters --
        # per direct feedback that the row didn't feel unified. Same
        # PANEL_BG/PANEL_BORDER treatment Card already uses elsewhere,
        # so this reads as "the same kind of panel" as the rest of the
        # app's chrome, not a new visual language.
        self.explore_panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.explore_panel.setStyleSheet(f"""
            QWidget#explorePanel {{
                background: {PANEL_BG};
                border: 1px solid {PANEL_BORDER};
                border-radius: 16px;
            }}
        """)
        # A 2-row grid was tried here and reverted: explore_panel's own
        # *height* in root's layout is hard-capped at ~91px (root's total
        # content already exceeds the 800x600 window's vertical budget by
        # about that much, and every other row -- nav_row,
        # button_holder -- is pinned/fixed, so a flexible two-row grid
        # simply gets compressed below any button's real minimum,
        # confirmed by measuring root's own children, not assumed). A
        # single row has no such problem (it always fit in that same
        # ~91px, before and after this change) -- only *width* is tight
        # now that Vent 1 has three buttons and Door is a fourth group,
        # so this stays a QHBoxLayout and the fix is in ExploreToggle's
        # own button/spacing sizes below, the same kind of squeeze this
        # spacing constant's own history already went through once
        # (20 -> 12px, for three groups; now tighter again, for four).
        # Small horizontal/vertical margins (was 0,0,0,0) give the new
        # shared panel background some breathing room around the
        # buttons rather than painting flush against them -- kept small
        # deliberately, the width budget this comment already describes
        # has no slack to give up to padding.
        explore_layout = QtWidgets.QHBoxLayout(self.explore_panel)
        explore_layout.setContentsMargins(10, 6, 10, 6)
        # This spacing applies between *every* adjacent pair of items in
        # the row, dividers (see _make_group_divider) included -- a
        # divider's own 2px width plus this 6px gap on each side of it
        # gives ~14px of visual separation between groups, clearly more
        # than the 6px ExploreToggle's own row.setSpacing uses *within*
        # a group, so the eye reads "Vent 1 | Candles | Vent 2 | Door" as
        # four distinct clusters rather than one continuous row (a real
        # complaint: at the old, roughly-equal within/between spacing,
        # it read as one row of loose buttons).
        explore_layout.setSpacing(6)
        self._explore_layout = explore_layout
        # Toggles (and the dividers between them) are appended at this
        # index (see set_explore_controls), which starts at 0 and
        # advances by one per widget, so each new one lands to the right
        # of the previous.
        self._explore_insert_index = 0
        self._explore_dividers: list = []
        self.explore_panel.hide()
        # AlignLeft, not a bare addWidget: a QVBoxLayout stretches a
        # plain child to the *layout's* full available width regardless
        # of the widget's own sizePolicy, which left this panel's own
        # background/border stretching well past its actual buttons into
        # a large stretch of empty painted panel (the old internal
        # addStretch(1) this replaced was pushing the *buttons* left
        # within that same over-wide box, not shrinking the box itself)
        # -- a real, screenshotted "the bar looks like it's enveloping
        # empty space, not just the buttons" complaint. The alignment
        # flag instead sizes explore_panel to its own sizeHint (i.e. to
        # its real content) and left-aligns *that*, so the painted panel
        # itself now ends where the last button/divider does.
        #
        # The alignment flag alone measurably was *not* enough on its
        # own (still stretched full-width in practice) -- QWidget's
        # default horizontal size policy is Preferred, which still lets
        # a QVBoxLayout hand it more than its sizeHint whenever more is
        # available. Maximum pins the *ceiling* at sizeHint's own width
        # (it can still shrink below that if the window is narrower, the
        # same "gets compressed toward each button's own 52px floor"
        # behaviour already established at 800x600), which is what
        # actually makes the alignment flag's left-alignment bind.
        self.explore_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        root.addWidget(self.explore_panel, 0, QtCore.Qt.AlignLeft)

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
        self.games_home_panel.setObjectName("gamesHomePanel")
        # A real panel background (same PANEL_BG/PANEL_BORDER language as
        # explore_panel/stat_panel/Card), not "background: transparent" --
        # this grid sits directly over the live scene, and a transparent
        # backdrop let the flame/room bleed through the gaps between
        # tiles (and faintly through each tile's own ~92%-opaque fill), a
        # real, screenshotted "the tiles look like they're floating in
        # the fire" complaint. WA_StyledBackground is what makes a bare
        # QWidget's own QSS background actually paint (explore_panel's
        # own comment explains why); scoped by #gamesHomePanel, not a
        # bare `QWidget {...}` rule, so it doesn't cascade onto the
        # BigButton tiles inside it.
        self.games_home_panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.games_home_panel.setStyleSheet(f"""
            QWidget#gamesHomePanel {{
                background: {PANEL_BG};
                border: 1px solid {PANEL_BORDER};
                border-radius: 16px;
            }}
        """)
        self._games_grid = QtWidgets.QGridLayout(self.games_home_panel)
        self._games_grid.setContentsMargins(14, 14, 14, 14)
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

        # --- mascot + speech bubble: absolutely positioned, bottom-right
        # (swapped with Dr. Funke -- see her own construction comment
        # below for why: her old bottom-right spot risked crowding the
        # experiment/room area as she grew, and the worker mascot's own
        # tight-corner bubble treatment -- min_width/tail_side/h_padding
        # below -- is exactly what her bubble used to need in that same
        # corner). Deliberately outside the layout. As layout items they
        # competed with the card for vertical space, and on a short
        # window the card (whose wrapped labels can shrink to one line)
        # lost and vanished entirely.
        self.mascot = Mascot(self)
        # tail_side="right": he now stands in the true bottom-right
        # corner (see _position_mascot), with the bubble to his *left* --
        # mirrors Dr. Funke's own old tight-corner treatment exactly
        # (min_width/h_padding included), since he now occupies the same
        # physical pocket next to the thermometer's own sidebar column
        # that she used to.
        self.bubble = SpeechBubble(self, min_width=130, tail_side="right", h_padding=12)
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

        # Dr. Frieda Funke: a second, ambient guide, now standing in the
        # true bottom-left corner (see Scientist's own docstring) --
        # periodically given a real, simplified fire-science fact by
        # PublicExperience, shown in scientist_bubble. Swapped out of her
        # old bottom-right spot (next to the thermometer's instrument
        # sidebar) per direct feedback that the corner risked crowding
        # the experiment/room area as she grew to full size -- the
        # worker mascot now occupies that pocket instead (see his own
        # construction comment above), and she takes his old spot, with
        # the same generous, un-cramped bubble treatment he used to get
        # there (min_width/h_padding below both revert to SpeechBubble's
        # own defaults). A second SpeechBubble instance, the same widget
        # the primary mascot uses (not a separate bubble class). Shares
        # the thermometer's own visibility gating (both are
        # PublicExperience.set_scientist_visible/set_thermometer_visible,
        # called together) rather than a second copy of that logic.
        self.scientist = Scientist(self)
        self.scientist.setAccessibleName(i18n.tr("scientist_name"))
        self.scientist.hide()
        # "#7C93A8": Scientist's own _SCI_COLLAR -- the one cool colour
        # in her own palette -- so her bubble reads as visually hers at
        # a glance (a tinted border/background, see SpeechBubble's own
        # accent handling) without forking the widget for it, regardless
        # of which corner she happens to stand in. tail_side left at its
        # default ("left"): she now stands in the true bottom-left
        # corner (see _position_scientist), with the bubble to her
        # *right* -- the same orientation the primary mascot's own
        # bubble always used in that corner.
        self.scientist_bubble = SpeechBubble(self, accent="#7C93A8")
        self.scientist_bubble.hide()

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
            # stay collision-free -- see _position_stat_group.
            self._position_stat_group()
            self.thermometer.raise_()
            # The meter chips' own sizeHint-driven geometry (read by
            # _position_stat_group's `top`) isn't always final the
            # instant set_meters_visible() returns -- Qt can defer a
            # child layout's own relayout by one event-loop turn past
            # the parent's activate(). Caught as a real, measured bug:
            # this call's own `top` used a stale (larger) meter-bottom
            # value, which _position_stat_group's height formula then
            # read as "less room available" than genuinely exists,
            # capping the thermometer shorter than it needed to be. One
            # deferred re-run corrects it once everything has settled;
            # guarded so a rapid run of phase changes never stacks more
            # than one pending correction.
            if self._thermometer_reposition_timer is None:
                # Parented to self, not a bare QTimer.singleShot(): a
                # bare one-shot's callback still fires even after this
                # widget is destroyed (between one test's teardown and
                # the next, or a real exit-public-mode) -- PublicExperience.
                # _defer already established this same fix elsewhere in
                # the public package; this mirrors it rather than
                # reintroducing the crash risk it exists to avoid.
                self._thermometer_reposition_timer = QtCore.QTimer(self)
                self._thermometer_reposition_timer.setSingleShot(True)
                self._thermometer_reposition_timer.timeout.connect(
                    self._reposition_thermometer_once_settled)
            self._thermometer_reposition_timer.start(0)
        else:
            self.thermometer.clear()

    def _reposition_thermometer_once_settled(self) -> None:
        if self.thermometer.isVisible():
            self._position_stat_group()
            self._position_mascot()
            # Dr. Funke's own speech bubble clamps its width against
            # self.thermometer.x() (see _position_scientist_bubble's own
            # comment), computed synchronously back in
            # set_thermometer_visible -- before *this* deferred
            # correction ran. If settling moved the thermometer even a
            # few px, that clamp is now stale and the bubble can run
            # past the thermometer's real left edge (a real, measured
            # overlap with whichever mascot stands in that corner --
            # the worker mascot's own spot since the corner swap, see
            # __init__'s own note). Re-flowing it here, against the
            # now-settled geometry, is what closes that gap.
            self._position_scientist()

    def update_thermometer(self, value_c: float, caption: str) -> None:
        self.thermometer.set_reading(value_c, caption)

    def update_thermometer_trail(self, values: list) -> None:
        self.thermometer.set_trail(values)

    # -- Dr. Funke (see Scientist's own docstring) -----------------------
    def set_scientist_visible(self, visible: bool) -> None:
        self.scientist.setVisible(visible)
        if visible:
            self._position_scientist()
            self.scientist.raise_()
        else:
            self.scientist_bubble.set_text("")

    def say_fact(self, text: str) -> None:
        """Show a fact in Dr. Funke's own speech bubble -- a second
        SpeechBubble instance (see its own docstring), not routed
        through the primary Mascot's say(), which is the primary guide's
        direct address to the child (see Scientist's own docstring for
        why the two are kept visually and semantically separate, and
        PublicExperience._show_next_fact for why they're also kept
        temporally separate -- deferred whenever the primary guide has
        spoken recently, so the two are never both up at once)."""
        self.scientist_bubble.set_text(text)
        if text:
            self._position_scientist_bubble()
            self.scientist_bubble.raise_()

    def clear_fact(self) -> None:
        self.scientist_bubble.set_text("")

    def pulse_explore_toggle(self, key: str) -> None:
        """See ExploreToggle.flash_highlight -- called right after a
        related tap in the scene itself (e.g. the candle) so the child's
        eye is drawn to the one real control that changes it, instead of
        a second in-scene picker duplicating it."""
        toggle = self._explore_toggles.get(key)
        if toggle is not None and not toggle.isHidden():
            toggle.flash_highlight()


    # -- explore controls (direct manipulation, see experiments.py) -----
    def _make_group_divider(self) -> QtWidgets.QFrame:
        """A thin, subtle vertical hairline between two control groups
        (see set_explore_controls) -- stretches to the full height of
        its row siblings (a toggle's own caption-plus-buttons column),
        not just the button row, so it separates "everything about Vent
        1" from "everything about Candles" rather than just the buttons.
        PANEL_BORDER, the same faint white this bar's own outer border
        already uses, so it reads as "part of this panel's own visual
        language" rather than a new, competing line style."""
        line = QtWidgets.QFrame(self.explore_panel)
        line.setFrameShape(QtWidgets.QFrame.NoFrame)
        line.setFixedWidth(2)
        line.setStyleSheet(f"background: {PANEL_BORDER}; border: none;")
        return line

    def set_explore_controls(self, controls: list) -> None:
        """Build one ExploreToggle per available control, with a thin
        divider between each pair (see _make_group_divider). Called
        once, at construction, with whatever `available_explore_
        controls()` found in this study's own manifest -- never a fixed
        UI list."""
        for toggle in self._explore_toggles.values():
            self._explore_layout.removeWidget(toggle)
            toggle.deleteLater()
        for divider in self._explore_dividers:
            self._explore_layout.removeWidget(divider)
            divider.deleteLater()
        self._explore_toggles = {}
        self._explore_dividers = []
        self._explore_insert_index = 0
        # Both real openings (Vent 1/vod, Vent 2/voc) get the same
        # "air can move here" color language -- INERT (grey) for a shut
        # state, AIRFLOW (cyan) for an open one -- instead of one of them
        # being fan-styled and the other falling back to generic ACCENT
        # (fire-orange, already meaning flame/candles elsewhere on
        # screen). This colors the *opening's own state* (a real factor
        # value), not a claim about measured effect size -- Vent 2's own
        # near-null airspeed effect is still reported honestly by the
        # noticeable_delta machinery in the science card, not hidden by
        # this button's color. Candles and Door aren't openings a fan
        # blows through, so they keep the plain ACCENT default.
        _VENT_CHECKED_COLORS = {
            "vent1": [AIRFLOW, INERT, AIRFLOW],   # open, closed, fan on
            "vent2": [AIRFLOW, INERT],            # open, closed
        }
        for control in controls:
            if self._explore_toggles:   # every group but the first gets a divider first
                divider = self._make_group_divider()
                self._explore_layout.insertWidget(self._explore_insert_index, divider)
                self._explore_insert_index += 1
                self._explore_dividers.append(divider)
            checked_colors = _VENT_CHECKED_COLORS.get(control.key)
            toggle = ExploreToggle(control.label, control.icon, control.options,
                                    checked_colors=checked_colors)
            toggle.value_changed.connect(
                lambda value, key=control.key: self.explore_changed.emit(key, value))
            self._explore_layout.insertWidget(self._explore_insert_index, toggle)
            self._explore_insert_index += 1
            self._explore_toggles[control.key] = toggle
        self.explore_panel.setVisible(bool(controls))
        # Same "hint isn't a floor" fix as ExploreToggle's own
        # setMinimumWidth (see its own comment): explore_panel relies on
        # root's QVBoxLayout only ever consulting its minimumSizeHint(),
        # which root's own reserved right-hand column (for the stat
        # panel) can and does compress below at 800x600 -- with each
        # ExploreToggle now a hard floor, that compression no longer
        # overlaps buttons *within* one group, but the outer
        # explore_layout still had no hard floor of its own and would
        # still compress *between* groups instead (Vent 1 sized
        # correctly, then Candles starting inside Vent 1's own span --
        # measured, not assumed). Setting this widget's own minimum
        # width to what its layout actually needs is what makes the
        # ancestor stop handing it a too-small rect in the first place.
        self.explore_panel.setMinimumWidth(self._explore_layout.minimumSize().width())
        # At 800x600 even the best available x (see _position_stat_group's
        # own button_rights clamp) can't fully clear explore_panel's real
        # minimum width without pushing stat_panel off the right edge of
        # the window -- the two panels' true minimum widths genuinely
        # don't both fit at that size. Explicit, not incidental: whichever
        # widget happens to paint later must not be the one that decides
        # whether a real control stays usable, so explore_panel is always
        # the one on top where they do overlap -- the real buttons stay
        # fully visible and clickable, and only the (already decorative,
        # non-interactive) stat_panel background is the one partly
        # covered.
        self.explore_panel.raise_()

    def set_explore_values(self, values: dict) -> None:
        """Snap each toggle's checked state to an explicit value per
        control key -- used to keep the toggle UI in sync with whatever
        real, possibly multi-factor scenario is actually loaded (see
        PublicExperience._sync_explore_controls_to_case), including
        right after a change composes on top of the others rather than
        resetting them (PublicExperience._on_explore_changed)."""
        for key, value in values.items():
            toggle = self._explore_toggles.get(key)
            if toggle is not None:
                toggle.set_value_silently(value)

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
        # Same "hint isn't a floor" fix as explore_panel's own
        # setMinimumWidth (see its own comment): without this,
        # games_home_panel only offered root's layout a soft
        # minimumSizeHint(), which root's own reserved right-hand column
        # (for the stat panel) compressed below at 800x600 -- and unlike
        # explore_panel's row, a QGridLayout squeezed that way doesn't
        # just clip each tile's text, it lays two tiles' real geometry on
        # top of each other (measured: a real, functional bug, not just
        # cosmetic -- a tap in the overlap could hit the wrong tile's
        # callback). Left at 0 (no floor) whenever there are no tiles, so
        # a hidden/empty grid never forces a phantom minimum width.
        self.games_home_panel.setMinimumWidth(
            self._games_grid.minimumSize().width() if tiles else 0)

    def set_games_home_visible(self, visible: bool) -> None:
        self.games_home_panel.setVisible(visible)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # lang_en_button/lang_de_button used to sit fixed to exit_button's
        # own left here -- now the top row of stat_panel (see
        # _position_stat_group), which reads their real size but doesn't
        # move them until that call below.
        self.exit_button.move(self.width() - self.exit_button.width() - 16, 16)
        self.countdown.setGeometry(0, 0, self.width(), self.height())
        strip_width = min(460, max(320, self.width() - 80))
        self.stages.resize(strip_width, self.stages.height())
        self.stages.move((self.width() - strip_width) // 2,
                         self.height() - self.stages.height() - 6)
        self._position_stat_group()
        self._position_scientist()
        self._position_mascot()

    # Extra clearance below the thermometer's computed bottom edge, on
    # top of bottom_limit's own margin -- bottom_limit already stops it
    # short of the stage strip (StageStrip really is visible, not dead
    # chrome, during EXPERIMENT/REVEAL/SCIENCE -- confirmed by checking
    # its actual isVisible()/geometry() in those phases, not assumed),
    # so this is a second, smaller safety margin, not the only one.
    # 104 -> 70 -> 20: the thermometer read as squeezed and, worse,
    # after the first trim its painted tube (_paint_tube) silently
    # stopped drawing at all below ~310px tall -- caught in a
    # screenshot, not by reasoning about the numbers -- because the
    # `top` this reserve gets subtracted from used to be measured before
    # the meter chips' own layout had actually settled (see
    # set_thermometer_visible's deferred re-position), overstating how
    # little room this control has to work with.
    _PROBE_DOCK_RESERVE = 20
    # Carved out of the thermometer's own height so whichever mascot
    # corner-hugs the true bottom-right (_position_mascot -- the worker,
    # since the corner swap; originally tuned for Dr. Funke before she
    # moved to bottom-left) has enough room -- 45px got her to her own
    # native design resolution (Scientist._DESIGN_H = 108) back when she
    # stood here; per a second round of direct feedback ("still too
    # small relative to the engineer... move her so both mascots sit at
    # the same vertical baseline"), 136 is what it actually takes for a
    # 191px-tall occupant to stand at the primary mascot's own baseline
    # (see _position_mascot's own margin=32) without overlapping the
    # thermometer -- not the exact minimum: a few px of real slack so
    # the pocket isn't immediately re-squeezed by the next sibling-
    # geometry change that shifts `top` by a few px. Both mascots share
    # the same 191px height (see Mascot/Scientist's own _DESIGN_H-driven
    # setFixedSize calls), so this value carries over unchanged now that
    # the worker occupies the corner it was tuned against. The
    # thermometer's own height floor (max(220, ...) below) is what stops
    # this from ever being raised further without also shrinking the
    # thermometer below its own established minimum.
    _BOTTOM_RIGHT_RESERVE_PX = 136

    # Horizontal/vertical padding inside stat_panel, and the gap between
    # its three stacked rows (language toggle / meters / thermometer) --
    # plain visual spacing, not tied to any of the reserve constants
    # below (those guard against *other* widgets, this is just "don't
    # paint flush against this panel's own border").
    _STAT_PANEL_PADDING = 16
    _STAT_PANEL_ROW_SPACING = 12
    # Each meter chip's fixed width inside this panel -- deliberately
    # much narrower than MeterChip's own un-flat 210px floor (see
    # MeterChip's own flat= docstring for why: at 210px each, two side
    # by side plus the thermometer made this whole panel wide enough to
    # visually collide with the explore-control row below it, a real,
    # measured overlap at 800x600). 115, not narrower still: the caption
    # elides cleanly at any width (see MeterChip's own resizeEvent), so
    # what actually floors this is the *value*/*phrase* labels, which
    # still word-wrap -- checked via MeterChip.heightForWidth against
    # every real reading/phrase this dataset can show, including the
    # longest ("Sehr starker Luftstrom", the German very-strong-airflow
    # phrase), for a still-reasonable 2-3 line wrap rather than an
    # excessively tall chip a narrower floor would force.
    _METER_CHIP_FLAT_WIDTH = 115

    def _position_stat_group(self) -> None:
        """Lay out the language toggle, both meter chips, and the
        thermometer as one visually connected panel, top to bottom --
        previously three separately-floating elements (a real, repeated
        complaint that they didn't read as belonging together). None of
        the three groups is reparented into stat_panel (see its own
        construction comment in __init__): every widget here stays a
        direct child of this overlay, positioned in *this widget's* own
        coordinate space exactly as before this panel existed, so every
        other reader of that geometry elsewhere (the scientist bubble's
        thermometer-column clamp, the worker mascot's thermometer-bottom
        safety net) keeps working unmodified -- stat_panel is nothing
        but the shared background/border wrapped around wherever this
        method puts them.

        A hidden row costs no space (a hidden meter row lets the
        thermometer ride up to meet the language row above it) -- the
        same "hidden means zero space, not a still-reserved gap" rule
        the rest of this file follows.
        """
        self.layout().activate()
        pad = self._STAT_PANEL_PADDING
        spacing = self._STAT_PANEL_ROW_SPACING
        # The exit button sits fixed at (width-60, 16, 44, 48) regardless
        # of anything else in this panel -- its right edge lines up with
        # the panel's own right edge, so the panel's top always clears
        # its bottom, the same fallback the old thermometer-only version
        # of this method used only when the meters were hidden; it's the
        # only anchor available now that the meters live inside this
        # panel rather than being laid out above it.
        top = self.exit_button.geometry().bottom() + 12
        bottom_limit = (self.stages.y() if self.stages.y() > 0 else self.height()) - 20

        lang_w = self.lang_en_button.width() + 6 + self.lang_de_button.width()
        content_w = lang_w

        chip_w = self._METER_CHIP_FLAT_WIDTH
        meters_visible = self.temperature_meter.isVisible()
        if meters_visible:
            # heightForWidth, not sizeHint -- these chips word-wrap now
            # (see MeterChip's own flat= docstring), so their *preferred*
            # (unwrapped) size isn't what they'll actually need at the
            # fixed width this panel gives them.
            temp_h = self.temperature_meter.heightForWidth(chip_w)
            flow_h = self.airflow_meter.heightForWidth(chip_w)
            meters_w = chip_w + 12 + chip_w
            content_w = max(content_w, meters_w)

        panel_w = max(content_w, self.thermometer.width()) + 2 * pad

        # Always right-docked, a fixed inset from the window edge -- an
        # earlier version docked this beside the room's real right wall
        # instead (a physical-geometry anchor, matching the fan/candle
        # markers' own convention), and it actively caused the exact
        # "control bar runs into the stat panel" bug this whole method
        # exists to avoid: that wall sits at a roughly constant *fraction*
        # of the window width, so its anchored x grew slower than
        # explore_panel's own right edge (which grows in lockstep with
        # the window, one-to-one) as the window widened -- fine at
        # 800x600, a measured 3px gap at 1024px, and a full, measured
        # -73px/-265px overlap at 1280/1920px. A fixed inset from the
        # window's own right edge grows at the same one-to-one rate
        # explore_panel's own margin-driven right edge already does, so
        # the gap between them stays constant instead of closing as the
        # window widens.
        x = self.width() - panel_w - 24
        button_rights = [b.geometry().right() for b in self._buttons if b.geometry().right() > 0]
        # explore_panel's own right edge, same reasoning as button_rights
        # above: ExploreToggle now enforces a hard minimum width per
        # group (see its own comment -- Vent 1's 3-button row can no
        # longer be squeezed thinner than real buttons need), so
        # explore_panel can legitimately be wider than the column this
        # panel used to assume it stayed inside. Without this, a visible
        # (measured, not assumed) ~97px overlap opens up between the
        # Door group's own buttons and this panel at 800x600 the moment
        # Vent 1 needs its real width -- exactly the class of bug this
        # method's own button_rights check already exists to avoid,
        # just for a row it didn't know about yet.
        if self.explore_panel.isVisible() and self.explore_panel.geometry().right() > 0:
            button_rights.append(self.explore_panel.geometry().right())
        if button_rights:
            x = max(x, max(button_rights) + 12)
        # int(): `anchor` (and so `x`) can be a numpy.float64 -- extent
        # values come straight from the store's own array metadata -- and
        # QWidget.move() rejects that type outright.
        panel_x = int(min(x, self.width() - panel_w - 4))

        y = top + pad
        self.lang_en_button.move(panel_x + (panel_w - lang_w) // 2, y)
        self.lang_de_button.move(
            self.lang_en_button.x() + self.lang_en_button.width() + 6, y)
        y += self.lang_en_button.height() + spacing

        if meters_visible:
            meters_x0 = panel_x + (panel_w - meters_w) // 2
            row_h = max(temp_h, flow_h)
            self.temperature_meter.resize(chip_w, temp_h)
            self.temperature_meter.move(meters_x0, y)
            self.airflow_meter.resize(chip_w, flow_h)
            self.airflow_meter.move(meters_x0 + chip_w + 12, y)
            y += row_h + spacing

        if self.thermometer.isVisible():
            # 380 -> 460: direct feedback that the thermometer read as
            # squeezed -- raises the ceiling this can grow to when the
            # sibling geometry (meters/stage strip) actually leaves the
            # room; the max()/min() clamp already keeps it from
            # overflowing whatever room there really is.
            #
            # - self._BOTTOM_RIGHT_RESERVE_PX: the worker mascot's own
            # pocket (see _position_mascot) is everything below wherever
            # this height formula ends up putting the panel's bottom
            # edge, down to the screen edge -- so without reserving space
            # for him here explicitly, this formula (which knows nothing
            # about him) would keep happily growing the thermometer into
            # room he needs, capping him at a sliver. Reserved
            # unconditionally, not just while he's visible: the pocket
            # must not jump around (and re-trigger this same squeeze)
            # every time visibility flips.
            height = max(220, min(460, bottom_limit - self._PROBE_DOCK_RESERVE
                                  - self._BOTTOM_RIGHT_RESERVE_PX - y))
            self.thermometer.resize(self.thermometer.width(), height)
            self.thermometer.move(panel_x + (panel_w - self.thermometer.width()) // 2, y)
            y += height

        self.stat_panel.setGeometry(panel_x, top, panel_w, y + pad - top)

    def _position_scientist(self) -> None:
        """Dr. Funke now stands in the true bottom-left corner (swapped
        with the worker mascot -- see the class-level swap note in
        __init__, and _position_mascot for his new bottom-right spot):
        same 32px margin on both the side and bottom edges the primary
        mascot's own placement always used in this corner, so both
        mascots' feet still sit on the exact same baseline
        (self.height() - margin) regardless of which of them is on
        which side."""
        if not self.scientist.isVisible():
            return
        margin = 32
        scientist_y = self.height() - self.scientist.height() - margin
        self.scientist.move(margin, max(0, scientist_y))
        self._position_scientist_bubble()

    def _position_scientist_bubble(self) -> None:
        """To Dr. Funke's *right* now that she corner-hugs the true
        bottom-left (swapped -- see _position_scientist), mirroring the
        worker mascot's own old bubble-to-the-right formula for this
        corner exactly, including the stat-panel-column clamp (a wide
        bubble reaching right could still clip its left edge, docked at
        the opposite corner -- checked against stat_panel as a whole,
        not just the thermometer, since the language toggle/meters can
        occupy that corner even when the thermometer itself is hidden)
        and the stage-strip clamp (a tall wrapped bubble dipping into
        that row)."""
        if not self.scientist_bubble.isVisible():
            return
        margin = 32
        available = self.width() - self.scientist.width() - 3 * margin
        if self.stat_panel.isVisible():
            available = min(available, self.stat_panel.x() - self.scientist.x()
                            - self.scientist.width() - 12)
        hint = self.scientist_bubble.sizeHint()
        width = max(240, min(hint.width(), available))
        # sizeHint()'s height assumes its own preferred width; recompute
        # at the width the bubble will actually get, or long text clips.
        self.scientist_bubble.resize(width, self.scientist_bubble.heightForWidth(width))
        y = self.height() - self.scientist_bubble.height() - margin - 6
        if self.stages.isVisible():
            y = min(y, self.stages.y() - self.scientist_bubble.height() - 6)
        self.scientist_bubble.move(self.scientist.x() + self.scientist.width() + 12, y)

    def _position_mascot(self, _settling: bool = False) -> None:
        """The worker mascot now stands in the true bottom-right corner
        (swapped with Dr. Funke -- see the class-level swap note in
        __init__), a full mirror of her own old bottom-right placement:
        same 32px margin on both the side and bottom edges, and the
        same safety-net clamp against stat_panel (the language toggle,
        meters, and thermometer all dock along this same edge) for a
        window size _BOTTOM_RIGHT_RESERVE_PX wasn't tuned against --
        stat_panel's own geometry, not just the thermometer's, since it
        stays current even when the thermometer itself is hidden but the
        meters/language row above it still occupy this corner.

        His speech bubble goes to his *left* here -- there's no room to
        his right once he's flush against the screen edge -- bounded by
        the same 32px screen margin rather than the thermometer's own x
        (a tighter thermometer-relative bound caused a real,
        screenshotted regression the one time it was tried for this
        corner: see the git history around Dr. Funke's own bubble, back
        when she stood here). Freely overlapping the scene area is fine:
        PublicExperience._show_next_fact's own timing deferral is what
        keeps the two bubbles from ever being up at once, not their
        bounding boxes staying apart."""
        margin = 32
        x = self.width() - self.mascot.width() - margin
        y = self.height() - self.mascot.height() - margin
        y = max(y, self.stat_panel.geometry().bottom() + 2)
        self.mascot.move(x, y)

        if not self.bubble.isVisible():
            return
        mascot = self.mascot.geometry()
        available_w = max(0, mascot.x() - 6 - margin)
        # PREDICTION's own three *tall* choice buttons (the one screen
        # button_row is this wide -- see _render_prediction) reach far
        # enough right, *and* far enough down to share this bubble's own
        # vertical band (unlike OBSERVE's shorter, higher-sitting Help/
        # Play row), that this bubble's usual (mascot-margin-only) width
        # let it extend clean across the button row rather than stopping
        # short of it -- measured, not assumed: the bubble's sizeHint at
        # the old bound spanned x=52-586/y=389-490 against a button row
        # at x=40-470/y=338-440, real ghosted-text-behind-the-buttons
        # overlap, not just visually close. The vertical check matters:
        # an earlier version of this fix keyed off button_row.count()
        # alone and clamped (then hid) the bubble on *every* screen with
        # any button_row content at all, including OBSERVE's ambient fact
        # bubble, whose row sits at y=354-424 while the mascot/bubble
        # band starts at mascot.y()=544 -- no real vertical overlap
        # there, and nothing needed clamping (a real, measured
        # regression: bubble geometry stops updating once actually
        # applied wrongly, since this is a per-attempt decision, not a
        # per-screen one).
        row_bottom = 0
        row_right = 0
        for i in range(self.button_row.count()):
            button = self.button_row.itemAt(i).widget()
            if button is None:
                continue
            # button_row's own geometry() understates how far its
            # buttons actually reach -- BigButton's real, measured size
            # can exceed the box a squeezed row layout reports owning
            # (each button still gets its own full sizeHint/minimum,
            # same "hard floor beats a too-small container" behaviour
            # ExploreToggle's own overlap fix relies on) -- the real
            # buttons' own mapped rects are the thing to actually check,
            # not the row's own geometry(), which can also simply be
            # stale the instant add_button() just inserted them (QLayout
            # only recomputes on an activation pass; even an explicit
            # .activate() call measured as *not* enough here, unlike
            # _position_stat_group's own equivalent case).
            top_left = button.mapTo(self, QtCore.QPoint(0, 0))
            row_right = max(row_right, top_left.x() + button.width())
            row_bottom = max(row_bottom, top_left.y() + button.height())
        vertical_overlap = row_bottom > mascot.y()
        if row_right > 0 and vertical_overlap:
            available_w = min(available_w, mascot.x() - 6 - row_right - 12)
        if not _settling and self.button_row.count() > 0:
            # One deferred re-run once the event loop has actually had a
            # turn to settle -- the staleness noted above means the very
            # first pass right after add_button() can still read every
            # button at a stale (0,0)-ish position, understating row_
            # bottom enough that vertical_overlap itself computes False
            # on this pass even on PREDICTION, where it's actually True
            # (a real, measured regression: gating the re-arm on this
            # same pass's vertical_overlap meant a wrong first reading
            # never got a chance to correct itself). Always scheduled
            # whenever there's a row to (re-)check, not just when this
            # pass already thinks it needs to -- the same fix
            # _reposition_thermometer_once_settled already established
            # for this exact class of problem. `_settling` stops this
            # scheduling itself again forever once the deferred
            # correction actually runs.
            if self._mascot_reposition_timer is None:
                # Parented to self, not a bare QtCore.QTimer.singleShot():
                # see _thermometer_reposition_timer's own comment on the
                # crash a bare one-shot risks after teardown.
                self._mascot_reposition_timer = QtCore.QTimer(self)
                self._mascot_reposition_timer.setSingleShot(True)
                self._mascot_reposition_timer.timeout.connect(
                    lambda: self._position_mascot(_settling=True))
            self._mascot_reposition_timer.start(0)
        available_h = max(0, self.height() - mascot.y() - 2)
        width = max(20, min(self.bubble.sizeHint().width(), available_w))
        x = mascot.x() - 6 - width
        # Position and size are always kept current -- even when the
        # result below is too narrow to be legible and this hides the
        # bubble -- so anything that reads .geometry() regardless of
        # visibility (this file's own tests included) never sees a stale
        # rect left over from before the row existed or from a wider
        # screen. Hiding, not skipping the update, is what avoids a real
        # unreadable-sliver bubble (PREDICTION's own real gap can be as
        # little as ~28px at 800x600, too narrow for legible text at any
        # font scale -- fit_to() only shrinks height-for-a-given-width)
        # or, worse, forcing it back into overlapping the row -- the
        # same "don't render something unreadable" principle this file's
        # other guards already follow (MeterChip elides rather than
        # showing garbage; Thermometer._paint_tube draws nothing sooner
        # than a corrupted shape). Not a real loss when it happens: "Make
        # a guess..."/a fact line is encouragement, not information the
        # child needs -- and say() unhides it again next time there's
        # room, since it always calls bubble.set_text(), which re-shows.
        self.bubble.fit_to(width, available_h)
        self.bubble.move(int(x), mascot.y())
        # fit_to() (not a bypassed, uncapped heightForWidth()) either
        # way -- a real, measured regression the first version of this
        # hide path had: skipping fit_to() to hide meant nothing capped
        # the height, and a narrow width you'd want to hide anyway can
        # still wrap into a very tall box (over 700px on the SCIENCE
        # phase's own real text, at a 122px width) -- geometry has to
        # stay sane even when hidden, since this file's own tests (and
        # potentially other code) read .geometry() regardless of
        # isVisible().
        if width < self.bubble._min_width and row_right > 0 and vertical_overlap:
            self.bubble.hide()

    # 12px clear gap above whichever mascot is taller, on top of the 32px
    # margin _position_mascot/_position_scientist already move each
    # mascot up from the true bottom edge -- without this a mascot's own
    # top edge would land exactly on PublicScene's new reserved-strip
    # boundary (see mascot_band_height_px), a hairline match that a
    # pixel of rounding either way could turn into a visible overlap.
    _MASCOT_BAND_GAP_PX = 12

    def mascot_band_height_px(self) -> int:
        """How many pixels PublicScene must reserve at its own bottom
        edge (see PublicScene.set_bottom_reserve_px) so neither mascot's
        real, fixed-size footprint ever overlaps the room/flame it
        renders -- both mascots' own 32px bottom margin plus a small
        clear gap, against whichever of the two is taller (they're equal
        today, but this reads off their real geometry rather than
        assuming that stays true). Called once, after both mascots exist
        -- their sizes are fixed for the life of the app, so this never
        needs to be recomputed."""
        margin = 32
        return max(self.mascot.height(), self.scientist.height()) + margin + self._MASCOT_BAND_GAP_PX

    # -- helpers --------------------------------------------------------
    def say(self, text: str, mood: str = IDLE) -> None:
        # TEMPORARY diagnostic instrumentation for the real, live SIGABRT
        # crash under investigation -- see PublicExperience._diag_depths'
        # own comment. self._say_depth is set up in __init__.
        self._say_depth += 1
        if self._say_depth > 1:
            print(f"[DIAG] REENTRANT: PublicOverlay.say depth={self._say_depth}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
        else:
            print("[DIAG] enter PublicOverlay.say", file=sys.stderr)
        try:
            self._say_inner(text, mood)
        finally:
            print(f"[DIAG] exit PublicOverlay.say (was depth {self._say_depth})", file=sys.stderr)
            self._say_depth -= 1

    def _say_inner(self, text: str, mood: str) -> None:
        self._last_say_at = time.monotonic()
        self.bubble.set_text(text)
        self.mascot.set_mood(mood)
        self._position_mascot()
        # Belt-and-braces on top of PublicExperience._show_next_fact's
        # own defer-before-starting check: that only stops a *new* fact
        # from beginning near a fresh say(), it does nothing about one
        # already showing when the buddy suddenly needs to speak (a
        # child's tap mid-fact triggers exactly that). Cutting her off
        # here is what makes "the two mascots never both have bubbles up
        # at once" hold in both directions, not just the common one.
        #
        # Deferred via a zero-ms parented QTimer, not called inline: say()
        # itself usually runs from deep inside a signal/slot chain (a tap
        # or toggle), and clearing scientist_bubble synchronously means
        # another widget's set_text()/updateGeometry()/update() executes
        # while that same call stack is still unwinding -- reentrant Qt
        # event handling this app has hit a real, hard-to-diagnose crash
        # from before. A 0ms singleShot still fires before the next
        # visible frame (no perceptible delay) but moves the work onto
        # its own fresh top-level event-loop turn instead. Parented to
        # self, not a bare QTimer.singleShot, for the same reason
        # PublicExperience._defer's own docstring gives: a bare one-shot's
        # callback still fires even after this widget is gone.
        if self.scientist_bubble.isVisible():
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.scientist_bubble.set_text(""))
            timer.start(0)

    def seconds_since_say(self) -> float:
        """How long since the primary mascot last spoke -- the signal
        Dr. Funke's own ambient timer defers on (see PublicExperience.
        _show_next_fact), so her facts never land while the buddy is
        also mid-utterance. Tracked here, at say()'s single choke point,
        rather than at each of its ~40 call sites across experience.py:
        the primary guide's own invitation text stays visible for the
        whole of OBSERVE once set (say() has no auto-hide), so a plain
        `bubble.isVisible()` check would block her forever -- this is
        specifically "how long since a *fresh* utterance", not "is a
        bubble currently on screen"."""
        return time.monotonic() - self._last_say_at

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
        # The meters used to live in root's own layout, which reflowed
        # everything below them automatically on a visibility change --
        # now that they're one row of the manually-positioned stat_group
        # (see _position_stat_group), hiding/showing them has to trigger
        # that reflow explicitly, or the thermometer (and the panel's own
        # bounding box) stays sized for whichever state was last laid out.
        self._position_stat_group()

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
        # New content at the card's *current* width doesn't fire a
        # resizeEvent on its own (see Card._update_minimum_height's own
        # comment on why this is needed at all) -- called after layout()
        # has actually placed the new children, so heightForWidth()
        # reads their real wrapped height, not stale content from
        # whatever was shown here before.
        self.card.layout().activate()
        self.card._update_minimum_height()

    def shutdown(self) -> None:
        self.mascot.stop()
        self.scientist.stop()
