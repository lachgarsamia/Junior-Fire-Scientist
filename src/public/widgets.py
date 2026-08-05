"""Exhibition chrome: big buttons, live meters, explanation cards.

Sized for a science-museum display -- touch targets well past the 44 px
minimum, type readable from a couple of metres, and enough contrast to
survive a bright room. Styling is inline QSS rather than theme.py tokens
because the public experience deliberately runs one fixed dark look over
the fire, independent of the researcher app's light/dark setting.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from public import kid_language as kid

# Touch/readability floor for anything a visitor presses. The width floor
# has to leave three choice buttons side by side on a 1280-wide display
# (3 x 190 + spacing + margins), so it is smaller than it looks -- real
# buttons grow past it whenever there is room.
BUTTON_MIN_HEIGHT = 66
BUTTON_MIN_WIDTH = 190


def enable_height_for_width(widget) -> None:
    """Let a word-wrapped QLabel actually report its wrapped height.

    QLabel sets wordWrap but leaves QSizePolicy.hasHeightForWidth() False,
    so a layout sizes it from a single-line hint and clips every line
    after the first few. This is the standard fix and it has to be applied
    to the container too, or the container's own hint stays wrong.
    """
    policy = widget.sizePolicy()
    policy.setHeightForWidth(True)
    widget.setSizePolicy(policy)

ACCENT = "#FF7A18"
PANEL_BG = "rgba(14, 18, 26, 232)"
PANEL_BORDER = "rgba(255, 255, 255, 38)"
TEXT = "#F3F6FA"
TEXT_DIM = "#A9B4C4"
# The app's one other bright hue besides ACCENT -- already used ad hoc for
# the focus ring, the countdown glyph, and the mascot's visor (mascot.py),
# but never given a shared name. Named here so a second "this is delightful,
# not just informational" UI element (the Games entry) can reuse it on
# purpose instead of picking a new color out of the air.
DELIGHT = "#FFD166"
# Same cyan as scene.py's vent-activity glow (the patch that appears over
# the vent while vod == 2) -- reused here, not picked fresh, so the fan's
# ON state reads as "air is moving" instead of borrowing ACCENT's
# fire-orange, which already means flame/candles elsewhere on screen.
AIRFLOW = "#7DD3FC"
# Neutral, hue-less "off/inert" checked state -- paired with AIRFLOW on
# the fan toggle so OFF never reads as a dimmer version of some other
# meaning (it isn't cooling, isn't flame-adjacent, it's just off).
INERT = "#6B7280"


class BigButton(QtWidgets.QPushButton):
    """A large, touch-friendly choice button. `primary=True` gives the
    filled accent treatment used for the one action we want pressed."""

    def __init__(self, text: str, icon: str = "", primary: bool = False,
                 tall: bool = False, parent=None):
        """`tall` is the prediction-choice treatment: the icon sits on its
        own line above the words, so a child reads the picture first and
        the label second. Deliberately a variant of this button rather
        than a new widget -- same focus ring, same touch floor."""
        label = f"{icon}\n{text}" if (icon and tall) else (f"{icon}  {text}" if icon else text)
        super().__init__(label, parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        height = 112 if tall else BUTTON_MIN_HEIGHT
        # Pinned, not just minimum. setMinimumHeight alone loses to the
        # application-level QSS the researcher theme installs (which sets
        # its own QPushButton min-height), and with only a minimum the
        # layout stretched a lone button to fill the column. A touch
        # target must be exactly as big as it claims to be.
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAccessibleName(text)
        # Grow into whatever width is available, but never below the
        # touch floor -- three of these must still fit side by side.
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                           QtWidgets.QSizePolicy.Fixed)
        fill = ACCENT if primary else "rgba(24, 30, 42, 235)"
        hover = "#FF9440" if primary else "rgba(44, 54, 72, 245)"
        fg = "#1A1005" if primary else TEXT
        self.setStyleSheet(f"""
            QPushButton {{
                background: {fill};
                color: {fg};
                border: 2px solid {'transparent' if primary else PANEL_BORDER};
                border-radius: 18px;
                padding: 14px 22px;
                font-size: 19px;
                font-weight: 600;
                min-height: {height - 28}px;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ padding-top: 17px; padding-bottom: 11px; }}
            QPushButton:focus {{ border: 3px solid #FFD166; }}
            QPushButton:disabled {{ background: rgba(30,36,48,180); color: {TEXT_DIM}; }}
        """)
        if tall:
            self.setMinimumWidth(210)
            self.setStyleSheet(self.styleSheet().replace("font-size: 19px", "font-size: 22px"))


class ExploreToggle(QtWidgets.QWidget):
    """A real, physical-style control: OFF|ON, 1|2, etc. -- direct
    manipulation rather than a form field.

    Pressing an option flips its own pressed/checked visual state as part
    of Qt's normal button handling, *before* `value_changed` is even
    emitted -- so the control always acknowledges a tap instantly,
    independent of however long the caller takes to react to the signal
    (loading this dataset's cached scenarios is a few milliseconds, but
    the widget makes no assumption about that).
    """

    value_changed = QtCore.pyqtSignal(object)

    def __init__(self, label: str, icon: str, options, parent=None, checked_colors=None):
        """`checked_colors`, when given, is one color per option (same
        order as `options`) for that option's own checked state -- e.g.
        the fan toggle uses [INERT, AIRFLOW] so OFF and ON read as two
        different things rather than one generic "selected" hue. Falls
        back to ACCENT for every option, the original one-hue-fits-all
        treatment every other control still uses."""
        super().__init__(parent)
        self._flash_timer = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        caption = QtWidgets.QLabel(f"{icon} {label}")
        caption.setAlignment(QtCore.Qt.AlignCenter)
        caption.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 13px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(caption)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._values = []
        for index, option in enumerate(options):
            color = checked_colors[index] if checked_colors else ACCENT
            hover_color = QtGui.QColor(color).lighter(115).name()
            button = QtWidgets.QPushButton(f"{option.icon} {option.label}")
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setMinimumHeight(64)
            button.setMinimumWidth(88)
            button.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(24, 30, 42, 235); color: {TEXT};
                    border: 2px solid {PANEL_BORDER}; border-radius: 14px;
                    font-size: 16px; font-weight: 600; padding: 6px 12px;
                }}
                QPushButton:checked {{
                    background: {color}; color: #1A1005; border: 2px solid transparent;
                }}
                QPushButton:hover {{ background: rgba(44, 54, 72, 245); }}
                QPushButton:checked:hover {{ background: {hover_color}; }}
                QPushButton:focus {{ border: 3px solid #FFD166; }}
            """)
            self._group.addButton(button, index)
            self._values.append(option.value)
            row.addWidget(button)
        layout.addLayout(row)

        if self._values:
            self._group.button(0).setChecked(True)
        self._group.idClicked.connect(self._on_clicked)

    def _on_clicked(self, index: int) -> None:
        self.value_changed.emit(self._values[index])

    def flash_highlight(self, duration_ms: int = 900) -> None:
        """A brief glow around this control -- draws the eye here right
        after a related tap elsewhere in the scene (e.g. tapping the
        candle itself), so "here's the real control" never means
        building a second floating picker widget. A stylesheet swap +
        parented restore timer, the same pattern TitleBanner.flash()
        already established, not a geometry animation -- this widget is
        laid out by its parent's QHBoxLayout, and animating geometry
        directly would fight the very next layout pass."""
        if self._flash_timer is None:
            self._flash_timer = QtCore.QTimer(self)
            self._flash_timer.setSingleShot(True)
            self._flash_timer.timeout.connect(lambda: self.setStyleSheet(""))
        self.setStyleSheet("background: rgba(255, 209, 102, 70); border-radius: 14px;")
        self._flash_timer.start(duration_ms)

    def set_value_silently(self, value) -> None:
        """Reset the visual state without emitting -- used when another
        control's change means this one reverts to its default (see
        experiments.ExploreControl's one-factor-at-a-time convention)."""
        if value in self._values:
            button = self._group.button(self._values.index(value))
            block = self._group.blockSignals(True)
            button.setChecked(True)
            self._group.blockSignals(block)


class TemperatureSparkline(QtWidgets.QWidget):
    """A tiny live history trail for a pinned probe spot -- real sampled
    values only (see PublicExperience._update_thermometer's "pinned"
    branch), scaled to their own min/max so even a modest rise is
    visible rather than flat against a 0-470 C axis. Deliberately not a
    matplotlib chart (a plot widget would drag the researcher-grade
    rendering machinery in for a handful of pixels); a plain QPainter
    strip repainted on each new sample is plenty cheap at this size.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self._values: list = []

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(130, 24)

    def set_values(self, values: list) -> None:
        self._values = list(values)
        self.update()

    def paintEvent(self, event) -> None:
        if len(self._values) < 2:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QtGui.QPainter) -> None:
        w, h = self.width(), self.height()
        pad_x, pad_y = 4.0, 3.0
        lo, hi = min(self._values), max(self._values)
        span = max(hi - lo, 1e-6)
        n = len(self._values)
        step = (w - 2 * pad_x) / max(n - 1, 1)
        points = []
        for i, v in enumerate(self._values):
            x = pad_x + i * step
            frac = (v - lo) / span
            y = (h - pad_y) - frac * (h - 2 * pad_y)
            points.append(QtCore.QPointF(x, y))
        painter.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 2.0))
        painter.drawPolyline(QtGui.QPolygonF(points))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(ACCENT))
        painter.drawEllipse(points[-1], 3.0, 3.0)


class Thermometer(QtWidgets.QWidget):
    """A large, always-on vertical thermometer: real measured temperature,
    coloured by the same bands kid_language uses for its words ("Warm
    air", "Hot", "Flame"...), so a child never has to tap anything first
    to see that the scene has cooler and hotter regions.

    Three reading modes, set by the caller (PublicExperience), never by
    this widget:
      - "mean": the whole-scene average, refreshed every frame -- the
        default, so the thermometer means something before any tap.
      - "point": a single tapped location's value, held static until the
        next tap or a mode change.
      - "pinned": a "watch this spot" location, resampled over time, with
        a short real-value trail so the child can see it evolve.
    No value here is ever interpolated or invented -- every number this
    widget displays arrived from PublicScene's own frame-array lookups.
    """

    _DISPLAY_MAX_C = 470.0   # this dataset's flame ceiling (kid_language's calibration note)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._value: float = None
        # The rendered fill/bulb sweeps toward _value rather than
        # snapping to it -- purely a readability animation (Phase 3
        # spec: "the animation may interpolate visually... but the final
        # value must come from the actual simulation array"). The number
        # in _value_label is never animated; it is set to the exact real
        # target the instant set_reading() is called.
        self._display_value: float = None
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(280)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_value)
        # 200 -> 232 -> 184 -> 216: the painted tube below (see
        # _paint_tube) was thin relative to this widget's own footprint
        # and read as a flat HUD chip rather than an instrument (Design
        # Review §6) -- first widened, then pulled back in to fit inside
        # the scene's own reserved never-drawn-into column
        # (PublicScene.SCENE_WIDTH_FRAC) once that existed, then widened
        # again once direct feedback said that column-fitted size still
        # read as squeezed (the column itself grew wider at the same
        # time -- see SCENE_WIDTH_FRAC's own comment). All positioning
        # that depends on this width (PublicOverlay._position_thermometer)
        # already reads self.thermometer.width() rather than a hardcoded
        # constant, so this is a pure size change, not a layout rewrite.
        self.setFixedWidth(216)
        # Left at 220 (PublicOverlay._position_thermometer's own floor),
        # not raised further: a first attempt at 400 forced this widget
        # taller than _position_thermometer's own bottom_limit-derived
        # safety math allowed at the moment top happens to be large (the
        # meter chips' text length shifts it run to run) -- a real,
        # screenshotted overflow where the readout box ran off the
        # bottom of an 800x600 screen. That formula's own max() cap is
        # what actually has to grow to make this taller safely; see its
        # comment in overlay.py.
        self.setMinimumHeight(220)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._caption = QtWidgets.QLabel("🌡️ TEMPERATURE")
        self._caption.setAlignment(QtCore.Qt.AlignCenter)
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 14px; font-weight: 700; letter-spacing: 0.5px;"
            "background: transparent;")
        layout.addWidget(self._caption)

        # A simple 3-tier mood (🥶/🙂/🥵, kid_language.mood_emoji) -- the
        # big, immediately-readable part: a child reads "hot" from the
        # picture a beat before the number. Deliberately the coarser
        # 3-tier read here rather than temperature_icon()'s 6-tier band
        # icon (already shown elsewhere, e.g. the meter chip) -- "how does
        # this feel?" wants one simple answer, not a precise category.
        self._emoji_label = QtWidgets.QLabel("🌡️")
        self._emoji_label.setAlignment(QtCore.Qt.AlignCenter)
        self._emoji_label.setStyleSheet("font-size: 64px; background: transparent;")
        layout.addWidget(self._emoji_label)

        layout.addStretch(1)   # the tube itself is painted in paintEvent

        self._value_label = QtWidgets.QLabel("—")
        self._value_label.setAlignment(QtCore.Qt.AlignCenter)
        self._value_label.setStyleSheet(
            f"color: {TEXT}; font-size: 48px; font-weight: 800;"
            f"background: rgba(10, 13, 20, 210); border: 2px solid {DELIGHT};"
            "border-radius: 14px; padding: 4px 6px;")
        layout.addWidget(self._value_label)

        self._phrase_label = QtWidgets.QLabel("")
        self._phrase_label.setAlignment(QtCore.Qt.AlignCenter)
        self._phrase_label.setWordWrap(True)
        self._phrase_label.setStyleSheet(f"color: {ACCENT}; font-size: 14px; font-weight: 700; "
                                         "background: transparent;")
        layout.addWidget(self._phrase_label)

        # A live sparkline, not text -- "the child should visually SEE the
        # temperature increasing" (see the pinned-spot history this
        # renders). set_trail() feeds it the exact same real sampled
        # values a text trail would have shown; nothing here interpolates.
        self._sparkline = TemperatureSparkline()
        layout.addWidget(self._sparkline)

    def set_reading(self, value_c: float, caption: str = "") -> None:
        self._value = value_c
        self._value_label.setText(f"{value_c:.0f}°C")
        self._caption.setText(caption or "🌡️ TEMPERATURE")
        self._emoji_label.setText(kid.mood_emoji(value_c))
        self._phrase_label.setText(kid.temperature_phrase(value_c))
        start = self._display_value if self._display_value is not None else value_c
        self._anim.stop()
        self._anim.setStartValue(float(start))
        self._anim.setEndValue(float(value_c))
        self._anim.start()

    def _on_anim_value(self, value) -> None:
        self._display_value = float(value)
        self.update()

    def set_trail(self, values: list) -> None:
        """A tiny live sparkline of real sampled values for a pinned
        spot -- empty for a plain tap, where there is nothing to trail
        yet."""
        self._sparkline.set_values(values)

    def clear(self) -> None:
        self._anim.stop()
        self._value = None
        self._display_value = None
        self._value_label.setText("—")
        self._caption.setText("🌡️ TEMPERATURE")
        self._emoji_label.setText("🌡️")
        self._phrase_label.setText("")
        self._sparkline.set_values([])
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint_tube(painter)
        finally:
            painter.end()

    # Tick values along the tube -- round numbers spanning this dataset's
    # real calibration ceiling (_DISPLAY_MAX_C), not an arbitrary scale.
    _TICKS = (400, 300, 200, 100, 0)

    def _paint_tube(self, painter: QtGui.QPainter) -> None:
        # 34/20 -> 48/28: Design Review §6 -- "reads as a flat HUD chip,
        # not an instrument" -- room_wall_anchor already docks this
        # correctly against the room's real wall (verified independently
        # against PublicExperience._update_markers), so the fix here is
        # purely more visual weight for the tube/bulb themselves.
        tube_w = 48
        bulb_r = 28.0
        top = 132
        bottom_limit = self.height() - 118
        bottom = bottom_limit - (2 * bulb_r + 4)
        if bottom <= top:
            return
        cx = self.width() / 2
        rect = QtCore.QRectF(cx - tube_w / 2, top, tube_w, bottom - top)

        # A recognisable bulb-and-tube shape, like a real thermometer --
        # not just an information panel (see the class docstring's "make
        # it a toy" note).
        # The bulb colour and fill height both key off the animated
        # display value, not the raw target -- so the sweep looks like
        # one coherent instrument moving, not a bar animating under a
        # colour that already jumped.
        shown = self._display_value if self._display_value is not None else self._value
        fill_color = (QtGui.QColor(kid.temperature_color(shown))
                      if shown is not None else QtGui.QColor(58, 66, 80))
        # The *fill* itself is a fixed blue(bottom)->red(top) gradient
        # spanning the tube's whole real range, not just the currently-
        # shown value -- so warming up always sweeps through the same
        # cool-to-warm colours in the same places, rather than jumping
        # between flat band colours the instant a threshold is crossed.
        # Built once per paint from the same band boundaries temperature_
        # color() uses (kid.temperature_gradient_stops), so it can never
        # disagree with the bulb/phrase beside it.
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        for pos, color in kid.temperature_gradient_stops(self._DISPLAY_MAX_C):
            gradient.setColorAt(pos, QtGui.QColor(color))
        bulb_center = QtCore.QPointF(cx, rect.bottom() + bulb_r + 2)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(fill_color)
        painter.drawEllipse(bulb_center, bulb_r, bulb_r)

        # A warm, toy-instrument outline (DELIGHT, the app's one other
        # bright hue -- see its own module docstring) instead of the
        # plain translucent-white PANEL_BORDER every flat panel already
        # uses: this is the one widget the child should read as a real
        # object, not another info chip (Design Review §6).
        painter.setPen(QtGui.QPen(QtGui.QColor(DELIGHT), 3.0))
        painter.setBrush(QtGui.QColor(10, 13, 20, 210))
        painter.drawRoundedRect(rect, tube_w / 2, tube_w / 2)

        # Tick marks + numbers, so the tube reads as a real graduated
        # instrument rather than a plain filled bar.
        font = painter.font()
        font.setPointSizeF(max(9.0, font.pointSizeF() * 0.85))
        painter.setFont(font)
        for value in self._TICKS:
            frac = min(1.0, value / self._DISPLAY_MAX_C)
            y = rect.bottom() - frac * rect.height()
            painter.setPen(QtGui.QPen(QtGui.QColor(TEXT_DIM), 1.4))
            painter.drawLine(QtCore.QPointF(rect.left() - 9, y), QtCore.QPointF(rect.left() - 1, y))
            painter.drawText(QtCore.QRectF(0, y - 8, rect.left() - 12, 16),
                             QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, str(value))

        painter.setPen(QtGui.QPen(fill_color.lighter(140), 1.4))
        painter.setBrush(fill_color)
        painter.drawEllipse(bulb_center, bulb_r - 4.0, bulb_r - 4.0)

        if shown is None:
            return
        fraction = max(0.0, min(1.0, shown / self._DISPLAY_MAX_C))
        fill_h = rect.height() * fraction
        fill = QtCore.QRectF(rect.x() + 4, rect.bottom() - fill_h, rect.width() - 8,
                             max(8.0, fill_h))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawRoundedRect(fill, 11, 11)


class MeterChip(QtWidgets.QFrame):
    """A live readout: icon, a big value, and the plain-language phrase
    for it. Values come from the scene's real measurements; this widget
    only formats them."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {PANEL_BG};
                border: 1px solid {PANEL_BORDER};
                border-radius: 16px;
            }}
        """)
        self.setMinimumWidth(210)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(2)

        self._caption = QtWidgets.QLabel(caption)
        self._caption.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px; "
                                    "letter-spacing: 1px; border: none;")
        layout.addWidget(self._caption)

        self._value = QtWidgets.QLabel("—")
        self._value.setStyleSheet(f"color: {TEXT}; font-size: 30px; font-weight: 700; border: none;")
        layout.addWidget(self._value)

        self._phrase = QtWidgets.QLabel("")
        self._phrase.setStyleSheet(f"color: {ACCENT}; font-size: 16px; font-weight: 600; border: none;")
        layout.addWidget(self._phrase)

    def set_reading(self, value_text: str, phrase: str, icon: str = "") -> None:
        self._value.setText(f"{icon} {value_text}".strip())
        self._phrase.setText(phrase)
        self.setAccessibleDescription(f"{self._caption.text()}: {value_text}, {phrase}")

    def set_caption(self, caption: str) -> None:
        """Re-set the caption after construction -- the language toggle's
        only route to updating this one label, which set_reading() never
        touches (see PublicExperience._on_language_requested)."""
        self._caption.setText(caption)


class Card(QtWidgets.QFrame):
    """A translucent panel: the explanation card, the question card, and
    the science card are all this with different content."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {PANEL_BG};
                border: 1px solid {PANEL_BORDER};
                border-radius: 22px;
            }}
            QLabel {{ border: none; }}
        """)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(32, 15, 32, 15)
        # Tight: the science card stacks a title, a hero block, a table
        # and a caption, and at 800x600 a looser rhythm pushes the last
        # row out of the card.
        self._layout.setSpacing(7)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        enable_height_for_width(self)

    def body(self) -> QtWidgets.QVBoxLayout:
        return self._layout

    @staticmethod
    def title_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {TEXT}; font-size: 28px; font-weight: 700;")
        enable_height_for_width(label)
        return label

    @staticmethod
    def block_label(text: str) -> QtWidgets.QLabel:
        """A multi-line block of measured values. Word wrap is off and
        the rows are pre-shortened to fit, so the numbers stay aligned in
        a column instead of reflowing into prose."""
        label = QtWidgets.QLabel(text)
        label.setWordWrap(False)
        label.setTextFormat(QtCore.Qt.PlainText)
        label.setStyleSheet(f"color: {TEXT}; font-size: 15px;")
        return label

    @staticmethod
    def text_label(text: str, dim: bool = False) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        color = TEXT_DIM if dim else TEXT
        label.setStyleSheet(f"color: {color}; font-size: 19px; line-height: 150%;")
        enable_height_for_width(label)
        return label


class StageStrip(QtWidgets.QWidget):
    """A four-step "where am I" strip: WATCH -> GUESS -> TEST -> DISCOVER.

    Deliberately not a progress bar in the score/percentage/timing sense
    -- it answers one question a child actually asks, "what am I doing
    now?" -- but IS painted as a connected step-track (a filled line
    linking small dots, labels below) rather than four separate
    label-only segments. Real testing feedback: with no connecting line
    or dot, four bold labels in a row read as a row of nav tabs a child
    tried tapping, and (being WA_TransparentForMouseEvents) nothing
    happened -- confusing rather than "dim enough to ignore." A visibly
    connected track is a more universally-read "you are here on a fixed
    path" shape, closer to a map or a checkout progress bar than a menu.
    """

    STAGES = (("👀", "WATCH"), ("🤔", "GUESS"), ("🔥", "TEST"), ("🔎", "DISCOVER"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = -1
        self.setFixedHeight(46)
        self.setMinimumWidth(360)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def set_active(self, index: int) -> None:
        """-1 hides the strip (attract/intro, where there is no journey
        under way yet)."""
        if index != self._active:
            self._active = index
            self.setVisible(index >= 0)
            self.update()

    _DOT_R = 5.0
    _TRACK_Y = 10.0

    def paintEvent(self, event) -> None:
        if self._active < 0:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            n = len(self.STAGES)
            step = self.width() / n
            centers = [step * (i + 0.5) for i in range(n)]

            # The connecting track first, so the dots/labels paint on
            # top of it -- one continuous line reads as a single path,
            # not n separate items.
            painter.setPen(QtGui.QPen(QtGui.QColor("#3A4454"), 3.0))
            painter.drawLine(QtCore.QPointF(centers[0], self._TRACK_Y),
                             QtCore.QPointF(centers[-1], self._TRACK_Y))
            if self._active > 0:
                painter.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 3.0))
                painter.drawLine(QtCore.QPointF(centers[0], self._TRACK_Y),
                                 QtCore.QPointF(centers[self._active], self._TRACK_Y))

            font = painter.font()
            font.setPointSizeF(max(9.5, font.pointSizeF() * 0.8))
            font.setBold(True)
            painter.setFont(font)
            for i, (icon, label) in enumerate(self.STAGES):
                done = i <= self._active
                current = i == self._active
                color = QtGui.QColor(ACCENT if current else (TEXT_DIM if done else "#4A5568"))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(color)
                r = self._DOT_R * (1.4 if current else 1.0)
                painter.drawEllipse(QtCore.QPointF(centers[i], self._TRACK_Y), r, r)
                painter.setPen(color)
                painter.drawText(
                    QtCore.QRectF(i * step, self._TRACK_Y + 8, step, self.height() - self._TRACK_Y - 8),
                    QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, f"{icon} {label}")
        finally:
            painter.end()


class TitleBanner(QtWidgets.QLabel):
    """The big prompt across the top of the scene."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet(self._BASE_QSS)
        self._flash_timer = None
        self._flash_previous = ""

    _BASE_QSS = ("color: #FFFFFF; font-size: 26px; font-weight: 600;"
                 "background: rgba(10, 13, 20, 200); border-radius: 18px; padding: 16px 28px;")
    _FLASH_QSS = ("color: #1A1005; font-size: 26px; font-weight: 700;"
                  f"background: {ACCENT}; border-radius: 18px; padding: 16px 28px;")

    def set_prompt(self, text: str) -> None:
        """A legitimate new prompt always wins over a pending flash
        restore. Without cancelling the timer here, a WHOOSH flash
        started during EXPERIMENT (a couple of seconds, see flash()'s
        duration) could still fire _end_flash() after the phase had
        already moved on to REVEAL and called set_prompt("") -- silently
        re-showing the *old* experiment prompt on top of a screen that
        was supposed to have no banner at all (a real, reproducible bug
        found via automated overlap checking, not a hypothetical one).
        """
        if self._flash_timer is not None and self._flash_timer.isActive():
            self._flash_timer.stop()
            self.setStyleSheet(self._BASE_QSS)
        self.setText(text)
        self.setVisible(bool(text))

    def flash(self, text: str, duration_ms: int = 2600) -> None:
        """Briefly take over the banner in the accent colour, then fall
        back to whatever prompt was there. Used to make a detected event
        (the smoke reaching the ceiling) impossible to miss -- the guide
        says it, this makes a child look up.

        The restore runs on a QTimer *parented to this widget*, not
        QTimer.singleShot: a bare singleShot closure keeps calling back
        into the banner after Qt has destroyed it, which aborts the
        process. Same rule the kiosk filter and the mascot's animation
        timer already follow -- a timer must not outlive its owner.
        """
        if self._flash_timer is None:
            self._flash_timer = QtCore.QTimer(self)
            self._flash_timer.setSingleShot(True)
            self._flash_timer.timeout.connect(self._end_flash)
        # Only capture what to restore to when nothing is already flashing.
        # A second flash arriving while the first is still showing (a rapid
        # fan-toggle-toggle-toggle at a kiosk) must not capture the first
        # flash's own text as "the real prompt" -- that stale text would
        # then be what a later flash restores to, permanently displacing
        # the true underlying prompt with no way back to it. Restarting the
        # timer below still extends the display window on every flash, same
        # as before; only what gets restored afterward changes.
        if not self._flash_timer.isActive():
            self._flash_previous = self.text()
        self.setStyleSheet(self._FLASH_QSS)
        self.set_prompt(text)
        self._flash_timer.start(duration_ms)

    def _end_flash(self) -> None:
        self.setStyleSheet(self._BASE_QSS)
        self.set_prompt(self._flash_previous)


class BarCompare(QtWidgets.QWidget):
    """Two labelled bars, scaled to the larger value.

    The point is that a child reads the *shape* before the decimals: a
    6x difference has to look like a 6x difference. Painted rather than
    charted, because a real plotting widget would drag matplotlib into
    the overlay for two rectangles.
    """

    ROW_HEIGHT = 22
    COMPACT_ROW_HEIGHT = 21
    LABEL_WIDTH = 92
    VALUE_WIDTH = 104

    def __init__(self, label_a: str, value_a: float, label_b: str, value_b: float,
                 unit: str, decimals: int, compact: bool = False, parent=None):
        """Bars are scaled within this comparison only. Two BarCompare
        widgets on the same card therefore never share a scale, which is
        the point: air speed in m/s and temperature in °C are not
        commensurable, and drawing them against a common axis would imply
        a relationship that does not exist.

        `compact` is the visual demotion used for a secondary finding --
        shorter rows and smaller type, still legible at 800x600.
        """
        super().__init__(parent)
        self._rows = ((label_a, float(value_a)), (label_b, float(value_b)))
        self._unit = unit
        self._decimals = decimals
        self._compact = compact
        self._row_height = self.COMPACT_ROW_HEIGHT if compact else self.ROW_HEIGHT
        self.setFixedHeight(self._row_height * 2 + 6)
        self.setMinimumWidth(340)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                           QtWidgets.QSizePolicy.Fixed)
        self.setAccessibleName("Measured comparison")
        self.setAccessibleDescription(
            "; ".join(f"{label}: {value:.{decimals}f} {unit}" for label, value in self._rows))

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QtGui.QPainter) -> None:
        peak = max((v for _, v in self._rows), default=0.0)
        track_x = self.LABEL_WIDTH
        track_w = max(40, self.width() - self.LABEL_WIDTH - self.VALUE_WIDTH)

        font = painter.font()
        font.setPointSizeF(max(9.5 if self._compact else 11.0,
                               font.pointSizeF() * (0.82 if self._compact else 0.95)))
        painter.setFont(font)

        for i, (label, value) in enumerate(self._rows):
            y = i * self._row_height + 3
            bar = QtCore.QRectF(track_x, y + 4, track_w,
                                self._row_height - (11 if self._compact else 14))

            painter.setPen(QtGui.QColor(TEXT_DIM))
            painter.drawText(QtCore.QRectF(0, y, self.LABEL_WIDTH - 10, self._row_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, label)

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, 26))
            painter.drawRoundedRect(bar, 6, 6)

            # A zero-length bar would read as missing data rather than as
            # a small value, so every non-zero measurement keeps a stub.
            fraction = (value / peak) if peak > 0 else 0.0
            filled = QtCore.QRectF(bar)
            filled.setWidth(max(6.0, bar.width() * fraction) if value > 0 else 0.0)
            # The secondary comparison is drawn in a cool tone rather than
            # the hero's accent orange, so a glance ranks the two findings
            # before either is read.
            highlight = ACCENT if not self._compact else "#5AA9E6"
            painter.setBrush(QtGui.QColor(highlight if i else "#5B6B84"))
            painter.drawRoundedRect(filled, 6, 6)

            painter.setPen(QtGui.QColor(TEXT))
            painter.drawText(
                QtCore.QRectF(self.width() - self.VALUE_WIDTH, y, self.VALUE_WIDTH, self._row_height),
                QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight,
                f"{value:.{self._decimals}f} {self._unit}")


class HeroMetric(QtWidgets.QWidget):
    """The science card's headline finding: what changed most, by how
    much, and a bar comparison making the size of it obvious."""

    def __init__(self, headline: str, bars: BarCompare, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QtWidgets.QLabel(headline)
        title.setWordWrap(True)
        title.setStyleSheet(f"color: {ACCENT}; font-size: 20px; font-weight: 700;")
        enable_height_for_width(title)
        layout.addWidget(title)
        layout.addWidget(bars)


class SecondaryMetric(QtWidgets.QWidget):
    """A demoted version of HeroMetric: the second real finding, drawn
    smaller and in a cooler colour so the hierarchy is legible before any
    number is read. Its bars carry their own scale (see BarCompare)."""

    def __init__(self, headline: str, bars: BarCompare, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        title = QtWidgets.QLabel(headline)
        title.setWordWrap(True)
        title.setStyleSheet("color: #8FC5EC; font-size: 16px; font-weight: 600;")
        enable_height_for_width(title)
        layout.addWidget(title)
        layout.addWidget(bars)


# Note: an earlier version faded cards in with a QGraphicsOpacityEffect.
# It was removed deliberately. The effect starts the widget at opacity 0,
# so any interruption before the animation finishes -- a card repopulated
# mid-fade, or simply an event loop that does not tick -- leaves the card
# permanently invisible. An unattended exhibit that sometimes shows no
# explanation at all is a far worse outcome than an instant swap, and the
# instant swap also reads as more responsive under a visitor's finger.
