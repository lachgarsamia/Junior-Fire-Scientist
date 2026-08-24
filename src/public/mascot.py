"""The guide: a small QPainter-drawn character with a speech bubble.

Drawn with QPainter (like schematic.py's flames and door icons) rather
than as image assets, so it scales cleanly to any exhibition display and
follows the theme without shipping binaries. Animated on its own ~20 Hz
timer -- deliberately *not* on the scene's 30 Hz interpolation timer, so
mascot repaints never contend with the matplotlib blitting path.

The mascot is a guide, not the protagonist: it sits in a corner, is
roughly 100 px tall, and never occludes the fire.
"""

from __future__ import annotations

import math

from PyQt5 import QtCore, QtGui, QtWidgets

# Mood drives the eyes and the bob amplitude only -- the silhouette is
# constant, so the character reads as one person throughout.
IDLE = "idle"
POINTING = "pointing"
SURPRISED = "surprised"
THINKING = "thinking"
# Expressive states added for the exhibit. Each is driven by a real
# simulation or story event (see experience.MOOD_FOR_BEAT and the phase
# handlers), never by a timer -- the guide reacting to nothing is what
# makes a mascot feel like decoration instead of a companion.
CURIOUS = "curious"        # something is starting: ignition, a run beginning
WATCHING = "watching"      # a phenomenon is developing (smoke gathering)
EXCITED = "excited"        # the result is in
EXPLAINING = "explaining"  # walking through the science

# Moods that lean forward / bob harder. Kept as data so _paint stays a
# lookup rather than a chain of comparisons.
_ANIMATED_MOODS = frozenset({POINTING, SURPRISED, EXCITED, CURIOUS})
_WIDE_EYE_MOODS = frozenset({SURPRISED, EXCITED})
_ARM_UP_MOODS = frozenset({POINTING, WATCHING, EXPLAINING})

_HELMET = "#E8552D"
_HELMET_DARK = "#B93C1B"
_FACE = "#F5C9A6"
_BODY = "#2C4A73"
_VISOR = "#FFD166"


class Mascot(QtWidgets.QWidget):
    """A friendly fire-scientist guide. `set_mood()` changes expression;
    the bob animation runs continuously so the exhibit never looks
    frozen."""

    # The design is drawn against this fixed canvas (the widget's original
    # 120x130 footprint every coordinate in _paint below was authored
    # for), then uniformly scaled via painter.scale() to whatever size the
    # widget actually is -- see Scientist's own _DESIGN_W/_DESIGN_H for
    # the same pattern. Lets setFixedSize grow the mascot (per supervisor
    # feedback -- more prominent, more legible from across a room) without
    # every one of _paint's ~25 hand-placed coordinates needing to become
    # relative to self.width()/height() by hand.
    _DESIGN_W = 120.0
    _DESIGN_H = 130.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mood = IDLE
        self._phase = 0.0
        # 120x130 -> 144x156 -> 176x191: same 120:130 aspect ratio as the
        # design canvas throughout (scale_x == scale_y, so painter.scale()
        # below never distorts the character), per repeated feedback that
        # he still read as too small even after the first bump -- sized
        # while he still stood in the bottom-left corner, with real free
        # room and nothing pinned tightly against him there (he has since
        # swapped corners with Dr. Funke -- see PublicOverlay._position_
        # mascot and the class-level swap note in PublicOverlay.__init__
        # -- but the size itself, tuned against his old corner's room,
        # still fits his new one: _BOTTOM_RIGHT_RESERVE_PX exists
        # specifically to carve out the same amount of room for whichever
        # mascot stands there).
        self.setFixedSize(176, 191)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # 20 Hz

    def set_mood(self, mood: str) -> None:
        if mood != self._mood:
            self._mood = mood
            self.update()

    def stop(self) -> None:
        """Stop the animation timer -- called when the experience is torn
        down, so a hidden mascot isn't repainting forever."""
        self._timer.stop()

    def _tick(self) -> None:
        self._phase += 0.08
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    def _paint(self, painter: QtGui.QPainter) -> None:
        painter.save()
        scale = min(self.width() / self._DESIGN_W, self.height() / self._DESIGN_H)
        painter.scale(scale, scale)
        amplitude = 4.0 if self._mood in _ANIMATED_MOODS else 2.0
        bob = math.sin(self._phase) * amplitude
        cx = self._DESIGN_W / 2.0
        top = 18 + bob
        # A slight lean, so an engaged mood reads at a glance even before
        # the face is legible from across a room.
        if self._mood in (CURIOUS, WATCHING):
            painter.save()
            painter.translate(cx, top + 60)
            painter.rotate(-5.0)
            painter.translate(-cx, -(top + 60))

        painter.setPen(QtCore.Qt.NoPen)

        # body / shoulders
        painter.setBrush(QtGui.QColor(_BODY))
        body = QtCore.QRectF(cx - 34, top + 62, 68, 58)
        painter.drawRoundedRect(body, 22, 22)

        # face
        painter.setBrush(QtGui.QColor(_FACE))
        painter.drawEllipse(QtCore.QPointF(cx, top + 44), 28, 27)

        # helmet: dome + front crest + brim
        painter.setBrush(QtGui.QColor(_HELMET))
        dome = QtCore.QRectF(cx - 32, top + 8, 64, 56)
        painter.drawChord(dome, 0, 180 * 16)
        painter.setBrush(QtGui.QColor(_HELMET_DARK))
        painter.drawRoundedRect(QtCore.QRectF(cx - 36, top + 32, 72, 11), 5, 5)
        painter.setBrush(QtGui.QColor(_VISOR))
        painter.drawEllipse(QtCore.QPointF(cx, top + 22), 8, 8)

        # eyes
        painter.setBrush(QtGui.QColor("#22303F"))
        eye_y = top + 46
        if self._mood in _WIDE_EYE_MOODS:
            painter.drawEllipse(QtCore.QPointF(cx - 10, eye_y), 5, 6)
            painter.drawEllipse(QtCore.QPointF(cx + 10, eye_y), 5, 6)
        elif self._mood == THINKING:
            # half-lidded: a flatter pair of ovals reads as pondering
            painter.drawEllipse(QtCore.QPointF(cx - 10, eye_y), 4.0, 2.2)
            painter.drawEllipse(QtCore.QPointF(cx + 10, eye_y), 4.0, 2.2)
        else:
            painter.drawEllipse(QtCore.QPointF(cx - 10, eye_y), 3.4, 4.2)
            painter.drawEllipse(QtCore.QPointF(cx + 10, eye_y), 3.4, 4.2)

        # raised brows for the "something is happening" moods
        if self._mood in (CURIOUS, SURPRISED, EXCITED):
            brow = QtGui.QPen(QtGui.QColor("#22303F"), 2.0)
            brow.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(brow)
            painter.setBrush(QtCore.Qt.NoBrush)
            for sign in (-1, 1):
                painter.drawArc(
                    QtCore.QRectF(cx + sign * 10 - 7, eye_y - 13, 14, 10), 0, 180 * 16)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor("#22303F"))

        # mouth
        pen = QtGui.QPen(QtGui.QColor("#8A4A32"), 2.2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        mouth = QtCore.QRectF(cx - 9, top + 52, 18, 12)
        if self._mood in _WIDE_EYE_MOODS:
            painter.setBrush(QtGui.QColor("#8A4A32"))
            radius = 7 if self._mood == EXCITED else 5
            painter.drawEllipse(QtCore.QPointF(cx, top + 59), radius, radius + 1)
        elif self._mood == EXPLAINING:
            # a small open mouth that pulses -- "I am talking to you"
            painter.setBrush(QtGui.QColor("#8A4A32"))
            height = 3.0 + 2.0 * abs(math.sin(self._phase * 1.6))
            painter.drawEllipse(QtCore.QPointF(cx, top + 58), 6.0, height)
        else:
            painter.drawArc(mouth, 200 * 16, 140 * 16)

        # raised arm: pointing at the scene, or gesturing while explaining
        if self._mood in _ARM_UP_MOODS:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(_FACE))
            painter.save()
            painter.translate(cx + 30, top + 78)
            swing = 6 if self._mood == POINTING else 3
            painter.rotate(-25 + math.sin(self._phase * 2) * swing)
            painter.drawRoundedRect(QtCore.QRectF(0, -6, 34, 12), 6, 6)
            painter.restore()

        if self._mood in (CURIOUS, WATCHING):
            painter.restore()

        painter.restore()


class SpeechBubble(QtWidgets.QWidget):
    """A rounded bubble with a tail, sized to its text. Text is set
    whole (no typewriter effect): at an exhibition, visitors read at very
    different speeds and animated text is the first thing that makes a
    display feel slow.

    Shared by the primary mascot and Dr. Funke (see PublicOverlay's own
    construction of each) -- one widget, not two near-duplicates, so
    their bubbles read as the same *kind* of thing even though they
    never appear at once (PublicExperience._show_next_fact defers her
    fact whenever the primary guide has spoken recently). Which
    character gets which constructor arguments below depends on which
    *corner* they currently stand in, not on who they are -- the two
    mascots have swapped corners once already (see PublicOverlay.
    __init__'s own swap note), and whichever pairing is current, the
    corner rules stay the same: `min_width`/`h_padding` tighter, and
    fit_to() (not resize()) used, for whoever is in the tight corner
    next to the thermometer's own sidebar (~124-150px total); the
    generous corner's occupant keeps the 280px/24px defaults and never
    calls fit_to(). `accent`: an optional border/background tint (a
    signature colour tied to *this* SpeechBubble instance's owner
    permanently, unlike the corner-dependent args, so their bubble stays
    recognizably theirs regardless of which corner they're in) so the
    two bubbles still read as visually distinct characters at a glance,
    not just distinguished by which one happens to be showing -- without
    forking the widget itself. None keeps the original plain styling
    exactly as it was. `tail_side`: "left" (default -- the owner stands
    to the bubble's left, the generous corner's own convention) or
    "right" for a bubble whose owner stands to its right instead (the
    tight corner's convention), mirroring the tail *and* the accent
    badge so both still point toward whoever is actually talking rather
    than out into empty space. `h_padding`: left/right text inset, 24 by
    default (the generous corner's own spacing) -- the tight corner's is
    smaller: between the 24px default padding on both sides plus the
    accent badge's own 16px, only ~60px was left for actual text there,
    forcing a real, screenshotted over-shrunk/clipped font for the
    longer facts. A smaller inset buys back real text width without
    touching font size at all."""

    def __init__(self, parent=None, min_width: int = 280, accent: str = None,
                 tail_side: str = "left", h_padding: int = 24):
        super().__init__(parent)
        self._text = ""
        self._font_scale = 1.0
        self._min_width = min_width
        self._accent = accent
        self._tail_side = tail_side
        self._h_padding = h_padding
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        # No setMinimumWidth here, deliberately -- min_width only feeds
        # sizeHint()'s own preference below. A widget-level minimum
        # fights the caller's own resize()/fit_to() the instant the
        # real available space drops under it: this exact bug already
        # shipped once (ThoughtBubble.setMinimumWidth, fixed by removing
        # it) and reappeared here when this class replaced ThoughtBubble
        # -- the tight corner's position (whichever mascot currently
        # stands there) leaves only ~124px next to it, under the 130
        # this constructor used to pass straight into setMinimumWidth,
        # and Qt's resize() silently snapping back up to that 130 pushed
        # the bubble's drawn width 6px past what the corner's own
        # positioning method actually computed. The caller is the only
        # one who knows the real available space, so it's the only one
        # allowed to set a floor.

    def set_text(self, text: str) -> None:
        self._text = text or ""
        self._font_scale = 1.0
        self.setVisible(bool(self._text))
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    def _font(self) -> QtGui.QFont:
        font = self.font()
        # 15.0 -> 17.0: per supervisor feedback, more legible speech-
        # bubble text for both mascots. fit_to()'s own shrink loop (down
        # to its 0.35 floor) is what keeps this from overflowing the
        # narrower of the two bubbles (Dr. Funke's sidebar pocket) --
        # re-verified against every fact/language after this bump, the
        # same check that first established that floor.
        font.setPointSizeF(max(17.0, font.pointSizeF() * 1.25) * self._font_scale)
        return font

    # Extra right-hand padding reserved for the small accent "spark"
    # badge paintEvent draws in the top-right corner when self._accent
    # is set (None -- the primary mascot's own bubble -- draws no badge
    # and needs none of this). Subtracted from the text-wrap width in
    # both sizeHint and heightForWidth, not just at paint time: those
    # two are what the caller (PublicOverlay._position_scientist_bubble)
    # actually sizes and positions the widget from, so if paintEvent
    # alone narrowed the text rect without this, the real text could
    # wrap onto a line heightForWidth never budgeted room for -- the
    # exact class of "computed size vs. drawn content disagree" bug this
    # whole file's other fit_to()/font-shrink additions exist to avoid.
    # 16 -> 10: tuned while Dr. Funke's bubble stood in the tight corner
    # (~124px total available, before the corner swap -- see PublicOverlay.
    # __init__'s own swap note), where every px this reserves was a px
    # fewer for actual text -- a smaller badge is still clearly visible
    # at this scale, and the value carries over unchanged now that her
    # bubble has more room to work with.
    _ACCENT_BADGE_W = 10

    def sizeHint(self) -> QtCore.QSize:
        if not self._text:
            return QtCore.QSize(0, 0)
        metrics = QtGui.QFontMetrics(self._font())
        width = min(560, max(self._min_width,
                             metrics.horizontalAdvance(self._text) + 2 * self._h_padding))
        return QtCore.QSize(width, self.heightForWidth(width))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        """Bubble height for a given width: the wrapped text plus the
        padding and the tail. The owner resizes this widget to whatever
        width is actually available, which is usually narrower than
        sizeHint()'s preferred width, so this has to be exact."""
        if not self._text:
            return 0
        metrics = QtGui.QFontMetrics(self._font())
        badge = self._ACCENT_BADGE_W if self._accent else 0
        content_w = max(40, width - 2 * self._h_padding - badge)
        rect = metrics.boundingRect(QtCore.QRect(0, 0, content_w, 10_000),
                                    QtCore.Qt.TextWordWrap, self._text)
        return rect.height() + 44

    # Called by whichever bubble stands in the tight corner (see
    # PublicOverlay._position_mascot / _position_scientist_bubble,
    # whichever currently applies -- the two mascots have swapped
    # corners once already, see PublicOverlay.__init__'s own swap note)
    # -- that corner is a fixed pocket, not elastic. ThoughtBubble, the
    # widget this replaced for Dr. Funke, needed the exact same kind of
    # shrink-to-fit after a real screenshot caught its own font
    # overflowing that pocket for German's longer compound-word facts.
    # 0.6 -> 0.35: the tight corner leaves only ~124x85px for the whole
    # bubble (tighter than the ~150-160px-wide spot this floor was
    # originally tuned for), and even at 0.6 the single longest German
    # fact still needed 108px of height in an 85px box -- a real,
    # screenshotted clip, not a hypothetical one. Measured every fact key
    # in both languages against the actual current width/badge/padding:
    # the worst case needs 0.45, so 0.35 leaves real margin rather than
    # landing exactly on today's specific numbers.
    def fit_to(self, width: int, max_height: int) -> None:
        self._font_scale = 1.0
        while self.heightForWidth(width) > max_height and self._font_scale > 0.35:
            self._font_scale -= 0.05
        self.resize(width, min(self.heightForWidth(width), max(0, max_height)))

    def paintEvent(self, event) -> None:
        if not self._text:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            body = QtCore.QRectF(0, 0, self.width(), self.height() - 12)
            path = QtGui.QPainterPath()
            path.addRoundedRect(body, 18, 18)
            # tail, pointing down toward whichever side its owner
            # actually stands on. "right" is the exact mirror image of
            # the original "left" points (x -> width - x), not a
            # separately hand-tuned shape, so both stay visually
            # identical apart from which way they lean.
            if self._tail_side == "left":
                p1, p2, p3 = 34, 30, 56
            else:
                w = body.width()
                p1, p2, p3 = w - 34, w - 30, w - 56
            tail = QtGui.QPainterPath()
            tail.moveTo(p1, body.bottom() - 2)
            tail.lineTo(p2, body.bottom() + 12)
            tail.lineTo(p3, body.bottom() - 2)
            tail.closeSubpath()
            path = path.united(tail)

            if self._accent:
                # A visible tinted border plus a background lightly
                # mixed toward the same colour -- distinct enough from
                # the primary mascot's plain neutral bubble (accent=None
                # keeps that exact original styling) to read as a
                # different character's voice at a glance, without the
                # two ever needing to be told apart by which one happens
                # to be on screen (they're never both up at once, see
                # PublicExperience._show_next_fact / PublicOverlay.say).
                accent = QtGui.QColor(self._accent)
                bg = QtGui.QColor(
                    int(18 + (accent.red() - 18) * 0.16),
                    int(22 + (accent.green() - 22) * 0.16),
                    int(30 + (accent.blue() - 30) * 0.16), 235)
                border = QtGui.QColor(accent)
                border.setAlpha(150)
                painter.setPen(QtGui.QPen(border, 2.0))
                painter.setBrush(bg)
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 40), 1.5))
                painter.setBrush(QtGui.QColor(18, 22, 30, 235))
            painter.drawPath(path)

            if self._accent:
                # A small "here's a thought" marker -- opposite corner
                # from the tail (top-right for a left-pointing tail,
                # top-left for a right-pointing one), the same spatial
                # logic a real thought-bubble/speech-bubble pair would
                # use. A plain drawn shape, not an emoji glyph: this app
                # already hit real missing-glyph fallbacks on this
                # exhibition build's fonts elsewhere (see story.py's own
                # CEILING_BEAT_ICON comment), so anything new here stays
                # vector-drawn rather than risk the same thing.
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(self._accent))
                # 20 -> 15, 6.0/2.6 -> 4.5/1.9: scaled down to match
                # _ACCENT_BADGE_W's own 16 -> 10 shrink, so the badge
                # still sits fully inside its own reserved strip instead
                # of bleeding into the text area right next to it.
                cx = body.right() - 15 if self._tail_side == "left" else body.left() + 15
                cy = body.top() + 15
                spark = QtGui.QPainterPath()
                for i in range(8):
                    angle = i * math.pi / 4
                    r = 4.5 if i % 2 == 0 else 1.9
                    point = QtCore.QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle))
                    (spark.moveTo(point) if i == 0 else spark.lineTo(point))
                spark.closeSubpath()
                painter.drawPath(spark)

            painter.setPen(QtGui.QColor("#F3F6FA"))
            painter.setFont(self._font())
            badge = self._ACCENT_BADGE_W if self._accent else 0
            badge_left = badge if self._tail_side == "right" else 0
            badge_right = badge if self._tail_side == "left" else 0
            text_rect = body.adjusted(self._h_padding + badge_left, 14,
                                      -self._h_padding - badge_right, -10)
            painter.drawText(text_rect,
                             QtCore.Qt.TextWordWrap | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                             self._text)
        finally:
            painter.end()


_SCI_SKIN = "#F0C9A0"
_SCI_HAIR = "#DCD7CB"
_SCI_HAIR_SHADE = "#C4BFB2"
_SCI_COAT = "#F2F1EC"
_SCI_COAT_SHADE = "#D6D4CB"
_SCI_COLLAR = "#7C93A8"  # a small cardigan-blue collar, her one cool colour
_SCI_ACCENT = "#E8552D"   # same fire-orange the main mascot's helmet uses


class Scientist(QtWidgets.QWidget):
    """Dr. Frieda Funke -- a second, ambient guide, visually distinct
    from the primary firefighter mascot (grey hair in a bun, round
    glasses, a lab coat, a small fire-orange pin tying her to the same
    accent colour the rest of the app already uses). She does not react
    to specific events the way the primary mascot's moods do --
    PublicExperience periodically gives her a real, simplified
    fire-science fact to share in her own SpeechBubble (see
    PublicOverlay.scientist_bubble; a second instance of the same
    widget the primary mascot uses, not a distinct bubble class),
    independent of anything the child just did -- but only during a
    genuine lull: _show_next_fact defers whenever the primary guide has
    spoken, or the child has interacted, in the last few seconds, so
    the two never both have something up at once. She is decoration
    with something to say, not a second narrator answering the scene."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        # She now stands in the true bottom-left corner (see
        # PublicOverlay._position_scientist and the class-level swap
        # note in PublicOverlay.__init__ -- she used to stand bottom-
        # right, next to the thermometer's own instrument sidebar,
        # until that spot's crowding risk against the experiment/room
        # area moved her here and gave it to the worker mascot instead).
        # Same 32px margin, same baseline either way.
        #
        # 100x128 -> 149x191: height matches Mascot's own 191 exactly
        # (real parity, not just "bigger again"); width scaled to match
        # (84/108 * 191 = 148.5 -> 149) so the *same* uniform scale
        # (painter.scale() below, min(width/_DESIGN_W, height/_DESIGN_H))
        # drives both -- not 191 tall but still width-bottlenecked at her
        # old 100, which would waste the new height as empty padding
        # rather than actually rendering her bigger. This sizing predates
        # the corner swap (established back when she stood bottom-right,
        # tight against the thermometer) but carries over unchanged: her
        # bottom-left spot has at least as much room as bottom-right ever
        # did, so nothing here needed re-tuning for the move.
        self.setFixedSize(149, 191)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(60)

    def stop(self) -> None:
        """Stop the animation timer -- called when the experience is
        torn down, the same reason Mascot.stop() exists."""
        self._timer.stop()

    def _tick(self) -> None:
        self._phase += 0.045
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            self._paint(painter)
        finally:
            painter.end()

    # The design is drawn against this fixed canvas, then uniformly
    # scaled to whatever size the widget actually is -- lets the
    # sidebar shrink her (see PublicOverlay._position_scientist) without
    # every coordinate below needing to be relative to self.width()/
    # height() by hand.
    _DESIGN_W = 84.0
    _DESIGN_H = 108.0

    def _paint(self, painter: QtGui.QPainter) -> None:
        painter.save()
        scale = min(self.width() / self._DESIGN_W, self.height() / self._DESIGN_H)
        painter.scale(scale, scale)
        # A slower, smaller sway than the primary mascot's bob -- she is
        # standing and thinking, not actively reacting to the scene.
        sway = math.sin(self._phase) * 1.5
        cx = self._DESIGN_W / 2.0 + sway
        top = 8.0
        painter.setPen(QtCore.Qt.NoPen)

        # bun, drawn before everything else so the fringe (below) overlaps
        # its lower edge -- reads as hair pulled up and back, not a
        # separate ball floating above her head
        painter.setBrush(QtGui.QColor(_SCI_HAIR_SHADE))
        painter.drawEllipse(QtCore.QPointF(cx, top + 10), 8.5, 8.5)

        # coat: a simple A-line silhouette, deliberately not the primary
        # mascot's rounded-rect torso, so the two never read as the same
        # character in a different colour
        painter.setBrush(QtGui.QColor(_SCI_COAT))
        coat = QtGui.QPainterPath()
        coat.moveTo(cx - 15, top + 56)
        coat.lineTo(cx + 15, top + 56)
        coat.lineTo(cx + 25, top + 98)
        coat.lineTo(cx - 25, top + 98)
        coat.closeSubpath()
        painter.drawPath(coat)
        painter.setBrush(QtGui.QColor(_SCI_COAT_SHADE))
        painter.drawRect(QtCore.QRectF(cx - 1.5, top + 56, 3, 42))  # coat seam

        # a soft diagonal fold-highlight down the coat's own left half --
        # the one bit of dimensional shading she had none of before,
        # keeping the A-line from reading as a single flat paper cutout
        painter.setBrush(QtGui.QColor(255, 255, 255, 45))
        fold = QtGui.QPainterPath()
        fold.moveTo(cx - 12, top + 59)
        fold.lineTo(cx - 6, top + 59)
        fold.lineTo(cx - 15, top + 95)
        fold.lineTo(cx - 21, top + 95)
        fold.closeSubpath()
        painter.drawPath(fold)

        # a small cardigan-blue V collar -- her one cool colour against an
        # otherwise warm palette, and a softer "settled in her chair"
        # detail than a bare coat neckline
        painter.setBrush(QtGui.QColor(_SCI_COLLAR))
        collar = QtGui.QPainterPath()
        collar.moveTo(cx - 9, top + 56)
        collar.lineTo(cx, top + 66)
        collar.lineTo(cx + 9, top + 56)
        collar.closeSubpath()
        painter.drawPath(collar)

        # lapels, folded open over the collar -- what actually reads as
        # "lab coat" rather than just "coat": a bare V neckline with no
        # lapels looks like a plain robe at this scale. Layered on top of
        # the collar's own outer edges, extending a little past its tip,
        # the way a real folded lapel sits proud of the shirt underneath.
        painter.setBrush(QtGui.QColor(_SCI_COAT_SHADE))
        for side in (-1, 1):
            lapel = QtGui.QPainterPath()
            lapel.moveTo(cx + side * 15, top + 56)
            lapel.lineTo(cx + side * 4, top + 71)
            lapel.lineTo(cx + side * 9.5, top + 57)
            lapel.closeSubpath()
            painter.drawPath(lapel)

        # a small fire-orange pin -- the one warm colour link to the rest
        # of the app's palette, standing in for a helmet she doesn't wear
        painter.setBrush(QtGui.QColor(_SCI_ACCENT))
        painter.drawEllipse(QtCore.QPointF(cx + 11, top + 60), 4.0, 4.0)

        # neck, bridging face and collar so the head doesn't read as
        # glued directly onto the coat
        painter.setBrush(QtGui.QColor(_SCI_SKIN))
        painter.drawRect(QtCore.QRectF(cx - 5, top + 50, 10, 8))

        # face
        painter.setBrush(QtGui.QColor(_SCI_SKIN))
        painter.drawEllipse(QtCore.QPointF(cx, top + 38), 19, 18)
        # soft cheek/jaw shading -- the face was a single flat skin-tone
        # disc before; a low-opacity underlay gives it a little roundness
        # without needing real gradient support
        painter.setBrush(QtGui.QColor(200, 150, 110, 55))
        painter.drawEllipse(QtCore.QPointF(cx, top + 45), 13, 8)

        # hair: a soft grey fringe over the bun drawn earlier -- the
        # clearest "this is a different, older character" cue at a glance
        painter.setBrush(QtGui.QColor(_SCI_HAIR))
        painter.drawEllipse(QtCore.QPointF(cx, top + 24), 20, 15)
        painter.setBrush(QtGui.QColor(_SCI_HAIR_SHADE))
        painter.drawEllipse(QtCore.QPointF(cx, top + 11), 7, 7)  # the bun's own crown, restated on top of the fringe join
        # a single loose wisp escaping the bun -- softens the otherwise
        # perfectly round fringe into something a little more lived-in
        wisp = QtGui.QPen(QtGui.QColor(_SCI_HAIR), 1.6)
        wisp.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(wisp)
        painter.drawArc(QtCore.QRectF(cx + 13, top + 16, 8, 12), 250 * 16, 120 * 16)
        painter.setPen(QtCore.Qt.NoPen)

        # round glasses -- the clearest "scientist" cue at a glance
        pen = QtGui.QPen(QtGui.QColor("#4A4A46"), 1.8)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(cx - 7, top + 38), 6.5, 5.5)
        painter.drawEllipse(QtCore.QPointF(cx + 7, top + 38), 6.5, 5.5)
        painter.drawLine(QtCore.QPointF(cx - 0.5, top + 38), QtCore.QPointF(cx + 0.5, top + 38))

        # eyes
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#22303F"))
        painter.drawEllipse(QtCore.QPointF(cx - 7, top + 38), 2.0, 2.4)
        painter.drawEllipse(QtCore.QPointF(cx + 7, top + 38), 2.0, 2.4)

        # a small warm, closed-mouth smile
        pen = QtGui.QPen(QtGui.QColor("#8A4A32"), 1.6)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawArc(QtCore.QRectF(cx - 5, top + 43, 10, 7), 200 * 16, 140 * 16)

        # a small notebook held at her chest -- the "expert taking notes"
        # cue, drawn last so it sits in front of the coat. Deliberately a
        # static prop rather than an articulated raised arm: an earlier
        # attempt at a hand-to-chin thinking pose was the same colour as
        # the coat and only a few px across, and was completely invisible
        # in a real screenshot at this scale -- a flat-colour prop with
        # its own distinct tone reads far more reliably this small.
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#E9E4D8"))
        notebook = QtCore.QRectF(cx + 9, top + 73, 13, 17)
        painter.drawRoundedRect(notebook, 1.5, 1.5)
        pen = QtGui.QPen(QtGui.QColor("#B8A888"), 1.0)
        painter.setPen(pen)
        for line in range(3):
            y = notebook.top() + 5 + line * 4
            painter.drawLine(QtCore.QPointF(notebook.left() + 2, y),
                             QtCore.QPointF(notebook.right() - 2, y))
        painter.restore()
