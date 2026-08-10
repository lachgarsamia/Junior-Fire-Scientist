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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mood = IDLE
        self._phase = 0.0
        self.setFixedSize(120, 130)
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
        amplitude = 4.0 if self._mood in _ANIMATED_MOODS else 2.0
        bob = math.sin(self._phase) * amplitude
        cx = self.width() / 2.0
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


class SpeechBubble(QtWidgets.QWidget):
    """A rounded bubble with a tail, sized to its text. Text is set
    whole (no typewriter effect): at an exhibition, visitors read at very
    different speeds and animated text is the first thing that makes a
    display feel slow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setMinimumWidth(280)

    def set_text(self, text: str) -> None:
        self._text = text or ""
        self.setVisible(bool(self._text))
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    def _font(self) -> QtGui.QFont:
        font = self.font()
        font.setPointSizeF(max(15.0, font.pointSizeF() * 1.25))
        return font

    def sizeHint(self) -> QtCore.QSize:
        if not self._text:
            return QtCore.QSize(0, 0)
        metrics = QtGui.QFontMetrics(self._font())
        width = min(560, max(300, metrics.horizontalAdvance(self._text) + 48))
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
        rect = metrics.boundingRect(QtCore.QRect(0, 0, max(80, width - 48), 10_000),
                                    QtCore.Qt.TextWordWrap, self._text)
        return rect.height() + 44

    def paintEvent(self, event) -> None:
        if not self._text:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            body = QtCore.QRectF(0, 0, self.width(), self.height() - 12)
            path = QtGui.QPainterPath()
            path.addRoundedRect(body, 18, 18)
            # tail, pointing down-left toward the mascot
            tail = QtGui.QPainterPath()
            tail.moveTo(34, body.bottom() - 2)
            tail.lineTo(30, body.bottom() + 12)
            tail.lineTo(56, body.bottom() - 2)
            tail.closeSubpath()
            path = path.united(tail)

            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 40), 1.5))
            painter.setBrush(QtGui.QColor(18, 22, 30, 235))
            painter.drawPath(path)

            painter.setPen(QtGui.QColor("#F3F6FA"))
            painter.setFont(self._font())
            painter.drawText(body.adjusted(24, 14, -24, -10),
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
    """Dr. Frieda Funke -- a second, ambient guide standing in the
    instrument sidebar, visually distinct from the primary firefighter
    mascot (grey hair in a bun, round glasses, a lab coat, a small
    fire-orange pin tying her to the same accent colour the rest of the
    app already uses). She does not react to specific events the way
    the primary mascot's moods do -- PublicExperience periodically gives
    her a real, simplified fire-science fact to muse over in a thought
    bubble (see ThoughtBubble), independent of anything the child just
    did. She is decoration with something to say, not a second
    narrator answering the scene."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        # She stands in the sidebar's lower-right pocket, below the
        # thermometer (see PublicOverlay._position_scientist for why
        # there specifically) -- at the 800x600 exhibit target that
        # pocket is only ~86px tall, tighter than the gap above the
        # thermometer an earlier version of this used, hence the smaller
        # size here than that version had. _DESIGN_W/_DESIGN_H below are
        # drawn at native scale and shrunk to this via painter.scale(),
        # so this is a pure size change, not a redraw.
        self.setFixedSize(52, 66)
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

        # a small fire-orange pin -- the one warm colour link to the rest
        # of the app's palette, standing in for a helmet she doesn't wear
        painter.setBrush(QtGui.QColor(_SCI_ACCENT))
        painter.drawEllipse(QtCore.QPointF(cx + 11, top + 60), 4.0, 4.0)

        # neck, bridging face and collar so the head doesn't read as
        # glued directly onto the coat
        painter.setBrush(QtGui.QColor(_SCI_SKIN))
        painter.drawRect(QtCore.QRectF(cx - 5, top + 50, 10, 8))

        # face
        painter.drawEllipse(QtCore.QPointF(cx, top + 38), 19, 18)

        # hair: a soft grey fringe over the bun drawn earlier -- the
        # clearest "this is a different, older character" cue at a glance
        painter.setBrush(QtGui.QColor(_SCI_HAIR))
        painter.drawEllipse(QtCore.QPointF(cx, top + 24), 20, 15)
        painter.setBrush(QtGui.QColor(_SCI_HAIR_SHADE))
        painter.drawEllipse(QtCore.QPointF(cx, top + 11), 7, 7)  # the bun's own crown, restated on top of the fringe join

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
        painter.restore()


class ThoughtBubble(QtWidgets.QWidget):
    """A cloud-shaped bubble with trailing circles, not a tail -- the
    standard "thinking" convention, and visually distinct from
    SpeechBubble's rounded-rect-plus-tail: Dr. Funke's facts are her
    own ambient musing, not the primary guide speaking directly to the
    child. Sized to its text the same way SpeechBubble is."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._font_pt = 11.5
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        # No setMinimumWidth here, deliberately: the sidebar gutter she
        # lives in (see PublicOverlay._position_thought_bubble) is only
        # ~150-170px wide, and a widget-level minimum used to fight that
        # call's own width clamp -- QWidget.resize() silently snaps back
        # up to whatever setMinimumWidth demands, which pushed the
        # bubble a real, screenshotted 10px past the right edge of an
        # 800x600 screen. The caller is the only one with the actual
        # available space, so it's the only one allowed to set a floor.

        # A quick pop -- three little circles growing into the bubble
        # itself, like the standard "..." thought-bubble trail turning
        # into a real thought -- rather than the text just appearing.
        # Purely decorative timing (_POP_MS), same "it may animate, the
        # final value is still the real one" spirit as the thermometer's
        # own sweep animation (see Thermometer._anim).
        self._pop = QtCore.QVariantAnimation(self)
        self._pop.setDuration(self._POP_MS)
        self._pop.setStartValue(0.0)
        self._pop.setEndValue(1.0)
        self._pop.setEasingCurve(QtCore.QEasingCurve.OutBack)
        self._pop.valueChanged.connect(self._on_pop)
        self._pop_t = 1.0

    _POP_MS = 420

    def _on_pop(self, value) -> None:
        self._pop_t = float(value)
        self.update()

    def set_text(self, text: str) -> None:
        was_empty = not self._text
        self._text = text or ""
        self._font_pt = 11.5
        self.setVisible(bool(self._text))
        self.updateGeometry()
        if self._text and was_empty:
            self._pop.stop()
            self._pop.start()
        else:
            self._pop_t = 1.0
        self.update()

    def text(self) -> str:
        return self._text

    def _font(self) -> QtGui.QFont:
        font = self.font()
        font.setPointSizeF(self._font_pt)
        return font

    # Her sidebar spot (see PublicOverlay._position_thought_bubble) is a
    # fixed strip above the thermometer, not elastic -- there's nowhere
    # for a too-tall bubble to grow into without covering the tube. A
    # first pass at this sized purely off the text overlapped the
    # thermometer's own caption in a real screenshot, for German's
    # longer compound-word facts. Shrinking the font here (down to a
    # floor of 9pt) is what keeps every fact, in either language,
    # inside the strip it actually has -- not a cosmetic touch.
    def fit_to(self, width: int, max_height: int) -> None:
        self._font_pt = 11.5
        while self.heightForWidth(width) > max_height and self._font_pt > 9.0:
            self._font_pt -= 0.5
        self.resize(width, min(self.heightForWidth(width), max(0, max_height)))

    # Extra height below the main body reserved for the trailing
    # circles that make this read as a *thought* bubble, not a speech
    # one -- excluded from the body rect itself in paintEvent. She
    # stands to this bubble's lower-left (see PublicOverlay._position_
    # thought_bubble -- the sidebar gutter has no room to stack a full
    # thermometer, her, and a bubble in one column, so the two sit
    # side by side above the thermometer instead), so the trail hugs
    # the body's own left edge rather than its centre, to visually
    # land near her rather than drifting into empty space to her right.
    _TRAIL_H = 20

    def sizeHint(self) -> QtCore.QSize:
        if not self._text:
            return QtCore.QSize(0, 0)
        metrics = QtGui.QFontMetrics(self._font())
        width = min(210, max(160, metrics.horizontalAdvance(self._text) + 32))
        return QtCore.QSize(width, self.heightForWidth(width))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if not self._text:
            return 0
        metrics = QtGui.QFontMetrics(self._font())
        rect = metrics.boundingRect(QtCore.QRect(0, 0, max(80, width - 32), 10_000),
                                    QtCore.Qt.TextWordWrap, self._text)
        return rect.height() + 30 + self._TRAIL_H

    def paintEvent(self, event) -> None:
        if not self._text:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            body = QtCore.QRectF(0, 0, self.width(), self.height() - self._TRAIL_H)
            # trailing circles' own anchor, down-and-left toward Dr.
            # Funke standing beside her own bubble -- also the pivot the
            # whole bubble pops in from (_pop below), so it reads as
            # growing out of that same trail rather than the screen's
            # own corner.
            cx = body.left() + 20
            anchor = QtCore.QPointF(cx - 15, body.bottom() + 14)

            t = max(0.02, self._pop_t)
            painter.save()
            painter.translate(anchor)
            painter.scale(t, t)
            painter.translate(-anchor)

            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 40), 1.5))
            painter.setBrush(QtGui.QColor(30, 26, 44, 235))
            painter.drawRoundedRect(body, 16, 16)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(cx - 6, body.bottom() + 6), 6.0, 6.0)
            painter.drawEllipse(anchor, 3.5, 3.5)

            painter.setPen(QtGui.QColor("#F3F6FA"))
            painter.setFont(self._font())
            painter.drawText(body.adjusted(16, 9, -16, -7),
                             QtCore.Qt.TextWordWrap | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                             self._text)
            painter.restore()

            # A couple of extra little bubbles fizzing up past the trail
            # while the real one is still popping in -- gone by the time
            # it settles, so they read as the announcement, not a
            # permanent decoration.
            if self._pop_t < 0.88:
                fade = max(0.0, 1.0 - self._pop_t / 0.88)
                grow = min(1.0, self._pop_t / 0.5)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(255, 255, 255, int(150 * fade)))
                r1 = 4.5 * (0.35 + 0.65 * grow)
                painter.drawEllipse(anchor + QtCore.QPointF(-3, -11), r1, r1)
                painter.setBrush(QtGui.QColor(255, 255, 255, int(110 * fade)))
                r2 = 3.0 * (0.35 + 0.65 * grow)
                painter.drawEllipse(anchor + QtCore.QPointF(5, -20), r2, r2)
        finally:
            painter.end()
