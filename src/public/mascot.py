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
