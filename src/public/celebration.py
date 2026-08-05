"""A brief confetti/star burst for "you got it!" moments in the Games
section -- a reward that never blocks or gates anything: it paints over
the scene for about a second and then gets out of the way on its own.

Deliberately a Qt widget layered on top of PublicOverlay, not a
matplotlib artist added to PublicScene: a celebration has to appear in
front of every button/card, and PublicScene sits *below* the overlay in
z-order (see overlay.py's own module docstring on why cartoon elements
never touch the matplotlib scene). This follows the same "one shared
QVariantAnimation drives a paintEvent" idiom Thermometer's own fill sweep
already uses (widgets.py) -- not a new animation mechanism.
"""

from __future__ import annotations

import math
import random

from PyQt5 import QtCore, QtGui, QtWidgets

# A handful of festive colours -- distinct from the app's own ACCENT/
# DELIGHT palette on purpose, so the burst reads as "something extra just
# happened" rather than reusing the everyday UI hues.
_COLORS = ("#FF5C8A", "#5AC8FA", "#FFD166", "#7ED957", "#B388FF", "#FF8A3D", "#FF3B30", "#4ADEDE")
# Bumped from an earlier, sparser 26/1100ms pass: at 26 particles falling
# straight down together, the burst read as a thin confetti "belt"
# crossing the screen for a blink rather than an exciting reward (real
# testing feedback -- "games are mid... I wanted some confetti"). More
# particles, a radial launch instead of uniform top-down rain, and a
# longer hold read as a real firework/popper burst instead.
_PARTICLE_COUNT = 70
_DURATION_MS = 1650


class _Particle:
    __slots__ = ("x0", "y0", "vx", "vy", "size", "color", "shape", "spin", "spin_speed", "delay")

    def __init__(self, width: int, height: int):
        # Launched outward from a point low-center-ish (where the guide/
        # discovery moment sits), like a popper going off, rather than
        # spawning already spread across the top edge -- the outward
        # motion in the first third of the burst is what reads as an
        # "explosion" instead of gentle rain.
        self.x0 = width * random.uniform(0.35, 0.65)
        self.y0 = height * random.uniform(0.55, 0.75)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(220, 620)
        self.vx = math.cos(angle) * speed
        # Bias upward so the launch reads as "up and out", gravity (see
        # paintEvent) pulls it back down over the run.
        self.vy = math.sin(angle) * speed - random.uniform(80, 220)
        self.size = random.uniform(8, 18)
        self.color = random.choice(_COLORS)
        self.shape = random.choice(("rect", "star"))
        self.spin = random.uniform(0, 360)
        self.spin_speed = random.uniform(180, 620) * random.choice((-1, 1))
        self.delay = random.uniform(0, 0.12)   # staggered start, as a fraction of the run


class CelebrationOverlay(QtWidgets.QWidget):
    """Raised over the whole PublicOverlay for the duration of one burst.
    Transparent to clicks and hidden whenever nothing is playing, so it
    never competes with or blocks the real interface underneath it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self._particles: list = []
        self._progress = 0.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(_DURATION_MS)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutQuad)
        self._anim.valueChanged.connect(self._on_progress)
        self._anim.finished.connect(self.hide)
        self.hide()

    def burst(self) -> None:
        """Start one celebration over the full parent area. Safe to call
        again while a previous burst is still fading -- it just restarts
        with a fresh set of particles."""
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
        self._particles = [_Particle(max(1, self.width()), max(1, self.height()))
                          for _ in range(_PARTICLE_COUNT)]
        self.raise_()
        self.show()
        self._anim.stop()
        self._anim.start()

    def _on_progress(self, value) -> None:
        self._progress = float(value)
        self.update()

    # Pixels/second^2 -- tuned so a particle launched at the speeds
    # _Particle picks arcs back down well before _DURATION_MS ends,
    # instead of still sailing offscreen when the burst fades out.
    _GRAVITY = 900.0

    def paintEvent(self, event) -> None:
        if not self._particles:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        try:
            duration_s = _DURATION_MS / 1000.0
            for p in self._particles:
                local_t = max(0.0, min(1.0, (self._progress - p.delay) / max(1e-6, 1 - p.delay)))
                if local_t <= 0.0:
                    continue
                t = local_t * duration_s
                x = p.x0 + p.vx * t
                y = p.y0 + p.vy * t + 0.5 * self._GRAVITY * t * t
                # Fade in fast, hold, then fade out over the last third --
                # a burst that's gone before it can feel like it's
                # blocking anything.
                opacity = 1.0
                if local_t > 0.6:
                    opacity = max(0.0, 1.0 - (local_t - 0.6) / 0.4)
                if opacity <= 0.0:
                    continue
                painter.save()
                painter.translate(x, y)
                painter.rotate(p.spin + p.spin_speed * local_t)
                color = QtGui.QColor(p.color)
                color.setAlphaF(opacity)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(color)
                if p.shape == "rect":
                    painter.drawRoundedRect(
                        QtCore.QRectF(-p.size / 2, -p.size / 3, p.size, p.size * 0.66), 2, 2)
                else:
                    painter.drawPath(_star_path(p.size / 2))
                painter.restore()
        finally:
            painter.end()


def _star_path(radius: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    full_turn = 2 * math.pi
    for i in range(10):
        angle = full_turn * i / 10 - full_turn / 4
        r = radius if i % 2 == 0 else radius * 0.45
        point = QtCore.QPointF(r * math.cos(angle), r * math.sin(angle))
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return path
