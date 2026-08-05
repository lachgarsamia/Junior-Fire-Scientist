"""The exhibit's landing page -- shown before a visitor commits to Fire
Explorer, and reached again via the exit (X) affordance so leaving the
kids experience never surfaces the researcher app underneath.

Deliberately standalone: unlike everything else under public/, this
widget has no live scene and no PublicExperience state to drive, so it
does not import PublicOverlay (which assumes both). It hand-matches the
overlay's dark background and BigButton visual language instead of
depending on it -- the only thing it shares with the rest of the public
package is the i18n table.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from public.i18n import tr

_BG = "#07090E"
_ACCENT = "#FF7A18"
_ACCENT_HOVER = "#FF9440"
_TEXT = "#F3F6FA"
_TEXT_DIM = "#A9B4C4"
_PANEL = "rgba(24, 30, 42, 235)"
_PANEL_HOVER = "rgba(44, 54, 72, 245)"
_PANEL_BORDER = "rgba(255, 255, 255, 38)"
_FOCUS = "#FFD166"

_KIDS_ICON = "🧒"
_GROWNUPS_ICON = "🧑"


class WelcomeWidget(QtWidgets.QWidget):
    """"Who's exploring today?" -- a heading and two large touch choices.
    "Grown-ups" is inert on purpose: researcher access has no route
    through here yet (see main_window.exit_public_mode's own docstring
    on why the X must never reach the researcher shell)."""

    def __init__(self, on_kids, parent=None):
        super().__init__(parent)
        self._on_kids = on_kids
        # A bare QWidget doesn't paint stylesheet backgrounds on its own --
        # WA_StyledBackground is what makes Qt actually consult the
        # styleSheet during paintEvent instead of just filling the default
        # (light) palette. PublicExperience's own setStyleSheet("background:
        # #07090E;") never needed this because its scene canvas is opaque
        # and covers every pixel; this page has no such covering child, so
        # the gap is visible instead of silently masked.
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background: {_BG};")

        root = QtWidgets.QVBoxLayout(self)
        root.addStretch(2)

        self.heading = QtWidgets.QLabel(self)
        self.heading.setAlignment(QtCore.Qt.AlignCenter)
        self.heading.setStyleSheet(
            f"color: {_TEXT}; font-size: 48px; font-weight: 800; background: transparent;")
        root.addWidget(self.heading)

        self.subtitle = QtWidgets.QLabel(self)
        self.subtitle.setAlignment(QtCore.Qt.AlignCenter)
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 18px; font-weight: 500; background: transparent;")
        root.addSpacing(12)
        root.addWidget(self.subtitle)

        root.addSpacing(36)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)

        self.kids_button = self._make_button(primary=True)
        self.kids_button.clicked.connect(self._on_kids)
        button_row.addWidget(self.kids_button)
        button_row.addSpacing(28)

        grownups_col = QtWidgets.QVBoxLayout()
        grownups_col.setSpacing(6)
        self.grownups_button = self._make_button(primary=False)
        self.grownups_button.setEnabled(False)
        grownups_col.addWidget(self.grownups_button)
        self.coming_soon = QtWidgets.QLabel(self)
        self.coming_soon.setAlignment(QtCore.Qt.AlignCenter)
        self.coming_soon.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 12px; font-weight: 600; background: transparent;")
        grownups_col.addWidget(self.coming_soon)
        button_row.addLayout(grownups_col)

        button_row.addStretch(1)
        root.addLayout(button_row)
        root.addStretch(3)

        self.retranslate()

    def _make_button(self, primary: bool) -> QtWidgets.QPushButton:
        """Hand-matches BigButton's own visual language (public/widgets.py)
        without importing it -- this page has no scene/state to hand a
        real BigButton's touch-target machinery, just the same look."""
        button = QtWidgets.QPushButton(self)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setMinimumHeight(96)
        button.setMinimumWidth(240)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        button.setFocusPolicy(QtCore.Qt.StrongFocus)
        fill = _ACCENT if primary else _PANEL
        hover = _ACCENT_HOVER if primary else _PANEL_HOVER
        fg = "#1A1005" if primary else _TEXT
        button.setStyleSheet(f"""
            QPushButton {{
                background: {fill};
                color: {fg};
                border: 2px solid {'transparent' if primary else _PANEL_BORDER};
                border-radius: 20px;
                font-size: 24px;
                font-weight: 700;
                padding: 14px 22px;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ padding-top: 17px; padding-bottom: 11px; }}
            QPushButton:focus {{ border: 3px solid {_FOCUS}; }}
            QPushButton:disabled {{
                background: rgba(30,36,48,180); color: {_TEXT_DIM};
                border: 2px solid transparent;
            }}
        """)
        return button

    def retranslate(self) -> None:
        """Re-read every static label from the current language -- same
        "fully re-declare from live tr() calls" convention as
        PublicOverlay.retranslate(), since this widget is constructed
        once and reused (see main_window's own construct-once pattern
        for public_experience) but may be shown after the language was
        switched during a Kids session."""
        self.heading.setText(tr("welcome_heading"))
        self.subtitle.setText(tr("welcome_subtitle"))
        self.kids_button.setText(f"{_KIDS_ICON}  {tr('welcome_kids_button')}")
        self.grownups_button.setText(f"{_GROWNUPS_ICON}  {tr('welcome_grownups_button')}")
        self.grownups_button.setAccessibleName(tr("welcome_grownups_button"))
        self.kids_button.setAccessibleName(tr("welcome_kids_button"))
        self.coming_soon.setText(tr("welcome_coming_soon"))
