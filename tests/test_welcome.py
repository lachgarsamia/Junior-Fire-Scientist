"""Tests for public.welcome.WelcomeWidget -- the Welcome landing page's
two buttons. Kids routes in-process (on_kids); Grown-ups launches
FireScope as a separate process (on_grownups, see
public/firescope_launcher.py) and is no longer disabled/"coming soon"."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from public.welcome import WelcomeWidget  # noqa: E402


def _widget(qapp):
    calls = {"kids": 0, "grownups": 0}
    widget = WelcomeWidget(
        on_kids=lambda: calls.__setitem__("kids", calls["kids"] + 1),
        on_grownups=lambda: calls.__setitem__("grownups", calls["grownups"] + 1))
    return widget, calls


def test_grownups_button_is_enabled(qapp):
    widget, _ = _widget(qapp)
    assert widget.grownups_button.isEnabled()


def test_coming_soon_caption_is_hidden(qapp):
    widget, _ = _widget(qapp)
    assert widget.coming_soon.isVisible() is False


def test_clicking_grownups_calls_on_grownups_not_on_kids(qapp):
    widget, calls = _widget(qapp)
    widget.grownups_button.click()
    assert calls["grownups"] == 1
    assert calls["kids"] == 0


def test_clicking_kids_calls_on_kids_not_on_grownups(qapp):
    widget, calls = _widget(qapp)
    widget.kids_button.click()
    assert calls["kids"] == 1
    assert calls["grownups"] == 0
