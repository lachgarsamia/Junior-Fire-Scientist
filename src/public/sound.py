"""A short, synthesized "you got it!" chime for the Games celebration.

This repo bundles no audio assets and has no existing playback code (grep
confirms: no QSound/QSoundEffect/QMediaPlayer, no .wav/.mp3 anywhere) --
so rather than inventing a fake "real" sound-effect file, this synthesizes
a small, cheerful three-note ascending arpeggio (a plain success-chime
shape, not music) directly with the stdlib `wave`/`struct`/`math` modules,
writes it once to a temp file, and plays it with PyQt5's QSoundEffect
(confirmed available: PyQt5.QtMultimedia is already an installed
dependency of this project, just never used yet).

Never fatal: an exhibit kiosk must keep working with no sound at all if
QtMultimedia can't initialize an audio device in some environment (no
speakers wired up, an unusual Linux audio stack, etc.) -- every entry
point here catches broadly and just no-ops rather than raising.
"""

from __future__ import annotations

import logging
import math
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 44100
# A short major-triad-ish rise (C6, E6, G6) -- an "arpeggio up" is the
# simplest shape that reads as "success" rather than a plain beep, three
# notes is enough for that without becoming actual music.
_NOTES_HZ = (1046.5, 1318.5, 1568.0)
_NOTE_MS = 110
_GAP_MS = 15

_cached_path: Optional[str] = None
_sound_effect = None   # QSoundEffect, created lazily (needs a QApplication)


def _synthesize_chime() -> str:
    """Write the chime to a temp .wav file once per process, return its
    path. A plain sine tone per note with a short linear fade-in/out
    (avoids the audible "click" a hard-edged tone would have), stitched
    back to back with tiny silent gaps."""
    samples = []
    fade_n = int(_SAMPLE_RATE * 0.012)   # ~12ms fade, enough to kill clicks
    for note_hz in _NOTES_HZ:
        n = int(_SAMPLE_RATE * _NOTE_MS / 1000)
        for i in range(n):
            t = i / _SAMPLE_RATE
            envelope = 1.0
            if i < fade_n:
                envelope = i / fade_n
            elif i > n - fade_n:
                envelope = (n - i) / fade_n
            value = math.sin(2 * math.pi * note_hz * t) * envelope * 0.5
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))
        gap_n = int(_SAMPLE_RATE * _GAP_MS / 1000)
        samples.extend([0] * gap_n)

    fd, path = tempfile.mkstemp(prefix="fire_explorer_chime_", suffix=".wav")
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SAMPLE_RATE)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def play_success_chime() -> None:
    """Fire-and-forget: play the chime, or silently do nothing if audio
    isn't available in this environment. Never raises."""
    global _cached_path, _sound_effect
    try:
        from PyQt5 import QtCore, QtMultimedia
        if _cached_path is None:
            _cached_path = _synthesize_chime()
        if _sound_effect is None:
            _sound_effect = QtMultimedia.QSoundEffect()
            _sound_effect.setSource(QtCore.QUrl.fromLocalFile(_cached_path))
            _sound_effect.setVolume(0.6)
        _sound_effect.play()
    except Exception as e:  # noqa: BLE001 - a missing sound must never break the exhibit
        logger.info("success chime unavailable (%s)", e)
