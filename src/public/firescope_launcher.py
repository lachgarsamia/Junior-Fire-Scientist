"""Launches the researcher-facing FireScope app from the "Grown-ups"
button on the Welcome page (public/welcome.py), or activates it if it's
already running rather than spawning a duplicate.

Deliberately NOT an in-process mode switch, unlike enter_public_mode/
_show_welcome's root_stack swap in main_window.py: FireScope and this
kids app are two separate repositories with independently evolving
history (fds-analysis-platform vs. Junior-Fire-Scientist, diverged since
commit 011c98c). Keeping them separate -- Grown-ups launches FireScope as
its own process rather than embedding its GUI here -- means Grown-ups
always opens whatever is *currently* installed/checked out for
FireScope, with no merge or sync step ever required to keep it current.

Already-running detection, without any real IPC: launch_firescope()
returns the Popen it started, and main_window.py's launch_researcher_app
holds onto it across clicks (self._firescope_process). On a later
Grown-ups click, if that process is still alive, its window is activated
(macOS only, via System Events/osascript) instead of starting a second
FireScope process. FireScope's own "Back" button does the mirror of
this: it's launched with JUNIOR_FIRE_SCIENTIST_PID set to this process's
pid (see below), so its kids_app_launcher.py can activate *this*
window instead of relaunching, when it's still alive.

Locating FireScope: FIRESCOPE_APP_PATH, if set, points directly at the
FireScope checkout's root. Otherwise this assumes the conventional local
layout on this machine -- a sibling directory next to this repo's own
root (both as direct children of the same parent, e.g. ~/Desktop/
FireScope next to ~/Desktop/fds_visualizer kids/). Interpreter:
FIRESCOPE_PYTHON, if set, otherwise the anaconda interpreter FireScope's
own dev workflow already runs on; this process's own sys.executable is
the last-resort fallback if that path doesn't exist on this machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

_DEFAULT_SIBLING_NAME = "FireScope"
_DEFAULT_PYTHON = "/opt/anaconda3/bin/python"


def find_firescope_root() -> Optional[Path]:
    """FireScope repo root, or None if it can't be located. Checked in
    order: FIRESCOPE_APP_PATH env var, then the conventional sibling-
    directory layout. Verified by the presence of src/main.py, not just
    the directory, so a stale/half-moved checkout is reported as missing
    rather than launched broken."""
    override = os.environ.get("FIRESCOPE_APP_PATH")
    if override:
        candidate = Path(override).expanduser()
        return candidate if (candidate / "src" / "main.py").is_file() else None

    this_repo_root = Path(__file__).resolve().parents[2]
    candidate = this_repo_root.parent / _DEFAULT_SIBLING_NAME
    return candidate if (candidate / "src" / "main.py").is_file() else None


def pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists. Signal 0 probes
    existence/permission without actually sending a signal -- the
    standard portable way to check a pid without owning it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    except OSError:
        return False
    return True


def activate_pid(pid: int) -> bool:
    """Bring pid's window(s) to the front. macOS only (System Events via
    osascript -- the standard no-extra-dependency way to activate
    another process's window by pid, no PyObjC/extra package needed);
    returns False elsewhere or on any failure, so the caller can fall
    back to a fresh launch instead of silently doing nothing."""
    if sys.platform != "darwin":
        return False
    script = (
        f'tell application "System Events" to set frontmost of '
        f'(first process whose unix id is {pid}) to true'
    )
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def launch_firescope() -> Tuple[Optional[subprocess.Popen], str]:
    """Best-effort launch of FireScope as an independent, detached
    process (start_new_session=True: it must keep running and stay its
    own process group after this one continues/exits, the same as
    launching any standalone sibling app). Passes this process's own pid
    via JUNIOR_FIRE_SCIENTIST_PID so FireScope's Back button can find its
    way back to (activate) this window later. Returns (process, message)
    -- process is the launched Popen on success (None on failure),
    message is empty on success or a short, honest explanation on
    failure for the caller to show the user, never swallowed."""
    root = find_firescope_root()
    if root is None:
        return None, (
            "FireScope isn't installed where this launcher expects it. "
            "Set the FIRESCOPE_APP_PATH environment variable to its "
            f"checkout, or place it as a sibling folder named "
            f"'{_DEFAULT_SIBLING_NAME}'."
        )

    python = os.environ.get("FIRESCOPE_PYTHON", _DEFAULT_PYTHON)
    if not Path(python).is_file():
        python = sys.executable

    src_dir = root / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    env["JUNIOR_FIRE_SCIENTIST_PID"] = str(os.getpid())
    try:
        process = subprocess.Popen(
            [python, "main.py"],
            cwd=str(src_dir),
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return None, f"Couldn't start FireScope: {exc}"
    return process, ""
