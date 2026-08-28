"""Launches the researcher-facing FireScope app as an independent process
from the "Grown-ups" button on the Welcome page (public/welcome.py).

Deliberately NOT an in-process mode switch, unlike enter_public_mode/
_show_welcome's root_stack swap in main_window.py: FireScope and this
kids app are two separate repositories with independently evolving
history (fds-analysis-platform vs. Junior-Fire-Scientist, diverged since
commit 011c98c). Keeping them separate -- Grown-ups launches FireScope as
its own process rather than embedding its GUI here -- means Grown-ups
always opens whatever is *currently* installed/checked out for
FireScope, with no merge or sync step ever required to keep it current.

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


def launch_firescope() -> Tuple[bool, str]:
    """Best-effort launch of FireScope as an independent, detached
    process (start_new_session=True: it must keep running and stay its
    own process group after this one continues/exits, the same as
    launching any standalone sibling app). Returns (started, message) --
    message is empty on success, or a short, honest explanation on
    failure for the caller to show the user, never swallowed."""
    root = find_firescope_root()
    if root is None:
        return False, (
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
    try:
        subprocess.Popen(
            [python, "main.py"],
            cwd=str(src_dir),
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return False, f"Couldn't start FireScope: {exc}"
    return True, ""
