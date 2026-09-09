#!/bin/bash
# Launcher for Junior Fire Scientist.
#
# First run creates a virtual environment at ~/.venvs/junior_fire_scientist
# and installs the project into it; later runs just launch the app. The venv
# lives outside the repo on purpose: under an iCloud-synced folder (e.g.
# ~/Desktop) Qt's macOS platform plugin cannot enumerate a venv kept inside
# the working tree and the app aborts at startup.

set -u

cd "$(dirname "$0")"
REPO_DIR="$PWD"

VENV="$HOME/.venvs/junior_fire_scientist"
PYTHON="$VENV/bin/python"

for f in pyproject.toml src/main.py; do
    if [ ! -f "$f" ]; then
        echo "error: $f not found -- run this from a full checkout of the repository." >&2
        exit 1
    fi
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found. Install Python 3 (python.org or 'brew install python') and retry." >&2
    exit 1
fi

if [ ! -f fds/sim/manifest.json ]; then
    cat >&2 <<'EOF'
FDS dataset not found.

Expected:
  ./fds/sim/

Please copy the provided FDS simulation dataset into that directory.
EOF
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "Creating virtual environment at $VENV"
    python3 -m venv "$VENV" || exit 1
    if [ ! -x "$PYTHON" ]; then
        echo "error: venv creation did not produce $PYTHON" >&2
        exit 1
    fi
fi

if ! "$PYTHON" -m pip show fdsvis >/dev/null 2>&1; then
    echo "Installing Junior Fire Scientist and its dependencies"
    "$PYTHON" -m pip install -e . || exit 1
fi

echo "Launching Junior Fire Scientist"
if [ "$#" -eq 0 ]; then
    set -- --welcome
fi
exec env PYTHONPATH="$REPO_DIR/src" "$PYTHON" src/main.py "$@"
