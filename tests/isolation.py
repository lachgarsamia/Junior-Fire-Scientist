"""Repository isolation checks.

This project has a sibling checkout of the same application on the same
machine, and twice now code from that checkout has silently executed in
this one:

  * `fds/sim/manifest.json` stored absolute scenario paths into the
    sibling, so the app served the other directory's simulation data;
  * a copied `tests/__pycache__` held bytecode whose `co_filename` was
    the sibling's, so pytest ran the other checkout's test code while
    reporting a green suite.

Both failures were invisible: everything "passed". A global editable
install (`__editable__.fdsvis-*.pth`) still puts the sibling's `src` on
`sys.path` for every interpreter on this machine, so the hazard is live.

These helpers are pure and take explicit arguments so they can be unit
tested against synthetic paths; conftest.py wires them into pytest.

Severity is deliberately split:

  * a module that *actually resolved* outside this checkout is
    contamination -- always fatal;
  * a foreign source root that *shadows* this one on sys.path would make
    the next import wrong -- always fatal;
  * a foreign source root sitting after ours is currently harmless but
    fragile (one `cd` away from taking over), so it is reported loudly on
    every run rather than ignored.

Nothing here mutates sys.path, uninstalls anything, or deletes files:
the environment conflict is the operator's to resolve, and a test suite
that quietly repairs its own environment is how this class of bug hides
in the first place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Files that identify a directory as a checkout of *this* application's
# source root. Both must be present, so an unrelated `src/` directory on
# the path is not mistaken for a rival copy.
SOURCE_ROOT_MARKERS = ("main_window.py", "data_provider.py")

# Modules whose import location proves which checkout is in play.
CORE_MODULES = ("main_window", "data_provider", "public.experience", "public.experiments")


def project_root() -> Path:
    """This checkout's root, derived from this file's location -- never a
    hardcoded absolute path, so the guard survives being cloned anywhere."""
    return Path(__file__).resolve().parent.parent


def is_inside(path, root: Path) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    return True


def looks_like_source_root(path) -> bool:
    try:
        directory = Path(path)
        return all((directory / marker).is_file() for marker in SOURCE_ROOT_MARKERS)
    except OSError:
        return False


def foreign_source_roots(paths, root: Path) -> list:
    """Entries of `paths` that are a source root of this application but
    belong to a different checkout."""
    return [p for p in paths if p and looks_like_source_root(p) and not is_inside(p, root)]


def own_source_roots(paths, root: Path) -> list:
    return [p for p in paths if p and looks_like_source_root(p) and is_inside(p, root)]


def shadowing_roots(paths, root: Path) -> list:
    """Foreign source roots that would win an import against this
    checkout: any that precede our own, or all of them when ours is
    absent from the path entirely."""
    foreign = foreign_source_roots(paths, root)
    if not foreign:
        return []
    mine = own_source_roots(paths, root)
    if not mine:
        return foreign
    first_mine = min(paths.index(p) for p in mine)
    return [p for p in foreign if paths.index(p) < first_mine]


def contaminated_modules(modules, root: Path) -> list:
    """(name, file) for every already-imported project module loaded from
    outside this checkout. Reads `__file__` rather than re-importing, so
    it reports what actually happened."""
    offenders = []
    for name in CORE_MODULES:
        module = modules.get(name)
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        if origin and not is_inside(origin, root):
            offenders.append((name, origin))
    return offenders


def contaminated_test_modules(paths, root: Path) -> list:
    """Collected test files that live outside this checkout -- the exact
    shape of the stale-bytecode incident."""
    return [str(p) for p in paths if not is_inside(p, root)]


def _remedy(root: Path) -> str:
    return (
        "How to fix:\n"
        f"  1. rm -rf {root}/**/__pycache__ {root}/.pytest_cache\n"
        "  2. Check for a stale editable install pointing at another checkout:\n"
        "       python -c \"import sys; print([p for p in sys.path if 'src' in p])\"\n"
        "       ls $(python -c \"import site; print(site.getsitepackages()[0])\")/__editable__*fdsvis*\n"
        "  3. Remove it (`pip uninstall fdsvis`) and, if you need this checkout\n"
        f"     importable, reinstall from here: pip install -e {root}\n"
        "     Note: only one checkout can hold the editable install at a time."
    )


def fatal_report(root: Path, modules=None, sys_paths=None, test_paths=()) -> str:
    """A single message describing every fatal isolation problem, or ""
    when there are none."""
    modules = sys.modules if modules is None else modules
    sys_paths = list(sys.path) if sys_paths is None else list(sys_paths)

    problems = []
    for name, origin in contaminated_modules(modules, root):
        problems.append(
            f"  module {name!r}\n"
            f"    imported from : {origin}\n"
            f"    expected under: {root}")
    for path in shadowing_roots(sys_paths, root):
        problems.append(
            f"  sys.path entry {path}\n"
            f"    is another checkout's source root and takes priority over this one\n"
            f"    expected under: {root}")
    for path in contaminated_test_modules(test_paths, root):
        problems.append(
            f"  test module {path}\n"
            f"    collected from outside this checkout\n"
            f"    expected under: {root}")

    if not problems:
        return ""
    return ("Repository isolation failure -- code outside this checkout is in play.\n\n"
            + "\n".join(problems) + "\n\n" + _remedy(root))


def warning_report(root: Path, sys_paths=None) -> str:
    """Non-fatal but load-bearing: a rival checkout is importable, just
    not currently winning. Reported on every run because the ordering
    that keeps it harmless is one working directory away from flipping."""
    sys_paths = list(sys.path) if sys_paths is None else list(sys_paths)
    foreign = [p for p in foreign_source_roots(sys_paths, root)
               if p not in shadowing_roots(sys_paths, root)]
    if not foreign:
        return ""
    listed = "\n".join(f"    {p}" for p in foreign)
    return ("Repository isolation warning: another checkout of this application is on\n"
            "  sys.path (currently after this one, so imports still resolve here):\n"
            f"{listed}\n"
            "  pytest's own `pythonpath = [\"src\"]` is what keeps this correct; any\n"
            "  interpreter started without it will import the other checkout instead.\n"
            f"  {os.path.basename(__file__)}: see tests/test_repo_isolation.py")
