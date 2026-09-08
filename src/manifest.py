"""Scenario manifest: the single source of truth mapping fds/sim/ folders to
factor levels (candles/door/vod/voc), replacing the previous implicit
assumption in scenario_store.build_data_matrix() that folder sort order
exactly matches a hardcoded nested-loop counting order.

Folder names encode the factor levels directly (e.g. "c1_d0_vod0_voc0");
this module parses that instead of assuming it. It doesn't change which
scenario ends up at which case_index (that's still folder sort order, via
list_scenario_folders()) -- it makes the candles/door/vod/voc -> case_index
mapping an explicit, derived fact instead of an unverified assumption.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass

import numpy as np

from scenario_store import list_scenario_folders

logger = logging.getLogger(__name__)

_FOLDER_RE = re.compile(r'^c(\d+)_d(\d+)_vod(\d+)_voc(\d+)(?:_.+)?$')

# Order matters: matches the (candles, door, vod, voc) axis order used
# throughout the app (config.py's N_CANDLES/N_DOORS/N_VOD/N_VOC, and the
# old build_data_matrix's nested-loop order).
_FACTORS = ('candles', 'door', 'vod', 'voc')


@dataclass(frozen=True)
class ScenarioEntry:
    """One scenario folder's identity: where it lives, and its position
    along each factor axis (0-indexed, derived from the sorted set of raw
    values actually present on disk -- not assumed from the literal digit
    in the folder name)."""
    case_index: int
    folder: str    # basename, e.g. "c1_d0_vod0_voc0"
    path: str      # absolute path
    candles: int
    door: int
    vod: int
    voc: int

    def factor_index(self, factor: str) -> int:
        return getattr(self, factor)


def _parse_folder_name(folder: str) -> dict:
    """Extract raw factor-level strings from a scenario folder's basename.
    Raises ValueError if the name doesn't match the expected pattern --
    callers should skip or report such folders rather than guess."""
    m = _FOLDER_RE.match(folder)
    if not m:
        raise ValueError(f"folder name '{folder}' doesn't match c<n>_d<n>_vod<n>_voc<n>")
    raw = dict(zip(_FACTORS, m.groups()))
    return raw


def scan_scenarios(sim_root: str) -> list:
    """Scan sim_root for scenario folders and build ScenarioEntry list.

    case_index assignment matches list_scenario_folders()'s sort order
    (unchanged from the pre-manifest behavior); factor indices are derived
    by ranking each folder's raw factor value among the sorted set of
    distinct raw values seen for that factor, so a factor's index reflects
    its actual position among what's really on disk, not an assumed count.

    A folder name may carry an arbitrary trailing suffix after its
    c<n>_d<n>_vod<n>_voc<n> stem (e.g. the cluster-run output directories
    are named "c1_d0_vod0_voc0_stage1_pleiades", and a dataset may also
    carry a bare "c1_d0_vod0_voc0" input-prep stub alongside). When two
    folder names share the same stem, only the longer (more specific)
    name is kept as that scenario's entry; this stays a pure name
    comparison, no disk access beyond the directory listing
    list_scenario_folders() already did, matching this module's existing
    "names only, not contents" scanning contract.
    """
    folders = list_scenario_folders(sim_root)

    raw_by_folder = {}
    skipped = []
    for folder in folders:
        base = os.path.basename(os.path.normpath(folder))
        try:
            raw_by_folder[folder] = _parse_folder_name(base)
        except ValueError:
            skipped.append(base)

    if skipped:
        logger.warning("manifest: skipping %d folder(s) with unrecognized names: %s",
                        len(skipped), skipped)

    by_stem = {}
    for folder, raw in raw_by_folder.items():
        stem = tuple(raw[factor] for factor in _FACTORS)
        current = by_stem.get(stem)
        if current is None or len(os.path.basename(os.path.normpath(folder))) > \
                len(os.path.basename(os.path.normpath(current))):
            by_stem[stem] = folder
    raw_by_folder = {folder: raw_by_folder[folder] for folder in by_stem.values()}
    folders = [f for f in folders if f in raw_by_folder]

    raw_levels = {factor: sorted({raw[factor] for raw in raw_by_folder.values()})
                  for factor in _FACTORS}

    entries = []
    for case_index, folder in enumerate(f for f in folders if f in raw_by_folder):
        raw = raw_by_folder[folder]
        indices = {factor: raw_levels[factor].index(raw[factor]) for factor in _FACTORS}
        entries.append(ScenarioEntry(
            case_index=case_index,
            folder=os.path.basename(os.path.normpath(folder)),
            path=os.path.abspath(folder),
            **indices,
        ))
    return entries


def _has_smv(directory: str) -> bool:
    import glob
    return bool(glob.glob(os.path.join(directory, '*.smv')))


def scan_generic_study(root: str) -> list:
    """Scan an arbitrary FDS-output directory that is NOT the candle
    factorial (V2 roadmap M2.5 -- multi-study). Two layouts are handled,
    both producing ScenarioEntry objects with the four factor indices
    defaulted to 0 (a generic study has no candle/door/vent factor axes):

      - `root` itself contains `.smv` output -> a single-scenario study
        (e.g. one FDS case). The degenerate manifest the roadmap requires
        to be first-class, not an error.
      - `root` contains subfolders that each have `.smv` output -> a
        multi-scenario study identified only by folder name.

    Returns [] if no FDS output is found anywhere (caller reports that).
    """
    if _has_smv(root):
        return [ScenarioEntry(
            case_index=0, folder=os.path.basename(os.path.normpath(root)),
            path=os.path.abspath(root), candles=0, door=0, vod=0, voc=0)]

    entries = []
    for folder in list_scenario_folders(root):
        if _has_smv(folder):
            entries.append(ScenarioEntry(
                case_index=len(entries),
                folder=os.path.basename(os.path.normpath(folder)),
                path=os.path.abspath(folder), candles=0, door=0, vod=0, voc=0))
    return entries


def scan_study(root: str) -> tuple:
    """Scan `root` and return (entries, is_factorial) for the study there
    (V2 roadmap M2.5). A candle factorial (folders matching
    c<n>_d<n>_vod<n>_voc<n>) is loaded via scan_scenarios and reported as
    factorial; anything else falls back to scan_generic_study and is
    reported as non-factorial (its candle/door/vent controls are hidden
    by the UI)."""
    factorial = scan_scenarios(root)
    if factorial:
        return factorial, True
    return scan_generic_study(root), False


def save_manifest(entries: list, manifest_path: str):
    payload = {
        "version": 1,
        "scenarios": [asdict(e) for e in entries],
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_manifest(manifest_path: str) -> list:
    with open(manifest_path, 'r') as f:
        payload = json.load(f)
    return [ScenarioEntry(**s) for s in payload["scenarios"]]


def get_manifest(sim_root: str, manifest_path: str = None, force_regenerate: bool = False) -> list:
    """Load the manifest from disk if present, else scan and write it.

    manifest_path defaults to <sim_root>/manifest.json. fds/sim/ is
    entirely gitignored (regenerable, large simulation output), so the
    manifest lives alongside it rather than in the repo.
    """
    if manifest_path is None:
        manifest_path = os.path.join(sim_root, 'manifest.json')

    if not force_regenerate and os.path.exists(manifest_path):
        try:
            return load_manifest(manifest_path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning("manifest at %s is unreadable (%s); regenerating", manifest_path, e)

    entries = scan_scenarios(sim_root)
    try:
        save_manifest(entries, manifest_path)
    except OSError as e:
        logger.warning("could not write manifest at %s (%s); continuing without persisting it", manifest_path, e)
    return entries


def foreign_path_entries(entries: list, sim_root: str) -> list:
    """Entries whose `path` points outside `sim_root` -- i.e. this study's
    manifest is serving data from some *other* checkout.

    The manifest stores absolute paths (see ScenarioEntry.path), so a
    directory that was copied rather than re-scanned keeps pointing at the
    original. That silently works as long as the original still exists,
    which makes it the worst kind of failure: an exhibition machine loads
    fine on the bench and falls back to demo data once the sibling
    directory moves. Public mode calls this at startup (see
    public/experience.py) rather than trusting a manifest it did not write.
    """
    root = os.path.abspath(sim_root)
    return [e for e in entries if os.path.commonpath([root, os.path.abspath(e.path)]) != root]


def factor_counts(entries: list) -> tuple:
    """(n_candles, n_door, n_vod, n_voc) actually present across entries."""
    return tuple(max((e.factor_index(f) for e in entries), default=-1) + 1 for f in _FACTORS)


def data_matrix_from_manifest(entries: list) -> np.ndarray:
    """Build the [candles, door, vod, voc] -> case_index array from actual
    parsed folder names, replacing build_data_matrix()'s nested-loop
    assumption with an explicit mapping."""
    n_candles, n_door, n_vod, n_voc = factor_counts(entries)
    data_matrix = np.zeros((n_candles, n_door, n_vod, n_voc), dtype=int)
    for e in entries:
        data_matrix[e.candles, e.door, e.vod, e.voc] = e.case_index
    return data_matrix
