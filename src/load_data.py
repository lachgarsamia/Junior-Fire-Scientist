import os
import logging

import numpy as np

import fds.slice.slice as fds
from slice_key import SliceKey, DEFAULT_SLICE_KEY, SOOT_QUANTITY, DIRECTION_TO_AXIS

logger = logging.getLogger(__name__)

# SOOT DENSITY is stored in kg/m3 but is tiny (~1e-2 kg/m3 peak for these
# candle fires); the app displays it in mg/m3 (x1e6) so its values and the
# integer display-scale slider read in human-friendly numbers, same as
# TEMPERATURE's degrees. The low-level fds/s3d reader stays in kg/m3
# (its cross-validated unit); only this display-facing loader scales.
SOOT_DISPLAY_SCALE = 1.0e6

# fds/sim/ is resolved relative to this file, not the process cwd, so the
# loader works regardless of where the application is launched from.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_SIM_ROOT = os.path.join(_SRC_DIR, '..', 'fds', 'sim')

# The candle-factorial study now runs on Pleiades (see
# FireScope/fds/sim_stage1_prep/manifest.json, which already maps each
# c<n>_d<n>_vod<n>_voc<n> case to its "..._stage1_pleiades" output folder)
# rather than the local fds/sim/ checkout that data was originally
# generated into. Preferred by default when present on this machine so the
# app shows the current runs, not the older local set; FDSVIS_SIM_ROOT
# overrides either, and a machine without that checkout (CI, another
# developer) transparently falls back to fds/sim/.
_PLEIADES_SIM_ROOT = "/Users/samialachgar/Desktop/FireScope/fds/sim_stage1_prep"
if os.environ.get('FDSVIS_SIM_ROOT'):
    SIM_ROOT = os.environ['FDSVIS_SIM_ROOT']
elif os.path.isdir(_PLEIADES_SIM_ROOT):
    SIM_ROOT = _PLEIADES_SIM_ROOT
else:
    SIM_ROOT = _LOCAL_SIM_ROOT

# Deprecated aliases for DEFAULT_SLICE_KEY's fields -- kept because
# ScenarioStore's disk-cache filenames were already built from these names
# before M2.1 (see git history); not worth a filename-format migration for
# a purely-derived cache. Prefer slice_key.DEFAULT_SLICE_KEY in new code.
QUANTITY = DEFAULT_SLICE_KEY.quantity
DIRECTION = DEFAULT_SLICE_KEY.direction
OFFSET = DEFAULT_SLICE_KEY.offset


def load_data(root_dir: str, key: SliceKey = DEFAULT_SLICE_KEY) -> np.ndarray:
    """Load one slice for one scenario folder, shape (n_times, n_row, n_col).

    For SOOT DENSITY (M2.2) the data is a plane extracted from the
    volumetric `.s3d` files (key.plane_pos gives the physical position
    along key.direction's axis) rather than a `.sf` slice -- already
    ceiling-first flipped by extract_soot_plane, so it's not re-flipped
    here, and scaled to mg/m3 for display.
    """
    if key.quantity == SOOT_QUANTITY:
        from fds.s3d.s3d import extract_soot_plane
        axis = DIRECTION_TO_AXIS[key.direction]
        _times, _extent, frames = extract_soot_plane(root_dir, axis=axis, offset=key.plane_pos)
        return frames * SOOT_DISPLAY_SCALE
    data = fds.readDataOnly(root_dir, direction=key.direction, offset=key.offset, quantity=key.quantity)
    data = np.flip(data, axis=1)
    return data


def load_times(root_dir: str, key: SliceKey = DEFAULT_SLICE_KEY) -> np.ndarray:
    """Real simulation timestamps (seconds), shape (n_times,), for the same
    slice load_data() would return the data array for -- the per-frame
    clock needed to align two different quantities recorded on different
    output schedules (e.g. TEMPERATURE's .sf slices vs SOOT DENSITY's
    .s3d dumps, ~481 vs ~1001 frames over the same real interval; see
    cinema/real_smoke.py for why frame-index arithmetic alone can't do
    this alignment).

    Cheap for .sf quantities: reads only the header/time records off one
    representative mesh block via Slice.readAllTimes(), no full data
    decode. SOOT DENSITY has no equivalently cheap partial read (the
    RLE-encoded .s3d format interleaves each frame's time with that
    frame's own payload) -- this pays the same cost as load_data() itself
    for that quantity. Callers needing both should still call load_data()
    and load_times() separately rather than assuming one implies the
    other; nothing here shares a cache with ScenarioStore.get()."""
    if key.quantity == SOOT_QUANTITY:
        from fds.s3d.s3d import extract_soot_plane
        axis = DIRECTION_TO_AXIS[key.direction]
        times, _extent, _frames = extract_soot_plane(root_dir, axis=axis, offset=key.plane_pos)
        return times
    smv_fn = fds.scanDirectory(root_dir)
    sc = fds.readSliceInfos(os.path.join(root_dir, smv_fn))
    meshes = fds.readMeshes(os.path.join(root_dir, smv_fn))
    sids = fds.findSlices(sc.slices, meshes, key.quantity, key.direction, key.offset)
    if not sids:
        raise ValueError(f"no matching slices for {key} in {root_dir}")
    sids[0].readAllTimes(root_dir)
    return sids[0].all_times


def load_data_with_times(root_dir: str, key: SliceKey = DEFAULT_SLICE_KEY) -> tuple:
    """Like load_data() + load_times() together, but paying the real
    parse cost only once -- for SOOT DENSITY specifically, calling
    load_data() and load_times() separately each run their own full
    extract_soot_plane() decode of the same `.s3d` RLE stream (measured:
    ~5.4s combined for a scenario's first real access, vs ~2.7s for one
    decode alone), which is exactly the redundant read ScenarioStore's
    disk cache exists to avoid paying twice. Callers that need both the
    data and its timestamps (ScenarioStore._load_with_disk_cache, on a
    cache miss) should prefer this over two separate calls."""
    if key.quantity == SOOT_QUANTITY:
        from fds.s3d.s3d import extract_soot_plane
        axis = DIRECTION_TO_AXIS[key.direction]
        times, _extent, frames = extract_soot_plane(root_dir, axis=axis, offset=key.plane_pos)
        return frames * SOOT_DISPLAY_SCALE, times
    mesh, extent, data, mask, times = fds.readSlice(
        root_dir, direction=key.direction, offset=key.offset, quantity=key.quantity, data_only=False)
    data = np.flip(data, axis=1)
    return data, times


def load_slice_geometry(root_dir: str, key: SliceKey = DEFAULT_SLICE_KEY):
    """Return (mesh, extent, mask) for one slice without reading frame data."""
    if key.quantity == SOOT_QUANTITY:
        from fds.s3d.s3d import soot_plane_geometry
        axis = DIRECTION_TO_AXIS[key.direction]
        return soot_plane_geometry(root_dir, axis=axis, offset=key.plane_pos)
    return fds.readSliceGeometry(root_dir, direction=key.direction, offset=key.offset, quantity=key.quantity)


def check_scenario_count(n_scenarios: int, c: int, d: int, vod: int, voc: int):
    """Warn if the folder count on disk doesn't match the assumed factor-level counts."""
    expected = c * d * vod * voc
    if n_scenarios != expected:
        logger.warning(
            "found %d scenario folders in %s but factor levels (c=%d, d=%d, vod=%d, voc=%d) "
            "imply %d scenarios; data_matrix indexing may not match folder contents",
            n_scenarios, SIM_ROOT, c, d, vod, voc, expected)
