"""Disk-backed storage for cube grids, and a low-memory .cube file parser.

Grids are the bulk of ChemInsight3D's memory (a 191 x 199 x 200 cube is 61 MB
as float64), and the viewer used to keep every one of them in RAM. Here each
grid is written once to a managed ``.npy`` file and served as a read-only
memory map (``np.load(..., mmap_mode='r')``). Its pages belong to the OS file
cache: clean, shared, and dropped under memory pressure instead of swapped,
so the process holds no private copy of any grid. A small LRU keeps the
recently used maps open; an evicted grid is unmapped and reopened on use.

CubeStore    owns the cache files of one viewer process.
CubeRecord   a cube dict (origin, spacing, atoms, ...) whose 'data' entry is
             fetched from the store on access, so code that reads cube['data']
             keeps working while the grid itself is not held in the dict.
parse_cube_file
             reads a Gaussian .cube file into a NumPy array in chunks, without
             building a Python list of every value.

Cache policy
------------
* Files live in <cache root>/run-<pid>-<time>-<random>/ (one directory per
  process). The root is $CI3D_CUBE_CACHE_DIR, else the platform cache
  directory (~/.cache/ChemInsight3D/cube-cache, or
  %LOCALAPPDATA%\\ChemInsight3D\\cube-cache on Windows).
* The budget for open maps is $CI3D_CUBE_CACHE_MB megabytes (default 256).
  The most recently used grid always stays open, even if larger.
* A grid's file is deleted when its last cube is released, and the run
  directory is removed when the store closes. Files that cannot be deleted
  yet (still mapped, on Windows) are retried at close.
* Run directories left by processes that ended without closing (crash, kill)
  are removed at the next start once they are older than STALE_AGE_HOURS.
* Cache files are disposable. Durable copies for sessions are written by
  export_grid() into the session's own assets folder.
"""

from __future__ import annotations

import atexit
import collections
import os
import shutil
import threading
import time
import uuid
import warnings

import numpy as np

DEFAULT_BUDGET_MB = 256
STALE_AGE_HOURS = 12
_PARSE_CHUNK_BYTES = 8 * 2**20


class CubeStoreError(OSError):
    """A grid could not be written to or read from the cache."""


def default_cache_root():
    env = os.environ.get("CI3D_CUBE_CACHE_DIR")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "ChemInsight3D", "cube-cache")


def _budget_from_env():
    try:
        return max(1, int(os.environ.get("CI3D_CUBE_CACHE_MB", DEFAULT_BUDGET_MB))) * 2**20
    except ValueError:
        return DEFAULT_BUDGET_MB * 2**20


def _write_npy_atomically(array, path):
    """np.save to a temporary name, then rename, so a crash never leaves a partial file."""
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    check = np.load(path, mmap_mode="r")
    if check.shape != array.shape or check.dtype != array.dtype:
        raise CubeStoreError(f"Cache file {path} did not read back as written")
    return check


class CubeStore:
    """Grids by id, on disk as .npy (Fortran order) with an LRU of resident arrays."""

    def __init__(self, root=None, budget_bytes=None):
        self.root = root or default_cache_root()
        self.budget = int(budget_bytes) if budget_bytes else _budget_from_env()
        self.run_dir = None                         # created on first write
        self._paths = {}                            # grid id -> file path
        self._owned = set()                         # ids whose files this store may delete
        self._resident = collections.OrderedDict()  # grid id -> ndarray, most recent last
        self._resident_bytes = 0
        self._undeleted = set()                     # files to retry deleting at close
        self._lock = threading.RLock()
        self.cleanup_stale()
        atexit.register(self.close)                 # backstop if the app never closes it

    # ── writing and reading ──────────────────────────────────────────────────

    def _ensure_run_dir(self):
        if self.run_dir is None:
            name = f"run-{os.getpid()}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            path = os.path.join(self.root, name)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                raise CubeStoreError(f"Cannot create the cube cache folder {path}: {e}") from e
            self.run_dir = path
        return self.run_dir

    def put(self, array, keep_resident=True):
        """
        Write a grid to the cache and return its id. The array is stored in
        Fortran order (so VTK can use it without a copy) and its dtype is kept.
        With keep_resident the new file's memory map joins the LRU (its pages
        are still in the file cache from the write); the caller's array is not
        kept, so it is freed once the caller drops it.
        """
        arr = np.asfortranarray(np.asarray(array))
        with self._lock:
            grid_id = uuid.uuid4().hex
            path = os.path.join(self._ensure_run_dir(), grid_id + ".npy")
            try:
                mapped = _write_npy_atomically(arr, path)
            except CubeStoreError:
                raise
            except OSError as e:
                raise CubeStoreError(
                    f"Could not write a cube grid to the cache folder {self.run_dir} "
                    f"({arr.nbytes / 1e6:.0f} MB): {e}") from e
            self._paths[grid_id] = path
            self._owned.add(grid_id)
            if keep_resident:
                self._remember(grid_id, mapped)
            return grid_id

    def add_external(self, path):
        """Register an existing .npy file (e.g. a session asset); it is never deleted."""
        if not os.path.isfile(path):
            raise CubeStoreError(f"Grid file not found: {path}")
        with self._lock:
            grid_id = uuid.uuid4().hex
            self._paths[grid_id] = os.path.abspath(path)
            return grid_id

    def get(self, grid_id):
        """The grid as a read-only memory map (reopened if it was evicted)."""
        with self._lock:
            arr = self._resident.get(grid_id)
            if arr is not None:
                self._resident.move_to_end(grid_id)
                return arr
            path = self._paths.get(grid_id)
            if path is None:
                raise KeyError(f"Unknown or released cube grid {grid_id}")
            try:
                arr = np.load(path, mmap_mode="r")
            except (OSError, ValueError) as e:
                raise CubeStoreError(f"Cannot read cached cube grid {path}: {e}") from e
            self._remember(grid_id, arr)
            return arr

    def _remember(self, grid_id, arr):
        if grid_id in self._resident:
            self._resident_bytes -= self._resident.pop(grid_id).nbytes
        self._resident[grid_id] = arr
        self._resident_bytes += arr.nbytes
        while self._resident_bytes > self.budget and len(self._resident) > 1:
            _, old = self._resident.popitem(last=False)
            self._resident_bytes -= old.nbytes

    def path(self, grid_id):
        return self._paths[grid_id]

    @property
    def resident_bytes(self):
        return self._resident_bytes

    def resident_ids(self):
        return list(self._resident)

    # ── durable copies and lifetimes ─────────────────────────────────────────

    def export_grid(self, grid_id, dest_path):
        """Copy a grid to a durable location (atomically). Returns dest_path."""
        with self._lock:
            src = self._paths[grid_id]
        tmp = f"{dest_path}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest_path)
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise CubeStoreError(f"Could not write {dest_path}: {e}") from e
        return dest_path

    def release(self, grid_id):
        """Forget a grid; its cache file is deleted if this store created it."""
        with self._lock:
            arr = self._resident.pop(grid_id, None)
            if arr is not None:
                self._resident_bytes -= arr.nbytes
            path = self._paths.pop(grid_id, None)
            if grid_id in self._owned:
                self._owned.discard(grid_id)
                self._delete(path)

    def _delete(self, path):
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            self._undeleted.add(path)       # still mapped (Windows): retry at close

    def close(self):
        """Drop everything and remove this process's cache folder."""
        with self._lock:
            self._resident.clear()
            self._resident_bytes = 0
            for grid_id in list(self._owned):
                self._delete(self._paths.get(grid_id))
            self._owned.clear()
            self._paths.clear()
            for path in list(self._undeleted):
                self._undeleted.discard(path)
                self._delete(path)
            if self.run_dir and os.path.isdir(self.run_dir):
                shutil.rmtree(self.run_dir, ignore_errors=True)
            self.run_dir = None

    def cleanup_stale(self, max_age_hours=STALE_AGE_HOURS):
        """Remove run folders left behind by earlier processes that did not close."""
        if not os.path.isdir(self.root):
            return []
        removed = []
        now = time.time()
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            if not (name.startswith("run-") and os.path.isdir(path)) or path == self.run_dir:
                continue
            try:
                age_hours = (now - os.path.getmtime(path)) / 3600
            except OSError:
                continue
            if age_hours < max_age_hours or _process_alive(name):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
        return removed


def _process_alive(run_dir_name):
    """Best effort (POSIX only): is the process that owns this run folder still running?"""
    if os.name == "nt":
        return False                       # age alone decides; os.kill(pid, 0) would kill on Windows
    try:
        pid = int(run_dir_name.split("-")[1])
        os.kill(pid, 0)
        return True
    except (ValueError, IndexError, ProcessLookupError):
        return False
    except PermissionError:
        return True


class CubeRecord(dict):
    """
    A cube dict whose 'data' grid lives in a CubeStore.

    Everything except the grid (origin, spacing, dimensions, atoms,
    coordinates, bonds, unit_bohr, ...) is kept in the dict. cube['data']
    returns the grid from the store; the dict never holds a reference to it,
    so evicted grids really leave memory. Extra fields:
      id           stable identity (does not change when list rows move)
      grid_id      the store key for the grid
      data_min/max cached so whole-list statistics never page grids in
      source_path  the .cube file it was read from or saved to, if any
      label        display name
      unsaved      True for computed grids not yet exported by the user
      source_details  basis/coefficient metadata for population analysis
    """

    def __init__(self, store, cube, *, source_path=None, label=None, unsaved=False,
                 source_details=None, grid_id=None):
        super().__init__({k: v for k, v in cube.items() if k != "data"})
        self.store = store
        if grid_id is None:
            data = np.asarray(cube["data"])
            grid_id = store.put(data)
            lo, hi = (float(np.nanmin(data)), float(np.nanmax(data))) if data.size else (0.0, 0.0)
        else:
            lo, hi = cube.get("data_min"), cube.get("data_max")
            if lo is None or hi is None:
                data = store.get(grid_id)
                lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        dict.update(self, id=cube.get("id") or uuid.uuid4().hex, grid_id=grid_id,
                    data_min=lo, data_max=hi, source_path=source_path,
                    label=label, unsaved=bool(unsaved), source_details=source_details)

    def __getitem__(self, key):
        if key == "data":
            return self.store.get(dict.__getitem__(self, "grid_id"))
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key == "data":
            return self["data"]
        return dict.get(self, key, default)

    def __contains__(self, key):
        return key == "data" or dict.__contains__(self, key)

    def __setitem__(self, key, value):
        if key == "data":
            old = dict.get(self, "grid_id")
            data = np.asarray(value)
            dict.update(self, grid_id=self.store.put(data),
                        data_min=float(np.nanmin(data)), data_max=float(np.nanmax(data)))
            if old:
                self.store.release(old)
        else:
            dict.__setitem__(self, key, value)

    def release(self):
        """Give the grid back to the store (deletes its cache file)."""
        grid_id = dict.get(self, "grid_id")
        if grid_id:
            self.store.release(grid_id)
            dict.__setitem__(self, "grid_id", None)

    def metadata(self):
        """Everything but the grid and the store handle, for sessions."""
        return {k: v for k, v in dict.items(self) if k not in ("grid_id",)}


def _parse_numbers(text, path):
    """Whitespace-separated floats; a token that is not a number is an error.
    (np.fromstring otherwise stops at it with only a DeprecationWarning.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            return np.fromstring(text, sep=" ")
        except (DeprecationWarning, ValueError) as e:
            raise ValueError(f"Unreadable value in the data block of {path}") from e


def parse_cube_file(path):
    """
    Read a Gaussian .cube file. Returns a dict with origin, spacing, dimensions,
    atoms, coordinates, coord_lines, unit_bohr and data (float64, shape
    (nx, ny, nz), x slowest as in the file).

    The value block is parsed in ~8 MB chunks straight into a preallocated
    array, so peak memory stays close to the grid's own size (a Python list of
    every value took ~5x the grid). Fortran 'D' exponents are accepted. A
    negative atom count (orbital cubes) is followed by a line of orbital ids,
    which is skipped; files with more than one orbital are rejected.
    """
    with open(path, "r") as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        natoms_signed = int(parts[0])
        origin = np.array([float(x) for x in parts[1:4]])
        axes = []
        for _ in range(3):
            n, *vec = f.readline().split()
            axes.append((int(n), [float(x) for x in vec]))
        (nx, xvec), (ny, yvec), (nz, zvec) = axes
        # Same rule as the viewer's original reader (positive x spacing = bohr),
        # kept so existing files load exactly as before.
        unit_bohr = xvec[0] > 0
        nx, ny, nz = abs(nx), abs(ny), abs(nz)
        atoms, coords, coord_lines = [], [], []
        for _ in range(abs(natoms_signed)):
            p = f.readline().split()
            an = abs(int(float(p[0])))
            c = [float(x) for x in p[2:5]]
            atoms.append(an)
            coords.append(c)
            coord_lines.append([an] + c)
        if natoms_signed < 0:
            ids = f.readline().split()
            if ids and int(ids[0]) > 1:
                raise ValueError(f"{path} holds {ids[0]} orbitals; only single-orbital cubes are supported")
        count = nx * ny * nz
        data = np.empty(count, dtype=np.float64)
        filled = 0
        remainder = ""
        while True:
            chunk = f.read(_PARSE_CHUNK_BYTES)
            if not chunk:
                text, last = remainder, True
            else:
                chunk = remainder + chunk
                cut = chunk.rfind("\n")
                text, remainder = (chunk, "") if cut < 0 else (chunk[:cut], chunk[cut + 1:])
                last = False
            if "D" in text or "d" in text:
                text = text.replace("D", "E").replace("d", "e")
            values = _parse_numbers(text, path) if text.strip() else np.empty(0)
            if filled + values.size > count:
                raise ValueError(f"Data size mismatch in {path}: more than {count} values")
            data[filled:filled + values.size] = values
            filled += values.size
            if last:
                break
    if filled != count:
        raise ValueError(f"Data size mismatch in {path}: {filled} values, expected {count}")
    return dict(
        origin=origin,
        spacing=(xvec[0], yvec[1], zvec[2]),
        dimensions=(nx, ny, nz),
        atoms=np.array(atoms),
        coordinates=np.array(coords),
        coord_lines=coord_lines,
        unit_bohr=unit_bohr,
        data=data.reshape((nx, ny, nz)),
    )
