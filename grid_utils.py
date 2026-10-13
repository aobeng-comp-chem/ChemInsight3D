"""Shared cube-grid construction.

Every cube-producing path (nbo_read, fchk_read, read_molden, density_analysis)
needs the same regular Cartesian grid spanning the molecule's bounding box plus
a margin.  Keeping one implementation here means a change to the grid geometry
cannot apply to some source formats and not others.
"""

from __future__ import annotations

import time

import numpy as np


def build_uniform_grid(coordinates_ang, grid_quality, ext_dist, bohr_const):
    """
    Build a regular Cartesian grid in bohr covering the molecule plus a margin.

    Parameters
    ----------
    coordinates_ang : (natoms, 3) array-like of float
        Atomic coordinates in Angstrom.
    grid_quality : int
        Number of grid points along the widest axis; the other two axes get
        whatever point count the same isotropic spacing yields.
    ext_dist : float
        Margin in bohr added past the molecular bounding box on every side.
    bohr_const : float
        Angstrom per bohr.

    Returns
    -------
    points : (npoints, 3) ndarray
        Grid points in bohr, C-ordered with the x index varying slowest.
    shape : tuple[int, int, int]
        (nx, ny, nz).
    spacing : (3,) ndarray
        Isotropic spacing repeated per axis, in bohr.
    origin : (3,) ndarray
        Lowest grid corner in bohr.
    coordinates_bohr : (natoms, 3) ndarray
        The input coordinates converted to bohr, which callers need anyway to
        evaluate basis functions.
    """
    coordinates_bohr = np.asarray(coordinates_ang, dtype=float) / bohr_const

    ext_min = coordinates_bohr.min(axis=0) - ext_dist
    ext_max = coordinates_bohr.max(axis=0) + ext_dist
    ranges = ext_max - ext_min
    spacing = ranges[int(np.argmax(ranges))] / (grid_quality - 1)

    nx = int(round(ranges[0] / spacing)) + 1
    ny = int(round(ranges[1] / spacing)) + 1
    nz = int(round(ranges[2] / spacing)) + 1

    origin = ext_min
    x = np.arange(nx, dtype=float) * spacing + origin[0]
    y = np.arange(ny, dtype=float) * spacing + origin[1]
    z = np.arange(nz, dtype=float) * spacing + origin[2]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    return (points, (nx, ny, nz), np.array([spacing, spacing, spacing]),
            origin, coordinates_bohr)


def _orbital_evaluator(basis, coord_bohr, points, shape):
    """
    Build the per-orbital grid evaluator.

    Returns (evaluate, engine_name).  The compiled OpenMP engine is used when
    it is importable for this interpreter's ABI; otherwise a vectorised NumPy
    fallback computes the same sum over primitives.
    """
    try:
        import electron_density_opt_omp as cpp_engine
    except ImportError:
        cpp_engine = None

    if cpp_engine is not None:
        def evaluate(cmo):
            return cpp_engine.electron_density(
                basis, coord_bohr, points, cmo, None).reshape(shape)
        return evaluate, "C++ OpenMP"

    from angular_funct import ang_res_lamda

    def evaluate(cmo):
        """Pure-Python / NumPy vectorised fallback."""
        density = np.zeros(len(points))
        for basis_fn, c in zip(basis, cmo):
            if abs(c) <= 1e-15:
                continue
            atom_c = coord_bohr[basis_fn['CENTER'] - 1][:, np.newaxis]
            dx, dy, dz = points.T - atom_c
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            ang = ang_res_lamda(dx, dy, dz, basis_fn['orb_val'])
            for coeff, zeta in zip(basis_fn['coeffs'], basis_fn['exps']):
                density += np.round(c * coeff * ang * np.exp(-zeta * r**2), 99)
        return density.reshape(shape)

    return evaluate, "Python (NumPy)"


def evaluate_orbital_grids(basis, coordinates_ang, atom_info, cmos,
                           orbital_indices, label_base,
                           grid_quality, ext_dist, bohr_const):
    """
    Evaluate one cube grid per supplied MO coefficient vector.

    This is the body that nbo_read.compute_cube_data, fchk_read's and
    read_molden's equivalents all shared verbatim; they now differ only in how
    they obtain *basis* and *cmos*.

    Returns
    -------
    List of dicts, one per orbital:
        {'index': int,          # 1-based orbital index
         'label': str,          # e.g. "molecule-7"
         'grid':  ndarray,      # shape (nx, ny, nz)
         'nx': int, 'ny': int, 'nz': int,
         'spacing': ndarray,    # [sx, sy, sz] in bohr
         'origin':  ndarray,    # [ox, oy, oz] in bohr
         'atom_info': list,     # (Z, x_ang, y_ang, z_ang) tuples
         'bohr_const': float}
    """
    points, shape, spacing, origin, coord_bohr = build_uniform_grid(
        coordinates_ang, grid_quality, ext_dist, bohr_const)
    nx, ny, nz = shape
    evaluate, engine_name = _orbital_evaluator(basis, coord_bohr, points, shape)

    results = []
    started = time.time()
    for cmo, idx in zip(cmos, orbital_indices):
        results.append({
            'index':      idx,
            'label':      f"{label_base}-{idx}",
            'grid':       evaluate(cmo),
            'nx': nx, 'ny': ny, 'nz': nz,
            'spacing':    spacing.copy(),
            'origin':     origin.copy(),
            'atom_info':  atom_info,
            'bohr_const': bohr_const,
        })
    print(f"[{engine_name}] total grid generation: {time.time() - started:.3f}s "
          f"for {len(orbital_indices)} orbital(s)")
    return results
