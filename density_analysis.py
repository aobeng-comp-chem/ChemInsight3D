import os

import numpy as np

from angular_funct import ang_res_lamda
from localization_io import (
    _get_occupation_arrays,
    _num_occupied_from_arrays,
    _recognize_source_type,
    get_localization_inputs,
)


def _prepare_basis_set(basis_set):
    """
    Convert the basis-set container to an ordered list.

    The ordering must be exactly the same as the AO ordering used
    in the density matrix.
    """
    if isinstance(basis_set, dict):
        basis_functions = list(basis_set.values())
    else:
        basis_functions = list(basis_set)

    required_keys = {"CENTER", "orb_val", "coeffs", "exps"}

    for index, basis in enumerate(basis_functions):
        missing = required_keys.difference(basis)

        if missing:
            raise KeyError(
                f"Basis function {index} is missing keys: "
                f"{sorted(missing)}"
            )

    return basis_functions


def basis_values_at_points(
    basis_set,
    coordinates,
    points,
):
    """
    Evaluate all contracted AO basis functions at multiple points.

    Parameters
    ----------
    basis_set : sequence of dict or dict of dict
        Basis-function information. Each basis function must contain:

        CENTER
            One-based index of the atom on which the AO is centered.
        orb_val
            Orbital/angular-function identifier accepted by
            ``ang_res_lamda``.
        coeffs
            Primitive contraction coefficients.
        exps
            Primitive Gaussian exponents.

    coordinates : array_like, shape (n_atoms, 3)
        Atomic coordinates.

    points : array_like, shape (n_points, 3)
        Points where the basis functions are evaluated.

    Returns
    -------
    ao_values : ndarray, shape (n_points, n_basis)
        ``ao_values[p, i]`` is basis function ``i`` evaluated at
        point ``p``.
    """
    basis_functions = _prepare_basis_set(basis_set)

    coordinates = np.asarray(coordinates, dtype=float)
    points = np.asarray(points, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape (n_atoms, 3)."
        )

    # Permit a single point with shape (3,).
    if points.ndim == 1:
        if points.shape != (3,):
            raise ValueError(
                "A single point must have shape (3,)."
            )
        points = points[None, :]

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points must have shape (n_points, 3)."
        )

    n_points = points.shape[0]
    n_basis = len(basis_functions)

    # Complex dtype also supports complex-valued angular functions.
    ao_values = np.zeros(
        (n_points, n_basis),
        dtype=np.complex128,
    )

    for basis_index, basis in enumerate(basis_functions):
        center_index = int(basis["CENTER"]) - 1

        if not 0 <= center_index < len(coordinates):
            raise IndexError(
                f"Basis function {basis_index} has CENTER="
                f"{basis['CENTER']}, but there are only "
                f"{len(coordinates)} atoms."
            )

        center = coordinates[center_index]

        # Displacements from the AO center to every requested point.
        displacement = points - center

        dx = displacement[:, 0]
        dy = displacement[:, 1]
        dz = displacement[:, 2]

        r_squared = np.einsum(
            "pi,pi->p",
            displacement,
            displacement,
        )

        coeffs = np.asarray(
            basis["coeffs"],
            dtype=np.complex128,
        )

        exponents = np.asarray(
            basis["exps"],
            dtype=float,
        )

        if coeffs.ndim != 1 or exponents.ndim != 1:
            raise ValueError(
                f"coeffs and exps for basis function "
                f"{basis_index} must be one-dimensional."
            )

        if coeffs.size != exponents.size:
            raise ValueError(
                f"Basis function {basis_index} has "
                f"{coeffs.size} coefficients but "
                f"{exponents.size} exponents."
            )

        if np.any(exponents <= 0.0):
            raise ValueError(
                f"Basis function {basis_index} contains a "
                "nonpositive Gaussian exponent."
            )

        angular_part = np.asarray(
            [
                ang_res_lamda(
                    dx_value,
                    dy_value,
                    dz_value,
                    basis["orb_val"],
                )
                for dx_value, dy_value, dz_value
                in zip(dx, dy, dz)
            ],
            dtype=np.complex128,
        )

        primitive_exponentials = np.exp(
            -r_squared[:, None] * exponents[None, :]
        )

        radial_part = primitive_exponentials @ coeffs

        ao_values[:, basis_index] = angular_part * radial_part

    return ao_values


def _density_from_ao_values(
    ao_values,
    denmat,
    *,
    check_hermitian=True,
    atol=1.0e-10,
):
    """
    Contract precomputed AO values against a density matrix:

        rho(r) = chi(r)^H @ denmat @ chi(r)

    Factored out of spin_density_at_points so that
    compute_spin_density_cube_data can evaluate the (expensive) AO basis
    on the grid once and reuse it for every requested density type,
    instead of re-evaluating the basis functions per type.
    """
    denmat = np.asarray(denmat)
    n_basis = ao_values.shape[1]

    if denmat.shape != (n_basis, n_basis):
        raise ValueError(
            f"Density matrix has shape {denmat.shape}; "
            f"expected {(n_basis, n_basis)}."
        )

    if not np.all(np.isfinite(denmat)):
        raise ValueError(
            "The density matrix contains non-finite values."
        )

    if check_hermitian and not np.allclose(
        denmat,
        denmat.conj().T,
        atol=atol,
        rtol=0.0,
    ):
        maximum_error = np.max(np.abs(denmat - denmat.conj().T))
        raise ValueError(
            "The density matrix is not Hermitian. "
            f"Maximum deviation: {maximum_error:.3e}"
        )

    densities = np.einsum(
        "pi,ij,pj->p",
        ao_values.conj(),
        denmat,
        ao_values,
        optimize=True,
    )

    densities = np.real_if_close(densities, tol=1000)

    if np.iscomplexobj(densities):
        maximum_imaginary = np.max(np.abs(densities.imag))
        raise ValueError(
            "The calculated density has a significant imaginary "
            f"component: {maximum_imaginary:.3e}"
        )

    return np.asarray(densities.real, dtype=float)


def spin_density_at_points(
    basis_set,
    coordinates,
    points,
    denmat,
    *,
    check_hermitian=True,
    atol=1.0e-10,
):
    """
    Evaluate the spin density at one or more spatial points.

    The density is

        rho_s(r) = chi(r)^H @ denmat @ chi(r)
    """
    basis_functions = _prepare_basis_set(basis_set)

    ao_values = basis_values_at_points(
        basis_functions,
        coordinates,
        points,
    )

    return _density_from_ao_values(
        ao_values,
        denmat,
        check_hermitian=check_hermitian,
        atol=atol,
    )


def nuclear_spin_density(
    basis_set,
    coordinates,
    point,
    denmat,
    *,
    check_hermitian=True,
):
    """
    Evaluate the spin density at one point.
    """
    densities = spin_density_at_points(
        basis_set=basis_set,
        coordinates=coordinates,
        points=point,
        denmat=denmat,
        check_hermitian=check_hermitian,
    )

    return float(densities[0])


def _build_uniform_grid(coordinates_ang, grid_quality, ext_dist, bohr_const):
    """Build a regular Cartesian grid in bohr units for cube generation."""
    coordinates_ang = np.asarray(coordinates_ang, dtype=float)
    coordinates_bohr = coordinates_ang / bohr_const

    ext_min = coordinates_bohr.min(axis=0) - ext_dist
    ext_max = coordinates_bohr.max(axis=0) + ext_dist
    ranges = ext_max - ext_min
    spacing = ranges[np.argmax(ranges)] / (grid_quality - 1)

    nx = int(round(ranges[0] / spacing)) + 1
    ny = int(round(ranges[1] / spacing)) + 1
    nz = int(round(ranges[2] / spacing)) + 1

    origin = ext_min
    x = np.arange(nx, dtype=float) * spacing + origin[0]
    y = np.arange(ny, dtype=float) * spacing + origin[1]
    z = np.arange(nz, dtype=float) * spacing + origin[2]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    return points, (nx, ny, nz), np.array([spacing, spacing, spacing]), origin, coordinates_bohr


_DENSITY_LABELS = {
    "alpha": "ALPHA_DENSITY",
    "beta": "BETA_DENSITY",
    "total": "TOTAL_DENSITY",
    "spin": "SPIN_DENSITY",
}


def _occupied_densities_at_points(basis_set, coordinates, points, alpha_occ, beta_occ):
    """
    Evaluate alpha and beta electron density on a grid from the occupied-
    orbital coefficient matrices alone (P_spin = C_occ @ C_occ.T), without
    ever forming the n_basis x n_basis density matrix:

        rho_spin(r) = chi(r)^H @ (C_occ @ C_occ.T) @ chi(r)
                    = sum_k |sum_i C_occ[i, k] * chi_i(r)|^2

    Prefers the compiled electron_density_opt_omp.occupied_density()
    kernel -- the same OpenMP/solid-harmonic engine used for orbital cube
    generation (electron_density()), reformulated to evaluate chi(r) once
    per grid point and reuse it across every occupied orbital, which costs
    O(n_basis * n_occ) per point instead of the O(n_basis^2) of an
    explicit density-matrix contraction. Falls back to a NumPy matmul
    when the extension isn't built.
    """
    basis_functions = _prepare_basis_set(basis_set)

    try:
        import electron_density_opt_omp as _cpp
    except ImportError:
        _cpp = None

    if _cpp is not None and hasattr(_cpp, "occupied_density"):
        alpha_density, beta_density = _cpp.occupied_density(
            basis_functions, coordinates, points, alpha_occ, beta_occ,
        )
        return np.asarray(alpha_density), np.asarray(beta_density)

    ao_values = basis_values_at_points(basis_functions, coordinates, points)
    alpha_density = (np.abs(ao_values @ alpha_occ) ** 2).sum(axis=1)
    beta_density = (np.abs(ao_values @ beta_occ) ** 2).sum(axis=1)
    return alpha_density, beta_density


def compute_spin_density_cube_data(
    path,
    key_path=None,
    grid_quality=75,
    ext_dist=4.0,
    bohr_const=0.529177249,
    density_type="spin",
):
    """
    Build one or more one-particle density cubes on a shared regular 3D
    grid, from occupied-orbital coefficients (see
    _occupied_densities_at_points).

    ``density_type`` is a single string, or an iterable of strings, drawn
    from:

        alpha   alpha density
        beta    beta density
        total   alpha + beta
        spin    alpha - beta   (default)

    Alpha and beta densities are evaluated once no matter how many types
    are requested, and every requested type is then a cheap elementwise
    combination of those two grids.

    For closed-shell systems alpha == beta, so "beta" duplicates
    "alpha", "total" is twice "alpha", and "spin" is identically zero --
    callers that want a meaningful cube for a closed-shell system should
    request "total".

    Returns a dictionary with:

        cubes           {density_type: cube_dict, ...}, one entry per
                        requested type. Each cube_dict matches the cube
                        format expected by the viewer (``grid``, ``nx``,
                        ``ny``, ``nz``, ``spacing``, ``origin``,
                        ``atom_info``, ``bohr_const``).
        is_open_shell   bool
        n_occ_alpha     int
        n_occ_beta      int
    """
    requested_types = (
        (density_type,) if isinstance(density_type, str) else tuple(density_type)
    )

    if not requested_types:
        raise ValueError("density_type must name at least one density.")

    unknown = [dt for dt in requested_types if dt not in _DENSITY_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown density_type(s) {unknown}; expected values from "
            f"{sorted(_DENSITY_LABELS)}."
        )

    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        key_path = os.path.splitext(path)[0] + ".40"
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Spin density for NBO sources requires a sibling .40 key file "
                f"(the canonical AO-basis MOs) -- expected {key_path}."
            )
        import nbo_read as _nr
        final_basis, coordinates_ang, atom_info = _nr.load_basis_headless(path)
    elif source_type == "fchk":
        key_path = None
        import fchk_read as _fr
        final_basis, coordinates_ang, atom_info = _fr.load_basis_from_fchk(path)
    elif source_type == "molden":
        key_path = None
        import read_molden as _mr
        final_basis, coordinates_ang, atom_info = _mr.load_basis_from_molden(path)
    else:
        raise ValueError(f"Unrecognized source file: {path}")

    occ_source_type, occ_alpha, occ_beta = _get_occupation_arrays(path, key_path=key_path)
    is_open_shell = occ_beta is not None

    n_occ_alpha = _num_occupied_from_arrays(
        occ_source_type, occ_alpha, occ_beta, spin="alpha", path=path
    )
    n_occ_beta = (
        _num_occupied_from_arrays(
            occ_source_type, occ_alpha, occ_beta, spin="beta", path=path
        )
        if is_open_shell else 0
    )

    alpha_cmo, _, _ = get_localization_inputs(path, key_path=key_path, spin="alpha")
    alpha_occ = alpha_cmo[:, :n_occ_alpha]

    if is_open_shell:
        beta_cmo, _, _ = get_localization_inputs(path, key_path=key_path, spin="beta")
        beta_occ = beta_cmo[:, :n_occ_beta]
    else:
        # Closed-shell alpha and beta densities are identical.
        beta_occ = alpha_occ

    points, shape, spacing, origin, coordinates_bohr = _build_uniform_grid(
        coordinates_ang,
        grid_quality=grid_quality,
        ext_dist=ext_dist,
        bohr_const=bohr_const,
    )

    alpha_grid, beta_grid = _occupied_densities_at_points(
        final_basis, coordinates_bohr, points, alpha_occ, beta_occ,
    )

    density_grids = {
        "alpha": alpha_grid,
        "beta": beta_grid,
        "total": alpha_grid + beta_grid,
        "spin": alpha_grid - beta_grid,
    }

    base = os.path.splitext(os.path.basename(path))[0]
    cubes = {}
    for dtype in requested_types:
        cubes[dtype] = {
            "index": 1,
            "label": f"{base}_{_DENSITY_LABELS[dtype]}",
            "grid": density_grids[dtype].reshape(shape),
            "nx": shape[0],
            "ny": shape[1],
            "nz": shape[2],
            "spacing": spacing,
            "origin": origin,
            "atom_info": atom_info,
            "bohr_const": bohr_const,
        }

    return {
        "cubes": cubes,
        "is_open_shell": is_open_shell,
        "n_occ_alpha": n_occ_alpha,
        "n_occ_beta": n_occ_beta,
    }

