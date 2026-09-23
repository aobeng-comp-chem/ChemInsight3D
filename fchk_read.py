"""
fchk_read.py
============
Reads a Gaussian formatted checkpoint (.fchk) file and exposes the same
interface that nbo_read.py provides for .47/.31 files, so that
chemview.py can call it uniformly.

Public API (mirrors nbo_read equivalents)
------------------------------------------
load_basis_from_fchk(fchk_path)
    → (final_norm_basis, coordinates_ang, atom_info)
    Parses basis set, normalises basis functions via the same pipeline
    used by nbo_read (convert_to_molden → normalize_by_self_overlap →
    iterative overlap normalisation).

get_orbital_count_fchk(fchk_path)
    → (orbital_type_str, nbas, is_open_shell)

get_orbital_energies_and_occupations_fchk(fchk_path)
    → (ene_alpha, occ_alpha, ene_beta, occ_beta)

load_cmos_from_fchk(fchk_path, orbital_indices, spin='alpha')
    → list of 1-D numpy arrays (one per requested orbital)

compute_cube_data_fchk(fchk_path, orbital_indices, spin,
                        grid_quality, ext_dist, bohr_const)
    → same list-of-dicts format as nbo_read.compute_cube_data()
"""

import re
import os
import math
import copy
import time
import numpy as np
from scipy.constants import physical_constants
from source_cache import ComputationCache, file_cache_key


_SOURCE_CACHE = ComputationCache()


def clear_source_cache():
    """Clear all parsed and derived fchk data (primarily useful in tests)."""
    _SOURCE_CACHE.clear()


def _read_fchk_lines(fchk_path):
    cache_key = ("lines", file_cache_key(fchk_path))

    def load():
        with open(fchk_path, "r") as handle:
            return handle.read().splitlines()

    return _SOURCE_CACHE.get(cache_key, load)

# ---------------------------------------------------------------------------
# Low-level fchk parser
# ---------------------------------------------------------------------------

def _parse_scalar(lines, name, dtype=float):
    """Parse a scalar (non-array) fchk entry: 'Name   R   value'."""
    for line in lines:
        if line.startswith(name):
            parts = line.split()
            try:
                return dtype(parts[-1])
            except (ValueError, IndexError):
                pass
    return None


def _unpack_triangular(flat, n):
    """
    Unpack a Gaussian fchk packed lower-triangular matrix (row-major,
    n*(n+1)/2 elements — e.g. 'Total SCF Density') into a full symmetric
    n x n array.
    """
    mat = np.zeros((n, n))
    tril_idx = np.tril_indices(n)
    mat[tril_idx] = flat
    mat[(tril_idx[1], tril_idx[0])] = flat
    return mat


def _parse_array(lines, name, dtype=float):
    """
    Parse an array fchk section:
        Name   [IRC]   N=   <count>
        <data lines...>
    Returns a numpy array, or None if section not found.
    """
    header_re = re.compile(
        r'^' + re.escape(name) + r'\s+[IRCL]\s+N=\s*(\d+)$'
    )
    data = []
    reading = False
    n_expected = None

    for line in lines:
        if not reading:
            m = header_re.match(line.rstrip())
            if m:
                n_expected = int(m.group(1))
                reading = True
        else:
            # Stop at the next section header
            if re.match(r'^[A-Za-z]', line.strip()) and line.strip():
                break
            tokens = line.strip().split()
            if dtype == float:
                data.extend(float(t) for t in tokens)
            elif dtype == int:
                data.extend(int(t) for t in tokens)
            else:
                data.extend(tokens)

    if not reading:
        return None

    arr = np.array(data, dtype=dtype)
    if n_expected is not None and arr.size != n_expected:
        raise ValueError(
            f"fchk section '{name}': expected {n_expected} values, got {arr.size}"
        )
    return arr


# ---------------------------------------------------------------------------
# Basis-set extraction from fchk
# ---------------------------------------------------------------------------

# Angular-momentum label maps matching nbo_read conventions
_COMP_MAP = {
    "s": 1, "p": 3, "d": 5, "f": 7, "g": 9, "h": 11, "i": 13, "j": 15
}
_LABEL_MAP = {
    "s":  ["s"],
    "p":  ["px", "py", "pz"],
    "d":  ["d0", "ds1", "dc1", "dc2", "ds2"],
    "f":  ["f0", "fc1", "fs1", "fc2", "fs2", "fc3", "fs3"],
    "g":  ["g0", "gc1", "gs1", "gc2", "gs2", "gc3", "gs3", "gc4", "gs4"],
    "h":  ["h0", "hc1", "hs1", "hc2", "hs2", "hc3", "hs3",
           "hc4", "hs4", "hc5", "hs5"],
    "i":  ["i0", "ic1", "is1", "ic2", "is2", "ic3", "is3",
           "ic4", "is4", "ic5", "is5", "ic6", "is6"],
    "j":  ["j0", "jc1", "js1", "jc2", "js2", "jc3", "js3",
           "jc4", "js4", "jc5", "js5", "jc6", "js6", "jc7", "js7"],
}

# Gaussian shell-type integers → angular-momentum string
_TYPE_MAP = {
    0: "s",  1: "p",  2: "d",  3: "f",  4: "g",  5: "h",  6: "i",  7: "j",
    -1: "sp",
    -2: "d", -3: "f", -4: "g", -5: "h", -6: "i", -7: "j",
}


def _extract_basis_set(lines):
    """
    Parse fchk basis-set sections and return a list of basis-function dicts
    using the same field names as nbo_read:
        N, CENTER, shell_num, type, orb_val, exps, coeffs,
        xcenter, ycenter, zcenter
    Coordinates are in Angstrom (converted from bohr as stored in fchk).
    """
    bohr_to_ang = physical_constants['Bohr radius'][0] * 1e10

    shell_types    = _parse_array(lines, "Shell types",                        int)
    num_primitives = _parse_array(lines, "Number of primitives per shell",     int)
    shell_to_atom  = _parse_array(lines, "Shell to atom map",                  int)
    prim_exponents = _parse_array(lines, "Primitive exponents",                float)
    contr_coeffs   = _parse_array(lines, "Contraction coefficients",           float)
    shell_coords   = _parse_array(lines, "Coordinates of each shell",          float)
    
    LABEL_MAPPING = {
    1: 's', 51: 's',
    101: 'px', 102: 'py', 103: 'pz',
    151: 'px', 152: 'py', 153: 'pz',
    251: 'ds2', 252: 'ds1', 253: 'dc1', 254: 'dc2', 255: 'd0',
    351: 'f0', 352: 'fc1', 353: 'fs1', 354: 'fc2', 355: 'fs2',
    356: 'fc3', 357: 'fs3',
    451: 'g0', 452: 'gc1', 453: 'gs1', 454: 'gc2', 455: 'gs2',
    456: 'gc3', 457: 'gs3', 458: 'gc4', 459: 'gs4',
    551: 'h0', 552: 'hc1', 553: 'hs1', 554: 'hc2', 555: 'hs2',
    556: 'hc3', 557: 'hs3', 558: 'hc4', 559: 'hs4', 560: 'hc5', 561: 'hs5',
    651: 'i0', 652: 'ic1', 653: 'is1', 654: 'ic2', 655: 'is2',
    656: 'ic3', 657: 'is3', 658: 'ic4', 659: 'is4',
    660: 'ic5', 661: 'is5', 662: 'ic6', 663: 'is6',
    751: 'j0', 752: 'jc1', 753: 'js1', 754: 'jc2', 755: 'js2',
    756: 'jc3', 757: 'js3', 758: 'jc4', 759: 'js4',
    760: 'jc5', 761: 'js5', 762: 'jc6', 763: 'js6', 764: 'jc7', 765: 'js7',
}
    
    ORBVAL_TO_LABELCODE = {}
    for code, lbl in LABEL_MAPPING.items():
        
        ORBVAL_TO_LABELCODE.setdefault(lbl, code)


    if any(a is None for a in [shell_types, num_primitives, shell_to_atom,
                                prim_exponents, contr_coeffs, shell_coords]):
        raise ValueError("fchk file is missing one or more required basis sections.")

    shell_coords = shell_coords.reshape(-1, 3) #* bohr_to_ang 

    # P(S=P) coefficients (only present when SP shells exist)
    psp_coeffs = _parse_array(lines, "P(S=P) Contraction coefficients", float)

    basis = []
    prim_idx    = 0
    fn_counter  = 0

    for shell_idx, stype in enumerate(shell_types):
        shell_type = _TYPE_MAP.get(int(stype), f"type{stype}")
        n_prim     = int(num_primitives[shell_idx])
        atom_idx   = int(shell_to_atom[shell_idx])
        coords     = shell_coords[shell_idx]                    
        exps       = prim_exponents[prim_idx:prim_idx + n_prim].tolist()
        s_coeffs   = contr_coeffs[prim_idx:prim_idx + n_prim].tolist()

        if shell_type == "sp":
            # S component
            basis.append({
                "N":         fn_counter + 1,
                "CENTER":    atom_idx,
                "shell_num": shell_idx + 1,
                "type":  "s",
                "orb_val":   "s",
                "exps":      exps,
                "coeffs":    s_coeffs,
                "xcenter":   float(coords[0]),
                "ycenter":   float(coords[1]),
                "zcenter":   float(coords[2]),
            })
            fn_counter += 1

            # P components — use P(S=P) coefficients
            if psp_coeffs is None:
                raise RuntimeError(
                    "SP shell found but 'P(S=P) Contraction coefficients' "
                    "section is missing from fchk file."
                )
            p_coeffs = psp_coeffs[prim_idx:prim_idx + n_prim].tolist()
            for label in ("px", "py", "pz"):
                basis.append({
                    "N":         fn_counter + 1,
                    "CENTER":    atom_idx,
                    "shell_num": shell_idx + 1,
                    "type":  label,
                    "orb_val":   label,
                    "exps":      exps,
                    "coeffs":    p_coeffs,
                    "xcenter":   float(coords[0]),
                    "ycenter":   float(coords[1]),
                    "zcenter":   float(coords[2]),
                })
                fn_counter += 1

        else:
            n_comp      = _COMP_MAP.get(shell_type, 1)
            comp_labels = _LABEL_MAP.get(shell_type, [shell_type] * n_comp)
            for j in range(n_comp):
                basis.append({
                    "N":         fn_counter + 1,
                    "CENTER":    atom_idx,
                    "shell_num": shell_idx + 1,
                    "type":  comp_labels[j],
                    "orb_val":   comp_labels[j],
                    "exps":      exps,
                    "coeffs":    s_coeffs,
                    "xcenter":   float(coords[0]),
                    "ycenter":   float(coords[1]),
                    "zcenter":   float(coords[2]),
                })
                fn_counter += 1

        prim_idx += n_prim
        
    for bf in basis:
        ov = bf["orb_val"]
        bf["LABEL"] = ORBVAL_TO_LABELCODE.get(ov)

    return basis


def _extract_atoms(lines):
    """
    Return atom coordinates and atomic numbers from the fchk file.

    fchk stores:
        'Atomic numbers'           I  N= <natoms>
        'Current cartesian coordinates'  R  N= 3*natoms   (in bohr)

    Returns:
        coordinates_ang : list of (x, y, z) in Angstrom
        atom_info       : list of (Z, x, y, z) in Angstrom
    """
    bohr_to_ang = physical_constants['Bohr radius'][0] * 1e10

    atomic_nums = _parse_array(lines, "Atomic numbers", int)
    cart_coords = _parse_array(lines, "Current cartesian coordinates", float)

    if atomic_nums is None or cart_coords is None:
        raise ValueError(
            "fchk file is missing 'Atomic numbers' or "
            "'Current cartesian coordinates' sections."
        )

    natoms = len(atomic_nums)
    coords_bohr = cart_coords.reshape(natoms, 3)
    coords_ang  = coords_bohr * bohr_to_ang

    coordinates_ang = [tuple(row) for row in coords_ang]
    atom_info       = [(int(atomic_nums[i]),) + coordinates_ang[i]
                       for i in range(natoms)]
    return coordinates_ang, atom_info


# ---------------------------------------------------------------------------
# Normalisation (delegates to nbo_read pipeline)
# ---------------------------------------------------------------------------

def _normalise_basis(raw_basis):
    """
    Apply the same two-stage normalisation used by nbo_read.load_basis_headless:
        1. convert_to_molden  (primitive Gaussian normalisation)
        2. normalize_by_self_overlap  (contracted-function normalisation)

    Returns final_norm_basis ready for compute_cube_data.
    Note: fchk MOs are already in the orthonormal MO basis, so the additional
    iterative_basis_modification step used for .47 files is NOT needed here.
    """
    import nbo_read as _nr
    from overlap_matrix import get_overlap_matrix as getSmat
    from bas_dict import dict_keys

  
    molden_basis = _nr.convert_to_molden(raw_basis)
    basis_info_dict = _nr.normalize_by_self_overlap(molden_basis)
    Smat = getSmat(basis_info_dict, dict_keys, normalize_primitives=False, diagonal_only=False)
    final_basis = _nr.normalize_basis_info(basis_info_dict, Smat)
    norm_basis = _sort_basis_by_shell_label_order(final_basis)

    
    return norm_basis


# ---------------------------------------------------------------------------
# Public API — mirrors nbo_read
# ---------------------------------------------------------------------------

def load_basis_from_fchk(fchk_path):
    """
    Parse and normalise the basis set from a .fchk file.

    Returns
    -------
    final_norm_basis : list of basis-function dicts (same format as nbo_read)
    coordinates_ang  : list of (x, y, z) in Angstrom, one per atom
    atom_info        : list of (Z, x, y, z) in Angstrom, one per atom
    """
    cache_key = ("basis", file_cache_key(fchk_path))

    def load():
        lines = _read_fchk_lines(fchk_path)
        raw_basis = _extract_basis_set(lines)
        coordinates_ang, atom_info = _extract_atoms(lines)
        final_norm_basis = _normalise_basis(raw_basis)
        return final_norm_basis, coordinates_ang, atom_info

    return _SOURCE_CACHE.get(cache_key, load)


def get_ao_overlap_matrix(fchk_path):
    """Return the final normalized-basis overlap, computed once per file."""
    cache_key = ("ao_overlap", file_cache_key(fchk_path))

    def load():
        from overlap_matrix import get_overlap_matrix
        from bas_dict import dict_keys

        final_basis, _, _ = load_basis_from_fchk(fchk_path)
        return get_overlap_matrix(
            final_basis, dict_keys,
            normalize_primitives=False, diagonal_only=False,
        )

    return _SOURCE_CACHE.get(cache_key, load)


def _get_orbital_count_fchk_uncached(fchk_path):
    """
    Return (orbital_type_str, nbas, is_open_shell) for a .fchk file.

    orbital_type_str is always 'CMO' (canonical MOs from fchk).
    nbas is the number of basis functions.
    is_open_shell is True when Beta MO coefficients are present.
    """
    lines = _read_fchk_lines(fchk_path)

    nbas_val = _parse_scalar(lines, "Number of basis functions", int)
    if nbas_val is None:
        # fall back: count from basis extraction
        raw_basis = _extract_basis_set(lines)
        nbas_val  = len(raw_basis)

    # Open-shell: fchk has separate Alpha and Beta MO coefficient blocks
    has_beta = any(
        line.startswith("Beta MO coefficients") for line in lines
    )
    return "CMO", int(nbas_val), has_beta

def get_orbital_count_fchk(fchk_path):
    """Return cached fchk orbital count and open-shell metadata."""
    cache_key = ("orbital_count", file_cache_key(fchk_path))
    return _SOURCE_CACHE.get(
        cache_key, lambda: _get_orbital_count_fchk_uncached(fchk_path)
    )



def _get_density_matrices(lines, nbas):
    """
    Derive alpha/beta density matrices from fchk's packed density sections.
        Total SCF Density (T) = alpha density + beta density
        Spin SCF Density  (S) = alpha density - beta density   (open-shell only)
            alpha density = (T + S) / 2
            beta density  = (T - S) / 2
    Returns (density_alpha, density_beta); both None if 'Total SCF Density'
    is absent, and beta falls back to T/2 when 'Spin SCF Density' is absent
    (closed-shell: alpha and beta densities are identical).
    """
    tot_flat = _parse_array(lines, "Total SCF Density", float)
    if tot_flat is None:
        return None, None
    total_density = _unpack_triangular(tot_flat, nbas)

    spin_flat = _parse_array(lines, "Spin SCF Density", float)
    if spin_flat is not None:
        spin_density  = _unpack_triangular(spin_flat, nbas)
        density_alpha = (total_density + spin_density) / 2
        density_beta  = (total_density - spin_density) / 2
    else:
        density_alpha = total_density / 2
        density_beta  = total_density / 2
    return density_alpha, density_beta


def _get_orbital_energies_and_occupations_fchk_uncached(fchk_path):
    """
    Return (ene_alpha, occ_alpha, ene_beta, occ_beta) from a .fchk file.
    All energies are in Hartree.  Beta arrays are None for closed-shell.
    """
    lines = _read_fchk_lines(fchk_path)

    ene_alpha = _parse_array(lines, "Alpha Orbital Energies", float)
    occ_alpha = _parse_array(lines, "Alpha Orbital occupancies", float)

    # Closed-shell fchk stores only Alpha sections even for a RHF
    ene_beta  = _parse_array(lines, "Beta Orbital Energies", float)
    occ_beta  = _parse_array(lines, "Beta Orbital occupancies", float)

    need_alpha = occ_alpha is None and ene_alpha is not None
    need_beta  = occ_beta is None and ene_beta is not None

    if need_alpha or need_beta:
        nbas = ene_alpha.size if ene_alpha is not None else ene_beta.size
        density_alpha, density_beta = _get_density_matrices(lines, nbas)

        # Restricted (closed-shell) fchk files store only one MO set for
        # both spins -- there is no "Beta Orbital Energies" section at all.
        # Its orbitals must report the *total* (0-2) occupation of that
        # spatial orbital, matching every other source-format backend
        # (nbo_read.py folds NBO's total density into the same "alpha"
        # slot for closed shell; molden's file format does the same), not
        # just the alpha-spin share (0-1) that
        # calculate_occupation_numbers(..., density_alpha) alone gives.
        closed_shell = ene_beta is None

        # Occupancy sections are rarely written for plain SCF/DFT jobs, but
        # the density matrix is: derive real occupation numbers from it
        # (n_i = C_i^T S D S C_i) instead of assuming aufbau filling.
        smat = None
        if (need_alpha and density_alpha is not None) or (need_beta and density_beta is not None):
            smat = get_ao_overlap_matrix(fchk_path)

        if need_alpha:
            if density_alpha is not None:
                C = _load_cmo_matrix_fchk(fchk_path, "alpha").T
                density_for_occ = (
                    density_alpha + density_beta if closed_shell else density_alpha
                )
                occ_alpha = calculate_occupation_numbers(C, smat, density_for_occ)
            else:
                n_elec_alpha = _parse_scalar(lines, "Number of alpha electrons", int)
                occ_alpha    = np.zeros(nbas)
                if n_elec_alpha:
                    occ_alpha[:n_elec_alpha] = 2.0 if closed_shell else 1.0

        if need_beta:
            if density_beta is not None:
                C = _load_cmo_matrix_fchk(fchk_path, "beta").T
                occ_beta = calculate_occupation_numbers(C, smat, density_beta)
            else:
                n_elec_beta = _parse_scalar(lines, "Number of beta electrons", int)
                occ_beta     = np.zeros(nbas)
                if n_elec_beta:
                    occ_beta[:n_elec_beta] = 1.0

    return (
        ene_alpha if ene_alpha is not None else np.array([]),
        occ_alpha if occ_alpha is not None else np.array([]),
        ene_beta,
        occ_beta,
    )


def get_orbital_energies_and_occupations_fchk(fchk_path):
    """Cached form of the fchk orbital metadata extraction."""
    cache_key = ("orbital_metadata", file_cache_key(fchk_path))
    return _SOURCE_CACHE.get(
        cache_key,
        lambda: _get_orbital_energies_and_occupations_fchk_uncached(fchk_path),
    )


def _sort_indices_by_energy(fchk_path, spin="alpha", ascending=True):
    """Return 1-based MO indices sorted by energy."""
    ene_alpha, _, ene_beta, _ = get_orbital_energies_and_occupations_fchk(fchk_path)
    if spin.lower().startswith("b"):
        if ene_beta is None:
            raise ValueError("Beta orbital energies not present in this fchk")
        energies = np.asarray(ene_beta)
    else:
        energies = np.asarray(ene_alpha)
    if energies.size == 0:
        _, nbas, _ = get_orbital_count_fchk(fchk_path)
        return list(range(1, nbas + 1))
    order = np.argsort(energies)
    if not ascending:
        order = order[::-1]
    return [int(i) + 1 for i in order]


def load_cmos_from_fchk(fchk_path, orbital_indices, spin="alpha"):
    """
    Load CMO row vectors for requested 1-based orbital_indices from a .fchk.

    Parameters
    ----------
    fchk_path       : str
    orbital_indices : list of int (1-based)
    spin            : 'alpha' or 'beta'

    Returns
    -------
    List of 1-D numpy arrays, one per requested orbital.
    """
    # fchk stores MOs row-by-row: row i = orbital i coefficients.
    mo_matrix = _load_cmo_matrix_fchk(fchk_path, spin)
    return [mo_matrix[i - 1] for i in orbital_indices]


def _load_cmo_matrix_fchk(fchk_path, spin="alpha"):
    spin_key = "beta" if spin.lower().startswith("b") else "alpha"
    cache_key = ("cmo_matrix", spin_key, file_cache_key(fchk_path))

    def load():
        lines = _read_fchk_lines(fchk_path)
        _, nbas, is_open = get_orbital_count_fchk(fchk_path)
        section = (
            "Beta MO coefficients"
            if spin_key == "beta" and is_open
            else "Alpha MO coefficients"
        )
        mo_flat = _parse_array(lines, section, float)
        if mo_flat is None:
            raise ValueError(f"Section '{section}' not found in {fchk_path}")

        expected = nbas * nbas
        if mo_flat.size < expected:
            raise ValueError(
                f"Expected {expected} MO coefficients, got {mo_flat.size}"
            )
        return mo_flat[:expected].reshape(nbas, nbas)

    return _SOURCE_CACHE.get(cache_key, load)


def compute_cube_data_fchk(fchk_path, orbital_indices, spin,
                            grid_quality, ext_dist, bohr_const,
                            precomputed_cmos=None, precomputed_basis=None):
    """
    Compute orbital grids directly from a .fchk file.

    Parameters match nbo_read.compute_cube_data() exactly, except that
    fchk_path replaces (basis_path, key_filepath).

    precomputed_cmos : optional list of 1-D AO-coefficient arrays, one per
        orbital_indices entry, already in hand (e.g. from
        localization_io.localize_orbitals). When given, the normal
        load_cmos_from_fchk() call -- and its "all orbitals -> energy
        order" reindexing -- is skipped, so orbital_indices stays aligned
        with the supplied coefficients.

    Returns the same list-of-dicts format so _load_computed_cubes in
    chemview.py can handle both sources identically.
    """
    from angular_funct import ang_res_lamda

    try:
        import electron_density_opt_omp as _cpp
        _use_cpp = True
    except ImportError:
        _use_cpp = False

    if precomputed_basis is None:
        final_norm_basis, coordinates_ang, atom_info = load_basis_from_fchk(fchk_path)
    else:
        final_norm_basis, coordinates_ang, atom_info = precomputed_basis
    if precomputed_cmos is not None:
        cmos = precomputed_cmos
    else:
        _, nbas, _ = get_orbital_count_fchk(fchk_path)
        if orbital_indices == list(range(1, nbas + 1)):
            orbital_indices = _sort_indices_by_energy(fchk_path, spin=spin, ascending=True)
        cmos = load_cmos_from_fchk(fchk_path, orbital_indices, spin)

    coord_bohr = np.array(coordinates_ang) / bohr_const
    ext_min    = coord_bohr.min(axis=0) - ext_dist
    ext_max    = coord_bohr.max(axis=0) + ext_dist
    ranges     = ext_max - ext_min
    spc        = ranges[int(np.argmax(ranges))] / (grid_quality - 1)
    nx = int(round(ranges[0] / spc)) + 1
    ny = int(round(ranges[1] / spc)) + 1
    nz = int(round(ranges[2] / spc)) + 1
    origin  = ext_min
    spacing = np.array([spc, spc, spc])

    x = np.arange(nx) * spc + origin[0]
    y = np.arange(ny) * spc + origin[1]
    z = np.arange(nz) * spc + origin[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points  = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    def _eval_python(cmo):
        density = np.zeros(len(points))
        for basis, c in zip(final_norm_basis, cmo):
            if abs(c) <= 1e-15:
                continue
            atom_c = coord_bohr[basis["CENTER"] - 1][:, np.newaxis]
            dx, dy, dz = points.T - atom_c
            r   = np.sqrt(dx**2 + dy**2 + dz**2)
            ang = ang_res_lamda(dx, dy, dz, basis["orb_val"])
            for coeff, zeta in zip(basis["coeffs"], basis["exps"]):
                density += np.round(c * coeff * ang * np.exp(-zeta * r**2), 99)
        return density.reshape(nx, ny, nz)

    def _eval_cpp(cmo):
        psi = _cpp.electron_density(
            final_norm_basis, coord_bohr, points, cmo, None)
        return psi.reshape(nx, ny, nz)

    _eval = _eval_cpp if _use_cpp else _eval_python

    base    = os.path.splitext(os.path.basename(fchk_path))[0]
    results = []
    engine_name = "C++ OpenMP" if _use_cpp else "Python (NumPy)"
    _t0 = time.time()
    for cmo, idx in zip(cmos, orbital_indices):
        grid = _eval(cmo)
        results.append({
            "index":      idx,
            "label":      f"{base}-{idx}",
            "grid":       grid,
            "nx": nx, "ny": ny, "nz": nz,
            "spacing":    spacing.copy(),
            "origin":     origin.copy(),
            "atom_info":  atom_info,
            "bohr_const": bohr_const,
        })
    print(f"[{engine_name}] total grid generation: {time.time() - _t0:.3f}s "
          f"for {len(orbital_indices)} orbital(s)")
    return results

def _sort_basis_by_shell_label_order(basis):
    """
    Keep basis functions grouped by shell, with components ordered according to
    _LABEL_MAP. SP shells naturally become s, px, py, pz.
    """
    label_order = {}
    for labels in _LABEL_MAP.values():
        for idx, label in enumerate(labels):
            label_order[label] = idx
    label_order["s"] = 0

    ordered_basis = []
    for shell_num in dict.fromkeys(bf["shell_num"] for bf in basis):
        shell_basis = [bf for bf in basis if bf["shell_num"] == shell_num]
        shell_basis.sort(
            key=lambda bf: (
                1 if bf["orb_val"] in ("px", "py", "pz") else 0,
                label_order.get(bf["orb_val"], len(label_order)),
            )
        )
        ordered_basis.extend(shell_basis)

    for idx, bf in enumerate(ordered_basis, start=1):
        bf["N"] = idx
    return ordered_basis

def calculate_occupation_numbers(C, S, D):
        return np.diag(C.T @ S @ D @ S @ C)

import pandas as pd
if __name__ == '__main__':
    import sys
    import nbo_read as _nr
    from overlap_matrix import get_overlap_matrix as getSmat
    from bas_dict import dict_keys

    path = sys.argv[1] if len(sys.argv) > 1 else input("FCHK file path: ")
    print(f"\n{'='*70}")
    print(f"Testing FCHK orthonormality: {path}")
    print('='*70)

    with open(path, 'r') as f:
        lines = f.read().splitlines()

    final_norm_basis, coordinates_ang, atom_info = load_basis_from_fchk(path)
    final_smat = getSmat(final_norm_basis, dict_keys, normalize_primitives=False, diagonal_only=False)

    ortho_flat = _parse_array(lines, "Orthonormal basis", float)
    if ortho_flat is not None:
        nbas = len(final_norm_basis)
        if ortho_flat.size < nbas * nbas:
            raise ValueError(
                f"Orthonormal basis section too small: expected at least {nbas * nbas}, got {ortho_flat.size}"
            )
        ortho_basis = ortho_flat[:nbas * nbas].reshape(nbas, nbas)
        print("\nOrthonormal basis top-left 5x5 block")
        print(ortho_basis[:5, :5])
        ortho_overlap = np.linalg.inv(ortho_basis.T @ ortho_basis)
    else:
        print("\nSection 'Orthonormal basis' not found.")

    print(pd.DataFrame(ortho_overlap))
        

    alpha_coeff_flat = _parse_array(lines, "Alpha MO coefficients", float)
    if alpha_coeff_flat is not None:
        nbas = len(final_norm_basis)
        alpha_coeffs = alpha_coeff_flat[:nbas * nbas].reshape(nbas, nbas)
        print("\nAlpha MO coefficients top-left 5x5 block")
        alpha_coeffs = alpha_coeffs.T
        print(alpha_coeffs[:5, :5])
    else:
        print("\nSection 'Alpha MO coefficients' not found.")

    beta_coeff_flat = _parse_array(lines, "Beta MO coefficients", float)
    if beta_coeff_flat is not None:
        nbas = len(final_norm_basis)
        beta_coeffs = beta_coeff_flat[:nbas * nbas].reshape(nbas, nbas)
        beta_coeffs = beta_coeffs.T
        print("\nBeta MO coefficients top-left 5x5 block")
        print(beta_coeffs[:5, :5])
       
    print(ortho_overlap - final_smat)


    
   
    ortho_test_1 = alpha_coeffs.T @ ortho_overlap @ alpha_coeffs
   
    ortho_test_2 = alpha_coeffs.T @ final_smat @ alpha_coeffs
   
    print(pd.DataFrame(ortho_overlap - final_smat))
    print(np.diag(ortho_test_2).round(8))

    print(np.diag(ortho_test_1).round(8))


    density_alpha, density_beta = _get_density_matrices(lines, nbas)

    if density_alpha is not None:
        occ_alpha_test = calculate_occupation_numbers(alpha_coeffs, ortho_overlap, density_alpha)
        print("\nAlpha occupation numbers (from density matrix)")
        print(occ_alpha_test.round(8))

    if density_beta is not None and beta_coeff_flat is not None:
        occ_beta_test = calculate_occupation_numbers(beta_coeffs, ortho_overlap, density_beta)
        print("\nBeta occupation numbers (from density matrix)")
        print(occ_beta_test.round(8))


