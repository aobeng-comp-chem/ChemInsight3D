"""
molden_read.py
==============
Reads a Molden (.molden) file and exposes the same interface as
nbo_read.py and fchk_read.py so that chemview.py can call all three
sources uniformly.

Supported Molden features
--------------------------
- [Atoms]  AU (bohr) or Angs (angstrom)
- [GTO]    basis set — S, SP, P, D, F, G, H shells
- [5D] / [5D7F] / [7F]   pure spherical harmonics (default assumed)
- [9G] / [Cartesian] / cartesian=True  flags handled
- [MO]     Ene, Spin, Occup, coefficients (both Alpha and Beta)

Public API (mirrors nbo_read / fchk_read)
------------------------------------------
load_basis_from_molden(molden_path)
    → (final_norm_basis, coordinates_ang, atom_info)

get_orbital_count_molden(molden_path)
    → (orbital_type_str, nbas, is_open_shell)

get_orbital_energies_and_occupations_molden(molden_path)
    → (ene_alpha, occ_alpha, ene_beta, occ_beta)

load_cmos_from_molden(molden_path, orbital_indices, spin='alpha')
    → list of 1-D numpy arrays

compute_cube_data_molden(molden_path, orbital_indices, spin,
                          grid_quality, ext_dist, bohr_const)
    → list-of-dicts  (same format as nbo_read.compute_cube_data)
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
    """Clear all parsed and derived Molden data (primarily useful in tests)."""
    _SOURCE_CACHE.clear()


_BOHR_TO_ANG = physical_constants['Bohr radius'][0] * 1e10  # 0.529177…


# ────────────────────────────────────────────────────────────────────────────
# Angular-momentum label maps  (pure spherical, matching nbo_read convention)
# ────────────────────────────────────────────────────────────────────────────

_COMP_MAP = {
    "s": 1, "p": 3, "d": 5, "f": 7, "g": 9, "h": 11
}

# type  AND  orb_val labels — same strings nbo_read stores
#LABELS ordering based on nbo_read
_LABEL_MAP = {
    "s": [("s",   "s")],
    "p": [("px",  "px"),  ("py",  "py"),  ("pz",  "pz")],
    "d": [("d0",  "d0"),  ("ds1", "ds1"), ("dc1", "dc1"),
          ("dc2", "dc2"), ("ds2", "ds2")],
    "f": [("f0",  "f0"),  ("fc1", "fc1"), ("fs1", "fs1"),
          ("fc2", "fc2"), ("fs2", "fs2"), ("fc3", "fc3"), ("fs3", "fs3")],
    "g": [("g0",  "g0"),  ("gc1", "gc1"), ("gs1", "gs1"),
          ("gc2", "gc2"), ("gs2", "gs2"), ("gc3", "gc3"), ("gs3", "gs3"),
          ("gc4", "gc4"), ("gs4", "gs4")],
    "h": [("h0",  "h0"),  ("hc1", "hc1"), ("hs1", "hs1"),
          ("hc2", "hc2"), ("hs2", "hs2"), ("hc3", "hc3"), ("hs3", "hs3"),
          ("hc4", "hc4"), ("hs4", "hs4"), ("hc5", "hc5"), ("hs5", "hs5")],
}

# For SP shells (Molden "-1" type, stored as "sp" or "-1")
_SP_LABELS = [("s", "s"), ("px", "px"), ("py", "py"), ("pz", "pz")]


# ────────────────────────────────────────────────────────────────────────────
# Low-level Molden file reader
# ────────────────────────────────────────────────────────────────────────────

def _fortran_float(s):
    """Convert Fortran D-exponent notation to Python float."""
    return float(s.replace('D', 'E').replace('d', 'e'))


def _find_section(lines, name):
    """
    Return the line index of the [SectionName] header (case-insensitive),
    or None if not found.
    """
    pat = re.compile(r'^\s*\[' + re.escape(name) + r'\]', re.IGNORECASE)
    for i, line in enumerate(lines):
        if pat.match(line):
            return i
    return None


def _section_lines(lines, start_idx):
    """
    Yield lines belonging to the section starting at start_idx+1,
    stopping at the next '[' section header or end of file.
    """
    for line in lines[start_idx + 1:]:
        if re.match(r'^\s*\[', line):
            break
        yield line


# ────────────────────────────────────────────────────────────────────────────
# [Atoms] parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_atoms(lines):
    """
    Parse [Atoms] section.

    Returns
    -------
    atom_symbols  : list of str
    atomic_nums   : list of int
    coordinates_ang : list of (x, y, z) in Angstrom
    atom_info     : list of (Z, x, y, z) in Angstrom
    units         : 'AU' or 'Angs'
    """
    # Find section header and determine units
    for i, line in enumerate(lines):
        m = re.match(r'^\s*\[Atoms\]\s*(AU|Angs|Angstrom)?\s*$', line, re.IGNORECASE)
        if m:
            units = (m.group(1) or 'AU').upper()
            if units.startswith('ANG'):
                units = 'Angs'
            start = i
            break
    else:
        raise ValueError("No [Atoms] section found in molden file.")

    atom_symbols  = []
    atomic_nums   = []
    coords_raw    = []

    for line in _section_lines(lines, start):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        # Format: symbol  serial  atomic_num  x  y  z
        try:
            sym = parts[0]
            Z   = int(parts[2])
            x   = _fortran_float(parts[3])
            y   = _fortran_float(parts[4])
            z   = _fortran_float(parts[5])
        except (ValueError, IndexError):
            continue
        atom_symbols.append(sym)
        atomic_nums.append(Z)
        coords_raw.append((x, y, z))

    # Convert to Angstrom and keep Bohr coordinates for basis functions
    if units == 'AU':
        coordinates_ang = [
            (x * _BOHR_TO_ANG, y * _BOHR_TO_ANG, z * _BOHR_TO_ANG)
            for x, y, z in coords_raw
        ]
        coordinates_bohr = list(coords_raw)  # Keep in Bohr for overlap calculation
    else:
        coordinates_ang = list(coords_raw)
        coordinates_bohr = [
            (x / _BOHR_TO_ANG, y / _BOHR_TO_ANG, z / _BOHR_TO_ANG)
            for x, y, z in coords_raw
        ]

    atom_info = [
        (int(atomic_nums[i],),) + coordinates_ang[i]
        for i in range(len(atomic_nums))
    ]
    return atom_symbols, atomic_nums, coordinates_ang, atom_info, units, coordinates_bohr


# ────────────────────────────────────────────────────────────────────────────
# [GTO] parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_gto(lines, coordinates_bohr):
    """
    Parse [GTO] section.

    Returns a list of basis-function dicts using nbo_read field names:
        N, CENTER, shell_num, type, orb_val, exps, coeffs,
        xcenter, ycenter, zcenter
    Coordinates (xcenter etc.) are in Bohr.
    """
    gto_idx = _find_section(lines, 'GTO')
    if gto_idx is None:
        raise ValueError("No [GTO] section found in molden file.")

    basis = []
    fn_counter  = 0
    shell_num   = 0
    
        
    LABEL_MAPPING = {
    1: 's', 51: 's',
    101: 'px', 102: 'py', 103: 'pz',
    151: 'px', 152: 'py', 153: 'pz',
    255: 'd0', 252: 'ds1', 253: 'dc1',  254: 'dc2', 251: 'ds2',    
    351: 'f0', 352: 'fc1', 353: 'fs1', 354: 'fc2', 355: 'fs2', 356: 'fc3', 357: 'fs3',
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

    # Determine if pure spherical (5D/7F) or Cartesian
    pure_spherical = True   # Molden default is spherical for 5D7F flag
    for i, line in enumerate(lines):
        if re.match(r'^\s*\[9G\]', line, re.IGNORECASE):
            pure_spherical = False
        if re.match(r'^\s*\[Cartesian\]', line, re.IGNORECASE):
            pure_spherical = False
        if re.match(r'^\s*\[5D\]', line, re.IGNORECASE):
            pure_spherical = True
        if re.match(r'^\s*\[5D7F\]', line, re.IGNORECASE):
            pure_spherical = True
        if re.match(r'^\s*\[7F\]', line, re.IGNORECASE):
            pure_spherical = True

    current_atom_idx = None   # 1-based
    current_coords   = None

    i = gto_idx + 1
    while i < len(lines):
        line = lines[i]

        # Stop at next section header
        if re.match(r'^\s*\[', line):
            break

        stripped = line.strip()
        i += 1

        if not stripped or stripped.startswith('#'):
            continue

        parts = stripped.split()

        # Atom header line:  "<atom_idx>  0"
        if len(parts) == 2 and parts[1] == '0':
            try:
                current_atom_idx = int(parts[0])
                current_coords   = coordinates_bohr[current_atom_idx - 1]
            except (ValueError, IndexError):
                pass
            continue

        # Shell header line:  "<type> <nprim> <scale>"
        # type can be s, p, d, f, g, h, sp (case-insensitive)
        if len(parts) >= 2 and re.match(r'^[spdSPDfFgGhHi]{1,2}$', parts[0]):
            shell_type = parts[0].lower()
            try:
                n_prim = int(parts[1])
                # scale = float(parts[2]) if len(parts) > 2 else 1.0
            except ValueError:
                continue

            shell_num += 1

            # Read n_prim primitive lines
            exps   = []
            coeffs = []
            pcoeffs = []   # for SP shells
            for _ in range(n_prim):
                while i < len(lines):
                    pline = lines[i].strip()
                    i += 1
                    if pline and not pline.startswith('#'):
                        break
                pparts = pline.split()
                try:
                    exps.append(_fortran_float(pparts[0]))
                    coeffs.append(_fortran_float(pparts[1]))
                    if len(pparts) >= 3:
                        pcoeffs.append(_fortran_float(pparts[2]))
                except (ValueError, IndexError):
                    continue

            if current_atom_idx is None or current_coords is None:
                continue

            # Determine component labels
            if shell_type == 'sp':
                # SP shell: s component uses coeffs, p components use pcoeffs
                if not pcoeffs:
                    pcoeffs = coeffs[:]   # fallback
                component_pairs = [
                    ("s",  "s",  coeffs),
                    ("px", "px", pcoeffs),
                    ("py", "py", pcoeffs),
                    ("pz", "pz", pcoeffs),
                ]
                for (type, orb_val, c) in component_pairs:
                    basis.append({
                        "N":         fn_counter + 1,
                        "CENTER":    current_atom_idx,
                        "shell_num": shell_num,
                        "type":  type,
                        "orb_val":   orb_val,
                        "exps":      list(exps),
                        "coeffs":    list(c),
                        "xcenter":   current_coords[0],
                        "ycenter":   current_coords[1],
                        "zcenter":   current_coords[2],
                    })
                    fn_counter += 1
            else:
                labels = _LABEL_MAP.get(shell_type)
                if labels is None:
                    # Unknown shell type — skip
                    continue
                for (type, orb_val) in labels:
                    basis.append({
                        "N":         fn_counter + 1,
                        "CENTER":    current_atom_idx,
                        "shell_num": shell_num,
                        "type":  type,
                        "orb_val":   orb_val,
                        "exps":      list(exps),
                        "coeffs":    list(coeffs),
                        "xcenter":   current_coords[0],
                        "ycenter":   current_coords[1],
                        "zcenter":   current_coords[2],
                    })
                    fn_counter += 1
            continue
        
            
    for bf in basis:
        ov = bf["orb_val"]
        bf["LABEL"] = ORBVAL_TO_LABELCODE.get(ov)    

    return basis


# ────────────────────────────────────────────────────────────────────────────
# [MO] parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_mo(lines, nbas):
    """
    Parse [MO] section.

    Returns
    -------
    mos_alpha : np.ndarray shape (n_alpha, nbas)  — each row = one MO
    ene_alpha : np.ndarray shape (n_alpha,)
    occ_alpha : np.ndarray shape (n_alpha,)
    mos_beta  : np.ndarray or None
    ene_beta  : np.ndarray or None
    occ_beta  : np.ndarray or None
    """
    mo_idx = _find_section(lines, 'MO')
    if mo_idx is None:
        raise ValueError("No [MO] section found in molden file.")

    mos_alpha, ene_alpha, occ_alpha = [], [], []
    mos_beta,  ene_beta,  occ_beta  = [], [], []

    # Each MO block:
    #   Sym= ...
    #   Ene= <value>
    #   Spin= Alpha|Beta
    #   Occup= <value>
    #   <index>   <coeff>
    #   ...  (nbas lines)

    current_ene   = None
    current_spin  = None
    current_occup = None
    current_coeffs = {}   # {1-based index: coeff}

    def _flush():
        if current_spin is None or current_ene is None:
            return
        vec = np.array([current_coeffs.get(k, 0.0) for k in range(1, nbas + 1)])
        if current_spin.lower() == 'alpha':
            mos_alpha.append(vec)
            ene_alpha.append(current_ene)
            occ_alpha.append(current_occup)
        else:
            mos_beta.append(vec)
            ene_beta.append(current_ene)
            occ_beta.append(current_occup)

    for line in _section_lines(lines, mo_idx):
        stripped = line.strip()
        if not stripped:
            continue

        # New MO block starts when we hit Sym= or Ene= after coefficients
        m_ene = re.match(r'^Ene\s*=\s*(.+)', stripped, re.IGNORECASE)
        m_sym = re.match(r'^Sym\s*=', stripped, re.IGNORECASE)

        if m_sym:
            # Flush previous MO if any
            if current_ene is not None:
                _flush()
            current_ene    = None
            current_spin   = None
            current_occup  = None
            current_coeffs = {}
            continue

        if m_ene:
            current_ene = _fortran_float(m_ene.group(1).strip())
            continue

        m_spin = re.match(r'^Spin\s*=\s*(\S+)', stripped, re.IGNORECASE)
        if m_spin:
            current_spin = m_spin.group(1).strip()
            continue

        m_occ = re.match(r'^Occup\s*=\s*(.+)', stripped, re.IGNORECASE)
        if m_occ:
            current_occup = _fortran_float(m_occ.group(1).strip())
            continue

        # Coefficient line:  <index>   <value>
        parts = stripped.split()
        if len(parts) == 2:
            try:
                idx  = int(parts[0])
                coef = _fortran_float(parts[1])
                current_coeffs[idx] = coef
            except ValueError:
                pass

    # Flush the last MO
    if current_ene is not None:
        _flush()

    def _to_arrays(mos, ene, occ):
        if not mos:
            return None, None, None
        return (np.array(mos),
                np.array(ene, dtype=float),
                np.array(occ, dtype=float))

    mo_a, en_a, oc_a = _to_arrays(mos_alpha, ene_alpha, occ_alpha)
    mo_b, en_b, oc_b = _to_arrays(mos_beta,  ene_beta,  occ_beta)
    return mo_a, en_a, oc_a, mo_b, en_b, oc_b


# ────────────────────────────────────────────────────────────────────────────
# Normalisation (delegates to nbo_read pipeline)
# ────────────────────────────────────────────────────────────────────────────

def _normalise_basis(raw_basis):
    """
    Apply the same two-stage normalisation used by nbo_read for .31 files:
        1. convert_to_molden  (primitive Gaussian normalisation)
        2. normalize_by_self_overlap  (contracted-function normalisation)
        3. normalize_basis_info  (final overlap-matrix scaling)

    The extra iterative_basis_modification from .47 files is NOT applied
    because Molden MOs are already in the orthonormal MO basis.
    """
    import nbo_read as _nr
    from overlap_matrix import get_overlap_matrix as getSmat
    from bas_dict import dict_keys

    basis = _nr.convert_to_molden(raw_basis)
    basis = _nr.normalize_by_self_overlap(basis)
   
    
    # Smat  = getSmat(basis, dict_keys, normalize_primitives=True, diagonal_only=False)
    basis = _nr.normalize_by_self_overlap(basis)
    # print(Smat)
    return basis


# ────────────────────────────────────────────────────────────────────────────
# Internal full-parse helper (cached per file identity)
# ────────────────────────────────────────────────────────────────────────────

def _parse_molden_uncached(molden_path):
    """
    Parse all sections of a molden file and return a dict with keys:
        atom_symbols, atomic_nums, coordinates_ang, atom_info,
        raw_basis,
        mos_alpha, ene_alpha, occ_alpha,
        mos_beta,  ene_beta,  occ_beta,
        nbas, is_open_shell
    """
    with open(molden_path, 'r') as f:
        lines = f.read().splitlines()

    _, atomic_nums, coordinates_ang, atom_info, units, coordinates_bohr = _parse_atoms(lines)
    raw_basis = _parse_gto(lines, coordinates_bohr)
    nbas      = len(raw_basis)

    mo_a, en_a, oc_a, mo_b, en_b, oc_b = _parse_mo(lines, nbas)
     
   
    is_open = mo_b is not None and len(mo_b) > 0

    return {
        'atomic_nums':      atomic_nums,
        'coordinates_ang':  coordinates_ang,
        'atom_info':        atom_info,
        'raw_basis':        raw_basis,
        'mos_alpha':        mo_a,
        'ene_alpha':        en_a,
        'occ_alpha':        oc_a,
        'mos_beta':         mo_b,
        'ene_beta':         en_b,
        'occ_beta':         oc_b,
        'nbas':             nbas,
        'is_open_shell':    is_open,
        'units':            units,
    }


def _parse_molden(molden_path):
    cache_key = ("parsed", file_cache_key(molden_path))
    return _SOURCE_CACHE.get(
        cache_key, lambda: _parse_molden_uncached(molden_path)
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_basis_from_molden(molden_path):
    """
    Parse and normalise the basis set from a .molden file.

    Returns
    -------
    final_norm_basis : list of basis-function dicts (nbo_read format)
    coordinates_ang  : list of (x, y, z) in Angstrom, one per atom
    atom_info        : list of (Z, x, y, z) in Angstrom, one per atom
    """
    cache_key = ("basis", file_cache_key(molden_path))

    def load():
        data = _parse_molden(molden_path)
        final_norm_basis = _normalise_basis(data['raw_basis'])
        return final_norm_basis, data['coordinates_ang'], data['atom_info']

    return _SOURCE_CACHE.get(cache_key, load)


def get_ao_overlap_matrix(molden_path):
    """Return the final normalized-basis overlap, computed once per file."""
    cache_key = ("ao_overlap", file_cache_key(molden_path))

    def load():
        from overlap_matrix import get_overlap_matrix
        from bas_dict import dict_keys

        final_basis, _, _ = load_basis_from_molden(molden_path)
        return get_overlap_matrix(
            final_basis, dict_keys,
            normalize_primitives=False, diagonal_only=False,
        )

    return _SOURCE_CACHE.get(cache_key, load)


def get_orbital_count_molden(molden_path):
    """
    Return (orbital_type_str, nbas, is_open_shell).

    orbital_type_str is always 'CMO' (canonical MOs from molden).
    """
    data = _parse_molden(molden_path)
    return 'CMO', data['nbas'], data['is_open_shell']


def get_orbital_energies_and_occupations_molden(molden_path):
    """
    Return (ene_alpha, occ_alpha, ene_beta, occ_beta).
    All energies are in Hartree (Molden stores them in Hartree by default).
    Beta arrays are None for closed-shell.
    """
    data = _parse_molden(molden_path)
    return (
        data['ene_alpha'],
        data['occ_alpha'],
        data['ene_beta'],
        data['occ_beta'],
    )


def load_cmos_from_molden(molden_path, orbital_indices, spin='alpha'):
    """
    Return CMO row vectors for the requested 1-based orbital_indices.

    Parameters
    ----------
    molden_path     : str
    orbital_indices : list of int (1-based)
    spin            : 'alpha' or 'beta'

    Returns
    -------
    List of 1-D numpy arrays, one per requested orbital.
    """
    data = _parse_molden(molden_path)

    if spin.lower().startswith('b') and data['is_open_shell']:
        mo_matrix = data['mos_beta']
    else:
        mo_matrix = data['mos_alpha']

    if mo_matrix is None:
        raise ValueError(
            f"No {spin} MOs found in {molden_path}"
        )

    return [mo_matrix[i - 1] for i in orbital_indices]


def compute_cube_data_molden(molden_path, orbital_indices, spin,
                              grid_quality, ext_dist, bohr_const,
                              precomputed_cmos=None, precomputed_basis=None):
    """
    Compute orbital grids directly from a .molden file.

    Parameters match nbo_read.compute_cube_data() / fchk_read equivalents.

    precomputed_cmos : optional list of 1-D AO-coefficient arrays, one per
        orbital_indices entry, already in hand (e.g. from
        localization_io.localize_orbitals). When given, the normal
        load_cmos_from_molden() call is skipped.

    Returns the same list-of-dicts so _load_computed_cubes in chemview.py
    handles all three sources identically.
    """
    from angular_funct import ang_res_lamda

    try:
        import electron_density_opt_omp as _cpp
        _use_cpp = True
    except ImportError:
        _use_cpp = False

    if precomputed_basis is None:
        final_norm_basis, coordinates_ang, atom_info = \
            load_basis_from_molden(molden_path)
    else:
        final_norm_basis, coordinates_ang, atom_info = precomputed_basis
    cmos = precomputed_cmos if precomputed_cmos is not None else load_cmos_from_molden(molden_path, orbital_indices, spin)

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
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points  = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    def _eval_python(cmo):
        density = np.zeros(len(points))
        for basis, c in zip(final_norm_basis, cmo):
            if abs(c) <= 1e-15:
                continue
            atom_c = coord_bohr[basis['CENTER'] - 1][:, np.newaxis]
            dx, dy, dz = points.T - atom_c
            r   = np.sqrt(dx**2 + dy**2 + dz**2)
            ang = ang_res_lamda(dx, dy, dz, basis['orb_val'])
            for coeff, zeta in zip(basis['coeffs'], basis['exps']):
                density += np.round(c * coeff * ang * np.exp(-zeta * r**2), 99)
        return density.reshape(nx, ny, nz)

    def _eval_cpp(cmo):
        psi = _cpp.electron_density(
            final_norm_basis, coord_bohr, points, cmo, None)
        return psi.reshape(nx, ny, nz)

    _eval = _eval_cpp if _use_cpp else _eval_python

    base    = os.path.splitext(os.path.basename(molden_path))[0]
    results = []
    engine_name = "C++ OpenMP" if _use_cpp else "Python (NumPy)"
    _t0 = time.time()
    for cmo, idx in zip(cmos, orbital_indices):
        grid = _eval(cmo)
        results.append({
            'index':      idx,
            'label':      f"{base}-{idx}",
            'grid':       grid,
            'nx': nx, 'ny': ny, 'nz': nz,
            'spacing':    spacing.copy(),
            'origin':     origin.copy(),
            'atom_info':  atom_info,
            'bohr_const': bohr_const,
        })
    print(f"[{engine_name}] total grid generation: {time.time() - _t0:.3f}s "
          f"for {len(orbital_indices)} orbital(s)")
    return results


def canonical_label_order(label):
    """Returns position in canonical order for each angular momentum"""
    if label == 1: return 0
    if label in [101,102,103]: return label - 101
    if label == 255: return 0
    if label == 252: return 1
    if label == 253: return 2  
    if label == 254: return 3
    if label == 251: return 4
    if label == 351: return 0
    if label == 352: return 1
    if label == 353: return 2
    if label == 354: return 3
    if label == 355: return 4
    if label == 356: return 5
    if label == 357: return 6
    return 999  # unknown





# ────────────────────────────────────────────────────────────────────────────
# Quick diagnostic / test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from pprint import pprint

    path = sys.argv[1] if len(sys.argv) > 1 else input("Molden file path: ")

    print(f"\n{'='*70}")
    print(f"Testing Molden: {path}")
    print('='*70)

    data = _parse_molden(path)

    print(f"  Coordinate units: {data['units']}")

    # Extract CMOs and compute overlap test
    try:
        print(f"\nLoading basis and CMOs from {path}...")
        final_basis, coords, atom_info = load_basis_from_molden(path)
        nbas = len(final_basis)
        print(f"  Number of basis functions: {nbas}")
        print(f"  Number of atoms: {len(atom_info)}")

        # Load all CMOs for testing orthonormality
        orbital_indices = list(range(1, nbas + 1))
        cmos = load_cmos_from_molden(path, orbital_indices, spin='alpha')
        print(f"  Loaded {len(cmos)} CMOs (alpha spin, all {len(orbital_indices)} orbitals)")

        # Compute overlap matrix from basis functions
        print(f"\nComputing overlap matrix from basis functions...")
        from overlap_matrix import get_overlap_matrix as getSmat
        
        # Prepare dict_keys as required by get_overlap_matrix
        dict_keys = {i: i for i in range(len(final_basis))}
        from bas_dict import dict_keys
        
        # Compute overlap matrix
        # overlap = getSmat(final_basis, dict_keys)
        overlap  = getSmat(final_basis, dict_keys, normalize_primitives=False, diagonal_only=False)

        print(f"  Overlap matrix shape: {overlap.shape}")
        print(np.diag(overlap))
    
        #print(f"  Overlap matrix diagonal (should be ~1.0): {np.diag(overlap)[:min(5, nbas)]}")

        # Test orthonormality: cmo.T @ overlap @ cmo
        print(f"\nTesting orthonormality with C^T @ S @ C (should be identity for all orbitals)...")
        cmat = np.column_stack(cmos)
        orthonormality_test = cmat.T @ overlap @ cmat
        diag_values = np.diag(orthonormality_test)


        print(diag_values)
        import pandas as pd
        
        print(pd.DataFrame(overlap ))
        

        
    except Exception as e:
        import traceback
        print(f"  Error during overlap test: {e}")
        print(traceback.format_exc())
