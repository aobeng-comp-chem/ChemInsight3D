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
import numpy as np
from grid_utils import evaluate_orbital_grids
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

# Molden shell tag -> orbital label.  Module level: this is a constant, and it
# was previously rebuilt (along with its inverse) on every _parse_gto call.
_LABEL_MAPPING = {
    1: 's', 51: 's',
    101: 'px', 102: 'py', 103: 'pz',
    151: 'px', 152: 'py', 153: 'pz',
    255: 'd0', 252: 'ds1', 253: 'dc1', 254: 'dc2', 251: 'ds2',
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

# Inverse: orbital label -> lowest shell tag that maps to it (setdefault keeps
# the first, so 's' resolves to 1 rather than 51 and 'px' to 101 rather than 151).
_ORBVAL_TO_LABELCODE = {}
for _code, _lbl in _LABEL_MAPPING.items():
    _ORBVAL_TO_LABELCODE.setdefault(_lbl, _code)

# Shell-header regex: one or two letters naming the shell type (incl. "sp").
_SHELL_HEADER_RE = re.compile(r'^[spdSPDfFgGhHi]{1,2}$')


def _read_shell_primitives(lines, i, n_prim):
    """
    Read *n_prim* primitive lines starting at lines[i], skipping blanks and
    comments.

    Returns (next_index, exps, coeffs, pcoeffs).  pcoeffs holds the third
    column, which only SP shells populate.  A primitive whose exponent or
    coefficient will not parse is skipped, matching the original behaviour.
    """
    exps, coeffs, pcoeffs = [], [], []
    pline = ''
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
    return i, exps, coeffs, pcoeffs


def _shell_components(shell_type, coeffs, pcoeffs):
    """
    Expand a shell into its (label, orb_val, coefficients) components.

    Returns None for a shell type with no known component list, which the
    caller treats as "skip this shell".
    """
    if shell_type == 'sp':
        # s uses the second column, the three p components use the third.
        p = pcoeffs if pcoeffs else coeffs[:]
        return [("s", "s", coeffs), ("px", "px", p),
                ("py", "py", p), ("pz", "pz", p)]
    labels = _LABEL_MAP.get(shell_type)
    if labels is None:
        return None
    return [(label, orb_val, coeffs) for label, orb_val in labels]


def _parse_atom_header(parts, coordinates_bohr, current_idx, current_coords):
    """
    Resolve an atom header line to (1-based index, coordinates).

    An unparseable index, or one outside the coordinate list, leaves the current
    atom unchanged -- as the original inline `except (ValueError, IndexError):
    pass` did.
    """
    try:
        idx = int(parts[0])
        return idx, coordinates_bohr[idx - 1]
    except (ValueError, IndexError):
        return current_idx, current_coords


def _shell_basis_functions(shell_type, exps, coeffs, pcoeffs,
                           atom_idx, coords, shell_num, first_n):
    """
    Build the basis-function dicts for one shell, numbered from *first_n*.

    Returns [] when the shell has no known component list, or when no atom
    header has been seen yet and there is therefore nowhere to attach the
    functions -- both cases the caller treats as "skip this shell".
    """
    if atom_idx is None or coords is None:
        return []
    components = _shell_components(shell_type, coeffs, pcoeffs)
    if components is None:
        return []
    return [{
        "N":         first_n + offset,
        "CENTER":    atom_idx,
        "shell_num": shell_num,
        "type":      label,
        "orb_val":   orb_val,
        "exps":      list(exps),
        "coeffs":    list(c),
        "xcenter":   coords[0],
        "ycenter":   coords[1],
        "zcenter":   coords[2],
    } for offset, (label, orb_val, c) in enumerate(components)]


def _parse_gto(lines, coordinates_bohr):
    """
    Parse [GTO] section.

    Returns a list of basis-function dicts using nbo_read field names:
        N, CENTER, shell_num, type, orb_val, exps, coeffs,
        xcenter, ycenter, zcenter
    Coordinates (xcenter etc.) are in Bohr.

    Note: the [5D]/[5D7F]/[7F] and [9G]/[Cartesian] flags are NOT honoured --
    every shell is expanded with the pure-spherical component list in
    _LABEL_MAP.  An earlier version scanned the file for those headers but
    discarded the result without ever reading it, so this has always been the
    effective behaviour; a genuinely Cartesian file still needs support adding
    here.
    """
    gto_idx = _find_section(lines, 'GTO')
    if gto_idx is None:
        raise ValueError("No [GTO] section found in molden file.")

    basis = []
    fn_counter = 0
    shell_num = 0
    current_atom_idx = None   # 1-based
    current_coords = None

    i = gto_idx + 1
    while i < len(lines):
        line = lines[i]
        if re.match(r'^\s*\[', line):     # next section header
            break

        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()

        # Atom header line:  "<atom_idx>  0"
        if len(parts) == 2 and parts[1] == '0':
            current_atom_idx, current_coords = _parse_atom_header(
                parts, coordinates_bohr, current_atom_idx, current_coords)
            continue

        # Shell header line:  "<type> <nprim> <scale>"
        if not (len(parts) >= 2 and _SHELL_HEADER_RE.match(parts[0])):
            continue
        try:
            n_prim = int(parts[1])
        except ValueError:
            continue

        shell_num += 1
        i, exps, coeffs, pcoeffs = _read_shell_primitives(lines, i, n_prim)
        functions = _shell_basis_functions(
            parts[0].lower(), exps, coeffs, pcoeffs,
            current_atom_idx, current_coords, shell_num, fn_counter + 1)
        basis.extend(functions)
        fn_counter += len(functions)

    for bf in basis:
        bf["LABEL"] = _ORBVAL_TO_LABELCODE.get(bf["orb_val"])

    return basis


# ────────────────────────────────────────────────────────────────────────────
# [MO] parser
# ────────────────────────────────────────────────────────────────────────────

# [MO] block header keywords.  "Sym" has no capture group -- it only marks the
# start of a new block.
_MO_HEADER_RES = (
    ('sym',   re.compile(r'^Sym\s*=', re.IGNORECASE)),
    ('ene',   re.compile(r'^Ene\s*=\s*(.+)', re.IGNORECASE)),
    ('spin',  re.compile(r'^Spin\s*=\s*(\S+)', re.IGNORECASE)),
    ('occup', re.compile(r'^Occup\s*=\s*(.+)', re.IGNORECASE)),
)


def _match_mo_header(stripped):
    """Return (kind, captured text) for an [MO] header line, else (None, None)."""
    for kind, regex in _MO_HEADER_RES:
        match = regex.match(stripped)
        if match:
            return kind, (match.group(1).strip() if match.groups() else None)
    return None, None


class _MoBlock:
    """
    One [MO] block's header fields and coefficients, accumulated as the block
    is read.  Coefficients arrive as sparse "<index> <value>" lines, so they
    are collected in a dict and densified only on flush().
    """

    __slots__ = ('ene', 'spin', 'occup', 'coeffs')

    def __init__(self):
        self.ene = None
        self.spin = None
        self.occup = None
        self.coeffs = {}

    def set_header(self, kind, value):
        if kind == 'ene':
            self.ene = _fortran_float(value)
        elif kind == 'spin':
            self.spin = value
        elif kind == 'occup':
            self.occup = _fortran_float(value)

    def add_coefficient(self, stripped):
        """Absorb a "<index> <value>" line; silently ignore anything else."""
        parts = stripped.split()
        if len(parts) != 2:
            return
        try:
            self.coeffs[int(parts[0])] = _fortran_float(parts[1])
        except ValueError:
            pass

    def flush(self, buckets, nbas):
        """
        Append this block to its spin's bucket.  A block missing Ene= or Spin=
        is dropped, which is what the original inline _flush() did.
        """
        if self.spin is None or self.ene is None:
            return
        vec = np.array([self.coeffs.get(k, 0.0) for k in range(1, nbas + 1)])
        mos, ene, occ = buckets['alpha' if self.spin.lower() == 'alpha' else 'beta']
        mos.append(vec)
        ene.append(self.ene)
        occ.append(self.occup)


def _to_mo_arrays(mos, ene, occ):
    """Stack one spin's collected MOs, or (None, None, None) if there were none."""
    if not mos:
        return None, None, None
    return (np.array(mos),
            np.array(ene, dtype=float),
            np.array(occ, dtype=float))


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

    # One bucket of (mos, energies, occupations) per spin; 'beta' stays empty
    # for a closed-shell file and _to_mo_arrays then reports None for it.
    buckets = {'alpha': ([], [], []), 'beta': ([], [], [])}
    block = _MoBlock()

    for line in _section_lines(lines, mo_idx):
        stripped = line.strip()
        if not stripped:
            continue
        kind, value = _match_mo_header(stripped)
        if kind == 'sym':
            # Sym= opens a new block, so bank the one that just ended.
            block.flush(buckets, nbas)
            block = _MoBlock()
        elif kind is not None:
            block.set_header(kind, value)
        else:
            block.add_coefficient(stripped)

    block.flush(buckets, nbas)     # the final block has no Sym= after it

    return _to_mo_arrays(*buckets['alpha']) + _to_mo_arrays(*buckets['beta'])


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
    if precomputed_basis is None:
        final_norm_basis, coordinates_ang, atom_info = \
            load_basis_from_molden(molden_path)
    else:
        final_norm_basis, coordinates_ang, atom_info = precomputed_basis
    cmos = precomputed_cmos if precomputed_cmos is not None else load_cmos_from_molden(molden_path, orbital_indices, spin)

    return evaluate_orbital_grids(
        final_norm_basis, coordinates_ang, atom_info, cmos, orbital_indices,
        os.path.splitext(os.path.basename(molden_path))[0],
        grid_quality, ext_dist, bohr_const)


# Molden shell tag -> position within its angular-momentum block.  A table
# rather than a branch chain: the d block in particular is not in numeric order
# (255, 252, 253, 254, 251), which is easy to misread as a sequence of ifs.
_CANONICAL_LABEL_ORDER = {
    1: 0,                                                    # s
    101: 0, 102: 1, 103: 2,                                  # p
    255: 0, 252: 1, 253: 2, 254: 3, 251: 4,                  # d
    351: 0, 352: 1, 353: 2, 354: 3, 355: 4, 356: 5, 357: 6,  # f
}
_CANONICAL_LABEL_UNKNOWN = 999


def canonical_label_order(label):
    """Returns position in canonical order for each angular momentum"""
    return _CANONICAL_LABEL_ORDER.get(label, _CANONICAL_LABEL_UNKNOWN)





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
