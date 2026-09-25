"""
nbo_read.py  —  File 1 of 2
============================
Reads and normalises the basis set (.47 or .31), then reads NBO/CMO
coefficient files.

Run directly:
    python nbo_read.py

Exports after main() returns:
    final_norm_basis  – list of normalised basis function dicts
    coordinates       – list of (x, y, z) in Angstrom
    atom_info          list of (atomic_number, x, y, z)
    bohr              – Angstrom-per-bohr constant
    orbital_dict      – {filename: [cmo_vector, ...]}
    orbital_index     – list of requested orbital indices (1-based)
"""

import numpy as np
import math
import re
import copy
from itertools import groupby
from scipy.constants import physical_constants
import os
import sys
import overlap_matrix
from bas_dict import dict_keys
from bas_dict import get_term_info
from grid_utils import evaluate_orbital_grids
from source_cache import ComputationCache

getSmat = overlap_matrix.get_overlap_matrix
bohr    = physical_constants['Bohr radius'][0] * 1e10  # Angstrom per bohr


# ── Per-file caching ─────────────────────────────────────────────────────────
# Parsing/normalizing a .47 (and building its overlap matrix) is expensive
# and was being redone from scratch on every call -- once per key file shown
# in a picker, once for the orbital load, again for localization, again for
# population analysis, etc, all for the exact same on-disk file within one
# session. These caches make each expensive step (raw parse, normalization
# incl. the iterative self-overlap correction, $OVERLAP/$DENSITY/$FOCK
# extraction, and the analytic AO overlap matrix) happen at most once per
# file per process. Nothing downstream mutates the returned basis dicts or
# matrices in place (normalization always copies before modifying), so
# sharing the cached objects across callers is safe.
_parse_file47_cache      = {}
_parse_file31_cache      = {}
_process_47_file_cache   = {}
_load_basis_headless_cache = {}
_ao_overlap_cache        = {}
_key_source_cache = ComputationCache()


def clear_source_cache():
    """Clear NBO basis, matrix, key-file, and orbital metadata caches."""
    _parse_file47_cache.clear()
    _parse_file31_cache.clear()
    _process_47_file_cache.clear()
    _load_basis_headless_cache.clear()
    _ao_overlap_cache.clear()
    _key_source_cache.clear()


def _file_cache_key(path):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def double_factorial(n):
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)

def gaussian_norm(alpha, l, m, n):
    lmn = l + m + n
    prefactor = (2 ** (2 * lmn + 1.5)) * (alpha ** (lmn + 1.5)) / (math.pi ** 1.5)
    denom = double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1)
    return math.sqrt(prefactor / denom)


# NBO LABEL code -> (descriptive type, orb_val used by the angular functions).
# Module level: a constant, previously rebuilt on every parse_file47 call.
_ORB_MAPPING = {
    1: ('s', 's'), 51: ('s', 's'), 101: ('px', 'px'), 102: ('py', 'py'), 103: ('pz', 'pz'),
    151: ('px', 'px'), 152: ('py', 'py'), 153: ('pz', 'pz'),
    251: ('d_xy', 'ds2'), 252: ('d_xz', 'ds1'), 253: ('d_yz', 'dc1'),
    254: ('d_x2-y2', 'dc2'), 255: ('d_z2', 'd0'),
    351: ('fz(5z2-3r2)', 'f0'), 352: ('fx(5z2-r2)', 'fc1'), 353: ('fy(5z2-r2)', 'fs1'),
    354: ('fz(x2-y2)', 'fc2'), 355: ('fxyz', 'fs2'), 356: ('fx(x2-3y2)', 'fc3'),
    357: ('f(3x2-y2)', 'fs3'),
    451: ('g0', 'g0'), 452: ('gc1', 'gc1'), 453: ('gs1', 'gs1'), 454: ('gc2', 'gc2'),
    455: ('gs2', 'gs2'), 456: ('gc3', 'gc3'), 457: ('gs3', 'gs3'), 458: ('gc4', 'gc4'),
    459: ('gs4', 'gs4'),
    551: ('h0', 'h0'), 552: ('hc1', 'hc1'), 553: ('hs1', 'hs1'), 554: ('hc2', 'hc2'),
    555: ('hs2', 'hs2'), 556: ('hc3', 'hc3'), 557: ('hs3', 'hs3'), 558: ('hc4', 'hc4'),
    559: ('hs4', 'hs4'), 560: ('hc5', 'hc5'), 561: ('hs5', 'hs5'),
    651: ('i0', 'i0'), 652: ('ic1', 'ic1'), 653: ('is1', 'is1'), 654: ('ic2', 'ic2'),
    655: ('is2', 'is2'), 656: ('ic3', 'ic3'), 657: ('is3', 'is3'), 658: ('ic4', 'ic4'),
    659: ('is4', 'is4'), 660: ('ic5', 'ic5'), 661: ('is5', 'is5'), 662: ('ic6', 'ic6'),
    663: ('is6', 'is6'),
    751: ('j0', 'j0'), 752: ('jc1', 'jc1'), 753: ('js1', 'js1'), 754: ('jc2', 'jc2'),
    755: ('js2', 'js2'), 756: ('jc3', 'jc3'), 757: ('js3', 'js3'), 758: ('jc4', 'jc4'),
    759: ('js4', 'js4'), 760: ('jc5', 'jc5'), 761: ('js5', 'js5'), 762: ('jc6', 'jc6'),
    763: ('js6', 'js6'), 764: ('jc7', 'jc7'), 765: ('js7', 'js7')
}

# Components per shell for each angular momentum, used to infer shell
# boundaries from the orbital-type sequence alone.
_SHELL_TYPE_LIMITS = {'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'j': 15}

# The .47 coefficient columns, in the order their non-zero entries are taken.
_COEFF_COLUMNS = ('CS', 'CP', 'CD', 'CF', 'CG', 'CH', 'CI', 'CJ')
_FLOAT_VARS = ('EXP',) + _COEFF_COLUMNS
_INT_VARS = ('CENTER', 'LABEL', 'NSHELL', 'NEXP', 'NCOMP', 'NPRIM', 'NPTR')

_ATOM_COORD_RE = re.compile(
    r'\s+(\d+)\s+(\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)')


def _read_file47_text(filename):
    """Read a .47/.31 file, re-raising I/O failures with the filename attached."""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except IOError:
        raise IOError(f"Error reading file '{filename}'.")


def _parse_array_from_block(varname, content, dtype=float):
    """Collect every float in the `VARNAME = ...` blocks named *varname*."""
    pattern = re.compile(
        rf'(?<![A-Z]){varname}(?![A-Z])\s*=\s*((?:[-+]?\d+\.\d+(?:E[+-]?\d+)?\s+)+)')
    values = []
    for match in pattern.findall(content):
        values.extend(dtype(v) for v in match.split())
    return values


def _parse_int_array(varname, content):
    """Collect every integer in the `VARNAME = ...` blocks named *varname*."""
    pattern = re.compile(rf'(?<![A-Z]){varname}(?![A-Z])\s*=\s*([\d\s]+)')
    values = []
    for match in pattern.findall(content):
        values.extend(int(v) for v in match.split())
    return values


def _is_ungrouped(ncomp):
    """True when the file lists one primitive set per basis function."""
    return all(x == 1 for x in ncomp)


def _shell_numbers_from_ncomp(orb_type, ncomp):
    """
    Assign each basis function its 1-based shell index from the NCOMP counts,
    checking that every multi-component shell holds a single angular momentum.
    """
    shell_num = []
    idx = 0
    for shell_idx, nc in enumerate(ncomp, 1):
        group = orb_type[idx:idx + nc]
        if nc > 1:
            base_type = group[0][0] if group else None
            if not all(t[0] == base_type for t in group) or len(group) != nc:
                raise ValueError(f"Invalid NCOMP grouping at shell {shell_idx}: "
                                 f"{group} does not match NCOMP={nc}")
        shell_num.extend([shell_idx] * nc)
        idx += nc
    return shell_num


def _shell_numbers_from_types(orb_type):
    """
    Shell numbering inferred from the orbital-type sequence alone: a run of the
    same angular momentum stays in one shell until that momentum's component
    count is filled.  Used only to cross-check the NCOMP-derived numbering, so
    the absolute values do not matter -- only where the group boundaries fall.
    """
    shell_num = []
    count = 1
    orb_count = 0
    for i, t in enumerate(orb_type):
        base_type = t[0]
        continues_shell = (i > 0
                           and base_type in _SHELL_TYPE_LIMITS
                           and orb_type[i - 1][0] == base_type
                           and orb_count < _SHELL_TYPE_LIMITS[base_type])
        if continues_shell:
            shell_num.append(shell_num[-1])
            orb_count += 1
        else:
            count += 1
            shell_num.append(count)
            orb_count = 1
    return shell_num


def _warn_if_shell_grouping_differs(shell_num, orb_type):
    """
    Report a disagreement between the NCOMP-derived and type-derived shell
    sizes.  Diagnostic only: the NCOMP numbering is used either way.
    """
    ncomp_shells = [len(list(g)) for _, g in groupby(shell_num)]
    type_shells = [len(list(g)) for _, g in groupby(_shell_numbers_from_types(orb_type))]
    if ncomp_shells != type_shells:
        print(f"Warning: NCOMP-based shells {ncomp_shells} differ from "
              f"type-based shells {type_shells}. Using NCOMP-based.")


def process_orbital_labels(label, ncomp, orb_mapping=None):
    """
    Resolve LABEL codes to orbital types/values and assign shell numbers.

    Returns (orb_type, orb_val, shell_num), one entry per basis function.
    """
    orb_mapping = _ORB_MAPPING if orb_mapping is None else orb_mapping
    if sum(ncomp) != len(label):
        raise ValueError(f"NCOMP sum ({sum(ncomp)}) does not match "
                         f"LABEL length ({len(label)})")

    unknown = ('unknown', 'unknown')
    orb_type = [orb_mapping.get(l, unknown)[0] for l in label]
    orb_val = [orb_mapping.get(l, unknown)[1] for l in label]

    shell_num = _shell_numbers_from_ncomp(orb_type, ncomp)
    if _is_ungrouped(ncomp):
        shell_num = list(range(1, len(label) + 1))
    else:
        _warn_if_shell_grouping_differs(shell_num, orb_type)
    return orb_type, orb_val, shell_num


def _expand_shell_pointers(nprim, nptr, ncomp):
    """
    Repeat each shell's NPRIM/NPTR once per component.

    An ungrouped file already carries one entry per basis function, so its
    arrays pass through unchanged.
    """
    if _is_ungrouped(ncomp):
        return nprim, nptr
    if len(nprim) != len(ncomp) or len(nptr) != len(ncomp):
        raise ValueError(f"NPRIM ({len(nprim)}) or NPTR ({len(nptr)}) length "
                         f"does not match NCOMP ({len(ncomp)})")
    nprim_expanded = []
    nptr_expanded = []
    for shell_idx, nc in enumerate(ncomp):
        nprim_expanded.extend([nprim[shell_idx]] * nc)
        nptr_expanded.extend([nptr[shell_idx]] * nc)
    return nprim_expanded, nptr_expanded


def _gather_coefficients(coeff_columns, ptr, prim):
    """
    Collect one basis function's contraction coefficients.

    A .47 file keeps a separate column per angular momentum (CS, CP, CD, ...)
    and zero-fills the columns that do not apply to a shell, so the non-zero
    entries across all columns are exactly this function's coefficients.
    """
    coeffs = []
    for column in coeff_columns:
        window = column[ptr - 1: ptr - 1 + prim]
        coeffs.extend([c for c in window if c != 0.0])
    return coeffs


def _atom_data_from_content(content):
    """
    Parse the atom block.

    Returns (rows, to_bohr, coordinates_in_angstrom) where each row is
    (Z, charge, x, y, z) in the file's own units, to_bohr is the divisor that
    converts those into bohr, and the third element is the same rows converted
    to Angstrom.
    """
    bohr_to_ang = physical_constants['Bohr radius'][0] * 1e10
    use_bohr = "BOHR" in content.upper()
    to_bohr = 1 if use_bohr else bohr_to_ang

    atom_data = [(int(z), int(chg), float(x), float(y), float(z_))
                 for z, chg, x, y, z_ in _ATOM_COORD_RE.findall(content)]
    atom_data_ang = [
        (z, charge, x * bohr_to_ang, y * bohr_to_ang, z_ * bohr_to_ang)
        if use_bohr else (z, charge, x, y, z_)
        for (z, charge, x, y, z_) in atom_data
    ]
    return atom_data, to_bohr, atom_data_ang


def _system_info47(content):
    """Build the basis-function list and atom table from a .47 file's text."""
    atom_data, to_bohr, atom_data_ang = _atom_data_from_content(content)

    parsed_float = {v: _parse_array_from_block(v, content, float) for v in _FLOAT_VARS}
    parsed_int = {v: _parse_int_array(v, content) for v in _INT_VARS}

    exp = parsed_float['EXP']
    coeff_columns = [parsed_float.get(v, []) for v in _COEFF_COLUMNS]
    center = parsed_int['CENTER']
    label = parsed_int['LABEL']
    ncomp = parsed_int['NCOMP']

    orb_type, orb_val, shell_num = process_orbital_labels(label, ncomp, _ORB_MAPPING)
    nprim_expanded, nptr_expanded = _expand_shell_pointers(
        parsed_int['NPRIM'], parsed_int['NPTR'], ncomp)
    if len(nprim_expanded) != len(label) or len(nptr_expanded) != len(label):
        raise ValueError(f"Expanded NPRIM ({len(nprim_expanded)}) or NPTR "
                         f"({len(nptr_expanded)}) does not match LABEL ({len(label)})")

    bas_info_dict = []
    for i in range(len(label)):
        prim = nprim_expanded[i]
        ptr = nptr_expanded[i]
        atom_coord = atom_data[center[i] - 1][2:5]
        bas_info_dict.append({
            "N": i + 1, "CENTER": center[i], "LABEL": label[i],
            "shell_num": shell_num[i], "type": orb_type[i], "orb_val": orb_val[i],
            "exps": exp[ptr - 1: ptr - 1 + prim],
            "coeffs": _gather_coefficients(coeff_columns, ptr, prim),
            "xcenter": atom_coord[0] / to_bohr,
            "ycenter": atom_coord[1] / to_bohr,
            "zcenter": atom_coord[2] / to_bohr,
        })
    return bas_info_dict, atom_data_ang, to_bohr


def parse_file47(filename):
    cache_key = _file_cache_key(filename)
    cached = _parse_file47_cache.get(cache_key)
    if cached is not None:
        return cached

    print(f"Parsing {filename} as a .47 file")
    basis_info_dict, atom_data, to_bohr = _system_info47(_read_file47_text(filename))
    coordinates = [atom[2:] for atom in atom_data]
    atom_info = [(atom[0],) + tuple(atom[2:]) for atom in atom_data]
    result = (basis_info_dict, coordinates, atom_info, to_bohr)
    _parse_file47_cache[cache_key] = result
    return result


_FILE31_DASH = "-------------------------------"

# Row order of the exponent/coefficient table at the end of a .31 file.
_FILE31_COLUMNS = ('EXP', 'CS', 'CP', 'CD', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK')

# Components in a shell -> the coefficient column that shell draws from.
_FILE31_SHELL_COEFF_COLUMN = {1: 'CS', 3: 'CP', 5: 'CD', 7: 'CF',
                              9: 'CG', 11: 'CH', 13: 'CI', 15: 'CJ'}


def _file31_dash_lines(content):
    """1-based line numbers of the dashed separator lines."""
    return [i for i, line in enumerate(content, 1) if _FILE31_DASH in line]


def _parse_file31_coordinates(content, coord_line, num_atom):
    """Read the atom block as rows of [Z, x, y, z]."""
    rows = []
    for line in content[coord_line - 1: coord_line + num_atom - 1]:
        parts = line.split()
        rows.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(rows)


def _parse_file31_columns(content, start_line, num_exps):
    """
    Reshape the trailing exponent/coefficient block into one row per column.

    Columns the file does not reach are reported as None, which is what the
    shell-to-column lookup relies on to fail loudly rather than silently read
    the wrong angular momentum.
    """
    tokens = []
    for line in content[start_line:]:
        tokens.extend(line.split())
    dimen = int(len(tokens) / num_exps)
    table = np.array(tokens).reshape(dimen, num_exps)
    return {name: (table[i] if i < len(table) else None)
            for i, name in enumerate(_FILE31_COLUMNS)}


def _merge_wrapped_label_lines(lines):
    """
    Rejoin a label row that continued onto the next physical line.

    A shell's label row should hold as many entries as the preceding
    primitive-pointer row's second field says.  When it holds more than four
    but not that many, the remainder sits on the following line.
    """
    merged = []
    i = 0
    while i < len(lines):
        items = lines[i].split()
        wrapped = (len(items) > 4
                   and i > 0
                   and len(lines[i - 1].split()) >= 2
                   and len(items) != int(lines[i - 1].split()[1])
                   and i < len(lines) - 1)
        if wrapped:
            items.extend(lines[i + 1].split())
            i += 1
        merged.append(' '.join(items))
        i += 1
    return merged


def _split_prim_ptr_and_labels(lines):
    """Odd lines hold primitive pointers, even lines hold LABEL codes."""
    prim_ptr_list = []
    label_list = []
    for i, line in enumerate(lines, start=1):
        target = prim_ptr_list if i % 2 == 1 else label_list
        target.append([int(element) for element in line.split()])
    return prim_ptr_list, label_list


def _build_basis_info31(prim_ptr_list, label_list, orb_type, orb_val,
                        columns, coordinates, to_bohr):
    """Assemble the basis-function dicts for a .31 file."""
    basis_info_dict = []
    shell_num = 1
    n = 1
    for p_p_l, orb in zip(prim_ptr_list, label_list):
        lo = p_p_l[2] - 1
        hi = lo + p_p_l[3]
        for i in range(p_p_l[1]):
            # Looked up inside the loop, as the original did: a shell listing
            # zero primitives must not reach the column lookup at all.
            coeff = columns[_FILE31_SHELL_COEFF_COLUMN[len(orb)]]
            atom_coords = coordinates[p_p_l[0] - 1][1:4]
            basis_info_dict.append({
                'N': n, "CENTER": p_p_l[0], "shell_num": shell_num, "LABEL": orb[i],
                # NOTE: this subscript is wrong for any shell past the first
                # with more than one component -- it indexes orb_type by a
                # value derived from the component number rather than by the
                # shell, so a p shell reads the *first* shell's sublist and
                # raises IndexError. Preserved verbatim here: it is a
                # pre-existing defect, not something this refactor introduced,
                # and fixing it is a behaviour change needing its own review.
                "type":    orb_type[int((i - 1) / 2)][i],
                "orb_val": orb_val[int((i - 1) / 2)][i],
                "exps":    columns['EXP'][lo:hi],
                "coeffs":  coeff[lo:hi],
                "xcenter": atom_coords[0] / to_bohr,
                "ycenter": atom_coords[1] / to_bohr,
                "zcenter": atom_coords[2] / to_bohr,
            })
            n += 1
        shell_num += 1
    return basis_info_dict


def parse_file31(filename):
    cache_key = _file_cache_key(filename)
    cached = _parse_file31_cache.get(cache_key)
    if cached is not None:
        return cached

    print(f"Parsing {filename} as a .31 file")
    to_bohr = physical_constants['Bohr radius'][0] * 1e10
    with open(filename, 'r') as file:
        content = file.readlines()

    num_atom, num_shell, num_exps = (int(x) for x in content[3].split()[:3])
    dash_line_num = _file31_dash_lines(content)

    coord_line = dash_line_num[1] + 1
    coordinates = _parse_file31_coordinates(content, coord_line, num_atom)

    exp_and_coeff_line = dash_line_num[3]
    columns = _parse_file31_columns(content, exp_and_coeff_line, num_exps)

    label_sect_line = coord_line + num_atom + 1
    lines = _merge_wrapped_label_lines(
        content[label_sect_line - 1: exp_and_coeff_line - 1])
    prim_ptr_list, label_list = _split_prim_ptr_and_labels(lines)

    orb_type = [[_ORB_MAPPING.get(e)[0] for e in sublist] for sublist in label_list]
    orb_val = [[_ORB_MAPPING.get(e)[1] for e in sublist] for sublist in label_list]

    basis_info_dict = _build_basis_info31(
        prim_ptr_list, label_list, orb_type, orb_val, columns, coordinates, to_bohr)

    atom_numbers = coordinates[:, 0].tolist()
    coordinates = coordinates[:, 1:].tolist()
    atom_info = list(zip(atom_numbers, *zip(*coordinates)))
    result = (basis_info_dict, coordinates, atom_info, to_bohr)
    _parse_file31_cache[cache_key] = result
    return result


def load_aonao_matrix(filename):
    # Placeholder: Load the AO to NAO transformation matrix from .31 or related file
    # This needs to be implemented based on how AONAO is stored
    # For now, raise NotImplementedError
    raise NotImplementedError("Loading AONAO matrix is not implemented yet. Please provide the implementation.")

def is_normalized(S_diag, tol=1e-5):
    return all(abs(sii - 1.0) < tol for sii in S_diag)

# def convert_to_molden(full_basis):
#     new_basis = copy.deepcopy(full_basis)
#     for basis_func in new_basis:
#         ityp = basis_func['orb_val']
#         nc, cc, l_m_n = get_term_info(ityp)
#         x, y, z = l_m_n[0]
#         for iprim in range(len(basis_func['exps'])):
#             alpha = basis_func['exps'][iprim]
#             norm  = gaussian_norm(alpha, x, y, z)
#             basis_func['coeffs'][iprim] *= norm
#     return new_basis


def convert_to_molden(full_basis):
    new_basis = []

    for bf in full_basis:
        new_bf = {k: (list(v) if k == 'coeffs' or k == 'exps' else v) 
                  for k, v in bf.items()}
        new_basis.append(new_bf)
        
    for basis_func in new_basis:
        ityp = basis_func['orb_val']
        nc, cc, l_m_n = get_term_info(ityp)
        x, y, z = l_m_n[0]
        for iprim in range(len(basis_func['exps'])):
            alpha = basis_func['exps'][iprim]
            norm  = gaussian_norm(alpha, x, y, z)
            basis_func['coeffs'][iprim] *= norm
    return new_basis

def normalize_by_self_overlap(full_basis):
    new_basis = copy.deepcopy(full_basis)
    S_diag = np.diag(getSmat(new_basis, dict_keys, normalize_primitives=False, diagonal_only=True))
    for i, basis_func in enumerate(new_basis):
        if S_diag[i] <= 0:
            print(f"Warning: Non-positive self-overlap S_ii = {S_diag[i]} for basis function {i}. Skipping normalization.")
            continue
        sqrt_sii = math.sqrt(S_diag[i])
        for iprim in range(len(basis_func['exps'])):
            basis_func['coeffs'][iprim] /= sqrt_sii
    return new_basis

def normalize_basis_info(prev_basis_info_dict, ovlp_mat):
    self_overlap = np.diag(ovlp_mat)
    scaling_factors = 1.0 / np.sqrt(self_overlap)
    new_basis_info_dict = copy.deepcopy(prev_basis_info_dict)
    for idx, basis in enumerate(new_basis_info_dict):
        basis['coeffs'] = list(np.array(basis['coeffs']) * scaling_factors[idx])
    return new_basis_info_dict

def extract_floats_numpy(content, start_index, count):
    return np.fromstring(content[start_index:], sep=' ', count=count)

def create_symmetric_matrix_vectorized(lower_triangular, n):
    matrix = np.zeros((n, n))
    matrix[np.tril_indices(n)] = lower_triangular
    return matrix + matrix.T - np.diag(matrix.diagonal())

def process_47_file(file_path, nbas):
    cache_key = (_file_cache_key(file_path), nbas)
    cached = _process_47_file_cache.get(cache_key)
    if cached is not None:
        return cached

    with open(file_path, 'r') as file:
        content = file.read()
    lines         = content.split('\n')
    is_open_shell = 'OPEN'  in lines[0].upper()
    has_upper     = 'UPPER' in lines[0].upper()
    float_count   = int(nbas * (nbas + 1) / 2) if has_upper else int(nbas * nbas)
    keyword_dict  = {}
    for keyword in ['$OVERLAP', '$DENSITY', '$FOCK']:
        try:
            start_index = content.index(keyword) + len(keyword)
            if keyword in ['$DENSITY', '$FOCK'] and is_open_shell:
                all_arr = extract_floats_numpy(content, start_index, 2 * float_count)
                if has_upper:
                    keyword_dict[f"{keyword[1:]}_ALPHA"] = create_symmetric_matrix_vectorized(all_arr[:float_count], nbas)
                    keyword_dict[f"{keyword[1:]}_BETA"]  = create_symmetric_matrix_vectorized(all_arr[-float_count:], nbas)
                else:
                    keyword_dict[f"{keyword[1:]}_ALPHA"] = all_arr[:float_count].reshape(nbas, nbas)
                    keyword_dict[f"{keyword[1:]}_BETA"]  = all_arr[-float_count:].reshape(nbas, nbas)
            else:
                if has_upper:
                    keyword_dict[keyword[1:]] = create_symmetric_matrix_vectorized(
                        extract_floats_numpy(content, start_index, float_count), nbas)
                else:
                    keyword_dict[keyword[1:]] = extract_floats_numpy(content, start_index, float_count).reshape(nbas, nbas)
        except ValueError:
            zero = np.zeros((nbas, nbas))
            if keyword in ['$DENSITY', '$FOCK'] and is_open_shell:
                keyword_dict[f"{keyword[1:]}_ALPHA"] = zero.copy()
                keyword_dict[f"{keyword[1:]}_BETA"]  = zero.copy()
            else:
                keyword_dict[keyword[1:]] = zero.copy()
    result = (is_open_shell, keyword_dict)
    _process_47_file_cache[cache_key] = result
    return result

def calculate_overlap_matrix(primit_info_dict, nbo_overlap_mat):
    S         = getSmat(primit_info_dict, dict_keys, normalize_primitives=False, diagonal_only=False)
    n         = S.shape[0]
    S_round   = np.round(S, 8)
    nbo_round = np.round(nbo_overlap_mat, 8)
    abs_close        = np.isclose(np.abs(S_round), np.abs(nbo_round), atol=1e-5)
    sign_diff        = np.sign(S_round) != np.sign(nbo_round)
    value_close      = np.isclose(S_round, nbo_round, atol=1e-5)
    sign_change_mask = abs_close & sign_diff
    mismatch_mask    = ~value_close & ~sign_change_mask
    lower_tri_mask   = np.tril_indices(n)
    sign_change_count = np.sum(sign_change_mask[lower_tri_mask])
    mismatch_count    = np.sum(mismatch_mask[lower_tri_mask])
    diag_idx           = np.diag_indices(n)
    diag_mismatch_mask = mismatch_mask[diag_idx]
    diag_ratios        = S_round[diag_idx] / nbo_round[diag_idx]
    self_overlap_mismatches = {i+1: diag_ratios[i] for i in np.where(diag_mismatch_mask)[0]}
    return mismatch_count, sign_change_count, self_overlap_mismatches, S

def modify_basis_info(primit_info_dict, self_overlap_mismatches):
    scaling_factors = np.ones(len(primit_info_dict))
    for basis_idx, ratio in self_overlap_mismatches.items():
        scaling_factors[basis_idx-1] = 1.0 / np.sqrt(ratio)
    new_primit_info_dict = []
    for idx, basis_info in enumerate(primit_info_dict):
        new_basis_info = basis_info.copy()
        new_basis_info['coeffs'] = list(np.array(new_basis_info['coeffs']) * scaling_factors[idx])
        new_primit_info_dict.append(new_basis_info)
    return new_primit_info_dict

def iterative_basis_modification(initial_primit_info_dict, nbo_overlap_mat, max_iterations=5):
    primit_info_dict = initial_primit_info_dict
    prev_metrics = []
    for iteration in range(max_iterations):
        mismatch_count, sign_change_count, self_overlap_mismatches, Smat = \
            calculate_overlap_matrix(primit_info_dict, nbo_overlap_mat)
        print(f"\nIteration {iteration + 1}")
        print(f"Total mismatches: {mismatch_count}")
        print(f"Total sign changes: {sign_change_count}")
        prev_metrics.append((mismatch_count, sign_change_count))
        if len(prev_metrics) > 2:
            if prev_metrics[-1] == prev_metrics[-2] == prev_metrics[-3]:
                print("\nNo change in mismatches/sign changes over last three iterations. Exiting early.")
                break
        if mismatch_count == 0 or iteration == max_iterations - 1:
            print("\nFinal Results:")
            print(f"Total mismatches: {mismatch_count}")
            print(f"Total sign changes: {sign_change_count}")
            if self_overlap_mismatches:
                print("\nRemaining self-overlap mismatches (i : calculated/NBO ratio):")
                for i, ratio in self_overlap_mismatches.items():
                    print(f"{i} : {ratio:.8f}")
            else:
                print("\nNo remaining self-overlap mismatches.")
            return primit_info_dict, Smat
        primit_info_dict = modify_basis_info(primit_info_dict, self_overlap_mismatches)
    print("\nMaximum iterations reached or exited early due to stagnation.")
    return primit_info_dict, Smat


import pandas as pd
def _parse_basis_by_extension(filename):
    """Dispatch a basis file to the .47 or .31 parser, or exit if neither."""
    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".47":
        return file_ext, parse_file47(filename)
    if file_ext == ".31":
        return file_ext, parse_file31(filename)
    print(f"Unsupported file extension: {file_ext}")
    sys.exit(1)


def _detect_basis_convention(basis_info_dict):
    """
    Work out whether a basis's contraction coefficients already include the
    primitive normalization.

    Only s and p functions are tested, since those are the ones whose
    self-overlap distinguishes the two conventions unambiguously.

    Returns (needs_molden_conversion, message, convention_recognised).
    """
    sp_basis = [f for f in basis_info_dict if f['orb_val'] in ['s', 'px', 'py', 'pz']]
    s_diag_no_norm = np.diag(getSmat(
        sp_basis, dict_keys, normalize_primitives=False, diagonal_only=True))
    s_diag_with_norm = np.diag(getSmat(
        sp_basis, dict_keys, normalize_primitives=True, diagonal_only=True))

    if is_normalized(s_diag_with_norm) and not is_normalized(s_diag_no_norm):
        return True, ("Basis set is in Gaussian convention "
                      "(coefficients do NOT include normalization)."), True
    if is_normalized(s_diag_no_norm):
        return False, ("Basis set is in ORCA/Molden convention "
                       "(coefficients include normalization)."), True
    return True, ("Basis set is not normalized in either convention. "
                  "Assuming Gaussian convention."), False


def _normalise_basis_to_molden(basis_info_dict):
    """
    Bring a basis into ORCA/Molden convention and scale every function to
    S_ii = 1, reporting each step as the original inline code did.
    """
    needs_conversion, message, recognised = _detect_basis_convention(basis_info_dict)
    print(message)
    if needs_conversion:
        print("Converting all basis functions to ORCA/Molden convention...")
        basis_info_dict = convert_to_molden(basis_info_dict)
        print("Conversion to ORCA/Molden convention complete.")
    print("Normalizing coefficients by square root of self-overlap...")
    basis_info_dict = normalize_by_self_overlap(basis_info_dict)
    print("Final normalization complete. All basis functions now have S_ii = 1.")
    if not recognised:
        print('\n--------------------------------------------------------')
    return basis_info_dict


def _refine_basis_against_overlap(filename, norm_basis_info, smat, file_ext, nbf):
    """
    For a .47 file, refine the basis against the file's own $OVERLAP block and
    dump the result; a .31 file has no such block, so it passes straight
    through.
    """
    if file_ext == '.31':
        return norm_basis_info

    _is_open, matrix_dict = process_47_file(filename, nbf)
    nbo_overlap_mat = matrix_dict.get('OVERLAP')
    final_norm_basis, _final_S = iterative_basis_modification(
        norm_basis_info, nbo_overlap_mat)

    for info in final_norm_basis:
        for key, value in info.items():
            print(f"{key}: {value}")
        print("---------------------------")
    print(pd.DataFrame(smat))
    return final_norm_basis


def _prompt_spin_start_line(lines, orbital_file):
    """
    Ask which spin to read from an open-shell key file and return the first
    data line of that block; a closed-shell file needs no prompt.
    """
    if "ALPHA" not in lines[3].strip():
        print(orbital_file, " is a closed shell system...")
        return 3

    print(orbital_file, " is an open-shell system")
    alpha_or_beta = input("Enter A or B to select Alpha or Beta spin: ")
    if alpha_or_beta.lower() == 'a':
        return 4
    if alpha_or_beta.lower() == 'b':
        return _beta_start_line(lines)
    # Behaviour change, deliberately: the original left start_line unbound here
    # and died with UnboundLocalError on any answer other than A or B. Raising
    # ValueError lets the caller's existing handler report it and skip the file.
    raise ValueError(f"Expected 'A' or 'B' for spin selection, got {alpha_or_beta!r}")


def _read_cmos_interactive(orbital_file, nbas):
    """
    Read a key file's coefficient matrix, prompting for the spin when the file
    is open-shell.  Returns None (after reporting) if the file is unusable.
    """
    try:
        with open(orbital_file, "r") as file:
            lines = file.readlines()
        if len(lines) < 4:
            raise ValueError("File structure is incorrect or file is too short.")
        orbital_type = lines[1].strip().split()[0]
        print(orbital_type, "in AO basis")

        start_line = _prompt_spin_start_line(lines, orbital_file)
        words = _read_limited_floats(lines, start_line, nbas * nbas)
        if len(words) % nbas != 0:
            raise ValueError("Please provide a correct NBO file.\n")
        num_cmos = int(len(words) / nbas)
        return np.array(words).reshape(nbas, num_cmos)
    except ValueError as e:
        print(e)
        return None


def _parse_orbital_index_spec(text):
    """
    Expand an orbital selection such as "1,5,8-12" into a list of 1-based
    indices.  Raises ValueError on a malformed or descending range.
    """
    indices = []
    for item in text.split(","):
        if "-" in item:
            start, end = map(int, item.split("-"))
            if start > end:
                raise ValueError("Start of range must be less than end of range.")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(item))
    return indices


def _prompt_orbital_indices():
    """Prompt until a valid orbital selection is entered."""
    while True:
        try:
            return _parse_orbital_index_spec(
                input("Enter an orbital index (e.g., 1,5,8-12): "))
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")


def _collect_orbital_dict(orbital_files, orbital_index, nbas):
    """Read the requested orbitals out of each existing key file."""
    orbital_dict = {}
    for orbital_file in orbital_files:
        if not os.path.exists(orbital_file):
            print(f"The file {orbital_file} does not exist. Please try again.")
            continue
        orbital_arr = _read_cmos_interactive(orbital_file, nbas)
        if orbital_arr is not None:
            orbital_dict[orbital_file] = [orbital_arr[i - 1] for i in orbital_index]
    return orbital_dict


def main():
    filename = input("Enter the filename (.47 recommended or .31): ")
    file_ext, parsed = _parse_basis_by_extension(filename)
    basis_info_dict, coordinates, atom_info, _to_bohr = parsed

    basis_info_dict = _normalise_basis_to_molden(basis_info_dict)

    smat = getSmat(basis_info_dict, dict_keys,
                   normalize_primitives=False, diagonal_only=False)
    norm_basis_info = normalize_basis_info(basis_info_dict, smat)

    final_norm_basis = _refine_basis_against_overlap(
        filename, norm_basis_info, smat, file_ext, len(basis_info_dict))

    print('Basis information extracted and renormalized...')

    orbital_files = input(
        "Enter NBO key files separated by commas: ").replace(" ", "").split(",")
    orbital_index = _prompt_orbital_indices()
    orbital_dict = _collect_orbital_dict(
        orbital_files, orbital_index, len(final_norm_basis))

    return {
        'final_norm_basis': final_norm_basis,   # normalised basis function list
        'coordinates':      coordinates,         # atom coords in Angstrom
        'atom_info':        atom_info,           # (Z, x, y, z) per atom
        'orbital_dict':     orbital_dict,        # {filename: [cmo_vector, ...]}
        'orbital_index':    orbital_index,       # requested indices (1-based)
        'bohr':             bohr,                # Angstrom-per-bohr constant
    }



def load_basis_headless(basis_filepath):
    """
    Parse and fully normalise a .47 or .31 basis file.
    Returns (final_norm_basis, coordinates_ang, atom_info).
    coordinates_ang : list of (x,y,z) tuples in Angstrom
    atom_info       : list of (Z, x_ang, y_ang, z_ang) tuples
    """
    cache_key = _file_cache_key(basis_filepath)
    cached = _load_basis_headless_cache.get(cache_key)
    if cached is not None:
        return cached

    ext = os.path.splitext(basis_filepath)[1].lower()
    if ext == '.47':
        basis_info_dict, coordinates, atom_info, to_bohr = parse_file47(basis_filepath)
    elif ext == '.31':
        basis_info_dict, coordinates, atom_info, to_bohr = parse_file31(basis_filepath)
    else:
        raise ValueError(f"Unsupported basis file extension: {ext}")

    sp_basis         = [f for f in basis_info_dict if f['orb_val'] in ['s','px','py','pz']]
    S_no   = np.diag(getSmat(sp_basis, dict_keys, normalize_primitives=False, diagonal_only=True))
    S_with = np.diag(getSmat(sp_basis, dict_keys, normalize_primitives=True,  diagonal_only=True))

    if is_normalized(S_with) and not is_normalized(S_no):
        basis_info_dict = convert_to_molden(basis_info_dict)
        basis_info_dict = normalize_by_self_overlap(basis_info_dict)
    elif is_normalized(S_no):
        basis_info_dict = normalize_by_self_overlap(basis_info_dict)
    else:
        basis_info_dict = convert_to_molden(basis_info_dict)
        basis_info_dict = normalize_by_self_overlap(basis_info_dict)

    Smat            = getSmat(basis_info_dict, dict_keys, normalize_primitives=False, diagonal_only=False)
    norm_basis_info = normalize_basis_info(basis_info_dict, Smat)

    nbf = len(basis_info_dict)
    if ext == '.31':
        final_norm_basis = norm_basis_info
    else:
        is_open, matrix_dict = process_47_file(basis_filepath, nbf)
        nbo_overlap_mat      = matrix_dict.get('OVERLAP')
        final_norm_basis, _  = iterative_basis_modification(norm_basis_info, nbo_overlap_mat)


    # print(final_norm_basis)
    result = (final_norm_basis, coordinates, atom_info)
    _load_basis_headless_cache[cache_key] = result
    return result


def get_ao_overlap_matrix(basis_filepath):
    """
    Analytic AO overlap matrix S for basis_filepath's normalized basis
    (load_basis_headless()'s final_norm_basis), computed once per file and
    cached -- this is the same (nbas, nbas) integral evaluation that both
    localization (localization_io.get_localization_inputs) and population
    analysis (chemview._load_overlap_for_details) need, so callers should
    go through here instead of recomputing it themselves.
    """
    cache_key = _file_cache_key(basis_filepath)
    cached = _ao_overlap_cache.get(cache_key)
    if cached is not None:
        return cached

    final_basis, _, _ = load_basis_headless(basis_filepath)
    overlap = getSmat(final_basis, dict_keys, normalize_primitives=False, diagonal_only=False)
    _ao_overlap_cache[cache_key] = overlap
    return overlap


def _read_key_lines(key_filepath):
    cache_key = ("key_lines", _file_cache_key(key_filepath))

    def load():
        with open(key_filepath, "r") as handle:
            return handle.readlines()

    return _key_source_cache.get(cache_key, load)


def _detect_open_shell_key(lines, key_filepath):
    """Detect whether a key-like orbital file should be treated as open shell."""
    if len(lines) > 3 and 'ALPHA' in lines[3].strip().upper():
        return True

    ext = os.path.splitext(key_filepath)[1].lower()
    if ext not in {'.32', '.33'}:
        return False

    base = os.path.splitext(key_filepath)[0]
    file47 = base + '.47'
    if not os.path.exists(file47):
        return False
    try:
        with open(file47, 'r') as f:
            first_line = f.readline().upper()
        return 'OPEN' in first_line
    except Exception:
        return False


def _single_block_open_shell_key(lines, key_filepath, is_open=None):
    """True for open-shell .32/.33 files that omit explicit ALPHA/BETA headers."""
    ext = os.path.splitext(key_filepath)[1].lower()
    if ext not in {'.32', '.33'}:
        return False
    if is_open is None:
        is_open = _detect_open_shell_key(lines, key_filepath)
    if not is_open:
        return False
    has_explicit_spins = any('ALPHA' in line.upper() for line in lines[:8]) or \
        any('BETA' in line.upper() for line in lines)
    return not has_explicit_spins


# Standard NBO extension-to-orbital-type mapping (no file I/O needed)
# Only includes extensions .32 to .41 (standard NBO output files)
_NBO_EXTENSION_TYPES = {
    '.32': 'PNAOs',    # Pre-orthogonal Natural Atomic Orbitals
    '.33': 'NAOs',     # Natural Atomic Orbitals
    '.34': 'PNHOs',    # Pre-orthogonal Natural Hybrid Orbitals
    '.35': 'NHOs',     # Natural Hybrid Orbitals
    '.36': 'PNBOs',    # Pre-orthogonal Natural Bond Orbitals
    '.37': 'NBOs',     # Natural Bond Orbitals
    '.38': 'PNLMOs',   # Pre-orthogonal Natural Localized Molecular Orbitals
    '.39': 'NLMOs',    # Natural Localized Molecular Orbitals
    '.40': 'MOs',      # Molecular Orbitals
    '.41': 'NOs',      # Natural Orbitals
}


def get_orbital_type_from_extension(key_filepath):
    """
    Fast lookup: return orbital type from file extension without reading the file.
    Returns the orbital type string (e.g. 'NBOs', 'NAOs') or None if extension unknown.
    For unknown extensions, the caller should list the file without an orbital type.
    """
    ext = os.path.splitext(key_filepath)[1].lower()
    return _NBO_EXTENSION_TYPES.get(ext)


def _get_orbital_count_uncached(key_filepath, basis_info_dict=None):
    """
    Peek at key file header to determine (orbital_type_str, nbas, is_open_shell).
    orbital_type_str is e.g. 'NBO', 'NHO', 'NAO' from line 2.
    nbas is the number of basis functions (= number of orbitals in the file).

    basis_info_dict : optional, the already-parsed basis (parse_file47/
        parse_file31's first return value) for this key file's sibling
        basis file. Callers that check many key files sharing the same
        basis file (e.g. a "pick a key file" listing) should parse the
        basis file once and pass it in here for every key file, instead
        of re-parsing it -- potentially a large, expensive file -- once
        per key file.
    """
    lines = _read_key_lines(key_filepath)
    if len(lines) < 4:
        raise ValueError("Key file too short to parse header")
    orbital_type = lines[1].strip().split()[0] if len(lines) > 1 else 'UNKNOWN'
    is_open      = _detect_open_shell_key(lines, key_filepath)

    if basis_info_dict is None:
        base, _ = os.path.splitext(key_filepath)
        path_47 = base + ".47"
        path_31 = base + ".31"

        if os.path.exists(path_47):
            basis_filepath = path_47
        elif os.path.exists(path_31):
            basis_filepath = path_31
        else:
            raise FileNotFoundError(
                f"No sibling .47 or .31 basis file found for {key_filepath}"
            )

        ext = os.path.splitext(basis_filepath)[1].lower()
        if ext == '.47':
            basis_info_dict, coordinates, atom_info, to_bohr = parse_file47(basis_filepath)
        elif ext == '.31':
            basis_info_dict, coordinates, atom_info, to_bohr = parse_file31(basis_filepath)
        else:
            raise ValueError(f"Unsupported basis file extension: {ext}")

    nbas = len(basis_info_dict)
    return orbital_type, nbas, is_open



def get_orbital_count(key_filepath, basis_info_dict=None):
    """Cached public wrapper for NBO key-file orbital metadata."""
    if basis_info_dict is not None:
        return _get_orbital_count_uncached(key_filepath, basis_info_dict)

    base = os.path.splitext(key_filepath)[0]
    basis_filepath = None
    for candidate in (base + ".47", base + ".31"):
        if os.path.exists(candidate):
            basis_filepath = candidate
            break
    basis_key = (
        _file_cache_key(basis_filepath) if basis_filepath is not None else None
    )
    cache_key = ("orbital_count", _file_cache_key(key_filepath), basis_key)
    return _key_source_cache.get(
        cache_key, lambda: _get_orbital_count_uncached(key_filepath)
    )


def load_cmos_headless(key_filepath, orbital_indices, spin='alpha'):
    """
    Load CMO row vectors for requested 1-based orbital_indices.
    spin : 'alpha' or 'beta' (open-shell files only).
    Returns list of 1-D numpy arrays, one per requested orbital.
    """
    orbital_arr = _load_cmo_matrix(key_filepath, spin)
    return [orbital_arr[i - 1] for i in orbital_indices]


def _read_limited_floats(lines, start_line, limit):
    """
    Collect up to *limit* floats from lines[start_line:], ignoring any token
    that will not parse as a number.
    """
    words = []
    for line in lines[start_line:]:
        for elem in line.split():
            if len(words) >= limit:
                return words
            try:
                words.append(float(elem))
            except ValueError:
                pass
    return words


def _beta_start_line(lines):
    """First data line after the BETA header of an open-shell key file."""
    for i, line in enumerate(lines):
        if 'BETA' in line.upper():
            return i + 1
    raise ValueError("BETA section not found in open-shell key file")


def _cmo_block_start_line(lines, is_open, duplicate_single_block, spin_key):
    """
    First data line of the requested spin's coefficient block.

    A closed-shell file -- or an open-shell one whose single block stands for
    both spins -- starts at line 3.  An open-shell alpha block starts at 4,
    and beta starts after its own BETA header.
    """
    if is_open and spin_key == 'beta' and not duplicate_single_block:
        return _beta_start_line(lines)
    return 3 if (not is_open or duplicate_single_block) else 4


def _load_cmo_matrix(key_filepath, spin='alpha'):
    spin_key = 'beta' if spin.lower().startswith('b') else 'alpha'
    cache_key = ("cmo_matrix", spin_key, _file_cache_key(key_filepath))

    def load():
        _, nbas, _ = get_orbital_count(key_filepath)
        lines = _read_key_lines(key_filepath)
        is_open = _detect_open_shell_key(lines, key_filepath)
        duplicate_single_block = _single_block_open_shell_key(
            lines, key_filepath, is_open=is_open
        )
        start_line = _cmo_block_start_line(
            lines, is_open, duplicate_single_block, spin_key)

        words = _read_limited_floats(lines, start_line, nbas * nbas)
        if len(words) < nbas * nbas:
            raise ValueError(
                f"Not enough data: expected {nbas*nbas} floats, got {len(words)}"
            )
        return np.array(words[:nbas * nbas]).reshape(nbas, nbas)

    return _key_source_cache.get(cache_key, load)


def load_transformation_matrix(key_filepath, spin='alpha'):
    """
    Load the full NBO key-file coefficient matrix.

    This is the same matrix used by load_cmos_headless(), returned as a
    square array so callers can use it as a basis transformation matrix.
    Rows are the key-file orbitals/basis functions, columns are AO basis
    functions, matching the orientation used for orbital plotting.
    """
    _, nbas, _ = get_orbital_count(key_filepath)
    rows = load_cmos_headless(
        key_filepath,
        list(range(1, nbas + 1)),
        spin=spin,
    )
    return np.asarray(rows, dtype=float)

# def process_47_file(file_path: str, nbas: int) -> tuple[bool, dict[str, np.ndarray]]:
#     """Extract OVERLAP, DENSITY, and FOCK matrices from .47 file."""
#     with open(file_path, 'r') as f:
#         content = f.read()

#     lines = content.split('\n')
#     is_open_shell = 'OPEN' in lines[0].upper()

#     float_count = int(nbas * (nbas + 1) / 2)          # lower triangular count

#     keyword_dict = {}
#     target_keywords = ['$OVERLAP', '$DENSITY', '$FOCK']

#     for keyword in target_keywords:
#         try:
#             start_idx = content.index(keyword) + len(keyword)

#             if keyword in ('$DENSITY', '$FOCK') and is_open_shell:
#                 # Open-shell: two spins
#                 spin_data = extract_floats_numpy(content, start_idx, 2 * float_count)
#                 alpha = create_symmetric_matrix_vectorized(spin_data[:float_count], nbas)
#                 beta  = create_symmetric_matrix_vectorized(spin_data[float_count:], nbas)

#                 keyword_dict[f"{keyword[1:]}_ALPHA"] = alpha
#                 keyword_dict[f"{keyword[1:]}_BETA"]  = beta
#             else:
#                 # Closed-shell or OVERLAP
#                 data = extract_floats_numpy(content, start_idx, float_count)
#                 matrix = create_symmetric_matrix_vectorized(data, nbas)
#                 dict_key = keyword[1:] if keyword.startswith('$') else keyword
#                 keyword_dict[dict_key] = matrix

#         except ValueError:
#             # Missing section
#             zero_mat = np.zeros((nbas, nbas))
#             if keyword in ('$DENSITY', '$FOCK') and is_open_shell:
#                 keyword_dict[f"{keyword[1:]}_ALPHA"] = zero_mat.copy()
#                 keyword_dict[f"{keyword[1:]}_BETA"]  = zero_mat.copy()
#             else:
#                 dict_key = keyword[1:] if keyword.startswith('$') else keyword
#                 keyword_dict[dict_key] = zero_mat.copy()

#             print(f"Warning: {keyword.replace('$', '')} matrix not found in {file_path}")

#     return is_open_shell, keyword_dict

def _get_orbital_energies_and_occupations_uncached(key_filepath: str, basis_filepath: str = None):
    """
    Return orbital energies (Hartree) and occupations for the requested key file.
    Works for both closed-shell and open-shell.
    Returns:
        (energies_alpha, occ_alpha, energies_beta, occ_beta)
        If closed-shell, beta arrays are None.
    """
    import nbo_read as _self   # for circular safety

    # Get MO coefficient matrix
    try:
        orbital_type, nbas, is_open = _self.get_orbital_count(key_filepath)
        cmos_alpha = _self.load_cmos_headless(key_filepath, list(range(1, nbas+1)), spin='alpha')
        cmos_alpha = np.column_stack(cmos_alpha)   # shape (nbas, nbas)

        if is_open:
            cmos_beta = _self.load_cmos_headless(key_filepath, list(range(1, nbas+1)), spin='beta')
            cmos_beta = np.column_stack(cmos_beta)
        else:
            cmos_beta = None
    except Exception as e:
        print("Failed to load MO coefficients:", e)
        return None, None, None, None

    # Find corresponding .47 file
    base = os.path.splitext(key_filepath)[0]
    file47 = base + ".47"
    if not os.path.exists(file47):
        print(f".47 file not found: {file47}")
        return None, None, None, None

    # Extract matrices from .47
    is_open_from47, matrices = process_47_file(file47, nbas)

    # Density, overlap & Fock
    overlap = matrices.get('OVERLAP', np.eye(nbas))
    if is_open:
        dm_alpha = matrices.get('DENSITY_ALPHA', np.zeros((nbas, nbas)))
        dm_beta  = matrices.get('DENSITY_BETA',  np.zeros((nbas, nbas)))
        f_alpha  = matrices.get('FOCK_ALPHA',    np.zeros((nbas, nbas)))
        f_beta   = matrices.get('FOCK_BETA',     np.zeros((nbas, nbas)))
    else:
        dm = matrices.get('DENSITY', np.zeros((nbas, nbas)))
        f  = matrices.get('FOCK',    np.zeros((nbas, nbas)))
        dm_alpha = dm
        f_alpha  = f
        dm_beta = f_beta = None

    # Compute energies and occupations
    try:
        occ_alpha = get_occupation(cmos_alpha, dm_alpha, overlap)
        ene_alpha = get_energy(cmos_alpha, f_alpha)

        if is_open and cmos_beta is not None:
            occ_beta = get_occupation(cmos_beta, dm_beta, overlap)
            ene_beta = get_energy(cmos_beta, f_beta)
        else:
            occ_beta = ene_beta = None
    except Exception as e:
        print("Error computing energies/occupations:", e)
        occ_alpha = ene_alpha = np.zeros(nbas)
        occ_beta = ene_beta = None

    return ene_alpha, occ_alpha, ene_beta, occ_beta


def get_orbital_energies_and_occupations(key_filepath: str, basis_filepath: str = None):
    """Return cached NBO orbital energies and occupations."""
    base = os.path.splitext(key_filepath)[0]
    file47 = base + ".47"
    file47_key = (
        _file_cache_key(file47)
        if os.path.exists(file47)
        else (os.path.abspath(file47), "missing")
    )
    basis_key = (
        _file_cache_key(basis_filepath)
        if basis_filepath and os.path.exists(basis_filepath)
        else None
    )
    cache_key = ("orbital_metadata", _file_cache_key(key_filepath), file47_key, basis_key)
    return _key_source_cache.get(
        cache_key,
        lambda: _get_orbital_energies_and_occupations_uncached(key_filepath, basis_filepath),
    )


# Keep your original helper functions (improved a bit)
def get_occupation(mat, dm, smat=None):
    """mat: orbital coefficients in AO basis, dm: density matrix, smat: overlap."""
    try:
        if smat is not None:
            return np.diag(mat.T @ smat @ dm @ smat @ mat)
        inv_mat = np.linalg.inv(mat)
        return np.diag(inv_mat @ dm @ inv_mat.T)
    except np.linalg.LinAlgError:
        return np.full(mat.shape[1], np.nan)


def get_energy(mat, fmat):
    """mat: MO coefficients, fmat: Fock matrix"""
    try:
        return np.diag(mat.T @ fmat @ mat)   # orbital energies in Hartree
    except:
        return np.full(mat.shape[1], np.nan)

def _write_cube_headless(filepath, grid_data, atom_info,
                         nx, ny, nz, spacing, origin, bohr_const):
    """
    Write a Gaussian cube file.
    atom_info : list of (Z, x_ang, y_ang, z_ang)
    origin, spacing : in bohr (cube spec requires bohr)
    """
    with open(filepath, 'w') as f:
        f.write("Generated by NBO2CUBE\n")
        f.write("Orbital\n")
        f.write(f"{len(atom_info):4d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        f.write(f"{nx:4d} {spacing[0]:12.6f}   0.000000   0.000000\n")
        f.write(f"{ny:4d}   0.000000 {spacing[1]:12.6f}   0.000000\n")
        f.write(f"{nz:4d}   0.000000   0.000000 {spacing[2]:12.6f}\n")
        for atom in atom_info:
            Z = int(round(float(atom[0])))
            xb = float(atom[1]) / bohr_const
            yb = float(atom[2]) / bohr_const
            zb = float(atom[3]) / bohr_const
            f.write(f"{Z:4d} {float(Z):10.6f} {xb:10.6f} {yb:10.6f} {zb:10.6f}\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{grid_data[i, j, k].real:>13.5E}")
                    if (k + 1) % 6 == 0:
                        f.write("\n")
                f.write("\n")


def compute_cube_data(final_norm_basis, coordinates_ang, atom_info,
                      orbital_indices, key_filepath, spin,
                      grid_quality, ext_dist, bohr_const,
                      precomputed_cmos=None):
    """
    Compute orbital grids for the requested orbitals from a pre-loaded basis.
    Returns in-memory data only — no files are written here.

    Parameters
    ----------
    final_norm_basis   : list from load_basis_headless()
    coordinates_ang    : list of (x,y,z) in Angstrom
    atom_info          : list of (Z, x_ang, y_ang, z_ang)
    orbital_indices    : list of 1-based int
    key_filepath       : path to NBO key file (used only for label/CMO loading)
    spin               : 'alpha' or 'beta'
    precomputed_cmos   : optional list of 1-D AO-coefficient arrays, one per
                          orbital_indices entry, already in hand (e.g. from
                          localization_io.localize_orbitals). When given,
                          key_filepath is only used for labeling and the
                          normal load_cmos_headless() call is skipped.
    grid_quality       : 50 / 75 / 100 / 125  (max grid points on widest axis)
    ext_dist           : float bohr extension past molecular bounds
    bohr_const         : Angstrom-per-bohr constant

    Returns
    -------
    List of dicts, one per orbital:
        {'index': int,          # 1-based orbital index
         'label': str,          # e.g. "molecule.31-7"
         'grid':  ndarray,      # shape (nx, ny, nz)
         'nx': int, 'ny': int, 'nz': int,
         'spacing': ndarray,    # [sx, sy, sz] in bohr
         'origin':  ndarray,    # [ox, oy, oz] in bohr
         'atom_info': list,     # (Z, x_ang, y_ang, z_ang) tuples
         'bohr_const': float}
    """
    cmos = precomputed_cmos if precomputed_cmos is not None else load_cmos_headless(key_filepath, orbital_indices, spin)

    return evaluate_orbital_grids(
        final_norm_basis, coordinates_ang, atom_info, cmos, orbital_indices,
        os.path.basename(os.path.splitext(key_filepath)[0]),
        grid_quality, ext_dist, bohr_const)


def write_cube_from_result(result_dict, filepath):
    """
    Save one compute_cube_data result dict to a .cube file.
    Call this when the user explicitly requests saving.
    """
    r = result_dict
    _write_cube_headless(
        filepath, r['grid'], r['atom_info'],
        r['nx'], r['ny'], r['nz'],
        r['spacing'], r['origin'], r['bohr_const'])

if __name__ == '__main__':
    main()
