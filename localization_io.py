"""
localization_io.py
===================
Single entry point for orbital-localization code to pull what it needs out
of an NBO (.47/.31), Gaussian checkpoint (.fchk/.fck), or Molden (.molden)
source, without having to know which loader (nbo_read / fchk_read /
read_molden) or file-format quirks apply, and to actually run Pipek-Mezey
localization once it has them.

    get_localization_inputs(path, key_path=None, spin='alpha')
        -> (cmo_matrix, overlap_matrix, final_basis)

    get_fock_matrix(path, key_path=None, spin='alpha', cmo=None, overlap=None)
        -> fock_matrix

    localize_orbitals(cmo, overlap, fock, final_basis,
                       space='occupied'|'virtual'|'range'|'occupied_valence', ...)
        -> (localized_cmo, loc_energy)

    compute_localized_cube_data(path, spin='alpha', space='occupied', ...)
        -> dict of localized cube-grid data, ready for a viewer
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from itertools import product

try:
    from localization_native import localize_orbitals_cpp as _localize_orbitals_cpp
except Exception:  # pragma: no cover - optional native extension
    _localize_orbitals_cpp = None


def _recognize_source_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".47", ".31"}:
        return "nbo"
    if ext in {".fchk", ".fck"}:
        return "fchk"
    if ext == ".molden":
        return "molden"
    return None


def get_localization_inputs(path, key_path=None, spin="alpha"):
    """
    Return (cmo_matrix, overlap_matrix, final_basis) for orbital localization.

    Parameters
    ----------
    path     : str
        NBO basis file (.47/.31), Gaussian checkpoint (.fchk/.fck), or
        Molden file (.molden).
    key_path : str, optional
        Required only for NBO sources — the paired key file (.31/.32/.33/...)
        holding the CMO coefficients. Ignored for fchk/molden, which store
        both the basis and the MOs in the same file.
    spin     : 'alpha' or 'beta'

    Returns
    -------
    cmo_matrix     : (nbas, norb) ndarray — columns are MOs, rows are AOs,
                      i.e. cmo_matrix[mu, i] is the coefficient of AO mu in
                      MO i.
    overlap_matrix : (nbas, nbas) ndarray — AO overlap matrix S.
    final_basis    : list of basis-function dicts, the shared schema used
                      by nbo_read / fchk_read / read_molden (N, CENTER,
                      type, orb_val, exps, coeffs, xcenter/ycenter/zcenter).
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        import nbo_read as _nr
        if key_path is None:
            raise ValueError(
                "NBO sources require key_path (the paired .31/.32/.33 "
                "orbital file holding the CMO coefficients)."
            )
        final_basis, _, _ = _nr.load_basis_headless(path)
        nbas = len(final_basis)
        cmo_rows = _nr.load_cmos_headless(key_path, list(range(1, nbas + 1)), spin=spin)
        # Cached per basis file (path) -- reused across every call for this
        # file within the process, instead of re-integrating the same
        # (nbas, nbas) overlap matrix on every localization/population-
        # analysis request. See nbo_read.get_ao_overlap_matrix().
        overlap_matrix = _nr.get_ao_overlap_matrix(path)

    elif source_type == "fchk":
        import fchk_read as _fr
        final_basis, _, _ = _fr.load_basis_from_fchk(path)
        nbas = len(final_basis)
        cmo_rows = _fr.load_cmos_from_fchk(path, list(range(1, nbas + 1)), spin=spin)
        overlap_matrix = _fr.get_ao_overlap_matrix(path)

    elif source_type == "molden":
        import read_molden as _mr
        final_basis, _, _ = _mr.load_basis_from_molden(path)
        nbas = len(final_basis)
        cmo_rows = _mr.load_cmos_from_molden(path, list(range(1, nbas + 1)), spin=spin)
        overlap_matrix = _mr.get_ao_overlap_matrix(path)

    else:
        raise ValueError(f"Unrecognized source file: {path}")

    # Loaders return one row per orbital (row i = MO i's AO coefficients);
    # localization code expects the standard AO x MO convention.
    cmo_matrix = np.asarray(cmo_rows, dtype=float).T

    return cmo_matrix, overlap_matrix, final_basis


def get_center_ranges(final_basis):
    """
    Group basis-function indices by their center (atom).

    Requires final_basis to already be ordered so all basis functions of a
    given CENTER form one contiguous run -- callers should pass final_basis
    through _reorder_for_contiguous_centers() first, since some sources
    (e.g. Molcas-generated NBO files) list extra shells for an atom after
    other atoms' shells instead of grouping every atom's functions
    together. Raises rather than silently mis-partitioning, because the
    native Pipek-Mezey extension can only represent one [bflo, bfhi] range
    per atom.

    Parameters
    ----------
    final_basis : list of basis-function dicts (must have a 'CENTER' key).

    Returns
    -------
    list of {'bflo': int, 'bfhi': int} per center, in center order, where
    bflo/bfhi are inclusive 0-based indices into final_basis.
    """
    ranges = []
    start = 0
    seen_centers = set()
    for i in range(1, len(final_basis) + 1):
        if i == len(final_basis) or final_basis[i]["CENTER"] != final_basis[start]["CENTER"]:
            center = final_basis[start]["CENTER"]
            if center in seen_centers:
                raise ValueError(
                    f"final_basis is not center-contiguous: CENTER {center!r} "
                    "reappears after another atom's basis functions. Call "
                    "_reorder_for_contiguous_centers() first."
                )
            seen_centers.add(center)
            ranges.append({"bflo": start, "bfhi": i - 1})
            start = i
    return ranges


def analyze_center_contiguity(final_basis):
    """
    Diagnose whether final_basis groups every atom's (CENTER's) basis
    functions into one contiguous run.

    This is the single source of truth for the assumption get_center_ranges()
    requires and _reorder_for_contiguous_centers() repairs -- used
    internally by both, and exposed publicly so external tooling (e.g.
    check_basis_ordering.py) can report on a source file without having to
    run localization at all.

    Parameters
    ----------
    final_basis : list of basis-function dicts (must have a 'CENTER' key).

    Returns
    -------
    dict with keys:
        contiguous        : bool -- True iff every atom's basis functions
                             form exactly one contiguous run.
        n_atoms           : number of distinct CENTER values.
        n_basis_functions : len(final_basis).
        atoms             : {center: [(lo, hi), ...]} -- 0-based inclusive
                             index ranges into final_basis for every run of
                             that atom's basis functions, in the order they
                             appear in the file.
        fragmented_atoms  : sorted list of CENTER values split across more
                             than one run.
    """
    atoms = {}
    current_center = None
    run_start = 0

    def _close_run(end_idx):
        atoms.setdefault(current_center, []).append((run_start, end_idx))

    for idx, bf in enumerate(final_basis):
        center = int(bf["CENTER"])
        if current_center is None:
            current_center = center
            run_start = idx
        elif center != current_center:
            _close_run(idx - 1)
            current_center = center
            run_start = idx
    if final_basis:
        _close_run(len(final_basis) - 1)

    fragmented_atoms = sorted(c for c, runs in atoms.items() if len(runs) > 1)

    return {
        "contiguous": not fragmented_atoms,
        "n_atoms": len(atoms),
        "n_basis_functions": len(final_basis),
        "atoms": atoms,
        "fragmented_atoms": fragmented_atoms,
    }


def _reorder_for_contiguous_centers(final_basis, cmo, overlap, fock):
    """
    Permute the AO axis so every atom's basis functions form one
    contiguous run, ordered by CENTER.

    get_center_ranges() -- and the native localization extension, which
    stores only a single [bflo, bfhi] range per atom -- both assume this.
    Most sources already satisfy it, in which case this is a no-op
    (returns the inputs unchanged). Some sources (e.g. Molcas-generated
    NBO files) don't: they can list extra shells for an atom after other
    atoms' shells, fragmenting that atom's functions across more than one
    run.

    Parameters
    ----------
    final_basis : list of basis-function dicts (must have a 'CENTER' key).
    cmo, overlap, fock : AO-indexed arrays sharing final_basis's AO order
        (cmo rows, overlap/fock rows *and* columns).

    Returns
    -------
    (final_basis, cmo, overlap, fock, inverse_perm)
        Reordered copies of the inputs, grouped by CENTER (or the original
        objects, untouched, when no reordering was needed -- inverse_perm
        is None in that case). Apply inverse_perm to any AO-indexed output
        (e.g. localized_cmo's rows) to restore the original basis-function
        order before it's compared against, or returned alongside, the
        original (unpermuted) final_basis/overlap/etc.
    """
    if analyze_center_contiguity(final_basis)["contiguous"]:
        return final_basis, cmo, overlap, fock, None

    centers = [int(bf["CENTER"]) for bf in final_basis]
    perm = np.argsort(centers, kind="stable")
    inverse_perm = np.argsort(perm)

    reordered_basis = [final_basis[i] for i in perm]
    reordered_cmo = cmo[perm, :]
    reordered_overlap = overlap[np.ix_(perm, perm)]
    reordered_fock = fock[np.ix_(perm, perm)]

    return reordered_basis, reordered_cmo, reordered_overlap, reordered_fock, inverse_perm


def get_fock_matrix(path, key_path=None, spin="alpha", cmo=None, overlap=None):
    """
    Return the AO-basis Fock matrix F for a source file.

    NBO .47 files store the Fock matrix directly (the $FOCK section) and it
    is simply read and returned as-is -- or, if the .47 doesn't have a
    $FOCK section at all (some job types omit it), a zero matrix of the
    right shape, with a RuntimeWarning. A zero Fock matrix doesn't affect
    Pipek-Mezey localization itself (that only uses the overlap matrix),
    but any energy/sort-order derived from it (e.g. loc_energy) becomes
    meaningless (all zero) in that case.

    fchk/molden sources don't store a
    Fock matrix, so it's rebuilt from the canonical orbital energies and MO
    coefficients:

        F = (S C) E (S C)^T = S C E C^T S

    where C is the AO x MO coefficient matrix, S is the AO overlap matrix,
    and E is the diagonal matrix of orbital energies. This holds because
    C^T S C = I (MO orthonormality in the AO metric) implies C^-1 = C^T S,
    so F_AO = S C F_MO C^-1 = S C E C^T S.

    Parameters
    ----------
    path, key_path, spin : same as get_localization_inputs.
    cmo, overlap : optional pre-computed cmo_matrix/overlap_matrix (e.g.
        from a prior get_localization_inputs call on the same source) to
        avoid re-parsing the file. Ignored for NBO sources, since the Fock
        matrix there is read directly rather than reconstructed.

    Returns
    -------
    fock_matrix : (nbas, nbas) ndarray, AO basis.
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        if os.path.splitext(path)[1].lower() != ".47":
            raise ValueError(
                "The Fock matrix is only stored in NBO .47 files (not .31) "
                "— point `path` at the .47 file."
            )
        import nbo_read as _nr
        final_basis, _, _ = _nr.load_basis_headless(path)
        nbas = len(final_basis)
        is_open, matrix_dict = _nr.process_47_file(path, nbas)
        key = ("FOCK_BETA" if spin.lower().startswith("b") else "FOCK_ALPHA") if is_open else "FOCK"
        fock = matrix_dict.get(key)
        if fock is None or not np.any(fock):
            warnings.warn(
                f"No $FOCK section found in {path} -- using a zero matrix. "
                "Pipek-Mezey localization itself is unaffected (it only uses "
                "the overlap matrix), but loc_energy/orbital sort order will "
                "be meaningless (all zero) since there's no real Fock matrix "
                "to derive them from.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros((nbas, nbas), dtype=float)
        return fock

    if source_type not in ("fchk", "molden"):
        raise ValueError(f"Unrecognized source file: {path}")

    if cmo is None or overlap is None:
        cmo, overlap, _ = get_localization_inputs(path, key_path=key_path, spin=spin)

    if source_type == "fchk":
        import fchk_read as _fr
        ene_alpha, _, ene_beta, _ = _fr.get_orbital_energies_and_occupations_fchk(path)
    else:
        import read_molden as _mr
        ene_alpha, _, ene_beta, _ = _mr.get_orbital_energies_and_occupations_molden(path)

    energies = ene_beta if (spin.lower().startswith("b") and ene_beta is not None) else ene_alpha
    if energies is None or len(energies) == 0:
        raise ValueError(f"No {spin} orbital energies found in {path}")

    E = np.diag(np.asarray(energies, dtype=float))
    return overlap @ cmo @ E @ cmo.T @ overlap



def _get_occupation_arrays(path, key_path=None):
    """
    Return (source_type, occ_alpha, occ_beta) occupation-number arrays for
    a source. occ_beta is None for closed-shell sources.
    """
    source_type = _recognize_source_type(path)

    if source_type == "nbo":
        if key_path is None:
            raise ValueError(
                "NBO sources require key_path (the paired .31/.32/.33 "
                "orbital file holding the CMO coefficients)."
            )
        import nbo_read as _nr
        _, occ_alpha, _, occ_beta = _nr.get_orbital_energies_and_occupations(key_path)

    elif source_type == "fchk":
        import fchk_read as _fr
        _, occ_alpha, _, occ_beta = _fr.get_orbital_energies_and_occupations_fchk(path)

    elif source_type == "molden":
        import read_molden as _mr
        _, occ_alpha, _, occ_beta = _mr.get_orbital_energies_and_occupations_molden(path)

    else:
        raise ValueError(f"Unrecognized source file: {path}")

    if occ_alpha is None or len(occ_alpha) == 0:
        raise ValueError(f"No orbital occupation numbers found in {path}")

    occ_alpha = np.asarray(occ_alpha, dtype=float)
    occ_beta = (
        np.asarray(occ_beta, dtype=float)
        if (occ_beta is not None and len(occ_beta) > 0)
        else None
    )
    return source_type, occ_alpha, occ_beta


def get_electron_count(path, key_path=None):
    """
    Return the total number of electrons for a source file, derived from
    its orbital occupation numbers.

    Every loader (nbo/fchk/molden) reports occ_beta=None for closed-shell
    sources, with the single alpha channel already on the orbital's total
    (0-2) occupation scale, so no per-loader scale correction is needed.

    Parameters
    ----------
    path     : str, same as get_localization_inputs.
    key_path : str, optional — required for NBO sources (see
        get_localization_inputs); ignored for fchk/molden.
    """
    source_type, occ_alpha, occ_beta = _get_occupation_arrays(path, key_path=key_path)

    if occ_beta is not None:
        return float(np.sum(occ_alpha) + np.sum(occ_beta))

    return float(np.sum(occ_alpha))


def _num_occupied_from_arrays(source_type, occ_alpha, occ_beta, spin="alpha", path="source"):
    if occ_beta is None:
        # Closed-shell: occ_alpha is already on the total (0-2) scale.
        n_electrons = np.sum(occ_alpha)
        if round(n_electrons) % 2 != 0:
            raise ValueError(
                f"{path} has an odd electron count ({n_electrons:g}) for a "
                "closed-shell source -- this looks like an open-shell "
                "system reporting only one spin channel."
            )
        return int(round(n_electrons)) // 2

    occ = occ_beta if spin.lower().startswith("b") else occ_alpha
    return int(round(float(np.sum(occ))))


def get_num_occupied_orbitals(path, key_path=None, spin="alpha"):
    """
    Number of occupied (spatial) molecular orbitals.

    Closed-shell : n_occ = n_electrons / 2 (same for either spin channel,
        since alpha and beta orbitals are identical).
    Open-shell    : each spin-orbital holds exactly one electron, so n_occ
        for the requested spin is just that spin channel's own electron
        count (n_occ_alpha = n_alpha_electrons, n_occ_beta = n_beta_electrons)
        -- no halving.

    Raises if a closed-shell source reports an odd electron count, since
    n_occ = n_electrons/2 wouldn't be an integer.
    """
    source_type, occ_alpha, occ_beta = _get_occupation_arrays(path, key_path=key_path)
    return _num_occupied_from_arrays(
        source_type, occ_alpha, occ_beta, spin=spin, path=path
    )


def _prepare_atom_grouping(basis, n_basis_fn):
    """
    Precompute the grouped-AO-index bookkeeping needed to sum any
    AO-indexed array over each atom's basis functions.

    Returns a `reduce_by_atom(values)` callable: sums `values` (shape
    (n_basis_fn, ...)) over each atom's ['bflo', 'bfhi'] range, giving an
    (natom, ...) array in the same atom order as `basis`. Shared by
    localize_orbitals() and atomic_populations() so both use identical
    atom bookkeeping and range validation.
    """
    natom = len(basis)
    atom_ranges = []
    atom_starts = np.empty(natom, dtype=np.intp)

    next_start = 0
    for atom_index, atom in enumerate(basis):
        bflo = int(atom["bflo"])
        bfhi = int(atom["bfhi"])

        if not (0 <= bflo <= bfhi < n_basis_fn):
            raise ValueError(
                f"Invalid basis-function range ({bflo}, {bfhi}) "
                f"for atom {atom_index}; valid basis indices are "
                f"0 through {n_basis_fn - 1}."
            )

        atom_starts[atom_index] = next_start
        indices = np.arange(bflo, bfhi + 1, dtype=np.intp)
        atom_ranges.append(indices)
        next_start += indices.size

    grouped_ao_indices = (
        np.concatenate(atom_ranges) if atom_ranges else np.empty(0, dtype=np.intp)
    )

    ao_ranges_are_contiguous = (
        grouped_ao_indices.size == n_basis_fn
        and np.array_equal(grouped_ao_indices, np.arange(n_basis_fn))
    )

    def reduce_by_atom(values):
        if values.shape[0] != n_basis_fn:
            raise ValueError(
                "The first dimension of values must equal the number "
                "of basis functions."
            )
        grouped_values = (
            values if ao_ranges_are_contiguous else values[grouped_ao_indices, ...]
        )
        return np.add.reduceat(grouped_values, atom_starts, axis=0)

    return reduce_by_atom


def atomic_populations(cmo, overlap, basis):
    """
    Per-atom Mulliken-style population q[A, i] of every orbital i (column
    of `cmo`) on every atom A (per-atom range in `basis`).

        q[A, i] = sum_{mu in A} C[mu, i] * (S C)[mu, i]

    For a canonical MO that is normalized in the AO metric (c_i^T S c_i =
    1, always true for orbitals straight out of an SCF), sum_A q[A, i] ==
    1, so q[A, i] doubles as that orbital's population *fraction* on atom
    A -- which is what find_valence_start() relies on.

    Parameters
    ----------
    cmo     : (nbas, norb) ndarray, AO x MO.
    overlap : (nbas, nbas) ndarray, AO overlap matrix S.
    basis   : per-atom AO ranges (each with inclusive 'bflo'/'bfhi'),
              sharing cmo/overlap's AO ordering -- see get_center_ranges().

    Returns
    -------
    (natom, norb) ndarray.
    """
    cmo = np.asarray(cmo, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    reduce_by_atom = _prepare_atom_grouping(basis, cmo.shape[0])
    return reduce_by_atom(cmo * (overlap @ cmo))


def find_valence_start(cmo, overlap, basis, n_occ, core_threshold=0.98):
    """
    Split the occupied orbitals into an inner/core block and an outer/
    valence block, based on how atom-centered each one is.

    Core orbitals are near-perfectly localized on a single atom already
    (typically >=98% of their population); Pipek-Mezey localization has
    little left to do for them and mixing them in only slows/destabilizes
    convergence for the orbitals that actually need it. So: starting from
    the lowest-energy occupied orbital (column 0), an orbital is core as
    long as its single largest atomic population share is >= core_threshold.
    The first occupied orbital that drops below core_threshold marks the
    start of the valence block; every orbital from there through the HOMO
    is valence, regardless of that orbital's own population share.

    Parameters
    ----------
    cmo     : (nbas, norb) ndarray, AO x MO, canonical/energy-ordered.
    overlap : (nbas, nbas) ndarray, AO overlap matrix S.
    basis   : per-atom AO ranges, see atomic_populations().
    n_occ   : number of occupied orbitals (columns 0..n_occ-1 are checked).
    core_threshold : float, default=0.98 -- minimum single-atom population
        fraction (0-1) for an orbital to count as core.

    Returns
    -------
    valence_start : int -- 0-based index, among the occupied orbitals, of
        the first valence orbital. 0 if none of them meet core_threshold
        (the whole occupied space is valence); n_occ if all of them do (no
        valence space to localize).
    """
    q_occ = atomic_populations(cmo[:, :n_occ], overlap, basis)
    max_atom_share = np.max(q_occ, axis=0)

    below_threshold = np.flatnonzero(max_atom_share < core_threshold)
    return int(below_threshold[0]) if below_threshold.size else n_occ


def _fock_diagonal(c, fock):
    """diag(c.T @ fock @ c) without forming the full n_sel x n_sel product."""
    return np.einsum('ij,ij->j', c, fock @ c)


# Numerical controls for the Pipek-Mezey Jacobi sweeps.
_PM_GAMMA_TOL = 1.0e-10        # smallest rotation angle worth applying
_PM_COUPLING_TOL = 1.0e-14     # smallest pair coupling worth rotating on
_PM_OBJECTIVE_ATOL = 1.0e-12
_PM_OBJECTIVE_RTOL = 1.0e-10
_PM_MAX_SWEEPS = 2000


def _validate_localization_inputs(cmo, overlap, fock, basis):
    """Check the AO-basis inputs agree on dimension before any work starts."""
    if cmo.ndim != 2:
        raise ValueError(
            f"cmo must be a two-dimensional array; got shape {cmo.shape}."
        )
    n_basis_fn = cmo.shape[0]
    if overlap.shape != (n_basis_fn, n_basis_fn):
        raise ValueError(
            f"overlap must have shape {(n_basis_fn, n_basis_fn)}; "
            f"got {overlap.shape}."
        )
    if fock.shape != (n_basis_fn, n_basis_fn):
        raise ValueError(
            f"fock must have shape {(n_basis_fn, n_basis_fn)}; "
            f"got {fock.shape}."
        )
    if len(basis) == 0:
        raise ValueError("basis must contain at least one atom.")


def _range_orbital_window(orbital_range, n_orbitals):
    """Convert a 1-based inclusive orbital_range to a half-open [lo, hi)."""
    if orbital_range is None:
        raise ValueError(
            "orbital_range=(first, last) is required "
            "when space='range'."
        )
    first, last = orbital_range
    if not (1 <= first <= last <= n_orbitals):
        raise ValueError(
            f"orbital_range {orbital_range} "
            f"(1-based, inclusive) is out of bounds for "
            f"{n_orbitals} orbitals."
        )
    return first - 1, last


def _valence_orbital_window(cmo, overlap, basis, n_occ, core_threshold):
    """
    Half-open window covering the outer/valence part of the occupied space.

    Returns None when every occupied orbital already meets core_threshold, so
    there is no valence block to localize.  The core orbitals are left
    untouched and are not part of the caller's output, same as any orbital
    outside the requested space.
    """
    valence_start = find_valence_start(
        cmo, overlap, basis, n_occ, core_threshold=core_threshold
    )
    if valence_start >= n_occ:
        print(
            f"All {n_occ} occupied orbitals meet the core threshold "
            f"({core_threshold:.0%} single-atom population) -- "
            "no valence orbitals to localize."
        )
        return None
    return valence_start, n_occ


def _select_orbital_window(space, cmo, overlap, basis, n_orbitals,
                           n_occ, orbital_range, core_threshold):
    """
    Resolve the requested subspace to a half-open orbital window [lo, hi).

    Returns None only for space='occupied_valence' with no valence block.
    """
    if space in ("occupied", "virtual", "occupied_valence") and n_occ is None:
        raise ValueError(
            f"n_occ is required when space={space!r}."
        )

    if space == "occupied":
        return 0, n_occ
    if space == "virtual":
        return n_occ, n_orbitals
    if space == "range":
        return _range_orbital_window(orbital_range, n_orbitals)
    if space == "occupied_valence":
        return _valence_orbital_window(
            cmo, overlap, basis, n_occ, core_threshold)
    raise ValueError(
        f"Unknown orbital space: {space!r}; expected "
        "'occupied', 'virtual', 'range', or 'occupied_valence'."
    )


class _JacobiWorkspace:
    """
    Buffers reused by every orbital pair in a sweep.

    Preallocating these keeps the pair loop free of per-rotation allocations,
    which dominate the runtime for a large basis.
    """

    __slots__ = ('rotated_c_s', 'rotated_c_t', 'rotated_sc_s', 'rotated_sc_t',
                 'cross_ao', 'cross_ao_tmp')

    def __init__(self, n_basis_fn, c, sc):
        self.rotated_c_s = np.empty(n_basis_fn, dtype=c.dtype)
        self.rotated_c_t = np.empty(n_basis_fn, dtype=c.dtype)
        self.rotated_sc_s = np.empty(n_basis_fn, dtype=sc.dtype)
        self.rotated_sc_t = np.empty(n_basis_fn, dtype=sc.dtype)
        self.cross_ao = np.empty(
            n_basis_fn, dtype=np.result_type(c.dtype, sc.dtype))
        self.cross_ao_tmp = np.empty_like(self.cross_ao)


def _pair_cross_population(c, sc, s, t, work, reduce_by_atom):
    """
    Symmetrized AO cross-population, reduced over atoms:

        q_st[A] = sum_{mu in A} 0.5 * (C[mu,t] SC[mu,s] + C[mu,s] SC[mu,t])
    """
    np.multiply(c[:, t], sc[:, s], out=work.cross_ao)
    np.multiply(c[:, s], sc[:, t], out=work.cross_ao_tmp)
    work.cross_ao += work.cross_ao_tmp
    work.cross_ao *= 0.5
    return reduce_by_atom(work.cross_ao)


def _jacobi_angle(qas, qat, qast):
    """
    Pipek-Mezey 2x2 rotation angle for one orbital pair.

    Returns None when the pair is already stationary -- either the coupling is
    below _PM_COUPLING_TOL, or the resulting angle is below _PM_GAMMA_TOL.
    """
    population_difference = qas - qat
    ast = (np.dot(qast, qast)
           - 0.25 * np.dot(population_difference, population_difference))
    bst = np.dot(qast, population_difference)

    denominator = np.hypot(ast, bst)
    if denominator < _PM_COUPLING_TOL:
        return None

    cos_arg = np.clip(-ast / denominator, -1.0, 1.0)
    gamma = 0.25 * np.arccos(cos_arg) * np.sign(bst)
    if abs(gamma) <= _PM_GAMMA_TOL:
        return None
    return gamma


def _apply_jacobi_rotation(c, sc, atomic_populations, s, t, gamma,
                           qas, qat, qast, work):
    """
    Rotate columns s and t of C and SC, and update the two affected
    atomic-population columns in place:

        q_s' = cos²(gamma) q_s + sin²(gamma) q_t + 2 cos(gamma) sin(gamma) q_st
        q_t' = sin²(gamma) q_s + cos²(gamma) q_t - 2 cos(gamma) sin(gamma) q_st
    """
    cosg = np.cos(gamma)
    sing = np.sin(gamma)

    np.multiply(c[:, s], cosg, out=work.rotated_c_s)
    work.rotated_c_s += c[:, t] * sing
    np.multiply(c[:, t], cosg, out=work.rotated_c_t)
    work.rotated_c_t -= c[:, s] * sing
    c[:, s] = work.rotated_c_s
    c[:, t] = work.rotated_c_t

    np.multiply(sc[:, s], cosg, out=work.rotated_sc_s)
    work.rotated_sc_s += sc[:, t] * sing
    np.multiply(sc[:, t], cosg, out=work.rotated_sc_t)
    work.rotated_sc_t -= sc[:, s] * sing
    sc[:, s] = work.rotated_sc_s
    sc[:, t] = work.rotated_sc_t

    cosg2 = cosg * cosg
    sing2 = sing * sing
    two_cos_sin = 2.0 * cosg * sing
    atomic_populations[:, s] = cosg2 * qas + sing2 * qat + two_cos_sin * qast
    atomic_populations[:, t] = sing2 * qas + cosg2 * qat - two_cos_sin * qast


def _jacobi_sweep(c, sc, atomic_populations, n_sel, rng, work, reduce_by_atom):
    """
    Run one full sweep over ordered orbital pairs; returns the rotation count.

    s is visited in a random order and t in normal index order, which is the
    pair ordering the original implementation used.  The pairs are generated
    lazily rather than materialised as a list.
    """
    rotations = 0
    for s in rng.permutation(n_sel):
        for t in range(n_sel):
            if t == s:
                continue
            # The atomic populations are maintained throughout the sweep, so
            # they never have to be recomputed from the AO coefficients here.
            qas = atomic_populations[:, s].copy()
            qat = atomic_populations[:, t].copy()
            qast = _pair_cross_population(c, sc, s, t, work, reduce_by_atom)

            gamma = _jacobi_angle(qas, qat, qast)
            if gamma is None:
                continue

            _apply_jacobi_rotation(c, sc, atomic_populations, s, t, gamma,
                                   qas, qat, qast, work)
            rotations += 1
    return rotations


def _pipek_mezey_objective(atomic_populations):
    """L_PM = sum_A sum_i q[A, i]^2."""
    return float(np.einsum("ai,ai->", atomic_populations,
                           atomic_populations, optimize=True))


def _report_localization(converged, final_sweep, current_objective,
                         objective_change, total_rotations, elapsed_time,
                         loc_energy):
    """Print the same convergence summary the inline code emitted."""
    if converged:
        print(f"Localization converged after {final_sweep} sweeps.")
    else:
        print(f"Localization not converged after "
              f"{_PM_MAX_SWEEPS} sweeps.")
    print(f"Pipek-Mezey objective: {current_objective:.15g}")
    print(f"Objective change: {objective_change:.3e}")
    print(f"Jacobi rotations: {total_rotations}")
    print(f"Localization runtime: {elapsed_time:.6f} seconds")
    print(f"Energy of localized orbitals:\n{loc_energy}\n")


def localize_orbitals(
    cmo,
    overlap,
    fock,
    basis,
    space="occupied",
    n_occ=None,
    orbital_range=None,
    seed=0,
    core_threshold=0.98,
):
    """
    Localize a subset of molecular orbitals with Pipek-Mezey localization
    using sequential Jacobi 2x2 rotations.

    This implementation vectorizes the atomic-population calculations.
    The orbital-pair loop remains sequential because each Jacobi rotation
    changes the orbitals used by subsequent rotations.

    Convergence is determined from the Pipek-Mezey objective

        L_PM = sum_i sum_A q_A(i)^2

    where

        q_A(i) = sum_{mu in A} C[mu, i] * (S C)[mu, i].

    Parameters
    ----------
    cmo : (nbas, norb) ndarray
        Full AO x MO coefficient matrix. Orbitals are assumed to be
        canonical and energy ordered.

    overlap : (nbas, nbas) ndarray
        AO overlap matrix S.

    fock : (nbas, nbas) ndarray
        AO Fock matrix F.

    basis : sequence of mappings
        Per-atom basis-function ranges. Each element must contain
        inclusive 'bflo' and 'bfhi' indices.

    space : {'occupied', 'virtual', 'range', 'occupied_valence'}, default='occupied'
        Orbital subspace to localize. 'occupied_valence' auto-splits the
        occupied space into an inner/core block and an outer/valence block
        (see find_valence_start()) and localizes -- and returns -- only the
        valence block; the core orbitals are left completely out of this
        call, same as any orbital outside the requested space/range.

    n_occ : int or None
        Number of occupied orbitals. Required for space='occupied',
        space='virtual', and space='occupied_valence'.

    orbital_range : tuple[int, int] or None
        First and last MO numbers using 1-based inclusive indexing.
        Required for space='range'.

    seed : int or None, default=0
        Random seed controlling the orbital-pair sweep order.

    core_threshold : float, default=0.98
        Only used for space='occupied_valence' -- see find_valence_start().

    Returns
    -------
    sorted_c : (nbas, n_selected) ndarray
        Localized orbitals of the requested subspace, sorted by Fock
        expectation value. For space='occupied_valence', n_selected is the
        size of the valence block alone (0 if every occupied orbital meets
        core_threshold -- see find_valence_start()), not n_occ.

    loc_energy : (n_selected,) ndarray
        Fock expectation values of the localized orbitals.
    """
    start_time = time.perf_counter()

    cmo = np.asarray(cmo)
    overlap = np.asarray(overlap)
    fock = np.asarray(fock)

    _validate_localization_inputs(cmo, overlap, fock, basis)
    n_basis_fn, n_orbitals = cmo.shape

    window = _select_orbital_window(
        space, cmo, overlap, basis, n_orbitals,
        n_occ, orbital_range, core_threshold)
    if window is None:      # occupied_valence with an all-core occupied space
        return cmo[:, :0].copy(), np.empty(0, dtype=float)
    lo, hi = window

    if not (0 <= lo < hi <= n_orbitals):
        raise ValueError(
            f"Orbital selection [{lo}, {hi}) is out of bounds "
            f"for {n_orbitals} orbitals."
        )
    n_sel = hi - lo

    # reduce_by_atom(values) sums an AO-indexed array over each atom's
    # basis functions via np.add.reduceat -- see _prepare_atom_grouping().
    reduce_by_atom = _prepare_atom_grouping(basis, n_basis_fn)

    # ---------------------------------------------------------------
    # Initialize the selected orbital space.
    # ---------------------------------------------------------------
    c = cmo[:, lo:hi].copy()
    sc = overlap @ c

    atomic_populations = reduce_by_atom(c * sc)
    prev_objective = _pipek_mezey_objective(atomic_populations)
    current_objective = prev_objective
    objective_change = 0.0

    rng = np.random.default_rng(seed)
    work = _JacobiWorkspace(n_basis_fn, c, sc)

    total_rotations = 0
    converged = False
    final_sweep = 0

    # ---------------------------------------------------------------
    # Sequential Jacobi sweeps.
    # ---------------------------------------------------------------
    for sweep in range(1, _PM_MAX_SWEEPS + 1):
        final_sweep = sweep
        rotations_this_sweep = _jacobi_sweep(
            c, sc, atomic_populations, n_sel, rng, work, reduce_by_atom)

        # Recompute the atomic populations from C and S C after each complete
        # sweep. This removes accumulated floating-point drift from the
        # incremental population updates.
        atomic_populations = reduce_by_atom(c * sc)
        current_objective = _pipek_mezey_objective(atomic_populations)
        objective_change = current_objective - prev_objective

        convergence_threshold = (
            _PM_OBJECTIVE_ATOL
            + _PM_OBJECTIVE_RTOL * max(abs(prev_objective),
                                       abs(current_objective))
        )
        if abs(objective_change) <= convergence_threshold:
            converged = True
            break

        # An entire sweep without a rotation is also converged.
        if rotations_this_sweep == 0:
            converged = True
            break

        prev_objective = current_objective

    # ---------------------------------------------------------------
    # Sort localized orbitals by Fock expectation value.
    # ---------------------------------------------------------------
    unsorted_energy = _fock_diagonal(c, fock)
    sort_energy_indices = np.argsort(unsorted_energy)

    sorted_c = c[:, sort_energy_indices]
    loc_energy = unsorted_energy[sort_energy_indices]

    _report_localization(
        converged, final_sweep, current_objective, objective_change,
        total_rotations, time.perf_counter() - start_time, loc_energy)

    return sorted_c, loc_energy


def localize_orbitals_cpp(
    cmo,
    overlap,
    fock,
    basis,
    space="occupied",
    n_occ=None,
    orbital_range=None,
    seed=0,
    core_threshold=0.98,
):
    """Thin wrapper around the native localization implementation.

    This is intended for benchmarking the speed of a C++ implementation
    against the pure-Python/NumPy version in :func:`localize_orbitals`.

    space='occupied_valence' is resolved in Python via find_valence_start()
    -- same as localize_orbitals() -- since the native extension itself
    only understands 'occupied'/'virtual'/'range'; it's handed the
    resolved valence block as an equivalent 'range' call, and only that
    block is returned (the core orbitals are left out of this call
    entirely, same as localize_orbitals()).
    """
    if _localize_orbitals_cpp is None:
        raise ImportError(
            "The native localization extension is not available. Build it first "
            "with python build_native_extensions.py"
        )

    if n_occ is None:
        n_occ = 0

    cmo_arr = np.asarray(cmo, dtype=float)
    overlap_arr = np.asarray(overlap, dtype=float)
    fock_arr = np.asarray(fock, dtype=float)
    n_orbitals = cmo_arr.shape[1]

    native_space = space
    native_orbital_range = orbital_range

    if space == "occupied_valence":
        valence_start = find_valence_start(
            cmo_arr, overlap_arr, basis, n_occ, core_threshold=core_threshold
        )

        if valence_start >= n_occ:
            print(
                f"All {n_occ} occupied orbitals meet the core threshold "
                f"({core_threshold:.0%} single-atom population) -- "
                "no valence orbitals to localize."
            )
            return cmo_arr[:, :0].copy(), np.empty(0, dtype=float)

        native_space = "range"
        native_orbital_range = (valence_start + 1, n_occ)

    if native_orbital_range is None:
        native_orbital_range = ()
    else:
        native_orbital_range = tuple(native_orbital_range)

    if native_space == "occupied":
        lo, hi = 0, n_occ
    elif native_space == "virtual":
        lo, hi = n_occ, n_orbitals
    elif native_space == "range":
        first, last = native_orbital_range
        lo, hi = first - 1, last
    else:
        raise ValueError(
            f"Unknown orbital space: {space!r}; expected 'occupied', "
            "'virtual', 'range', or 'occupied_valence'."
        )

    n_sel = hi - lo
    rng = np.random.default_rng(seed)
    sweep_orders = np.empty((2000, n_sel), dtype=np.intp)
    for sweep in range(2000):
        sweep_orders[sweep] = rng.permutation(n_sel)

    return _localize_orbitals_cpp(
        cmo_arr,
        overlap_arr,
        fock_arr,
        basis,
        space=native_space,
        n_occ=n_occ,
        orbital_range=native_orbital_range,
        seed=seed,
        sweep_orders=sweep_orders,
    )


def _localize_orbitals_with_fallback(
    cmo, overlap, fock, basis, space, n_occ, orbital_range, seed, core_threshold=0.98,
):
    """
    Prefer the native (C++) localization implementation; fall back to the
    pure-Python localize_orbitals if it's unavailable or fails.

    _localize_orbitals_cpp being non-None only means the extension
    *imported* successfully for this Python's ABI -- it doesn't guarantee
    a given call succeeds (natiTve crashes/errors, unexpected input shapes,
    etc. can still only surface when the compiled function actually runs).
    So this is a runtime check around the call itself, not just the
    import-time availability check localize_orbitals_cpp() already does.
    """
    if _localize_orbitals_cpp is not None:
        try:
            return localize_orbitals_cpp(
                cmo, overlap, fock, basis,
                space=space, n_occ=n_occ, orbital_range=orbital_range, seed=seed,
                core_threshold=core_threshold,
            )
        except Exception as exc:
            warnings.warn(
                f"Native localization failed at runtime ({exc!r}); "
                "falling back to the pure-Python implementation.",
                RuntimeWarning,
                stacklevel=2,
            )

    return localize_orbitals(
        cmo, overlap, fock, basis,
        space=space, n_occ=n_occ, orbital_range=orbital_range, seed=seed,
        core_threshold=core_threshold,
    )


def _load_source_basis(path, source_type):
    """
    Load the basis for a localization source, plus the NBO key-file path when
    the format needs one.

    Returns (key_path, final_basis, coordinates_ang, atom_info).
    """
    if source_type == "nbo":
        key_path = os.path.splitext(path)[0] + ".40"
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"NBO orbital localization requires a sibling .40 key file "
                f"(the canonical AO-basis MOs) -- expected {key_path}."
            )
        import nbo_read as _nr
        basis, coordinates_ang, atom_info = _nr.load_basis_headless(path)
        return key_path, basis, coordinates_ang, atom_info

    if source_type == "fchk":
        import fchk_read as _fr
        basis, coordinates_ang, atom_info = _fr.load_basis_from_fchk(path)
        return None, basis, coordinates_ang, atom_info

    if source_type == "molden":
        import read_molden as _mr
        basis, coordinates_ang, atom_info = _mr.load_basis_from_molden(path)
        return None, basis, coordinates_ang, atom_info

    raise ValueError(f"Unrecognized source file: {path}")


def _localized_occupations(space, n_sel, is_open_shell):
    """
    Per-orbital occupations for a localized set, or None where the selection
    leaves them ill-defined.
    """
    if space in ("occupied", "occupied_valence"):
        # Closed-shell orbitals hold 2 electrons each; open-shell spin-
        # orbitals (each spin solved independently) hold exactly 1.
        return np.full(n_sel, 1.0 if is_open_shell else 2.0)
    if space == "virtual":
        return np.full(n_sel, 0.0)
    # A 'range' selection may straddle the occ/virt boundary, where
    # per-orbital occupation isn't well-defined here.
    return None


def _compute_source_cubes(source_type, path, final_basis, coordinates_ang,
                          atom_info, orbital_indices, cmos_rows, spin,
                          grid_quality, ext_dist, bohr_const):
    """Dispatch cube generation to the reader that owns this source format."""
    if source_type == "nbo":
        import nbo_read as _nr
        return _nr.compute_cube_data(
            final_basis, coordinates_ang, atom_info,
            orbital_indices, path, spin,
            grid_quality, ext_dist, bohr_const,
            precomputed_cmos=cmos_rows,
        )
    if source_type == "fchk":
        import fchk_read as _fr
        return _fr.compute_cube_data_fchk(
            path, orbital_indices, spin, grid_quality, ext_dist, bohr_const,
            precomputed_cmos=cmos_rows,
            precomputed_basis=(final_basis, coordinates_ang, atom_info),
        )
    import read_molden as _mr
    return _mr.compute_cube_data_molden(
        path, orbital_indices, spin, grid_quality, ext_dist, bohr_const,
        precomputed_cmos=cmos_rows,
        precomputed_basis=(final_basis, coordinates_ang, atom_info),
    )


def compute_localized_cube_data(
    path,
    spin="alpha",
    space="occupied",
    n_occ=None,
    orbital_range=None,
    seed=0,
    core_threshold=0.98,
    grid_quality=75,
    ext_dist=4.0,
    bohr_const=0.529177249,
):
    """
    Localize a subspace of MOs and compute cube grids for them, ready for
    a viewer to render -- the single entry point tying together
    get_localization_inputs / get_fock_matrix / localize_orbitals and the
    per-format compute_cube_data* functions.

    Localization prefers the native (C++) implementation and transparently
    falls back to the pure-Python localize_orbitals if the extension isn't
    built for this Python's ABI, or if the native call itself fails at
    runtime (see _localize_orbitals_with_fallback) -- a RuntimeWarning is
    raised when that fallback happens.

    NBO sources always localize the sibling `.40` key file (the canonical
    AO-basis MOs), regardless of whichever key file is used elsewhere for
    picking un-localized orbitals; `path` must be the `.47` basis file
    (same requirement get_fock_matrix already has for the Fock matrix).
    fchk/molden sources localize whatever CMOs are in the source file
    itself.

    Parameters
    ----------
    path, spin, space, n_occ, orbital_range, seed, core_threshold : see
        localize_orbitals / get_localization_inputs. space='occupied_valence'
        auto-splits the occupied space (per core_threshold) and produces
        cubes for the outer/valence orbitals only -- inner/core orbitals
        are left untouched and are not part of the returned cubes at all.
    grid_quality, ext_dist, bohr_const : see compute_cube_data /
        compute_cube_data_fchk / compute_cube_data_molden.

    Returns
    -------
    dict with keys:
        cubes           : list of per-orbital cube-grid dicts (same shape
                           compute_cube_data* returns), labeled
                           "<base>_LOC_<space>-<n>" (no '.' before LOC --
                           os.path.splitext() is used elsewhere to recover
                           <base>, and a '.' there would be misparsed as a
                           file extension).
        final_basis     : basis-function list.
        atom_info       : list of (Z, x_ang, y_ang, z_ang).
        localized_cmo   : (nbas, n_selected) ndarray, AO x MO.
        energies        : (n_selected,) ndarray, Fock expectation values.
        occupations     : (n_selected,) ndarray, or None for space='range'
                           (ambiguous when the range straddles n_occ).
        overlap, fock   : AO-basis matrices used for the localization.
        orbital_indices : list(range(1, n_selected + 1)).
        n_occ           : the n_occ actually used (resolved if it was None).
        space           : echoed back, for convenience.
    """
    source_type = _recognize_source_type(path)
    key_path, final_basis, coordinates_ang, atom_info = _load_source_basis(
        path, source_type)

    cmo, overlap, final_basis = get_localization_inputs(path, key_path=key_path, spin=spin)
    fock = get_fock_matrix(path, key_path=key_path, spin=spin, cmo=cmo, overlap=overlap)

    ordered_basis, ordered_cmo, ordered_overlap, ordered_fock, inverse_perm = (
        _reorder_for_contiguous_centers(final_basis, cmo, overlap, fock)
    )
    basis_center = get_center_ranges(ordered_basis)

    occ_source_type, occ_alpha, occ_beta = _get_occupation_arrays(path, key_path=key_path)
    is_open_shell = occ_beta is not None

    if space in ("occupied", "virtual", "occupied_valence") and n_occ is None:
        n_occ = _num_occupied_from_arrays(
            occ_source_type, occ_alpha, occ_beta, spin=spin, path=path
        )

    localized_cmo, loc_energy = _localize_orbitals_with_fallback(
        ordered_cmo, ordered_overlap, ordered_fock, basis_center,
        space, n_occ, orbital_range, seed, core_threshold=core_threshold,
    )
    if inverse_perm is not None:
        localized_cmo = localized_cmo[inverse_perm, :]

    n_sel = localized_cmo.shape[1]
    orbital_indices = list(range(1, n_sel + 1))
    cmos_rows = list(localized_cmo.T)

    occupations = _localized_occupations(space, n_sel, is_open_shell)

    cubes = _compute_source_cubes(
        source_type, path, final_basis, coordinates_ang, atom_info,
        orbital_indices, cmos_rows, spin, grid_quality, ext_dist, bohr_const)

    base = os.path.splitext(os.path.basename(path))[0]
    for i, cube in enumerate(cubes, start=1):
        cube["label"] = f"{base}_LOC_{space}-{i}"

    return {
        "cubes": cubes,
        "final_basis": final_basis,
        "atom_info": atom_info,
        "localized_cmo": localized_cmo,
        "energies": loc_energy,
        "occupations": occupations,
        "overlap": overlap,
        "fock": fock,
        "orbital_indices": orbital_indices,
        "n_occ": n_occ,
        "space": space,
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanity-check get_localization_inputs() against an "
                    "NBO (.47/.31), Gaussian (.fchk/.fck), or Molden (.molden) source."
    )
    parser.add_argument("path", help="Basis/orbital source file")
    parser.add_argument(
        "--key", dest="key_path", default=None,
        help="NBO key file (.31/.32/.33/.40/...) holding the CMO coefficients "
             "— required when path is a .47/.31 NBO basis file, ignored otherwise.",
    )
    parser.add_argument("--spin", default="alpha", choices=["alpha", "beta"])
    parser.add_argument(
        "--space", default="occupied",
        choices=["occupied", "virtual", "range", "occupied_valence"],
        help="Orbital subspace to localize (default: occupied). 'occupied_valence' "
             "auto-splits the occupied space into core/valence via --core-threshold "
             "and localizes only the valence block.",
    )
    parser.add_argument(
        "--range", dest="orbital_range", default=None,
        help="Inclusive 1-based 'first-last' MO numbers, required when --space=range "
             "(e.g. '49-60').",
    )
    parser.add_argument(
        "--core-threshold", dest="core_threshold", type=float, default=0.98,
        help="Minimum single-atom population fraction (0-1) for an occupied orbital "
             "to be classified core; only used when --space=occupied_valence "
             "(default: 0.98).",
    )
    parser.add_argument(
        "--use-native", action="store_true",
        help="Use the native C++ localization implementation instead of the pure-Python version.",
    )
    parser.add_argument(
        "--compare-native", action="store_true",
        help="Run both the Python and native implementations and print their results.",
    )
    args = parser.parse_args()

    cmo, S, basis = get_localization_inputs(args.path, key_path=args.key_path, spin=args.spin)

    print(f"\n{'=' * 70}")
    print(f"Source type     : {_recognize_source_type(args.path)}")
    print(f"Source file     : {args.path}")
    if args.key_path:
        print(f"Key file        : {args.key_path}")
    print(f"Spin            : {args.spin}")
    print("=" * 70)
    print(f"Basis functions : {len(basis)}")
    print(f"CMO matrix      : {cmo.shape}  (AO x MO)")
    print(f"Overlap matrix  : {S.shape}")

    # Canonical MOs should be orthonormal in the AO metric: C^T S C == I.
    ortho     = cmo.T @ S @ cmo
    diag      = np.diag(ortho)
    off_diag  = ortho - np.diag(diag)
    print(f"\nC^T S C diagonal (first 10)   : {diag[:10].round(6)}")
    print(f"Max |diagonal - 1|            : {np.max(np.abs(diag - 1)):.3e}")
    print(f"Max |off-diagonal|            : {np.max(np.abs(off_diag)):.3e}")

    F = get_fock_matrix(args.path, key_path=args.key_path, spin=args.spin, cmo=cmo, overlap=S)
    print(f"\nFock matrix     : {F.shape}")
    # Back-transform to the MO basis: C^T F C should be ~diagonal, with the
    # diagonal equal to the canonical orbital energies.
    fock_mo  = cmo.T @ F @ cmo
    fock_off = fock_mo - np.diag(np.diag(fock_mo))
    print(f"C^T F C diagonal (first 10)   : {np.diag(fock_mo)[:10].round(6)}")
    print(f"Max |off-diagonal| of C^T F C : {np.max(np.abs(fock_off)):.3e}")

    ordered_basis, ordered_cmo, ordered_overlap, ordered_fock, inverse_perm = (
        _reorder_for_contiguous_centers(basis, cmo, S, F)
    )
    basis_center = get_center_ranges(ordered_basis)
    n_occ = get_num_occupied_orbitals(args.path, key_path=args.key_path, spin=args.spin)
    print(f"\nOccupied orbitals for {args.spin} : {n_occ}")

    orbital_range = None
    if args.orbital_range:
        first, last = (int(x) for x in args.orbital_range.split("-"))
        orbital_range = (first, last)

    primary_label = "Native (C++)" if args.use_native else "Python"
    primary_fn = localize_orbitals_cpp if args.use_native else localize_orbitals

    start = time.perf_counter()
    localized_cmo, loc_energy = primary_fn(
        ordered_cmo, ordered_overlap, ordered_fock, basis_center,
        space=args.space, n_occ=n_occ, orbital_range=orbital_range,
        core_threshold=args.core_threshold,
    )
    if inverse_perm is not None:
        localized_cmo = localized_cmo[inverse_perm, :]
    primary_elapsed = time.perf_counter() - start

    print(f"\n{primary_label} implementation ({primary_elapsed:.4f}s):")
    print(f"  localized_cmo shape : {localized_cmo.shape}")
    print(f"  loc_energy (sorted) : {np.sort(loc_energy).round(6)}")

    if args.compare_native:
        # Both implementations use the same objective-based convergence
        # criterion and sort their output by energy, and localize_orbitals_
        # cpp feeds the native side the exact sweep-order sequence
        # localize_orbitals itself would use for a given seed -- so results
        # should agree closely for well-conditioned systems. They can still
        # diverge when orbitals are confined to a single, genuinely
        # degenerate atom-localized subspace: the Pipek-Mezey objective has
        # a flat direction there, and tiny floating-point differences
        # between the two implementations can tip which point along it
        # either one lands on. Comparing sorted energy sets rather than
        # assuming index alignment is still the right approach for that
        # reason, not because of any remaining convergence-depth mismatch.
        other_label = "Python" if args.use_native else "Native (C++)"
        other_fn = localize_orbitals if args.use_native else localize_orbitals_cpp

        start = time.perf_counter()
        try:
            other_cmo, other_energy = other_fn(
                ordered_cmo, ordered_overlap, ordered_fock, basis_center,
                space=args.space, n_occ=n_occ, orbital_range=orbital_range,
                core_threshold=args.core_threshold,
            )
            if inverse_perm is not None:
                other_cmo = other_cmo[inverse_perm, :]
        except Exception as exc:
            print(f"\n{other_label} implementation unavailable: {exc}")
        else:
            other_elapsed = time.perf_counter() - start
            print(f"\n{other_label} implementation ({other_elapsed:.4f}s):")
            print(f"  localized_cmo shape : {other_cmo.shape}")
            print(f"  loc_energy (sorted) : {np.sort(other_energy).round(6)}")

            if args.use_native:
                py_elapsed, py_cmo, py_energy = other_elapsed, other_cmo, other_energy
                cpp_elapsed, cpp_cmo, cpp_energy = primary_elapsed, localized_cmo, loc_energy
            else:
                py_elapsed, py_cmo, py_energy = primary_elapsed, localized_cmo, loc_energy
                cpp_elapsed, cpp_cmo, cpp_energy = other_elapsed, other_cmo, other_energy

            print(f"\n{'=' * 70}\nComparison\n{'=' * 70}")
            print(f"Python runtime : {py_elapsed:.4f}s")
            print(f"Native runtime : {cpp_elapsed:.4f}s")
            if cpp_elapsed > 0:
                print(f"Speedup (Python / Native) : {py_elapsed / cpp_elapsed:.2f}x")

            py_sorted = np.sort(py_energy)
            cpp_sorted = np.sort(cpp_energy)
            if py_sorted.shape == cpp_sorted.shape:
                print(f"Max |energy diff| (sorted sets)   : {np.max(np.abs(py_sorted - cpp_sorted)):.3e}")
            else:
                print("Energy arrays have different shapes -- cannot compare directly.")

            py_ortho_err = np.max(np.abs(py_cmo.T @ S @ py_cmo - np.eye(py_cmo.shape[1])))
            cpp_ortho_err = np.max(np.abs(cpp_cmo.T @ S @ cpp_cmo - np.eye(cpp_cmo.shape[1])))
            print(f"Python orthonormality max error   : {py_ortho_err:.3e}")
            print(f"Native orthonormality max error   : {cpp_ortho_err:.3e}")

