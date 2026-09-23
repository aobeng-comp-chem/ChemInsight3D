"""
Compatibility layer for the Gaussian overlap-matrix backend.

This project originally shipped only a compiled extension built for CPython
3.10 (`overlap_matrix.cpython-310-...so`). When the app is run with a newer
Python, importing `overlap_matrix` fails before the GUI can start.

This module fixes that by:
1. Loading a native backend when a compatible one exists for the current ABI.
2. Falling back to a pure-Python implementation with the same public API.

The fallback is slower than the pybind11/OpenMP version, but it keeps the
application functional across Python versions without requiring a rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import importlib.util
import math
from pathlib import Path
import warnings

import numpy as np


def _load_native_backend():
    """Load a compatible compiled backend if one exists for this Python ABI."""
    module_dir = Path(__file__).resolve().parent
    module_stem = Path(__file__).stem

    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = module_dir / f"{module_stem}{suffix}"
        if not candidate.exists():
            continue

        spec = importlib.util.spec_from_file_location("_overlap_matrix_native", candidate)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return None


_native = None
_native_error = None
try:
    _native = _load_native_backend()
except Exception as exc:  # pragma: no cover - best effort warning path
    _native_error = exc
    _native = None


if _native is not None:
    get_overlap_matrix = _native.get_overlap_matrix
    get_overlap_diagonal_only = _native.get_overlap_diagonal_only
else:
    if _native_error is not None:
        warnings.warn(
            f"Compatible native overlap_matrix backend could not be loaded; "
            f"using slower pure-Python fallback instead. ({_native_error})",
            RuntimeWarning,
            stacklevel=2,
        )

    @dataclass(frozen=True)
    class _OrbitalTerm:
        coeff: float
        l: int
        m: int
        n: int


    @dataclass(frozen=True)
    class _BasisFunction:
        exps: tuple[float, ...]
        coeffs: tuple[float, ...]
        xcenter: float
        ycenter: float
        zcenter: float
        terms: tuple[_OrbitalTerm, ...]


    def double_factorial(n: int) -> int:
        if n <= 0:
            return 1
        result = 1
        for value in range(n, 0, -2):
            result *= value
        return result


    def binomial(a: int, b: int) -> float:
        if b < 0 or b > a:
            return 0.0
        return float(math.comb(a, b))


    def binomial_prefactor(s: int, ia: int, ib: int, xpa: float, xpb: float) -> float:
        total = 0.0
        for t in range(s + 1):
            if (s - ia) <= t <= ib:
                total += (
                    binomial(ia, s - t)
                    * binomial(ib, t)
                    * (xpa ** (ia - s + t))
                    * (xpb ** (ib - t))
                )
        return total


    def gaussian_norm(alpha: float, l: int, m: int, n: int) -> float:
        lmn = l + m + n
        prefactor = (2 ** (2 * lmn + 1.5)) * (alpha ** (lmn + 1.5)) / (math.pi ** 1.5)
        denom = (
            double_factorial(2 * l - 1)
            * double_factorial(2 * m - 1)
            * double_factorial(2 * n - 1)
        )
        return math.sqrt(prefactor / denom)


    def rsqr(
        x1: float, y1: float, z1: float,
        x2: float, y2: float, z2: float,
    ) -> float:
        return (
            (x1 - x2) * (x1 - x2)
            + (y1 - y2) * (y1 - y2)
            + (z1 - z2) * (z1 - z2)
        )


    def product_center_1d(alpha1: float, x1: float, alpha2: float, x2: float) -> float:
        return (alpha1 * x1 + alpha2 * x2) / (alpha1 + alpha2)


    def overlap_1d(l1: int, l2: int, pax: float, pbx: float, gamma: float) -> float:
        total = 0.0
        max_i = 1 + (l1 + l2) // 2
        for i in range(max_i):
            total += (
                binomial_prefactor(2 * i, l1, l2, pax, pbx)
                * double_factorial(2 * i - 1)
                / ((2 * gamma) ** i)
            )
        return total


    def overlap_int(
        alpha1: float, l1: int, m1: int, n1: int, xa: float, ya: float, za: float,
        alpha2: float, l2: int, m2: int, n2: int, xb: float, yb: float, zb: float,
    ) -> float:
        rab2 = rsqr(xa, ya, za, xb, yb, zb)
        gamma = alpha1 + alpha2

        xp = product_center_1d(alpha1, xa, alpha2, xb)
        yp = product_center_1d(alpha1, ya, alpha2, yb)
        zp = product_center_1d(alpha1, za, alpha2, zb)

        prefactor = ((math.pi / gamma) ** 1.5) * math.exp(-alpha1 * alpha2 * rab2 / gamma)

        wx = overlap_1d(l1, l2, xp - xa, xp - xb, gamma)
        wy = overlap_1d(m1, m2, yp - ya, yp - yb, gamma)
        wz = overlap_1d(n1, n2, zp - za, zp - zb, gamma)

        return prefactor * wx * wy * wz


    def _prepare_basis(primitives, dict_keys) -> list[_BasisFunction]:
        basis = []
        for primitive in primitives:
            orb_val = str(primitive["orb_val"])
            if orb_val not in dict_keys:
                raise KeyError(f"Unknown orbital key: {orb_val}")

            entry = dict_keys[orb_val]
            terms = tuple(
                _OrbitalTerm(float(coeff), int(l), int(m), int(n))
                for coeff, (l, m, n) in zip(entry["coe"], entry["var_cnts"])
            )

            basis.append(
                _BasisFunction(
                    exps=tuple(float(x) for x in primitive["exps"]),
                    coeffs=tuple(float(x) for x in primitive["coeffs"]),
                    xcenter=float(primitive["xcenter"]),
                    ycenter=float(primitive["ycenter"]),
                    zcenter=float(primitive["zcenter"]),
                    terms=terms,
                )
            )
        return basis


    def _basis_pair_overlap(a: _BasisFunction, b: _BasisFunction, normalize_primitives: bool) -> float:
        total = 0.0

        if normalize_primitives:
            for alpha1, coef1 in zip(a.exps, a.coeffs):
                for alpha2, coef2 in zip(b.exps, b.coeffs):
                    for term_a in a.terms:
                        norm1 = gaussian_norm(alpha1, term_a.l, term_a.m, term_a.n)
                        pref1 = term_a.coeff * coef1 * norm1
                        for term_b in b.terms:
                            norm2 = gaussian_norm(alpha2, term_b.l, term_b.m, term_b.n)
                            total += (
                                pref1
                                * term_b.coeff
                                * coef2
                                * norm2
                                * overlap_int(
                                    alpha1, term_a.l, term_a.m, term_a.n, a.xcenter, a.ycenter, a.zcenter,
                                    alpha2, term_b.l, term_b.m, term_b.n, b.xcenter, b.ycenter, b.zcenter,
                                )
                            )
        else:
            for alpha1, coef1 in zip(a.exps, a.coeffs):
                for alpha2, coef2 in zip(b.exps, b.coeffs):
                    primitive_scale = coef1 * coef2
                    for term_a in a.terms:
                        pref1 = term_a.coeff * primitive_scale
                        for term_b in b.terms:
                            total += (
                                pref1
                                * term_b.coeff
                                * overlap_int(
                                    alpha1, term_a.l, term_a.m, term_a.n, a.xcenter, a.ycenter, a.zcenter,
                                    alpha2, term_b.l, term_b.m, term_b.n, b.xcenter, b.ycenter, b.zcenter,
                                )
                            )

        return total


    def get_overlap_matrix(primitives, dict_keys, normalize_primitives=False, diagonal_only=False):
        """
        Pure-Python fallback matching the compiled extension API.

        Parameters mirror the pybind11 backend:
        - primitives: list of basis-function dicts
        - dict_keys: orbital term lookup from bas_dict
        - normalize_primitives: apply Gaussian primitive normalization
        - diagonal_only: only populate the diagonal of the returned matrix
        """
        basis = _prepare_basis(primitives, dict_keys)
        nbf = len(basis)
        overlap = np.zeros((nbf, nbf), dtype=float)

        if diagonal_only:
            for i, bf in enumerate(basis):
                overlap[i, i] = _basis_pair_overlap(bf, bf, bool(normalize_primitives))
            return overlap

        for j in range(nbf):
            for i in range(j + 1):
                value = _basis_pair_overlap(basis[i], basis[j], bool(normalize_primitives))
                overlap[j, i] = value
                if i != j:
                    overlap[i, j] = value

        return overlap


    def get_overlap_diagonal_only(primitives, dict_keys, normalize_primitives=False):
        """Return only the diagonal overlap elements as a 1D NumPy array."""
        basis = _prepare_basis(primitives, dict_keys)
        diag = np.zeros(len(basis), dtype=float)
        for i, bf in enumerate(basis):
            diag[i] = _basis_pair_overlap(bf, bf, bool(normalize_primitives))
        return diag


__all__ = ["get_overlap_matrix", "get_overlap_diagonal_only"]
