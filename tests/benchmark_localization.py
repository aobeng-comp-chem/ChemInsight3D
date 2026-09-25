import time
import numpy as np

from localization_io import localize_orbitals, localize_orbitals_cpp


def make_problem(n_basis=120, n_orbitals=60, n_atoms=8):
    rng = np.random.default_rng(7)
    cmo = rng.normal(size=(n_basis, n_orbitals))
    overlap = rng.normal(size=(n_basis, n_basis))
    overlap = overlap @ overlap.T
    overlap += np.eye(n_basis)
    fock = rng.normal(size=(n_basis, n_basis))
    fock = fock @ fock.T
    basis = []
    size = n_basis // n_atoms
    start = 0
    for atom in range(n_atoms):
        bfhi = min(start + size, n_basis) - 1
        if bfhi < start:
            bfhi = start
        basis.append({"bflo": start, "bfhi": bfhi})
        start = bfhi + 1
    if start < n_basis:
        basis[-1]["bfhi"] = n_basis - 1
    return cmo, overlap, fock, basis


if __name__ == "__main__":
    cmo, overlap, fock, basis = make_problem()

    start = time.perf_counter()
    localize_orbitals(cmo, overlap, fock, basis, space="occupied", n_occ=10, seed=1)
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    localize_orbitals_cpp(cmo, overlap, fock, basis, space="occupied", n_occ=10, seed=1)
    cpp_time = time.perf_counter() - start

    print(f"Python: {py_time:.6f}s")
    print(f"C++:    {cpp_time:.6f}s")
    print(f"Speedup: {py_time / cpp_time:.2f}x")
