import numpy as np

from localization_io import localize_orbitals, localize_orbitals_cpp


def _make_problem():
    rng = np.random.default_rng(0)
    cmo = rng.normal(size=(8, 4))
    overlap = np.eye(8)
    fock = np.diag(np.linspace(1.0, 8.0, 8))
    basis = [
        {"bflo": 0, "bfhi": 2},
        {"bflo": 3, "bfhi": 5},
        {"bflo": 6, "bfhi": 7},
    ]
    return cmo, overlap, fock, basis


def test_cpp_localization_smoke():
    cmo, overlap, fock, basis = _make_problem()

    out_c, out_e = localize_orbitals_cpp(
        cmo,
        overlap,
        fock,
        basis,
        space="occupied",
        n_occ=2,
        seed=1,
    )

    assert out_c.shape == (8, 2)
    assert out_e.shape == (2,)


def test_cpp_localization_accepts_default_orbital_range():
    cmo, overlap, fock, basis = _make_problem()

    out_c, out_e = localize_orbitals_cpp(
        cmo,
        overlap,
        fock,
        basis,
        space="occupied",
        n_occ=2,
        orbital_range=None,
        seed=1,
    )

    assert out_c.shape == (8, 2)
    assert out_e.shape == (2,)


def test_python_and_cpp_return_same_shapes():
    cmo, overlap, fock, basis = _make_problem()

    py_c, py_e = localize_orbitals(
        cmo,
        overlap,
        fock,
        basis,
        space="occupied",
        n_occ=2,
        seed=1,
    )
    cpp_c, cpp_e = localize_orbitals_cpp(
        cmo,
        overlap,
        fock,
        basis,
        space="occupied",
        n_occ=2,
        seed=1,
    )

    assert py_c.shape == cpp_c.shape == (8, 2)
    assert py_e.shape == cpp_e.shape == (2,)


def test_python_and_cpp_energies_are_close():
    cmo, overlap, fock, basis = _make_problem()

    _, py_e = localize_orbitals(
        cmo,
        overlap,
        fock,
        basis,
        space="occupied",
        n_occ=2,
        seed=1,
    )
    _, cpp_e = localize_orbitals_cpp(
        cmo,
        overlap,
        fock,
        basis,
        space="occupied",
        n_occ=2,
        seed=1,
    )

    np.testing.assert_allclose(py_e, cpp_e, rtol=1e-5, atol=1e-5)
