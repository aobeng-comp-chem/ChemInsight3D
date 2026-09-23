#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

struct AtomRange {
    int bflo;
    int bfhi;
};

inline py::array_t<double> make_2d(
    const std::vector<double>& data,
    std::size_t rows,
    std::size_t cols) {
    py::array_t<double> out({rows, cols});
    std::copy(data.begin(), data.end(), out.mutable_data());
    return out;
}

inline py::array_t<double> make_1d(const std::vector<double>& data) {
    py::array_t<double> out({static_cast<py::ssize_t>(data.size())});
    std::copy(data.begin(), data.end(), out.mutable_data());
    return out;
}

std::pair<int, int> select_orbital_range(
    int n_orbitals,
    const std::string& space,
    int n_occ,
    const py::tuple& orbital_range) {
    if (space == "occupied") {
        if (n_occ <= 0) {
            throw std::invalid_argument("n_occ must be positive for occupied space");
        }
        return {0, n_occ};
    }
    if (space == "virtual") {
        if (n_occ <= 0) {
            throw std::invalid_argument("n_occ must be positive for virtual space");
        }
        return {n_occ, n_orbitals};
    }
    if (space == "range") {
        if (orbital_range.size() != 2) {
            throw std::invalid_argument("orbital_range must be a 2-tuple");
        }
        const int first = orbital_range[0].cast<int>();
        const int last = orbital_range[1].cast<int>();
        if (!(1 <= first && first <= last && last <= n_orbitals)) {
            throw std::invalid_argument("orbital_range is out of bounds");
        }
        return {first - 1, last};
    }
    throw std::invalid_argument("Unknown orbital space");
}

void validate_inputs(
    const DoubleArray& cmo,
    const DoubleArray& overlap,
    const DoubleArray& fock) {
    if (cmo.ndim() != 2) throw std::invalid_argument("cmo must be a 2D array");
    if (overlap.ndim() != 2) throw std::invalid_argument("overlap must be a 2D array");
    if (fock.ndim() != 2) throw std::invalid_argument("fock must be a 2D array");
    if (overlap.shape(0) != cmo.shape(0) ||
        overlap.shape(1) != cmo.shape(0)) {
        throw std::invalid_argument(
            "overlap must be square with size equal to number of basis functions");
    }
    if (fock.shape(0) != cmo.shape(0) ||
        fock.shape(1) != cmo.shape(0)) {
        throw std::invalid_argument(
            "fock must be square with size equal to number of basis functions");
    }
}

// Population columns are contiguous for the two orbitals in a Jacobi pair.
// Atom-major traversal preserves Python's objective accumulation order.
inline double pipek_mezey_objective(
    const std::vector<double>& populations,
    int natom,
    int n_sel) {
    double total = 0.0;
    for (int atom = 0; atom < natom; ++atom) {
        for (int orbital = 0; orbital < n_sel; ++orbital) {
            const double value =
                populations[static_cast<std::size_t>(orbital) * natom + atom];
            total += value * value;
        }
    }
    return total;
}

inline double sign_of(double value) {
    if (value > 0.0) return 1.0;
    if (value < 0.0) return -1.0;
    return 0.0;
}

}  // namespace

py::tuple localize_orbitals_cpp(
    DoubleArray cmo_arr,
    DoubleArray overlap_arr,
    DoubleArray fock_arr,
    py::list basis_list,
    const std::string& space = "occupied",
    int n_occ = 0,
    py::tuple orbital_range = py::tuple(),
    int seed = 0,
    IntArray sweep_orders = IntArray()) {
    const auto start_time = std::chrono::steady_clock::now();
    validate_inputs(cmo_arr, overlap_arr, fock_arr);

    const int n_basis_fn = static_cast<int>(cmo_arr.shape(0));
    const int n_orbitals = static_cast<int>(cmo_arr.shape(1));
    const double* const cmo_data = cmo_arr.data();
    const double* const overlap_data = overlap_arr.data();
    const double* const fock_data = fock_arr.data();

    if (basis_list.size() == 0) {
        throw std::invalid_argument("basis must contain at least one atom");
    }

    std::vector<AtomRange> basis;
    basis.reserve(basis_list.size());
    for (py::handle item : basis_list) {
        const py::dict atom = item.cast<py::dict>();
        const AtomRange range{
            atom["bflo"].cast<int>(),
            atom["bfhi"].cast<int>()};
        if (!(0 <= range.bflo && range.bflo <= range.bfhi &&
              range.bfhi < n_basis_fn)) {
            throw std::invalid_argument("Invalid basis-function range");
        }
        basis.push_back(range);
    }

    const auto [lo, hi] =
        select_orbital_range(n_orbitals, space, n_occ, orbital_range);
    if (!(0 <= lo && lo < hi && hi <= n_orbitals)) {
        throw std::invalid_argument("Orbital selection is out of bounds");
    }

    const int n_sel = hi - lo;
    const int natom = static_cast<int>(basis.size());
    const std::size_t orbital_size = static_cast<std::size_t>(n_basis_fn);
    const bool have_orders =
        sweep_orders.ndim() == 2 &&
        sweep_orders.shape(0) > 0 &&
        sweep_orders.shape(1) == n_sel;
    const int order_count =
        have_orders ? static_cast<int>(sweep_orders.shape(0)) : 0;
    const int* const order_data =
        have_orders ? sweep_orders.data() : nullptr;

    // Orbital-major storage makes both columns in a Jacobi update contiguous.
    std::vector<double> c(orbital_size * n_sel);
    std::vector<double> sc(orbital_size * n_sel);
    for (int orbital = 0; orbital < n_sel; ++orbital) {
        double* const column =
            c.data() + static_cast<std::size_t>(orbital) * orbital_size;
        for (int mu = 0; mu < n_basis_fn; ++mu) {
            column[mu] =
                cmo_data[static_cast<std::size_t>(mu) * n_orbitals +
                         lo + orbital];
        }
    }

    // S*C columns are independent: thread columns and vectorize row dots.
    const long long initial_work =
        static_cast<long long>(n_basis_fn) * n_basis_fn * n_sel;
#pragma omp parallel for schedule(static) if(initial_work >= 1000000)
    for (int orbital = 0; orbital < n_sel; ++orbital) {
        const double* const column =
            c.data() + static_cast<std::size_t>(orbital) * orbital_size;
        double* const sc_column =
            sc.data() + static_cast<std::size_t>(orbital) * orbital_size;
        for (int mu = 0; mu < n_basis_fn; ++mu) {
            const double* const row =
                overlap_data + static_cast<std::size_t>(mu) * n_basis_fn;
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (int nu = 0; nu < n_basis_fn; ++nu) {
                sum += row[nu] * column[nu];
            }
            sc_column[mu] = sum;
        }
    }

    std::vector<double> populations(
        static_cast<std::size_t>(n_sel) * natom,
        0.0);
#pragma omp parallel for schedule(static) if(n_basis_fn * n_sel >= 20000)
    for (int orbital = 0; orbital < n_sel; ++orbital) {
        const double* const column =
            c.data() + static_cast<std::size_t>(orbital) * orbital_size;
        const double* const sc_column =
            sc.data() + static_cast<std::size_t>(orbital) * orbital_size;
        double* const q =
            populations.data() + static_cast<std::size_t>(orbital) * natom;
        for (int atom = 0; atom < natom; ++atom) {
            const AtomRange range = basis[atom];
            double sum = 0.0;
            for (int mu = range.bflo; mu <= range.bfhi; ++mu) {
                sum += column[mu] * sc_column[mu];
            }
            q[atom] = sum;
        }
    }

    constexpr double gamma_tol = 1.0e-10;
    constexpr double coupling_tol = 1.0e-14;
    constexpr double objective_atol = 1.0e-12;
    constexpr double objective_rtol = 1.0e-10;
    constexpr int max_sweeps = 2000;

    std::vector<int> order(n_sel);
    for (int orbital = 0; orbital < n_sel; ++orbital) {
        order[orbital] = orbital;
    }
    std::vector<double> qast(natom);

    double previous_objective =
        pipek_mezey_objective(populations, natom, n_sel);
    double current_objective = previous_objective;
    long long total_rotations = 0;
    int final_sweep = 0;
    bool converged = false;

    // Pairs stay sequential because each one changes the next pair's inputs.
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        final_sweep = sweep + 1;
        if (have_orders && sweep < order_count) {
            const int* const supplied =
                order_data + static_cast<std::size_t>(sweep) * n_sel;
            std::copy(supplied, supplied + n_sel, order.begin());
        } else {
            std::mt19937 rng(static_cast<unsigned>(seed + sweep));
            std::shuffle(order.begin(), order.end(), rng);
        }

        int sweep_rotations = 0;
        for (int s : order) {
            if (!(0 <= s && s < n_sel)) {
                throw std::invalid_argument(
                    "sweep_orders contains an invalid orbital index");
            }
            double* const c_s =
                c.data() + static_cast<std::size_t>(s) * orbital_size;
            double* const sc_s =
                sc.data() + static_cast<std::size_t>(s) * orbital_size;
            double* const q_s =
                populations.data() + static_cast<std::size_t>(s) * natom;

            for (int t = 0; t < n_sel; ++t) {
                if (t == s) continue;

                double* const c_t =
                    c.data() + static_cast<std::size_t>(t) * orbital_size;
                double* const sc_t =
                    sc.data() + static_cast<std::size_t>(t) * orbital_size;
                double* const q_t =
                    populations.data() + static_cast<std::size_t>(t) * natom;

                double ast = 0.0;
                double bst = 0.0;
                for (int atom = 0; atom < natom; ++atom) {
                    const AtomRange range = basis[atom];
                    double cross = 0.0;
                    for (int mu = range.bflo; mu <= range.bfhi; ++mu) {
                        cross +=
                            0.5 *
                            (c_t[mu] * sc_s[mu] +
                             c_s[mu] * sc_t[mu]);
                    }
                    qast[atom] = cross;
                    const double difference = q_s[atom] - q_t[atom];
                    ast +=
                        cross * cross -
                        0.25 * difference * difference;
                    bst += cross * difference;
                }

                const double denominator = std::hypot(ast, bst);
                if (denominator < coupling_tol) continue;

                const double gamma =
                    0.25 *
                    std::acos(std::clamp(
                        -ast / denominator, -1.0, 1.0)) *
                    sign_of(bst);
                if (std::abs(gamma) <= gamma_tol) continue;

                const double cosg = std::cos(gamma);
                const double sing = std::sin(gamma);
                const double cosg2 = cosg * cosg;
                const double sing2 = sing * sing;
                const double two_cos_sin = 2.0 * cosg * sing;

#pragma omp simd
                for (int mu = 0; mu < n_basis_fn; ++mu) {
                    const double old_c_s = c_s[mu];
                    const double old_c_t = c_t[mu];
                    const double old_sc_s = sc_s[mu];
                    const double old_sc_t = sc_t[mu];
                    c_s[mu] = cosg * old_c_s + sing * old_c_t;
                    c_t[mu] = cosg * old_c_t - sing * old_c_s;
                    sc_s[mu] = cosg * old_sc_s + sing * old_sc_t;
                    sc_t[mu] = cosg * old_sc_t - sing * old_sc_s;
                }

                for (int atom = 0; atom < natom; ++atom) {
                    const double old_q_s = q_s[atom];
                    const double old_q_t = q_t[atom];
                    q_s[atom] =
                        cosg2 * old_q_s +
                        sing2 * old_q_t +
                        two_cos_sin * qast[atom];
                    q_t[atom] =
                        sing2 * old_q_s +
                        cosg2 * old_q_t -
                        two_cos_sin * qast[atom];
                }
                ++sweep_rotations;
                ++total_rotations;
            }
        }

        // Independent columns can be recomputed in parallel deterministically.
#pragma omp parallel for schedule(static) if(n_basis_fn * n_sel >= 20000)
        for (int orbital = 0; orbital < n_sel; ++orbital) {
            const double* const column =
                c.data() +
                static_cast<std::size_t>(orbital) * orbital_size;
            const double* const sc_column =
                sc.data() +
                static_cast<std::size_t>(orbital) * orbital_size;
            double* const q =
                populations.data() +
                static_cast<std::size_t>(orbital) * natom;
            for (int atom = 0; atom < natom; ++atom) {
                const AtomRange range = basis[atom];
                double sum = 0.0;
                for (int mu = range.bflo; mu <= range.bfhi; ++mu) {
                    sum += column[mu] * sc_column[mu];
                }
                q[atom] = sum;
            }
        }

        current_objective =
            pipek_mezey_objective(populations, natom, n_sel);
        const double change =
            current_objective - previous_objective;
        const double threshold =
            objective_atol +
            objective_rtol *
                std::max(
                    std::abs(previous_objective),
                    std::abs(current_objective));
        if (std::abs(change) <= threshold ||
            sweep_rotations == 0) {
            converged = true;
            break;
        }
        previous_objective = current_objective;
    }

    // F*C columns are independent: thread columns and vectorize row dots.
    std::vector<double> energies(n_sel, 0.0);
    const long long energy_work =
        static_cast<long long>(n_basis_fn) * n_basis_fn * n_sel;
#pragma omp parallel for schedule(static) if(energy_work >= 1000000)
    for (int orbital = 0; orbital < n_sel; ++orbital) {
        const double* const column =
            c.data() + static_cast<std::size_t>(orbital) * orbital_size;
        double energy = 0.0;
        for (int mu = 0; mu < n_basis_fn; ++mu) {
            const double* const row =
                fock_data + static_cast<std::size_t>(mu) * n_basis_fn;
            double fc_mu = 0.0;
#pragma omp simd reduction(+:fc_mu)
            for (int nu = 0; nu < n_basis_fn; ++nu) {
                fc_mu += row[nu] * column[nu];
            }
            energy += column[mu] * fc_mu;
        }
        energies[orbital] = energy;
    }

    std::vector<int> sort_idx(n_sel);
    for (int orbital = 0; orbital < n_sel; ++orbital) {
        sort_idx[orbital] = orbital;
    }
    std::sort(
        sort_idx.begin(),
        sort_idx.end(),
        [&](int left, int right) {
            return energies[left] < energies[right];
        });

    std::vector<double> sorted_c(orbital_size * n_sel);
    std::vector<double> sorted_energies(n_sel);
    for (int sorted = 0; sorted < n_sel; ++sorted) {
        const int source = sort_idx[sorted];
        const double* const source_column =
            c.data() + static_cast<std::size_t>(source) * orbital_size;
        for (int mu = 0; mu < n_basis_fn; ++mu) {
            sorted_c[
                static_cast<std::size_t>(mu) * n_sel + sorted] =
                source_column[mu];
        }
        sorted_energies[sorted] = energies[source];
    }

    py::array_t<double> result_c =
        make_2d(sorted_c, n_basis_fn, n_sel);
    py::array_t<double> result_energies =
        make_1d(sorted_energies);

    const double seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time)
            .count();
    std::ostringstream message;
    message
        << "C++ localization runtime: "
        << std::fixed << std::setprecision(6)
        << seconds << " seconds ("
        << final_sweep << " sweeps, "
        << total_rotations << " rotations, "
        << (converged ? "converged" : "not converged")
        << ")";
    py::print(message.str());

    return py::make_tuple(
        std::move(result_c),
        std::move(result_energies));
}

PYBIND11_MODULE(localization_native, module) {
    module.def(
        "localize_orbitals_cpp",
        &localize_orbitals_cpp,
        py::arg("cmo"),
        py::arg("overlap"),
        py::arg("fock"),
        py::arg("basis"),
        py::arg("space") = "occupied",
        py::arg("n_occ") = 0,
        py::arg("orbital_range") = py::tuple(),
        py::arg("seed") = 0,
        py::arg("sweep_orders") = IntArray());
}
