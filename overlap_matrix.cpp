#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <string>
#include <omp.h>

namespace py = pybind11;

// ------------------ Math Utilities ------------------

inline int double_factorial(int n) {
    if (n <= 0) return 1;
    int res = 1;
    for (int i = n; i > 0; i -= 2) res *= i;
    return res;
}

inline double binomial(int a, int b) {
    if (b < 0 || b > a) return 0.0;
    double result = 1.0;
    for (int i = 1; i <= b; ++i)
        result *= (a - (b - i)) / double(i);
    return result;
}

double binomial_prefactor(int s, int ia, int ib, double xpa, double xpb) {
    double sum = 0.0;
    for (int t = 0; t <= s; ++t) {
        if ((s - ia <= t) && (t <= ib)) {
            sum += binomial(ia, s - t) * binomial(ib, t) *
                   std::pow(xpa, ia - s + t) * std::pow(xpb, ib - t);
        }
    }
    return sum;
}

inline double gaussian_norm(double alpha, int l, int m, int n) {
    int lmn = l + m + n;
    double prefactor = std::pow(2, 2 * lmn + 1.5) * std::pow(alpha, lmn + 1.5) / std::pow(M_PI, 1.5);
    return std::sqrt(prefactor /
        (double_factorial(2 * l - 1) *
         double_factorial(2 * m - 1) *
         double_factorial(2 * n - 1)));
}

inline double rsqr(double x1, double y1, double z1, double x2, double y2, double z2) {
    return (x1 - x2) * (x1 - x2) +
           (y1 - y2) * (y1 - y2) +
           (z1 - z2) * (z1 - z2);
}

inline double product_center_1D(double alpha1, double x1, double alpha2, double x2) {
    return (alpha1 * x1 + alpha2 * x2) / (alpha1 + alpha2);
}

double overlap_1D(int l1, int l2, double PAx, double PBx, double gamma) {
    double sum = 0.0;
    int max_i = 1 + (l1 + l2) / 2;
    for (int i = 0; i < max_i; ++i) {
        sum += binomial_prefactor(2 * i, l1, l2, PAx, PBx) * double_factorial(2 * i - 1) / std::pow(2 * gamma, i);
    }
    return sum;
}

double overlap_int(
    double alpha1, int l1, int m1, int n1, double xa, double ya, double za,
    double alpha2, int l2, int m2, int n2, double xb, double yb, double zb
) {
    double rab2 = rsqr(xa, ya, za, xb, yb, zb);
    double gamma = alpha1 + alpha2;

    double xp = product_center_1D(alpha1, xa, alpha2, xb);
    double yp = product_center_1D(alpha1, ya, alpha2, yb);
    double zp = product_center_1D(alpha1, za, alpha2, zb);

    double pre = std::pow(M_PI / gamma, 1.5) * std::exp(-alpha1 * alpha2 * rab2 / gamma);

    double wx = overlap_1D(l1, l2, xp - xa, xp - xb, gamma);
    double wy = overlap_1D(m1, m2, yp - ya, yp - yb, gamma);
    double wz = overlap_1D(n1, n2, zp - za, zp - zb, gamma);

    return pre * wx * wy * wz;
}

// ------------------ Data Structures ------------------

struct BasisFunction {
    int N, CENTER, LABEL, shell_num;
    std::string type, orb_val;
    std::vector<double> exps;
    std::vector<double> coeffs;
    double xcenter, ycenter, zcenter;
};

struct OrbitalType {
    int num_terms;
    std::vector<double> coe;
    std::vector<std::vector<int>> var_cnts;
};

// ------------------ Full Overlap Matrix ------------------

py::array_t<double> get_overlap_matrix(
    py::list primitives,
    py::dict dict_keys,
    bool normalize_primitives = false,
    bool diagonal_only = false
) {
    int nbf = primitives.size();

    std::vector<BasisFunction> basis(nbf);
    for (int i = 0; i < nbf; ++i) {
        py::dict d = primitives[i].cast<py::dict>();
        BasisFunction b;
        b.N = d["N"].cast<int>();
        b.CENTER = d["CENTER"].cast<int>();
        b.LABEL = d["LABEL"].cast<int>();
        b.shell_num = d["shell_num"].cast<int>();
        b.type = d["type"].cast<std::string>();
        b.orb_val = d["orb_val"].cast<std::string>();
        b.exps = d["exps"].cast<std::vector<double>>();
        b.coeffs = d["coeffs"].cast<std::vector<double>>();
        b.xcenter = d["xcenter"].cast<double>();
        b.ycenter = d["ycenter"].cast<double>();
        b.zcenter = d["zcenter"].cast<double>();
        basis[i] = b;
    }

    std::unordered_map<std::string, OrbitalType> orb_map;
    for (auto item : dict_keys) {
        std::string key = py::str(item.first);
        py::dict val = item.second.cast<py::dict>();
        OrbitalType o;
        o.num_terms = val["num_terms"].cast<int>();
        o.coe = val["coe"].cast<std::vector<double>>();
        o.var_cnts = val["var_cnts"].cast<std::vector<std::vector<int>>>();
        orb_map[key] = o;
    }

    std::vector<std::vector<double>> S(nbf, std::vector<double>(nbf, 0.0));

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < nbf; ++j) {
        for (int i = 0; i <= j; ++i) {
            if (diagonal_only && i != j) continue;

            const auto& A = basis[i], B = basis[j];
            const auto& orbA = orb_map.at(A.orb_val);
            const auto& orbB = orb_map.at(B.orb_val);

            for (size_t ip = 0; ip < A.exps.size(); ++ip) {
                for (size_t jp = 0; jp < B.exps.size(); ++jp) {
                    double alpha1 = A.exps[ip], alpha2 = B.exps[jp];
                    double coef1 = A.coeffs[ip], coef2 = B.coeffs[jp];

                    for (int ti = 0; ti < orbA.num_terms; ++ti) {
                        const auto& L1 = orbA.var_cnts[ti];
                        double cc1 = orbA.coe[ti];
                        for (int tj = 0; tj < orbB.num_terms; ++tj) {
                            const auto& L2 = orbB.var_cnts[tj];
                            double cc2 = orbB.coe[tj];

                            double norm1 = normalize_primitives ? gaussian_norm(alpha1, L1[0], L1[1], L1[2]) : 1.0;
                            double norm2 = normalize_primitives ? gaussian_norm(alpha2, L2[0], L2[1], L2[2]) : 1.0;

                            S[j][i] += cc1 * cc2 * coef1 * coef2 * norm1 * norm2 *
                                overlap_int(
                                    alpha1, L1[0], L1[1], L1[2], A.xcenter, A.ycenter, A.zcenter,
                                    alpha2, L2[0], L2[1], L2[2], B.xcenter, B.ycenter, B.zcenter
                                );
                        }
                    }
                }
            }

            if (!diagonal_only && i != j)
                S[i][j] = S[j][i];
        }
    }

    py::array_t<double> result({nbf, nbf});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < nbf; ++i)
        for (int j = 0; j < nbf; ++j)
            r(i, j) = S[i][j];

    return result;
}

// ------------------ Diagonal Only (1D Output) ------------------

py::array_t<double> get_overlap_diagonal_only(
    py::list primitives,
    py::dict dict_keys,
    bool normalize_primitives = false
) {
    int nbf = primitives.size();

    std::vector<BasisFunction> basis(nbf);
    for (int i = 0; i < nbf; ++i) {
        py::dict d = primitives[i].cast<py::dict>();
        BasisFunction b;
        b.N = d["N"].cast<int>();
        b.CENTER = d["CENTER"].cast<int>();
        b.LABEL = d["LABEL"].cast<int>();
        b.shell_num = d["shell_num"].cast<int>();
        b.type = d["type"].cast<std::string>();
        b.orb_val = d["orb_val"].cast<std::string>();
        b.exps = d["exps"].cast<std::vector<double>>();
        b.coeffs = d["coeffs"].cast<std::vector<double>>();
        b.xcenter = d["xcenter"].cast<double>();
        b.ycenter = d["ycenter"].cast<double>();
        b.zcenter = d["zcenter"].cast<double>();
        basis[i] = b;
    }

    std::unordered_map<std::string, OrbitalType> orb_map;
    for (auto item : dict_keys) {
        std::string key = py::str(item.first);
        py::dict val = item.second.cast<py::dict>();
        OrbitalType o;
        o.num_terms = val["num_terms"].cast<int>();
        o.coe = val["coe"].cast<std::vector<double>>();
        o.var_cnts = val["var_cnts"].cast<std::vector<std::vector<int>>>();
        orb_map[key] = o;
    }

    py::array_t<double> diag(nbf);
    auto d = diag.mutable_unchecked<1>();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nbf; ++i) {
        const auto& A = basis[i];
        const auto& orbA = orb_map.at(A.orb_val);
        double sum = 0.0;

        for (size_t ip = 0; ip < A.exps.size(); ++ip) {
            for (size_t jp = 0; jp < A.exps.size(); ++jp) {
                double alpha1 = A.exps[ip], alpha2 = A.exps[jp];
                double coef1 = A.coeffs[ip], coef2 = A.coeffs[jp];

                for (int ti = 0; ti < orbA.num_terms; ++ti) {
                    const auto& L1 = orbA.var_cnts[ti];
                    double cc1 = orbA.coe[ti];
                    for (int tj = 0; tj < orbA.num_terms; ++tj) {
                        const auto& L2 = orbA.var_cnts[tj];
                        double cc2 = orbA.coe[tj];

                        double norm1 = normalize_primitives ? gaussian_norm(alpha1, L1[0], L1[1], L1[2]) : 1.0;
                        double norm2 = normalize_primitives ? gaussian_norm(alpha2, L2[0], L2[1], L2[2]) : 1.0;

                        sum += cc1 * cc2 * coef1 * coef2 * norm1 * norm2 *
                            overlap_int(alpha1, L1[0], L1[1], L1[2], A.xcenter, A.ycenter, A.zcenter,
                                        alpha2, L2[0], L2[1], L2[2], A.xcenter, A.ycenter, A.zcenter);
                    }
                }
            }
        }
        d(i) = sum;
    }

    return diag;
}

// ------------------ Module ------------------

PYBIND11_MODULE(overlap_matrix, m) {
    m.doc() = "Gaussian Overlap Matrix (Parallel, Pybind11, GIL-free)";
    m.def("get_overlap_matrix", &get_overlap_matrix,
          py::arg("primitives"),
          py::arg("dict_keys"),
          py::arg("normalize_primitives") = false,
          py::arg("diagonal_only") = false,
          "Full overlap matrix with optional normalization and diagonal-only flag");

    m.def("get_overlap_diagonal_only", &get_overlap_diagonal_only,
          py::arg("primitives"),
          py::arg("dict_keys"),
          py::arg("normalize_primitives") = false,
          "Extract only diagonal overlap matrix elements (1D array)");
}
