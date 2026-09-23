#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;



inline double powi(double x, int n) {
    double result = 1.0;
    for (int i = 0; i < n; ++i) result *= x;
    return result;
}

// Angular-momentum tag, resolved once from the orb_val string at basis-load
// time so the hot per-point loop switches on an int instead of comparing
// strings (previously up to ~80 string compares per point per basis fn).
enum class OrbType {
    S, PX, PY, PZ,
    D0, DC1, DS1, DC2, DS2,
    F0, FC1, FS1, FC2, FS2, FC3, FS3,
    G0, GC1, GS1, GC2, GS2, GC3, GS3, GC4, GS4,
    H0, HC1, HS1, HC2, HS2, HC3, HS3, HC4, HS4, HC5, HS5,
    I0, IC1, IS1, IC2, IS2, IC3, IS3, IC4, IS4, IC5, IS5, IC6, IS6,
    J0, JC1, JS1, JC2, JS2, JC3, JS3, JC4, JS4, JC5, JS5, JC6, JS6, JC7, JS7,
    UNKNOWN
};

inline OrbType orb_type_from_string(const std::string& orb_val) {
    static const std::unordered_map<std::string, OrbType> table = {
        {"s", OrbType::S}, {"px", OrbType::PX}, {"py", OrbType::PY}, {"pz", OrbType::PZ},
        {"d0", OrbType::D0}, {"dc1", OrbType::DC1}, {"ds1", OrbType::DS1}, {"dc2", OrbType::DC2}, {"ds2", OrbType::DS2},
        {"f0", OrbType::F0}, {"fc1", OrbType::FC1}, {"fs1", OrbType::FS1}, {"fc2", OrbType::FC2}, {"fs2", OrbType::FS2}, {"fc3", OrbType::FC3}, {"fs3", OrbType::FS3},
        {"g0", OrbType::G0}, {"gc1", OrbType::GC1}, {"gs1", OrbType::GS1}, {"gc2", OrbType::GC2}, {"gs2", OrbType::GS2}, {"gc3", OrbType::GC3}, {"gs3", OrbType::GS3}, {"gc4", OrbType::GC4}, {"gs4", OrbType::GS4},
        {"h0", OrbType::H0}, {"hc1", OrbType::HC1}, {"hs1", OrbType::HS1}, {"hc2", OrbType::HC2}, {"hs2", OrbType::HS2}, {"hc3", OrbType::HC3}, {"hs3", OrbType::HS3}, {"hc4", OrbType::HC4}, {"hs4", OrbType::HS4}, {"hc5", OrbType::HC5}, {"hs5", OrbType::HS5},
        {"i0", OrbType::I0}, {"ic1", OrbType::IC1}, {"is1", OrbType::IS1}, {"ic2", OrbType::IC2}, {"is2", OrbType::IS2}, {"ic3", OrbType::IC3}, {"is3", OrbType::IS3}, {"ic4", OrbType::IC4}, {"is4", OrbType::IS4}, {"ic5", OrbType::IC5}, {"is5", OrbType::IS5}, {"ic6", OrbType::IC6}, {"is6", OrbType::IS6},
        {"j0", OrbType::J0}, {"jc1", OrbType::JC1}, {"js1", OrbType::JS1}, {"jc2", OrbType::JC2}, {"js2", OrbType::JS2}, {"jc3", OrbType::JC3}, {"js3", OrbType::JS3}, {"jc4", OrbType::JC4}, {"js4", OrbType::JS4}, {"jc5", OrbType::JC5}, {"js5", OrbType::JS5}, {"jc6", OrbType::JC6}, {"js6", OrbType::JS6}, {"jc7", OrbType::JC7}, {"js7", OrbType::JS7},
    };
    auto it = table.find(orb_val);
    return it != table.end() ? it->second : OrbType::UNKNOWN;
}

inline double solid_harmonic(double dx, double dy, double dz, OrbType t) {
    switch (t) {
    // s and p
    case OrbType::S:  return 1.0;
    case OrbType::PX: return dx;
    case OrbType::PY: return dy;
    case OrbType::PZ: return dz;

    // d
    case OrbType::D0: return 0.5 * (2 * dz * dz - dx * dx - dy * dy);
    case OrbType::DC1: return 1.7320508075688772 * dy * dz;
    case OrbType::DS1: return 1.7320508075688772 * dx * dz;
    case OrbType::DC2: return 0.8660254037844386 * (dx * dx - dy * dy);
    case OrbType::DS2: return 1.7320508075688772 * dx * dy;

    // f
    case OrbType::F0: return -1.5 * dx * dx * dz - 1.5 * dy * dy * dz + dz * dz * dz;
    case OrbType::FC1: return -0.6123724356957945 * dx * dx * dx - 0.6123724356957945 * dx * dy * dy + 2.449489742783178 * dx * dz * dz;
    case OrbType::FS1: return -0.6123724356957945 * dx * dx * dy - 0.6123724356957945 * dy * dy * dy + 2.449489742783178 * dy * dz * dz;
    case OrbType::FC2: return 1.9364916731037085 * dx * dx * dz - 1.9364916731037085 * dy * dy * dz;
    case OrbType::FS2: return 3.872983346207417 * dx * dy * dz;
    case OrbType::FC3: return 0.7905694150420949 * dx * dx * dx - 2.3717082451262845 * dx * dy * dy;
    case OrbType::FS3: return 2.3717082451262845 * dx * dx * dy - 0.7905694150420949 * dy * dy * dy;

    // g
    case OrbType::G0: return 0.375 * powi(dx,4) + 0.75 * powi(dx,2) * powi(dy,2) - 3.0 * powi(dx,2) * powi(dz,2) + 0.375 * powi(dy,4) - 3.0 * powi(dy,2) * powi(dz,2) + 1.0 * powi(dz,4);
    case OrbType::GC1: return -2.37170824512628 * powi(dx,3) * dz - 2.37170824512628 * dx * powi(dy,2) * dz + 3.16227766016838 * dx * powi(dz,3);
    case OrbType::GS1: return -2.37170824512628 * powi(dx,2) * dy * dz - 2.37170824512628 * powi(dy,3) * dz + 3.16227766016838 * dy * powi(dz,3);
    case OrbType::GC2: return -0.559016994374947 * powi(dx,4) + 3.35410196624968 * powi(dx,2) * powi(dz,2) + 0.559016994374947 * powi(dy,4) - 3.35410196624968 * powi(dy,2) * powi(dz,2);
    case OrbType::GS2: return -1.11803398874989 * powi(dx,3) * dy - 1.11803398874989 * dx * powi(dy,3) + 6.70820393249937 * dx * dy * powi(dz,2);
    case OrbType::GC3: return 2.09165006633519 * powi(dx,3) * dz - 6.27495019900557 * dx * powi(dy,2) * dz;
    case OrbType::GS3: return 6.27495019900557 * powi(dx,2) * dy * dz - 2.09165006633519 * powi(dy,3) * dz;
    case OrbType::GC4: return 0.739509972887452 * powi(dx,4) - 4.43705983732471 * powi(dx,2) * powi(dy,2) + 0.739509972887452 * powi(dy,4);
    case OrbType::GS4: return 2.95803989154981 * powi(dx,3) * dy - 2.95803989154981 * dx * powi(dy,3);

    // h
    case OrbType::H0: return 1.875 * powi(dx,4) * dz + 3.75 * powi(dx,2) * powi(dy,2) * dz - 5.0 * powi(dx,2) * powi(dz,3) + 1.875 * powi(dy,4) * dz - 5.0 * powi(dy,2) * powi(dz,3) + 1.0 * powi(dz,5);
    case OrbType::HC1: return 0.484122918275927 * powi(dx,5) + 0.968245836551854 * powi(dx,3) * powi(dy,2) - 5.80947501931113 * powi(dx,3) * powi(dz,2) + 0.484122918275927 * dx * powi(dy,4) - 5.80947501931113 * dx * powi(dy,2) * powi(dz,2) + 3.87298334620742 * dx * powi(dz,4);
    case OrbType::HS1: return 0.484122918275927 * powi(dx,4) * dy + 0.968245836551854 * powi(dx,2) * powi(dy,3) - 5.80947501931113 * powi(dx,2) * dy * powi(dz,2) + 0.484122918275927 * powi(dy,5) - 5.80947501931113 * powi(dy,3) * powi(dz,2) + 3.87298334620742 * dy * powi(dz,4);
    case OrbType::HC2: return -2.5617376914899 * powi(dx,4) * dz + 5.1234753829798 * powi(dx,2) * powi(dz,3) + 2.5617376914899 * powi(dy,4) * dz - 5.1234753829798 * powi(dy,2) * powi(dz,3);
    case OrbType::HS2: return -5.1234753829798 * powi(dx,3) * dy * dz - 5.1234753829798 * dx * powi(dy,3) * dz + 10.2469507659596 * dx * dy * powi(dz,3);
    case OrbType::HC3: return -0.522912516583797 * powi(dx,5) + 1.04582503316759 * powi(dx,3) * powi(dy,2) + 4.18330013267038 * powi(dx,3) * powi(dz,2) + 1.56873754975139 * dx * powi(dy,4) - 12.5499003980111 * dx * powi(dy,2) * powi(dz,2);
    case OrbType::HS3: return -1.56873754975139 * powi(dx,4) * dy - 1.04582503316759 * powi(dx,2) * powi(dy,3) + 12.5499003980111 * powi(dx,2) * dy * powi(dz,2) + 0.522912516583797 * powi(dy,5) - 4.18330013267038 * powi(dy,3) * powi(dz,2);
    case OrbType::HC4: return 2.21852991866236 * powi(dx,4) * dz - 13.3111795119741 * powi(dx,2) * powi(dy,2) * dz + 2.21852991866236 * powi(dy,4) * dz;
    case OrbType::HS4: return 8.87411967464942 * powi(dx,3) * dy * dz - 8.87411967464942 * dx * powi(dy,3) * dz;
    case OrbType::HC5: return 0.701560760020114 * powi(dx,5) - 7.01560760020114 * powi(dx,3) * powi(dy,2) + 3.50780380010057 * dx * powi(dy,4);
    case OrbType::HS5: return 3.50780380010057 * powi(dx,4) * dy - 7.01560760020114 * powi(dx,2) * powi(dy,3) + 0.701560760020114 * powi(dy,5);

    // i
    case OrbType::I0: return -0.3125 * powi(dx,6) - 0.9375 * powi(dx,4) * powi(dy,2) + 5.625 * powi(dx,4) * powi(dz,2) - 0.9375 * powi(dx,2) * powi(dy,4) + 11.25 * powi(dx,2) * powi(dy,2) * powi(dz,2) - 7.5 * powi(dx,2) * powi(dz,4) - 0.3125 * powi(dy,6) + 5.625 * powi(dy,4) * powi(dz,2) - 7.5 * powi(dy,2) * powi(dz,4) + 1.0 * powi(dz,6);
    case OrbType::IC1: return 2.8641098093474 * powi(dx,5) * dz + 5.7282196186948 * powi(dx,3) * powi(dy,2) * dz - 11.4564392373896 * powi(dx,3) * powi(dz,3) + 2.8641098093474 * dx * powi(dy,4) * dz - 11.4564392373896 * dx * powi(dy,2) * powi(dz,3) + 4.58257569495584 * dx * powi(dz,5);
    case OrbType::IS1: return 2.8641098093474 * powi(dx,4) * dy * dz + 5.7282196186948 * powi(dx,2) * powi(dy,3) * dz - 11.4564392373896 * powi(dx,2) * dy * powi(dz,3) + 2.8641098093474 * powi(dy,5) * dz - 11.4564392373896 * powi(dy,3) * powi(dz,3) + 4.58257569495584 * dy * powi(dz,5);
    case OrbType::IC2: return 0.45285552331842 * powi(dx,6) + 0.45285552331842 * powi(dx,4) * powi(dy,2) - 7.24568837309472 * powi(dx,4) * powi(dz,2) - 0.45285552331842 * powi(dx,2) * powi(dy,4) + 7.24568837309472 * powi(dx,2) * powi(dz,4) - 0.45285552331842 * powi(dy,6) + 7.24568837309472 * powi(dy,4) * powi(dz,2) - 7.24568837309472 * powi(dy,2) * powi(dz,4);
    case OrbType::IS2: return 0.90571104663684 * powi(dx,5) * dy + 1.81142209327368 * powi(dx,3) * powi(dy,3) - 14.4913767461894 * powi(dx,3) * dy * powi(dz,2) + 0.90571104663684 * dx * powi(dy,5) - 14.4913767461894 * dx * powi(dy,3) * powi(dz,2) + 14.4913767461894 * dx * dy * powi(dz,4);
    case OrbType::IC3: return -2.71713313991052 * powi(dx,5) * dz + 5.43426627982104 * powi(dx,3) * powi(dy,2) * dz + 7.24568837309472 * powi(dx,3) * powi(dz,3) + 8.15139941973156 * dx * powi(dy,4) * dz - 21.7370651192842 * dx * powi(dy,2) * powi(dz,3);
    case OrbType::IS3: return -8.15139941973156 * powi(dx,4) * dy * dz - 5.43426627982104 * powi(dx,2) * powi(dy,3) * dz + 21.7370651192842 * powi(dx,2) * dy * powi(dz,3) + 2.71713313991052 * powi(dy,5) * dz - 7.24568837309472 * powi(dy,3) * powi(dz,3);
    case OrbType::IC4: return -0.496078370824611 * powi(dx,6) + 2.48039185412305 * powi(dx,4) * powi(dy,2) + 4.96078370824611 * powi(dx,4) * powi(dz,2) + 2.48039185412305 * powi(dx,2) * powi(dy,4) - 29.7647022494766 * powi(dx,2) * powi(dy,2) * powi(dz,2) - 0.496078370824611 * powi(dy,6) + 4.96078370824611 * powi(dy,4) * powi(dz,2);
    case OrbType::IS4: return -1.98431348329844 * powi(dx,5) * dy + 19.8431348329844 * powi(dx,3) * dy * powi(dz,2) + 1.98431348329844 * dx * powi(dy,5) - 19.8431348329844 * dx * powi(dy,3) * powi(dz,2);
    case OrbType::IC5: return 2.32681380862329 * powi(dx,5) * dz - 23.2681380862329 * powi(dx,3) * powi(dy,2) * dz + 11.6340690431164 * dx * powi(dy,4) * dz;
    case OrbType::IS5: return 11.6340690431164 * powi(dx,4) * dy * dz - 23.2681380862329 * powi(dx,2) * powi(dy,3) * dz + 2.32681380862329 * powi(dy,5) * dz;
    case OrbType::IC6: return 0.671693289381396 * powi(dx,6) - 10.0753993407209 * powi(dx,4) * powi(dy,2) + 10.0753993407209 * powi(dx,2) * powi(dy,4) - 0.671693289381396 * powi(dy,6);
    case OrbType::IS6: return 4.03015973628838 * powi(dx,5) * dy - 13.4338657876279 * powi(dx,3) * powi(dy,3) + 4.03015973628838 * dx * powi(dy,5);

    // j
    case OrbType::J0: return -2.1875 * powi(dx,6) * dz - 6.5625 * powi(dx,4) * powi(dy,2) * dz + 13.125 * powi(dx,4) * powi(dz,3) - 6.5625 * powi(dx,2) * powi(dy,4) * dz + 26.25 * powi(dx,2) * powi(dy,2) * powi(dz,3) - 10.5 * powi(dx,2) * powi(dz,5) - 2.1875 * powi(dy,6) * dz + 13.125 * powi(dy,4) * powi(dz,3) - 10.5 * powi(dy,2) * powi(dz,5) + 1.0 * powi(dz,7);
    case OrbType::JC1: return -0.413398642353842 * powi(dx,7) - 1.24019592706153 * powi(dx,5) * powi(dy,2) + 9.92156741649221 * powi(dx,5) * powi(dz,2) - 1.24019592706153 * powi(dx,3) * powi(dy,4) + 19.8431348329844 * powi(dx,3) * powi(dy,2) * powi(dz,2) - 19.8431348329844 * powi(dx,3) * powi(dz,4) - 0.413398642353842 * dx * powi(dy,6) + 9.92156741649221 * dx * powi(dy,4) * powi(dz,2) - 19.8431348329844 * dx * powi(dy,2) * powi(dz,4) + 5.29150262212918 * dx * powi(dz,6);
    case OrbType::JS1: return -0.413398642353842 * powi(dx,6) * dy - 1.24019592706153 * powi(dx,4) * powi(dy,3) + 9.92156741649221 * powi(dx,4) * dy * powi(dz,2) - 1.24019592706153 * powi(dx,2) * powi(dy,5) + 19.8431348329844 * powi(dx,2) * powi(dy,3) * powi(dz,2) - 19.8431348329844 * powi(dx,2) * dy * powi(dz,4) - 0.413398642353842 * powi(dy,7) + 9.92156741649221 * powi(dy,5) * powi(dz,2) - 19.8431348329844 * powi(dy,3) * powi(dz,4) + 5.29150262212918 * dy * powi(dz,6);
    case OrbType::JC2: return 3.03784720237868 * powi(dx,6) * dz + 3.03784720237868 * powi(dx,4) * powi(dy,2) * dz - 16.2018517460197 * powi(dx,4) * powi(dz,3) - 3.03784720237868 * powi(dx,2) * powi(dy,4) * dz + 9.72111104761179 * powi(dx,2) * powi(dz,5) - 3.03784720237868 * powi(dy,6) * dz + 16.2018517460197 * powi(dy,4) * powi(dz,3) - 9.72111104761179 * powi(dy,2) * powi(dz,5);
    case OrbType::JS2: return 6.07569440475737 * powi(dx,5) * dy * dz + 12.1513888095147 * powi(dx,3) * powi(dy,3) * dz - 32.4037034920393 * powi(dx,3) * dy * powi(dz,3) + 6.07569440475737 * dx * powi(dy,5) * dz - 32.4037034920393 * dx * powi(dy,3) * powi(dz,3) + 19.4422220952236 * dx * dy * powi(dz,5);
    case OrbType::JC3: return 0.42961647140211 * powi(dx,7) - 0.42961647140211 * powi(dx,5) * powi(dy,2) - 8.5923294280422 * powi(dx,5) * powi(dz,2) - 2.14808235701055 * powi(dx,3) * powi(dy,4) + 17.1846588560844 * powi(dx,3) * powi(dy,2) * powi(dz,2) + 11.4564392373896 * powi(dx,3) * powi(dz,4) - 1.28884941420633 * dx * powi(dy,6) + 25.7769882841266 * dx * powi(dy,4) * powi(dz,2) - 34.3693177121688 * dx * powi(dy,2) * powi(dz,4);
    case OrbType::JS3: return 1.28884941420633 * powi(dx,6) * dy + 2.14808235701055 * powi(dx,4) * powi(dy,3) - 25.7769882841266 * powi(dx,4) * dy * powi(dz,2) + 0.42961647140211 * powi(dx,2) * powi(dy,5) - 17.1846588560844 * powi(dx,2) * powi(dy,3) * powi(dz,2) + 34.3693177121688 * powi(dx,2) * dy * powi(dz,4) - 0.42961647140211 * powi(dy,7) + 8.5923294280422 * powi(dy,5) * powi(dz,2) - 11.4564392373896 * powi(dy,3) * powi(dz,4);
    case OrbType::JC4: return -2.8497532787945 * powi(dx,6) * dz + 14.2487663939725 * powi(dx,4) * powi(dy,2) * dz + 9.49917759598167 * powi(dx,4) * powi(dz,3) + 14.2487663939725 * powi(dx,2) * powi(dy,4) * dz - 56.99506557589 * powi(dx,2) * powi(dy,2) * powi(dz,3) - 2.8497532787945 * powi(dy,6) * dz + 9.49917759598167 * powi(dy,4) * powi(dz,3);
    case OrbType::JS4: return -11.399013115178 * powi(dx,5) * dy * dz + 37.9967103839267 * powi(dx,3) * dy * powi(dz,3) + 11.399013115178 * dx * powi(dy,5) * dz - 37.9967103839267 * dx * powi(dy,3) * powi(dz,3);
    case OrbType::JC5: return -0.474958879799083 * powi(dx,7) + 4.27462991819175 * powi(dx,5) * powi(dy,2) + 5.699506557589 * powi(dx,5) * powi(dz,2) + 2.37479439899542 * powi(dx,3) * powi(dy,4) - 56.99506557589 * powi(dx,3) * powi(dy,2) * powi(dz,2) - 2.37479439899542 * dx * powi(dy,6) + 28.497532787945 * dx * powi(dy,4) * powi(dz,2);
    case OrbType::JS5: return -2.37479439899542 * powi(dx,6) * dy + 2.37479439899542 * powi(dx,4) * powi(dy,3) + 28.497532787945 * powi(dx,4) * dy * powi(dz,2) + 4.27462991819175 * powi(dx,2) * powi(dy,5) - 56.99506557589 * powi(dx,2) * powi(dy,3) * powi(dz,2) - 0.474958879799083 * powi(dy,7) + 5.699506557589 * powi(dy,5) * powi(dz,2);
    case OrbType::JC6: return 2.4218245962497 * powi(dx,6) * dz - 36.3273689437454 * powi(dx,4) * powi(dy,2) * dz + 36.3273689437454 * powi(dx,2) * powi(dy,4) * dz - 2.4218245962497 * powi(dy,6) * dz;
    case OrbType::JS6: return 14.5309475774982 * powi(dx,5) * dy * dz - 48.4364919249939 * powi(dx,3) * powi(dy,3) * dz + 14.5309475774982 * dx * powi(dy,5) * dz;
    case OrbType::JC7: return 0.647259849287749 * powi(dx,7) - 13.5924568350427 * powi(dx,5) * powi(dy,2) + 22.6540947250712 * powi(dx,3) * powi(dy,4) - 4.53081894501425 * dx * powi(dy,6);
    case OrbType::JS7: return 4.53081894501425 * powi(dx,6) * dy - 22.6540947250712 * powi(dx,4) * powi(dy,3) + 13.5924568350427 * powi(dx,2) * powi(dy,5) - 0.647259849287749 * powi(dy,7);

    default: return 0.0;
    }
}


namespace {

// Basis-set data unpacked once into flat C++ arrays/vectors, shared by
// electron_density() and occupied_density() so both hot loops switch on
// an int (OrbType) instead of comparing strings, and skip primitives that
// have decayed under 1e-15 via a precomputed r^2 cutoff instead of an
// exp() call.
struct BasisArrays {
    std::vector<int> centers;
    std::vector<OrbType> orb_types;
    std::vector<std::vector<double>> exps_list, coeffs_list;
    std::vector<double> r2_cutoff;

    explicit BasisArrays(const py::list& data) {
        constexpr double NEG_LOG_1EM15 = 34.538776394910684;  // -ln(1e-15)
        size_t n_basis = data.size();
        centers.resize(n_basis);
        orb_types.resize(n_basis);
        exps_list.resize(n_basis);
        coeffs_list.resize(n_basis);
        r2_cutoff.resize(n_basis);

        for (size_t i = 0; i < n_basis; ++i) {
            py::dict basis = data[i];
            centers[i] = basis["CENTER"].cast<int>() - 1;
            orb_types[i] = orb_type_from_string(basis["orb_val"].cast<std::string>());
            exps_list[i] = basis["exps"].cast<std::vector<double>>();
            coeffs_list[i] = basis["coeffs"].cast<std::vector<double>>();

            double zeta_small = *std::min_element(exps_list[i].begin(), exps_list[i].end());
            r2_cutoff[i] = NEG_LOG_1EM15 / zeta_small;
        }
    }

    size_t size() const { return centers.size(); }
};

}  // namespace


py::array_t<double> electron_density(
    py::list data,                   // List of dicts (basis functions)
    py::array_t<double> coordinates, // shape: (n_atoms, 3)
    py::array_t<double> points,      // shape: (n_points, 3)
    std::vector<double> cmo,         // MO coefficients, size: n_basis
    py::object /*ang_res_lambda*/    // Not used anymore, for interface compatibility
) {
    auto coords = coordinates.unchecked<2>();
    BasisArrays basis(data);
    size_t n_basis = basis.size();

    auto pts = points.unchecked<2>();
    size_t n_points = pts.shape(0);

    py::array_t<double> result(n_points);
    auto res = result.mutable_unchecked<1>();

    // Parallelize over grid points
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t ipt = 0; ipt < n_points; ++ipt) {
        double px = pts(ipt, 0);
        double py_ = pts(ipt, 1);
        double pz = pts(ipt, 2);
        double val = 0.0;

        for (size_t i = 0; i < n_basis; ++i) {
            double c = cmo[i];
            if (std::abs(c) < 1e-15) continue;

            int center = basis.centers[i];
            double dx = px - coords(center, 0);
            double dy = py_ - coords(center, 1);
            double dz = pz - coords(center, 2);
            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 > basis.r2_cutoff[i]) continue;

            double angular_part = solid_harmonic(dx, dy, dz, basis.orb_types[i]);
            const std::vector<double>& exps = basis.exps_list[i];
            const std::vector<double>& coeffs = basis.coeffs_list[i];

            double bas_res = 0.0;
            for (size_t j = 0; j < coeffs.size(); ++j) {
                bas_res += coeffs[j] * std::exp(-exps[j] * r2);
            }

            val += c * bas_res * angular_part;
        }
        res(ipt) = val;
    }

    return result;
}


// Alpha/beta electron density on a grid from occupied-orbital coefficient
// matrices (P_spin = C_occ @ C_occ.T), without ever forming the n_basis x
// n_basis density matrix:
//
//     rho_spin(r) = chi(r)^T P_spin chi(r) = sum_k (sum_i C_occ[i,k] chi_i(r))^2
//
// chi(r) is evaluated once per grid point (same basis-evaluation kernel as
// electron_density()) and reused across every occupied orbital, so this
// costs O(n_basis * n_occ) per point instead of the O(n_basis^2) of an
// explicit dense density-matrix contraction.
py::tuple occupied_density(
    py::list data,                    // List of dicts (basis functions)
    py::array_t<double> coordinates,  // shape: (n_atoms, 3)
    py::array_t<double> points,       // shape: (n_points, 3)
    py::array_t<double> alpha_occ,    // shape: (n_basis, n_occ_alpha)
    py::array_t<double> beta_occ      // shape: (n_basis, n_occ_beta)
) {
    auto coords = coordinates.unchecked<2>();
    BasisArrays basis(data);
    size_t n_basis = basis.size();

    auto a_occ = alpha_occ.unchecked<2>();
    auto b_occ = beta_occ.unchecked<2>();
    if (static_cast<size_t>(a_occ.shape(0)) != n_basis ||
        static_cast<size_t>(b_occ.shape(0)) != n_basis) {
        throw std::invalid_argument(
            "alpha_occ and beta_occ must both have n_basis rows");
    }
    size_t n_occ_alpha = a_occ.shape(1);
    size_t n_occ_beta = b_occ.shape(1);

    auto pts = points.unchecked<2>();
    size_t n_points = pts.shape(0);

    py::array_t<double> alpha_density(n_points);
    py::array_t<double> beta_density(n_points);
    auto ares = alpha_density.mutable_unchecked<1>();
    auto bres = beta_density.mutable_unchecked<1>();

    #pragma omp parallel
    {
        // Thread-local scratch, reused across grid points to avoid a
        // heap allocation per point.
        std::vector<double> chi(n_basis);
        std::vector<double> psi_alpha(n_occ_alpha);
        std::vector<double> psi_beta(n_occ_beta);

        #pragma omp for schedule(dynamic, 64)
        for (size_t ipt = 0; ipt < n_points; ++ipt) {
            double px = pts(ipt, 0);
            double py_ = pts(ipt, 1);
            double pz = pts(ipt, 2);

            std::fill(chi.begin(), chi.end(), 0.0);
            for (size_t i = 0; i < n_basis; ++i) {
                int center = basis.centers[i];
                double dx = px - coords(center, 0);
                double dy = py_ - coords(center, 1);
                double dz = pz - coords(center, 2);
                double r2 = dx*dx + dy*dy + dz*dz;

                if (r2 > basis.r2_cutoff[i]) continue;

                double angular_part = solid_harmonic(dx, dy, dz, basis.orb_types[i]);
                const std::vector<double>& exps = basis.exps_list[i];
                const std::vector<double>& coeffs = basis.coeffs_list[i];

                double bas_res = 0.0;
                for (size_t j = 0; j < coeffs.size(); ++j) {
                    bas_res += coeffs[j] * std::exp(-exps[j] * r2);
                }
                chi[i] = bas_res * angular_part;
            }

            // One pass over the (screened) basis, accumulating into every
            // occupied orbital at once -- keeps the a_occ/b_occ reads
            // contiguous along their row (n_occ-sized) dimension.
            std::fill(psi_alpha.begin(), psi_alpha.end(), 0.0);
            std::fill(psi_beta.begin(), psi_beta.end(), 0.0);
            for (size_t i = 0; i < n_basis; ++i) {
                double ci = chi[i];
                if (ci == 0.0) continue;
                for (size_t k = 0; k < n_occ_alpha; ++k) psi_alpha[k] += a_occ(i, k) * ci;
                for (size_t k = 0; k < n_occ_beta; ++k) psi_beta[k] += b_occ(i, k) * ci;
            }

            double a_sum = 0.0;
            for (double v : psi_alpha) a_sum += v * v;
            ares(ipt) = a_sum;

            double b_sum = 0.0;
            for (double v : psi_beta) b_sum += v * v;
            bres(ipt) = b_sum;
        }
    }

    return py::make_tuple(alpha_density, beta_density);
}


PYBIND11_MODULE(electron_density_opt_omp, m) {
    m.def("electron_density", &electron_density,
          py::arg("data"),
          py::arg("coordinates"),
          py::arg("points"),
          py::arg("cmo"),
          py::arg("ang_res_lambda") = py::none(), // Placeholder for compatibility
          "Vectorized electron density with solid harmonics and OpenMP");

    m.def("occupied_density", &occupied_density,
          py::arg("data"),
          py::arg("coordinates"),
          py::arg("points"),
          py::arg("alpha_occ"),
          py::arg("beta_occ"),
          "Alpha/beta electron density grids from occupied-orbital "
          "coefficient matrices, without forming the density matrix.");
}




