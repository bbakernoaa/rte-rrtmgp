#include "mo_rte_lw.h"
#include "mo_rte_util.h"
#include "mo_rte_solver_kernels.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace rte {

inline real_t fast_exp(real_t x) {
#ifdef ENABLE_FAST_EXP
    if (x < real_t(-87.0)) return real_t(0.0);
    if (x > real_t(88.0))  return std::numeric_limits<real_t>::infinity();

    constexpr real_t INV_LN2 = real_t(1.4426950408889634074);
    constexpr real_t LN2     = real_t(0.6931471805599453094);

    int k = static_cast<int>(x * INV_LN2 + (x < real_t(0.0) ? real_t(-0.5) : real_t(0.5)));
    real_t r = x - static_cast<real_t>(k) * LN2;

    real_t p = real_t(1.0) + r * (real_t(1.0) + r * (real_t(0.5) + r * (real_t(0.16666666666666667) + r * (real_t(0.041666666666666664) + r * real_t(0.008333333333333333)))));

    uint64_t bits = static_cast<uint64_t>(static_cast<int64_t>(k) + 1023) << 52;
    double twok;
    std::memcpy(&twok, &bits, sizeof(double));

    return p * twok;
#else
    return std::exp(x);
#endif
}

void SolverLw::solve_lw_noscat(
    ConstView3D tau,               // [ngpt, nlay, ncol]
    ConstView3D lay_source,        // [ngpt, nlay, ncol]
    ConstView3D lev_source,        // [ngpt, nlay+1, ncol]
    ConstView2D sfc_emis,          // [ngpt, ncol]
    ConstView2D sfc_src,           // [ngpt, ncol]
    ConstView2D incident_flux,     // [ngpt, ncol]
    View3D flux_up,                // [ngpt, nlay+1, ncol]
    View3D flux_dn                 // [ngpt, nlay+1, ncol]
) {
    const int gp_cnt = tau.extent(0);
    const int layers = tau.extent(1);
    const int columns = tau.extent(2);

    const real_t secant = 1.66;
    const real_t tau_thresh = std::sqrt(std::sqrt(1.1920929e-07));

    int num_tiles = gp_cnt * columns;
    #pragma omp parallel for
    for (int idx = 0; idx < num_tiles; ++idx) {
        int gp = idx % gp_cnt;
        int col = idx / gp_cnt;

        // Boundary condition Top
        flux_dn(gp, 0, col) = incident_flux(gp, col) / M_PI;

        // Transport down
        #pragma omp simd
        for (int lay = 0; lay < layers; ++lay) {
            real_t tau_loc = tau(gp, lay, col) * secant;
            real_t trans = fast_exp(-tau_loc);
            real_t fact;
            if (tau_loc > tau_thresh) {
                fact = (1.0 - trans) / tau_loc - trans;
            } else {
                fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * 1.0 / 8.0));
            }

            real_t source_dn = (1.0 - trans) * lev_source(gp, lay + 1, col) +
                               2.0 * fact * (lay_source(gp, lay, col) - lev_source(gp, lay + 1, col));

            flux_dn(gp, lay + 1, col) = trans * flux_dn(gp, lay, col) + source_dn;
        }

        // Surface boundary condition (Reflection and emission)
        real_t sfc_albedo = 1.0 - sfc_emis(gp, col);
        flux_up(gp, layers, col) = flux_dn(gp, layers, col) * sfc_albedo +
                                   sfc_emis(gp, col) * sfc_src(gp, col);

        // Transport up
        #pragma omp simd
        for (int lay = layers - 1; lay >= 0; --lay) {
            real_t tau_loc = tau(gp, lay, col) * secant;
            real_t trans = fast_exp(-tau_loc);
            real_t fact;
            if (tau_loc > tau_thresh) {
                fact = (1.0 - trans) / tau_loc - trans;
            } else {
                fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * 1.0 / 8.0));
            }

            real_t source_up = (1.0 - trans) * lev_source(gp, lay, col) +
                               2.0 * fact * (lay_source(gp, lay, col) - lev_source(gp, lay, col));

            flux_up(gp, lay, col) = trans * flux_up(gp, lay + 1, col) + source_up;
        }
    }
}

} // namespace rte
