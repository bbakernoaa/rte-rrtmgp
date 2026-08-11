#include "mo_rte_solver_kernels.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>

namespace rte::kernels {

inline real_t fast_exp(real_t x) {
#ifdef ENABLE_FAST_EXP
    if (x < real_t(-16.0)) return real_t(0.0);
    if (x > real_t(0.0)) return std::exp(x);

    real_t y = x * real_t(0.125);
    real_t p = real_t(1.0) + y * (real_t(1.0) + y * (real_t(0.5) + y * (real_t(0.16666666666666667) + y * (real_t(0.041666666666666664) + y * real_t(0.008333333333333333)))));
    p *= p; // p^2
    p *= p; // p^4
    return p * p; // p^8
#else
    return std::exp(x);
#endif
}

void sw_dif_and_source(
    int ncol, int nlay, int gp,
    ConstView2D mu0, ConstView2D sfc_alb_dir,
    ConstView3D tau, ConstView3D ssa, ConstView3D g,
    View2D Rdif, View2D Tdif,
    View2D source_dn, View2D source_up, View2D source_sfc,
    View2D flux_dn_dir
) {
    const real_t min_k = 1.0e4 * 1.1920929e-07;
    const real_t min_mu0 = std::sqrt(1.1920929e-07);

    // flux_dn_dir is incoming direct beam, sized [nlay+1, ncol]
    for (int lay = 0; lay < nlay; ++lay) {
        int lay_index = lay; // top_at_1 assumed true in C++ wrappers currently

        #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
            real_t tau_s = tau(gp, lay_index, col);
            real_t w0_s = ssa(gp, lay_index, col);
            real_t g_s = g(gp, lay_index, col);

            real_t gamma1 = (8.0 - w0_s * (5.0 + 3.0 * g_s)) * 0.25;
            real_t gamma2 = 3.0 * (w0_s * (1.0 - g_s)) * 0.25;

            real_t k = std::sqrt(std::max((gamma1 - gamma2) * (gamma1 + gamma2), min_k));
            real_t exp_minusktau = fast_exp(-tau_s * k);
            real_t exp_minus2ktau = exp_minusktau * exp_minusktau;

            real_t RT_term = 1.0 / (k * (1.0 + exp_minus2ktau) + gamma1 * (1.0 - exp_minus2ktau));

            Rdif(lay_index, col) = RT_term * gamma2 * (1.0 - exp_minus2ktau);
            Tdif(lay_index, col) = RT_term * 2.0 * k * exp_minusktau;

            real_t mu0_val = mu0(lay_index, col);
            real_t mu0_s = std::max(min_mu0, mu0_val);
            real_t k_mu = k * mu0_s;

            real_t denom = std::abs(1.0 - k_mu * k_mu) >= 1.1920929e-07 ? (1.0 - k_mu * k_mu) : 1.1920929e-07;
            RT_term = w0_s * RT_term / denom;

            real_t gamma3 = (2.0 - 3.0 * mu0_s * g_s) * 0.25;
            real_t gamma4 = 1.0 - gamma3;
            real_t alpha1 = gamma1 * gamma4 + gamma2 * gamma3;
            real_t alpha2 = gamma1 * gamma3 + gamma2 * gamma4;

            real_t k_gamma3 = k * gamma3;
            real_t k_gamma4 = k * gamma4;
            real_t Tnoscat = fast_exp(-tau_s / mu0_s);

            real_t Rdir = RT_term *
                ((1.0 - k_mu) * (alpha2 + k_gamma3) -
                 (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau -
                 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * Tnoscat);

            real_t Tdir = -RT_term *
                ((1.0 + k_mu) * (alpha1 + k_gamma4) * Tnoscat -
                 (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat -
                 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau);

            Rdir = std::max(0.0, std::min(Rdir, 1.0 - Tnoscat));
            Tdir = std::max(0.0, std::min(Tdir, 1.0 - Tnoscat - Rdir));

            real_t dir_flux_inc = flux_dn_dir(lay_index, col);
            source_up(lay_index, col) = Rdir * dir_flux_inc;
            source_dn(lay_index, col) = Tdir * dir_flux_inc;
            flux_dn_dir(lay_index + 1, col) = Tnoscat * dir_flux_inc;

            if (mu0_val <= 0.0) {
                source_up(lay_index, col) = 0.0;
                source_dn(lay_index, col) = 0.0;
            }
        }
    }

    #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
        real_t dir_flux_trans = flux_dn_dir(nlay, col);
        source_sfc(gp, col) = mu0(nlay - 1, col) > 0.0 ? dir_flux_trans * sfc_alb_dir(gp, col) : 0.0;
    }
}

void adding(
    int ncol, int nlay, int gp,
    ConstView2D sfc_alb_dif,
    ConstView2D Rdif, ConstView2D Tdif,
    ConstView2D source_dn, ConstView2D source_up, ConstView2D source_sfc,
    View2D flux_up, View2D flux_dn
) {
    #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
        real_t albedo_data[257] = {0.0};
        real_t src_data[257] = {0.0};

        albedo_data[nlay] = sfc_alb_dif(gp, col);
        src_data[nlay] = source_sfc(gp, col);

        // Upward sweep
        for (int lay = nlay - 1; lay >= 0; --lay) {
            real_t denom = 1.0 / (1.0 - Rdif(lay, col) * albedo_data[lay + 1]);
            albedo_data[lay] = Rdif(lay, col) + Tdif(lay, col) * Tdif(lay, col) * albedo_data[lay + 1] * denom;
            src_data[lay] = source_up(lay, col) + Tdif(lay, col) * denom *
                            (src_data[lay + 1] + albedo_data[lay + 1] * source_dn(lay, col));
        }

        flux_up(0, col) = src_data[0]; // Top boundary

        // Downward sweep
        for (int lay = 0; lay < nlay; ++lay) {
            real_t denom = 1.0 / (1.0 - Rdif(lay, col) * albedo_data[lay + 1]);
            flux_dn(lay + 1, col) = (Tdif(lay, col) * flux_dn(lay, col) +
                                     Rdif(lay, col) * src_data[lay + 1] +
                                     source_dn(lay, col)) * denom;
            flux_up(lay + 1, col) = flux_dn(lay + 1, col) * albedo_data[lay + 1] + src_data[lay + 1];
        }
    }
}

void lw_source_noscat(
    int ncol, int nlay, int gp,
    ConstView3D lay_source, ConstView3D lev_source, ConstView2D tau, ConstView2D trans,
    View2D source_dn, View2D source_up
) {
    const real_t tau_thresh = std::sqrt(std::sqrt(1.1920929e-07));

    for (int lay = 0; lay < nlay; ++lay) {
        for (int col = 0; col < ncol; ++col) {
            real_t t = tau(lay, col);
            real_t tr = trans(lay, col);
            real_t fact;

            if (t > tau_thresh) {
                fact = (1.0 - tr) / t - tr;
            } else {
                fact = t * (0.5 + t * (-1.0 / 3.0 + t * 1.0 / 8.0));
            }

            source_dn(lay, col) = (1.0 - tr) * lev_source(gp, lay + 1, col) +
                                  2.0 * fact * (lay_source(gp, lay, col) - lev_source(gp, lay + 1, col));
            source_up(lay, col) = (1.0 - tr) * lev_source(gp, lay, col) +
                                  2.0 * fact * (lay_source(gp, lay, col) - lev_source(gp, lay, col));
        }
    }
}

void lw_transport_noscat_dn(
    int ncol, int nlay,
    ConstView2D trans, ConstView2D source_dn, View2D radn_dn
) {
    for (int lay = 0; lay < nlay; ++lay) {
        for (int col = 0; col < ncol; ++col) {
            radn_dn(lay + 1, col) = trans(lay, col) * radn_dn(lay, col) + source_dn(lay, col);
        }
    }
}

void lw_transport_noscat_up(
    int ncol, int nlay,
    ConstView2D trans, ConstView2D source_up, View2D radn_up
) {
    for (int lay = nlay - 1; lay >= 0; --lay) {
        for (int col = 0; col < ncol; ++col) {
            radn_up(lay, col) = trans(lay, col) * radn_up(lay + 1, col) + source_up(lay, col);
        }
    }
}

} // namespace rte::kernels
