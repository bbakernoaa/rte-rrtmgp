#include "mo_rte_lw.h"
#include "mo_rte_util.h"
#include "mo_rte_solver_kernels.h"
#include <cmath>
#include <vector>

namespace rte {

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

    #pragma omp parallel for collapse(2)
    for (int gp = 0; gp < gp_cnt; ++gp) {
        for (int col = 0; col < columns; ++col) {
            // Boundary condition Top
            flux_dn(gp, 0, col) = incident_flux(gp, col) / M_PI;

            // Transport down
            for (int lay = 0; lay < layers; ++lay) {
                real_t tau_loc = tau(gp, lay, col) * secant;
                real_t trans = std::exp(-tau_loc);
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
            for (int lay = layers - 1; lay >= 0; --lay) {
                real_t tau_loc = tau(gp, lay, col) * secant;
                real_t trans = std::exp(-tau_loc);
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
}

} // namespace rte
