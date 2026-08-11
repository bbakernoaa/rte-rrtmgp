#include "mo_rte_sw.h"
#include "mo_rte_util.h"
#include "mo_rte_solver_kernels.h"
#include <cmath>
#include <vector>

namespace rte {

void SolverSw::solve_sw_noscat(
    ConstView3D tau,
    ConstView2D solar_zenith_angle, // [nlay, ncol]
    ConstView2D inc_flux_dir,       // [ngpt, ncol]
    View3D flux_dir                 // [ngpt, nlay+1, ncol]
) {
    const int gp_cnt = tau.extent(0);
    const int layers = tau.extent(1);
    const int columns = tau.extent(2);

    for (int gp = 0; gp < gp_cnt; ++gp) {
        #pragma omp parallel for
        for (int col = 0; col < columns; ++col) {
            real_t mu0 = solar_zenith_angle(layers - 1, col); // Assuming TOA is nlay - 1
            if (mu0 <= 0.0) {
                for (int lay = 0; lay <= layers; ++lay) flux_dir(gp, lay, col) = 0.0;
                continue;
            }

            flux_dir(gp, 0, col) = inc_flux_dir(gp, col) * mu0; // Top boundary

            for (int lay = 0; lay < layers; ++lay) {
                flux_dir(gp, lay + 1, col) = flux_dir(gp, lay, col) * std::exp(-tau(gp, lay, col) / mu0);
            }
        }
    }
}

void SolverSw::solve_sw_2stream(
    ConstView3D tau,
    ConstView3D ssa,
    ConstView3D g,
    ConstView2D solar_zenith_angle,
    ConstView2D sfc_alb_dir,
    ConstView2D sfc_alb_dif,
    ConstView2D inc_flux_dir,
    View3D flux_up,
    View3D flux_dn,
    View3D flux_dir
) {
    const int gp_cnt = tau.extent(0);
    const int layers = tau.extent(1);
    const int columns = tau.extent(2);

    std::vector<real_t> R_dir_buf(layers * columns, 0.0);
    std::vector<real_t> T_dir_buf(layers * columns, 0.0);
    std::vector<real_t> R_dif_buf(layers * columns, 0.0);
    std::vector<real_t> T_dif_buf(layers * columns, 0.0);
    std::vector<real_t> src_up_buf(layers * columns, 0.0);
    std::vector<real_t> src_dn_buf(layers * columns, 0.0);
    std::vector<real_t> src_sfc_buf(columns, 0.0);

    auto R_dir = View2D(R_dir_buf.data(), Extents2D(layers, columns));
    auto T_dir = View2D(T_dir_buf.data(), Extents2D(layers, columns));
    auto R_dif = View2D(R_dif_buf.data(), Extents2D(layers, columns));
    auto T_dif = View2D(T_dif_buf.data(), Extents2D(layers, columns));
    auto source_up = View2D(src_up_buf.data(), Extents2D(layers, columns));
    auto source_dn = View2D(src_dn_buf.data(), Extents2D(layers, columns));
    auto source_sfc = View2D(src_sfc_buf.data(), Extents2D(1, columns));

    for (int gp = 0; gp < gp_cnt; ++gp) {

        // 1. Set boundary condition direct beam
        #pragma omp parallel for
        for (int col = 0; col < columns; ++col) {
            flux_dir(gp, 0, col) = inc_flux_dir(gp, col) * solar_zenith_angle(0 /*top_layer*/, col);
            flux_dn(gp, 0, col) = 0.0; // Assume no diffuse boundary conditions
        }

        auto flux_dn_dir_slice = View2D(&flux_dir(gp, 0, 0), Extents2D(layers + 1, columns));

        // 2. Transmittance and reflectance for diffuse
        kernels::sw_dif_and_source(
            columns, layers, gp, solar_zenith_angle, sfc_alb_dir,
            tau, ssa, g,
            R_dif, T_dif, source_dn, source_up, source_sfc, flux_dn_dir_slice
        );

        // 3. Adding
        auto flux_up_slice = View2D(&flux_up(gp, 0, 0), Extents2D(layers + 1, columns));
        auto flux_dn_slice = View2D(&flux_dn(gp, 0, 0), Extents2D(layers + 1, columns));

        kernels::adding(
            columns, layers, gp, sfc_alb_dif,
            R_dif, T_dif, source_dn, source_up, source_sfc,
            flux_up_slice, flux_dn_slice
        );

        // adding() computes only diffuse flux; flux_dn is total
        #pragma omp parallel for collapse(2)
        for (int lay = 0; lay <= layers; ++lay) {
            for (int col = 0; col < columns; ++col) {
                flux_dn(gp, lay, col) += flux_dir(gp, lay, col);
            }
        }
    }
}

} // namespace rte
