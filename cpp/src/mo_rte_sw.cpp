#include "mo_rte_sw.h"
#include "mo_rte_util.h"
#include <stdexcept>
#include <cmath>
#include <vector>

namespace rte {

void SolverSw::solve_sw_noscat(
    ConstView3D tau,
    ConstView1D solar_zenith_angle,
    ConstView1D toa_flux,
    View2D flux_dir
) {
    const size_t gp_cnt = tau.extent(0);
    const size_t layers = tau.extent(1);
    const size_t columns = tau.extent(2);

    // Validate boundaries
    validate_extent(solar_zenith_angle, 0, columns, "solar_zenith_angle");
    validate_extent(toa_flux, 0, gp_cnt, "toa_flux");
    validate_extent(flux_dir, 0, gp_cnt, "flux_dir");
    validate_extent(flux_dir, 1, columns, "flux_dir");

    for (size_t col = 0; col < columns; ++col) {
        real_t mu0 = solar_zenith_angle(col); // assumed to be cos(zenith_angle) = mu0 directly
        if (mu0 <= 0.0) {
            for (size_t gp = 0; gp < gp_cnt; ++gp) {
                flux_dir(gp, col) = 0.0;
            }
            continue;
        }

        for (size_t gp = 0; gp < gp_cnt; ++gp) {
            real_t current_dir = toa_flux(gp) * mu0;
            for (int lay = static_cast<int>(layers) - 1; lay >= 0; --lay) {
                real_t t = tau(gp, lay, col);
                current_dir *= std::exp(-t / mu0);
            }
            flux_dir(gp, col) = current_dir;
        }
    }
}

void SolverSw::solve_sw_2stream(
    ConstView3D tau,
    ConstView3D ssa,
    ConstView3D g,
    ConstView1D solar_zenith_angle,
    ConstView2D sfc_albedo,
    ConstView1D toa_flux,
    View2D flux_up,
    View2D flux_dn,
    View2D flux_dir
) {
    const size_t gp_cnt = tau.extent(0);
    const size_t layers = tau.extent(1);
    const size_t columns = tau.extent(2);

    // Validate boundaries
    validate_extent(ssa, 0, gp_cnt, "ssa");
    validate_extent(ssa, 1, layers, "ssa");
    validate_extent(ssa, 2, columns, "ssa");
    validate_extent(g, 0, gp_cnt, "g");
    validate_extent(g, 1, layers, "g");
    validate_extent(g, 2, columns, "g");
    validate_extent(solar_zenith_angle, 0, columns, "solar_zenith_angle");
    validate_extent(sfc_albedo, 0, gp_cnt, "sfc_albedo");
    validate_extent(sfc_albedo, 1, columns, "sfc_albedo");
    validate_extent(toa_flux, 0, gp_cnt, "toa_flux");
    validate_extent(flux_up, 0, gp_cnt, "flux_up");
    validate_extent(flux_up, 1, columns, "flux_up");
    validate_extent(flux_dn, 0, gp_cnt, "flux_dn");
    validate_extent(flux_dn, 1, columns, "flux_dn");

    // 1. Solve direct beam extinction first
    solve_sw_noscat(tau, solar_zenith_angle, toa_flux, flux_dir);

    // Allocate temporary local storage buffers once at the API boundary
    // to prevent allocations in inner computational loops.
    // Reflectance/Transmittance (R/T) are 3D views (gpoints, layers, columns)
    std::vector<real_t> R_dir_buf(gp_cnt * layers * columns, 0.0);
    std::vector<real_t> T_dir_buf(gp_cnt * layers * columns, 0.0);
    std::vector<real_t> R_dif_buf(gp_cnt * layers * columns, 0.0);
    std::vector<real_t> T_dif_buf(gp_cnt * layers * columns, 0.0);
    std::vector<real_t> src_up_buf(gp_cnt * layers * columns, 0.0);
    std::vector<real_t> src_dn_buf(gp_cnt * layers * columns, 0.0);

    View3D R_dir(R_dir_buf.data(), Extents3D(gp_cnt, layers, columns));
    View3D T_dir(T_dir_buf.data(), Extents3D(gp_cnt, layers, columns));
    View3D R_dif(R_dif_buf.data(), Extents3D(gp_cnt, layers, columns));
    View3D T_dif(T_dif_buf.data(), Extents3D(gp_cnt, layers, columns));
    View3D source_up(src_up_buf.data(), Extents3D(gp_cnt, layers, columns));
    View3D source_dn(src_dn_buf.data(), Extents3D(gp_cnt, layers, columns));

    // 2. Invoke static physics kernels under rte::kernels
    kernels::sw_dif_and_source(
        tau, ssa, g, solar_zenith_angle, 
        R_dir, T_dir, R_dif, T_dif, source_up, source_dn
    );

    kernels::adding_doubling(
        R_dif, T_dif, source_up, source_dn, sfc_albedo,
        flux_up, flux_dn
    );
}

} // namespace rte
