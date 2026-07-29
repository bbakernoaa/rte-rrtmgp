#include "mo_rte_sw.h"
#include <stdexcept>

namespace rte {

void SolverSw::solve_sw_noscat(
    ConstView3D tau,
    ConstView1D solar_zenith_angle,
    ConstView1D toa_flux,
    View2D flux_dir
) {
    (void)tau;
    (void)solar_zenith_angle;
    (void)toa_flux;
    (void)flux_dir;
    throw std::runtime_error("SolverSw::solve_sw_noscat is not implemented yet");
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
    (void)tau;
    (void)ssa;
    (void)g;
    (void)solar_zenith_angle;
    (void)sfc_albedo;
    (void)toa_flux;
    (void)flux_up;
    (void)flux_dn;
    (void)flux_dir;
    throw std::runtime_error("SolverSw::solve_sw_2stream is not implemented yet");
}

} // namespace rte
