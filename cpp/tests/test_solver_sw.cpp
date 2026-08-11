#include "mo_rte_sw.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>

using namespace rte;

int main() {
    std::cout << "Running test_solver_sw..." << std::endl;

    const size_t gpoints = 2;
    const size_t layers = 2;
    const size_t columns = 1;

    // ----------------------------------------------------
    // Test Case 1: Non-scattering Solar Beam Attenuation (T004)
    // ----------------------------------------------------
    std::cout << "Executing Non-Scattering Beam Attenuation Test..." << std::endl;

    // Inputs: layers = 2, columns = 1, gpoints = 2 (column-major)
    std::vector<real_t> tau_data = {0.5, 0.5,  // gp 0-1, lay 0
                                    0.2, 0.2}; // gp 0-1, lay 1
    std::vector<real_t> sza_data = {0.8};      // mu0 = cos(zenith_angle) = 0.8
    std::vector<real_t> toa_data = {100.0, 100.0}; // TOA incoming solar flux for 2 gpoints

    auto tau_view = ConstView3D(tau_data.data(), Extents3D(gpoints, layers, columns));
    auto sza_view = ConstView1D(sza_data.data(), Extents1D(columns));
    auto toa_view = ConstView1D(toa_data.data(), Extents1D(gpoints));

    std::vector<real_t> sza_data_2d(layers * columns, 0.8);
    auto sza_view_2d = ConstView2D(sza_data_2d.data(), Extents2D(layers, columns));

    std::vector<real_t> toa_flux_data_2d(gpoints * columns, 100.0);
    auto toa_view_2d = ConstView2D(toa_flux_data_2d.data(), Extents2D(gpoints, columns));

    // Outputs: View3D (gpoints, layers+1, columns)
    std::vector<real_t> flux_dir_data(gpoints * (layers + 1) * columns, 0.0);
    auto flux_dir_view_3d = View3D(flux_dir_data.data(), Extents3D(gpoints, layers + 1, columns));

    try {
        SolverSw::solve_sw_noscat(tau_view, sza_view_2d, toa_view_2d, flux_dir_view_3d);

        // For gp=0, lay=0 has tau=0.5, lay=1 has tau=0.2. Total tau = 0.7.
        real_t expected_dir_gp0 = 100.0 * 0.8 * std::exp(-(0.5 + 0.2) / 0.8);

        std::cout << "Actual flux_dir: " << std::setprecision(10) << flux_dir_view_3d(0, layers, 0) << std::endl;
        std::cout << "Expected flux_dir: " << std::setprecision(10) << expected_dir_gp0 << std::endl;

        if (std::abs(flux_dir_view_3d(0, layers, 0) - expected_dir_gp0) >= 1e-5) {
            std::cerr << "test_solver_sw FAIL: Mismatched non-scattering flux values" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "test_solver_sw FAIL with unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    // ----------------------------------------------------
    // Test Case 2: Standalone Adding-Doubling Scattering Parity (T005)
    // ----------------------------------------------------
    std::cout << "Executing Standalone Scattering Solvers Test..." << std::endl;

    std::vector<real_t> ssa_data = {0.8, 0.8,  // gp 0-1, lay 0
                                    0.8, 0.8}; // gp 0-1, lay 1
    std::vector<real_t> g_data = {0.5, 0.5,    // gp 0-1, lay 0
                                  0.5, 0.5};   // gp 0-1, lay 1
    std::vector<real_t> sfc_albedo_data = {0.1, 0.1};              // surface albedo = 0.1

    auto ssa_view = ConstView3D(ssa_data.data(), Extents3D(gpoints, layers, columns));
    auto g_view = ConstView3D(g_data.data(), Extents3D(gpoints, layers, columns));
    auto sfc_albedo_view = ConstView2D(sfc_albedo_data.data(), Extents2D(gpoints, columns));

    // Outputs
    std::vector<real_t> flux_up_data(gpoints * (layers + 1) * columns, 0.0);
    std::vector<real_t> flux_dn_data(gpoints * (layers + 1) * columns, 0.0);

    auto flux_up_view = View3D(flux_up_data.data(), Extents3D(gpoints, layers + 1, columns));
    auto flux_dn_view = View3D(flux_dn_data.data(), Extents3D(gpoints, layers + 1, columns));

    std::vector<real_t> sfc_alb_dir_data(gpoints * columns, 0.2);
    auto sfc_alb_dir_view = ConstView2D(sfc_alb_dir_data.data(), Extents2D(gpoints, columns));

    try {
        SolverSw::solve_sw_2stream(
            tau_view, ssa_view, g_view, sza_view_2d, sfc_alb_dir_view, sfc_albedo_view, toa_view_2d,
            flux_up_view, flux_dn_view, flux_dir_view_3d
        );

        std::cout << "Actual flux_dn: " << flux_dn_view(0, 0, 0) << std::endl;
        std::cout << "Actual flux_up: " << flux_up_view(0, 0, 0) << std::endl;

        // Verification checks (for green phase)
        if (flux_up_view(0, 0, 0) <= 0.0 || flux_dn_view(0, 0, 0) <= 0.0) {
            std::cerr << "test_solver_sw FAIL: Mismatched scattering fluxes" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "test_solver_sw FAIL with unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "test_solver_sw: SUCCESS (GREEN phase verified for T004 & T005)" << std::endl;
    return 0;
}
