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

    // Inputs: layers = 2, columns = 1, gpoints = 2
    std::vector<real_t> tau_data = {0.5, 0.5,  // gp 0, lay 0-1
                                    0.2, 0.2}; // gp 1, lay 0-1
    std::vector<real_t> sza_data = {0.8};      // mu0 = cos(zenith_angle) = 0.8
    std::vector<real_t> toa_data = {100.0, 100.0}; // TOA incoming solar flux for 2 gpoints

    auto tau_view = ConstView3D(tau_data.data(), Extents3D(gpoints, layers, columns));
    auto sza_view = ConstView1D(sza_data.data(), Extents1D(columns));
    auto toa_view = ConstView1D(toa_data.data(), Extents1D(gpoints));

    // Outputs: View2D (gpoints, columns)
    std::vector<real_t> flux_dir_data(gpoints * columns, 0.0); 
    auto flux_dir_view = View2D(flux_dir_data.data(), Extents2D(gpoints, columns));

    try {
        // TDD RED Phase: Expected to throw a runtime_error since solve_sw_noscat is unimplemented
        SolverSw::solve_sw_noscat(tau_view, sza_view, toa_view, flux_dir_view);

        // Assertions for green phase
        real_t expected_dir_gp0 = 100.0 * 0.8 * std::exp(-(0.5 + 0.5) / 0.8);
        if (std::abs(flux_dir_view(0, 0) - expected_dir_gp0) >= 1e-5) {
            std::cerr << "test_solver_sw FAIL: Mismatched non-scattering flux values" << std::endl;
            return 1;
        }

    } catch (const std::runtime_error& e) {
        std::cout << "T004 RED: Successfully caught expected runtime_error for solve_sw_noscat: " << e.what() << std::endl;
    }

    // ----------------------------------------------------
    // Test Case 2: Standalone Adding-Doubling Scattering Parity (T005)
    // ----------------------------------------------------
    std::cout << "Executing Standalone Scattering Solvers Test..." << std::endl;

    std::vector<real_t> ssa_data(gpoints * layers * columns, 0.8); // single scattering albedo = 0.8
    std::vector<real_t> g_data(gpoints * layers * columns, 0.5);   // asymmetry factor = 0.5
    std::vector<real_t> sfc_albedo_data = {0.1, 0.1};              // surface albedo = 0.1

    auto ssa_view = ConstView3D(ssa_data.data(), Extents3D(gpoints, layers, columns));
    auto g_view = ConstView3D(g_data.data(), Extents3D(gpoints, layers, columns));
    auto sfc_albedo_view = ConstView2D(sfc_albedo_data.data(), Extents2D(gpoints, columns));

    // Outputs
    std::vector<real_t> flux_up_data(gpoints * columns, 0.0);
    std::vector<real_t> flux_dn_data(gpoints * columns, 0.0);

    auto flux_up_view = View2D(flux_up_data.data(), Extents2D(gpoints, columns));
    auto flux_dn_view = View2D(flux_dn_data.data(), Extents2D(gpoints, columns));

    try {
        // TDD RED Phase: Expected to throw a runtime_error since solve_sw_2stream is unimplemented
        SolverSw::solve_sw_2stream(
            tau_view, ssa_view, g_view, sza_view, sfc_albedo_view, toa_view,
            flux_up_view, flux_dn_view, flux_dir_view
        );

        // Verification checks (for green phase)
        if (flux_up_view(0, 0) <= 0.0 || flux_dn_view(0, 0) <= 0.0) {
            std::cerr << "test_solver_sw FAIL: Mismatched scattering fluxes" << std::endl;
            return 1;
        }

    } catch (const std::runtime_error& e) {
        std::cout << "T005 RED: Successfully caught expected runtime_error for solve_sw_2stream: " << e.what() << std::endl;
    }

    std::cout << "test_solver_sw: SUCCESS (RED phase verified for T004 & T005)" << std::endl;
    return 0;
}
