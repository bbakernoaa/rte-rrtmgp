#include "mo_rte_sw.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rte;

int main() {
    std::cout << "Running test_solver_sw..." << std::endl;

    const size_t gpoints = 2;
    const size_t layers = 2;
    const size_t columns = 1;

    // Create mock input arrays
    std::vector<real_t> tau_data(gpoints * layers * columns, 0.5);
    std::vector<real_t> sza_data = {0.5}; // mu0 = cos(zenith_angle) = 0.5
    std::vector<real_t> toa_data = {100.0, 100.0}; // TOA incoming solar flux for 2 gpoints

    auto tau_view = ConstView3D(tau_data.data(), Extents3D(gpoints, layers, columns));
    auto sza_view = ConstView1D(sza_data.data(), Extents1D(columns));
    auto toa_view = ConstView1D(toa_data.data(), Extents1D(gpoints));

    // Outputs
    std::vector<real_t> flux_dir_data(gpoints * columns, 0.0);
    auto flux_dir_view = View2D(flux_dir_data.data(), Extents2D(gpoints, columns));

    try {
        // TDD: This call is expected to fail at runtime with a runtime_error (RED state)
        SolverSw::solve_sw_noscat(tau_view, sza_view, toa_view, flux_dir_view);

        std::cerr << "test_solver_sw FAIL: solve_sw_noscat did not throw runtime_error!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "T003/T004 RED: Successfully caught expected runtime_error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_solver_sw FAIL with unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "test_solver_sw: SUCCESS (RED phase verified)" << std::endl;
    return 0;
}
