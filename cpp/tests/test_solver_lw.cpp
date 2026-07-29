#include "mo_rte_lw.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>

using namespace rte;

int main() {
    std::cout << "Running test_solver_lw..." << std::endl;

    const size_t gpoints = 2;
    const size_t layers = 2;
    const size_t columns = 1;

    // Create inputs (column-major layouts)
    // gp index varies fastest, then lay, then col
    std::vector<real_t> tau_data = {1.0, 1.0,  // gp 0-1, lay 0, col 0
                                    0.5, 0.5}; // gp 0-1, lay 1, col 0
    std::vector<real_t> source_data = {10.0, 10.0, // gp 0-1, lay 0, col 0
                                       20.0, 20.0}; // gp 0-1, lay 1, col 0

    auto tau_view = ConstView3D(tau_data.data(), Extents3D(gpoints, layers, columns));
    auto source_view = ConstView3D(source_data.data(), Extents3D(gpoints, layers, columns));

    // Outputs
    std::vector<real_t> flux_up_data(gpoints * columns, 0.0);
    std::vector<real_t> flux_dn_data(gpoints * columns, 0.0);

    auto flux_up_view = View2D(flux_up_data.data(), Extents2D(gpoints, columns));
    auto flux_dn_view = View2D(flux_dn_data.data(), Extents2D(gpoints, columns));

    try {
        SolverLw::solve_lw_noscat(tau_view, source_view, flux_up_view, flux_dn_view);

        // Hand-calculated expected downwelling for gp=0:
        // lay 1 (TOA): emit_1 = 20.0 * (1.0 - exp(-0.5)) = 7.86938
        // lay 0: dn_0 = 7.86938 * exp(-1.0) + 10.0 * (1.0 - exp(-1.0)) = 2.89498 + 6.321205 = 9.21619
        real_t expected_dn_gp0 = 20.0 * (1.0 - std::exp(-0.5)) * std::exp(-1.0) + 10.0 * (1.0 - std::exp(-1.0));
        
        std::cout << "Actual dn: " << std::setprecision(10) << flux_dn_view(0, 0) << std::endl;
        std::cout << "Expected dn: " << std::setprecision(10) << expected_dn_gp0 << std::endl;

        if (std::abs(flux_dn_view(0, 0) - expected_dn_gp0) >= 1e-5) {
            std::cerr << "test_solver_lw FAIL: Mismatched flux values" << std::endl;
            return 1;
        }

        std::cout << "test_solver_lw: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_solver_lw FAIL: " << e.what() << std::endl;
        return 1;
    }
}
