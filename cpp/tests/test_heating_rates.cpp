#include "mo_rte_extensions.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rte::extensions;
using namespace rrtmgp;

int main() {
    std::cout << "Running test_heating_rates..." << std::endl;

    const size_t layers = 1;
    const size_t columns = 1;

    // Boundary Levels = layers + 1 = 2 levels (Lower and Upper)
    std::vector<real_t> flux_up_data = {100.0, 101.0}; // Upwelling flux (Level 0 and 1) -> net increase of 1 W/m^2
    std::vector<real_t> flux_dn_data = {50.0, 50.0};   // Flat downwelling
    std::vector<real_t> p_level_data = {1000.0 * 100.0, 800.0 * 100.0}; // Pressure thickness delta p = 200 hPa = 20000 Pa

    auto flux_up = ConstView2D(flux_up_data.data(), Extents2D(2, columns));
    auto flux_dn = ConstView2D(flux_dn_data.data(), Extents2D(2, columns));
    auto p_level = ConstView2D(p_level_data.data(), Extents2D(2, columns));

    // Outputs
    std::vector<real_t> heating_data(layers * columns, 0.0);
    auto heating_view = View2D(heating_data.data(), Extents2D(layers, columns));

    try {
        compute_heating_rates(flux_up, flux_dn, p_level, heating_view);

        // Analytical solution:
        // net_low = 100.0 - 50.0 = 50.0
        // net_high = 101.0 - 50.0 = 51.0
        // dF = 50.0 - 51.0 = -1.0 W/m^2
        // dp = 20000.0 Pa
        // dT/dt = (9.80665 / 1004.6) * (-1.0 / 20000.0) * 86400 = -0.0421689 K/day
        real_t expected_rate = (gravity / cp_dry_air) * (-1.0 / 20000.0) * seconds_per_day;

        std::cout << "Computed heating rate: " << heating_view(0, 0) << std::endl;
        std::cout << "Expected heating rate: " << expected_rate << std::endl;

        if (std::abs(heating_view(0, 0) - expected_rate) >= 1e-6) {
            std::cerr << "test_heating_rates FAIL: Mismatched temperature trend!" << std::endl;
            return 1;
        }

        std::cout << "test_heating_rates: SUCCESS (GREEN phase verified)" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_heating_rates FAIL with exception: " << e.what() << std::endl;
        return 1;
    }
}
