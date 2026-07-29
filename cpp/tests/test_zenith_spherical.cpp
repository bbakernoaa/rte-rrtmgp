#include "mo_rte_extensions.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rte::extensions;
using namespace rrtmgp;

int main() {
    std::cout << "Running test_zenith_spherical..." << std::endl;

    const size_t columns = 1;
    const size_t layers = 2;

    std::vector<real_t> sza_data = {1.48353}; // 85 degrees in radians (shallow solar angle)
    std::vector<real_t> p_level_data = {1000.0 * 100.0, 500.0 * 100.0, 100.0 * 100.0}; // 3 levels (100000 Pa at surface)

    auto sza = ConstView1D(sza_data.data(), Extents1D(columns));
    auto p_level = ConstView2D(p_level_data.data(), Extents2D(layers + 1, columns));

    std::vector<real_t> corrected_data(columns, 0.0);
    auto corrected = View1D(corrected_data.data(), Extents1D(columns));

    try {
        zenith_angle_spherical_correction(sza, p_level, corrected);

        // Verification checks:
        // angle = 1.48353 -> cos(angle) = 0.0871556
        // p_bottom = 100000.0 Pa
        // correction = 0.05 * (1.0 - 100000.0 / 101325.0) = 0.05 * (1.0 - 0.986923) = 0.0006538
        // expected_mu = 0.0871556 + 0.0006538 = 0.087809
        real_t expected_mu = std::cos(sza_data[0]) + 0.05 * (1.0 - 100000.0 / 101325.0);

        std::cout << "Actual corrected cosine: " << corrected(0) << std::endl;
        std::cout << "Expected corrected cosine: " << expected_mu << std::endl;

        if (std::abs(corrected(0) - expected_mu) >= 1e-6) {
            std::cerr << "test_zenith_spherical FAIL: Cosine angle verification mismatched!" << std::endl;
            return 1;
        }

        std::cout << "test_zenith_spherical: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_zenith_spherical FAIL with exception: " << e.what() << std::endl;
        return 1;
    }
}
