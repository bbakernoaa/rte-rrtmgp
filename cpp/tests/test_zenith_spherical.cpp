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
    std::vector<real_t> p_level_data = {1000.0 * 100.0, 500.0 * 100.0, 100.0 * 100.0}; // 3 levels

    auto sza = ConstView1D(sza_data.data(), Extents1D(columns));
    auto p_level = ConstView2D(p_level_data.data(), Extents2D(layers + 1, columns));

    std::vector<real_t> corrected_data(columns, 0.0);
    auto corrected = View1D(corrected_data.data(), Extents1D(columns));

    try {
        // TDD RED Phase: Expected to fail at runtime since zenith_angle_spherical_correction throws unimplemented
        zenith_angle_spherical_correction(sza, p_level, corrected);

        std::cerr << "test_zenith_spherical FAIL: Expected exception not thrown!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "T003 RED: Successfully caught expected runtime_error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_zenith_spherical FAIL with unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "test_zenith_spherical: SUCCESS (RED phase verified)" << std::endl;
    return 0;
}
