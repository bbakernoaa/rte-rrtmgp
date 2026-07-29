#include "mo_cloud_optics.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rrtmgp;

int main() {
    std::cout << "Running test_cloud_optics..." << std::endl;

    // Create base optical props
    std::vector<int> gpoint_to_band_data = {0, 1}; // 2 gpoints, 2 bands
    std::vector<int> band_lims_data = {0, 1,
                                       1, 1}; // 2 rows, 2 bands
    
    auto gpoint_to_band_view = IntView1D(gpoint_to_band_data.data(), Extents1D(2));
    auto band_lims_view = IntView2D(band_lims_data.data(), Extents2D(2, 2));

    OpticalProps props(gpoint_to_band_view, band_lims_view);

    // Mock LUT tables
    std::vector<real_t> lut_liquid_data(2 * 2 * 2, 1.0); // gpoints x radius_sizes x parameter
    std::vector<real_t> lut_ice_data(2 * 2 * 2, 2.0);

    auto lut_liq_view = ConstView3D(lut_liquid_data.data(), Extents3D(2, 2, 2));
    auto lut_ice_view = ConstView3D(lut_ice_data.data(), Extents3D(2, 2, 2));

    try {
        // TDD RED Phase: Expected to fail at runtime with runtime_error since constructor throws
        CloudOptics optics(props, lut_liq_view, lut_ice_view);

        std::cerr << "test_cloud_optics FAIL: Constructor did not throw runtime_error!" << std::endl;
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "T009 RED: Successfully caught expected runtime_error for CloudOptics: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_cloud_optics FAIL with unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "test_cloud_optics: SUCCESS (RED phase verified)" << std::endl;
    return 0;
}
