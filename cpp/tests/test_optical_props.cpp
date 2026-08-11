#include "mo_gas_optics.h"
#include <iostream>
#include <stdexcept>

using namespace rrtmgp;

int main() {
    std::cout << "Running test_optical_props..." << std::endl;

    // Create spectral properties inputs
    std::vector<int> gpoint_to_band_data = {0, 0, 1, 1, 2}; // 5 gpoints, 3 bands
    std::vector<int> band_lims_data = {0, 2, 4,
                                       1, 3, 4}; // 2 rows, 3 bands

    auto gpoint_to_band_view = IntView1D(gpoint_to_band_data.data(), Extents1D(5));
    auto band_lims_view = IntView2D(band_lims_data.data(), Extents2D(2, 3));

    try {
        // Instantiate OpticalProps
        OpticalProps props(gpoint_to_band_view, band_lims_view);

        // Verify boundaries match
        if (props.get_gpoints() != 5 || props.get_bands() != 3) {
            std::cerr << "test_optical_props FAIL: Mismatched gpoints/bands" << std::endl;
            return 1;
        }

        std::cout << "test_optical_props: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_optical_props FAIL: " << e.what() << std::endl;
        return 1;
    }
}
