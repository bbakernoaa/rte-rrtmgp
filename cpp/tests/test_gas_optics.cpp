#include "mo_gas_optics.h"
#include <iostream>
#include <stdexcept>

using namespace rrtmgp;

int main() {
    std::cout << "Running test_gas_optics..." << std::endl;

    // Create spectral properties
    std::vector<int> gpoint_to_band_data = {0, 0, 1}; // 3 gpoints, 2 bands
    std::vector<int> band_lims_data = {0, 2,
                                       1, 2}; // 2 rows, 2 bands

    auto gpoint_to_band_view = IntView1D(gpoint_to_band_data.data(), Extents1D(3));
    auto band_lims_view = IntView2D(band_lims_data.data(), Extents2D(2, 2));

    OpticalProps props(gpoint_to_band_view, band_lims_view);

    // Mock coefficient data
    // kmajor dimensions: (gpoints=3, temp=2, press=2, species=1) -> 12 elements
    std::vector<real_t> kmajor_data(12, 1.5);
    // kminor dimensions: (gpoints=3, temp=2, species=1) -> 6 elements
    std::vector<real_t> kminor_data(6, 0.5);

    auto kmajor_view = ConstView4D(kmajor_data.data(), Extents4D(3, 2, 2, 1));
    auto kminor_view = ConstView3D(kminor_data.data(), Extents3D(3, 2, 1));

    try {
        // Instantiate GasOptics
        GasOptics optics(props, kmajor_view, kminor_view);

        // Verify bounds & extents
        if (optics.get_gpoints() != 3 || optics.get_bands() != 2) {
            std::cerr << "test_gas_optics FAIL: Mismatched gpoints/bands" << std::endl;
            return 1;
        }

        // Test boundary size validation failure
        std::vector<real_t> invalid_kmajor_data(8, 1.0);
        auto invalid_kmajor_view = ConstView4D(invalid_kmajor_data.data(), Extents4D(2, 2, 2, 1)); // gpoints=2 (expected 3)

        try {
            GasOptics invalid_optics(props, invalid_kmajor_view, kminor_view);
            std::cerr << "test_gas_optics FAIL: Expected invalid_argument exception on dimension mismatch but none thrown!" << std::endl;
            return 1;
        } catch (const std::invalid_argument& e) {
            std::cout << "test_gas_optics: Caught expected dimension mismatch exception: " << e.what() << std::endl;
        }

        std::cout << "test_gas_optics: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_gas_optics FAIL: " << e.what() << std::endl;
        return 1;
    }
}
