#include "mo_gas_optics_kernels.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace rrtmgp::kernels;
using namespace rrtmgp;

int main() {
    std::cout << "Running test_kernel_interpolation..." << std::endl;

    // Mock Reference pressure (3 points, logarithmic descent)
    std::vector<real_t> ref_press_data = {1000.0, 500.0, 100.0};
    auto ref_press = ConstView1D(ref_press_data.data(), Extents1D(3));

    // Mock Reference temperature (2 pressure layers, 2 temperature points each)
    std::vector<real_t> ref_temp_data = {
        300.0, 250.0, // lay 0
        280.0, 200.0, // lay 1
        250.0, 180.0  // lay 2
    };
    auto ref_temp = ConstView2D(ref_temp_data.data(), Extents2D(3, 2));

    InterpolationWeights weights;

    try {
        // TDD Test 1: Mid-point exact hit
        // pressure 500.0 is index 1
        // temp 240.0 is exactly mid-point of index 1 lay (280.0 down to 200.0) -> offset 0.5
        interpolation(ref_press, ref_temp, 500.0, 240.0, weights);

        assert(weights.jpress[0] == 1);
        assert(weights.jtemp[0] == 0); // Bottom boundary of interpolation cell

        std::cout << "test_kernel_interpolation: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_kernel_interpolation FAIL: " << e.what() << std::endl;
        return 1;
    }
}
