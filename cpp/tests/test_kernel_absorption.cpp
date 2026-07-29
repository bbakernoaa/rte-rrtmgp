#include "mo_gas_optics_kernels.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace rrtmgp::kernels;
using namespace rrtmgp;

int main() {
    std::cout << "Running test_kernel_absorption..." << std::endl;

    const size_t gpoints = 2;
    
    // Mock coefficients
    std::vector<real_t> kmajor_data(gpoints * 2 * 2 * 1, 2.0); // 2.0 everywhere
    std::vector<real_t> kminor_data(gpoints * 2 * 1, 0.5);

    auto kmajor = ConstView4D(kmajor_data.data(), Extents4D(gpoints, 2, 2, 1));
    auto kminor = ConstView3D(kminor_data.data(), Extents3D(gpoints, 2, 1));

    InterpolationWeights weights;
    weights.jpress[0] = 0; weights.jpress[1] = 1;
    weights.jtemp[0] = 0; weights.jtemp[1] = 1;
    weights.ftemp = 0.5;

    std::vector<real_t> tau_data(gpoints, 0.0);
    auto tau = View1D(tau_data.data(), Extents1D(gpoints));

    try {
        compute_tau_absorption(weights, kmajor, kminor, tau);

        // Kmajor is flat 2.0. Interpolation of 2.0 and 2.0 with weight 0.5 is 2.0.
        assert(std::abs(tau(0) - 2.0) < 1e-10);
        assert(std::abs(tau(1) - 2.0) < 1e-10);

        std::cout << "test_kernel_absorption: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_kernel_absorption FAIL: " << e.what() << std::endl;
        return 1;
    }
}
