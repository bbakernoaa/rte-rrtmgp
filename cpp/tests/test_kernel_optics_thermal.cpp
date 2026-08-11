#include "mo_gas_optics_kernels.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace rrtmgp::kernels;
using namespace rrtmgp;

int main() {
    std::cout << "Running test_kernel_optics_thermal..." << std::endl;

    const size_t gpoints = 2;

    // Mock coefficients
    std::vector<real_t> krayleigh_data(gpoints * 2 * 1, 0.05); // 0.05 everywhere
    std::vector<real_t> planck_data(gpoints * 2 * 1, 15.0);

    auto krayleigh = ConstView3D(krayleigh_data.data(), Extents3D(gpoints, 2, 1));
    auto planck_fraction = ConstView3D(planck_data.data(), Extents3D(gpoints, 2, 1));

    InterpolationWeights weights;
    weights.jtemp[0] = 0;
    weights.jtemp[1] = 1;

    std::vector<real_t> tau_rayleigh_data(gpoints, 0.0);
    auto tau_rayleigh = View1D(tau_rayleigh_data.data(), Extents1D(gpoints));

    std::vector<real_t> planck_source_data(gpoints, 0.0);
    auto planck_source = View1D(planck_source_data.data(), Extents1D(gpoints));

    try {
        compute_tau_rayleigh(weights, krayleigh, tau_rayleigh);
        compute_Planck_source(weights, planck_fraction, planck_source);

        assert(std::abs(tau_rayleigh(0) - 0.05) < 1e-10);
        assert(std::abs(planck_source(0) - 15.0) < 1e-10);

        std::cout << "test_kernel_optics_thermal: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_kernel_optics_thermal FAIL: " << e.what() << std::endl;
        return 1;
    }
}
