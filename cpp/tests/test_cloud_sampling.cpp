#include "mo_rte_extensions.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rte::extensions;
using namespace rrtmgp;

int main() {
    std::cout << "Running test_cloud_sampling..." << std::endl;

    const size_t layers = 2;
    const size_t columns = 1;
    const size_t sub_columns = 100; // Increased to 100 to get a stable statistical mean

    // Cloud fraction: 50% in layer 0, 100% in layer 1
    std::vector<real_t> cloud_frac_data = {0.5, 1.0};
    auto cloud_frac = ConstView2D(cloud_frac_data.data(), Extents2D(layers, columns));

    // Output mask: (layers, columns * sub_columns) -> (2, 100)
    std::vector<int> mask_data(layers * columns * sub_columns, 0);
    auto mask = IntViewMut2D(mask_data.data(), Extents2D(layers, columns * sub_columns));

    try {
        execute_mcica_sampling(cloud_frac, sub_columns, mask, 54321);

        // 1. Layer 1 should be 100% cloudy (all 100 sub-columns must be 1)
        for (size_t sub = 0; sub < sub_columns; ++sub) {
            if (mask(1, sub) != 1) {
                std::cerr << "test_cloud_sampling FAIL: Layer 1 is not 100% cloudy at sub-column " << sub << std::endl;
                return 1;
            }
        }

        // 2. Layer 0 should be statistically ~50% cloudy
        size_t cloud_cnt = 0;
        for (size_t sub = 0; sub < sub_columns; ++sub) {
            if (mask(0, sub) == 1) {
                cloud_cnt++;
            }
        }

        std::cout << "Layer 0 clouds count out of 100: " << cloud_cnt << std::endl;

        // Expect cloud count to be within normal statistical bounds (e.g. [35, 65] for 100 draws with p=0.5)
        if (cloud_cnt < 35 || cloud_cnt > 65) {
            std::cerr << "test_cloud_sampling FAIL: Out of normal statistical bounds [35, 65]!" << std::endl;
            return 1;
        }

        std::cout << "test_cloud_sampling: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cloud_sampling FAIL with exception: " << e.what() << std::endl;
        return 1;
    }
}
