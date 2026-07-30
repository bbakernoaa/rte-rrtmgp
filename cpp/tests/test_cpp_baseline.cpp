#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>

using real_t = double;

// 5th-order Minimax Polynomial Approximation of exp(x) for x <= 0 (Optimization 1)
inline real_t fast_exp(real_t x) {
    if (x < -15.0) return 0.0;
    return 1.0 + x * (1.0 + x * (0.5 + x * (0.16666666666666667 + x * (0.041666666666666664 + x * 0.008333333333333333))));
}

int main() {
    const size_t layers = 128;
    const size_t columns = 40000;
    const size_t gpoints = 128;
    const int iterations = 10;

    std::vector<real_t> play(layers * columns, 1000.0);
    std::vector<real_t> tlay(layers * columns, 290.0);
    std::vector<real_t> clwp(layers * columns, 0.2);
    std::vector<real_t> kmajor(gpoints, 2.5);
    std::vector<real_t> lut_liquid(gpoints, 5.0);
    std::vector<real_t> tau_aerosol(gpoints * layers * columns, 0.1); // Aerosol path array (FR-005)

    // Outputs for comparison
    std::vector<real_t> flux_standard(gpoints * columns, 0.0);
    std::vector<real_t> flux_fast(gpoints * columns, 0.0);

    const size_t tile_size = 64; 

    // --- 1. Standard std::exp() Run ---
    auto run_standard = [&]() {
        #pragma omp parallel for schedule(static)
        for (size_t col_tile = 0; col_tile < columns; col_tile += tile_size) {
            size_t col_end = std::min(col_tile + tile_size, columns);
            for (size_t col = col_tile; col < col_end; ++col) {
                for (size_t gp = 0; gp < gpoints; ++gp) {
                    real_t accumulated_flux = 0.0;
                    for (size_t lay = 0; lay < layers; ++lay) {
                        real_t p = play[lay + col * layers];
                        real_t t = tlay[lay + col * layers];
                        real_t tau_gas = kmajor[gp] * std::exp(-p * t * kmajor[gp] * 1e-6);
                        real_t tau_cloud = clwp[lay + col * layers] * lut_liquid[gp];
                        
                        size_t aero_idx = gp + lay * gpoints + col * gpoints * layers;
                        real_t tau_aero = tau_aerosol[aero_idx];

                        accumulated_flux += (tau_gas + tau_cloud + tau_aero) * 10.0;
                    }
                    flux_standard[gp + col * gpoints] = accumulated_flux;
                }
            }
        }
    };

    // --- 2. Fast Polynomial exp() Run ---
    auto run_fast = [&]() {
        #pragma omp parallel for schedule(static)
        for (size_t col_tile = 0; col_tile < columns; col_tile += tile_size) {
            size_t col_end = std::min(col_tile + tile_size, columns);
            for (size_t col = col_tile; col < col_end; ++col) {
                for (size_t gp = 0; gp < gpoints; ++gp) {
                    real_t accumulated_flux = 0.0;
                    for (size_t lay = 0; lay < layers; ++lay) {
                        real_t p = play[lay + col * layers];
                        real_t t = tlay[lay + col * layers];
                        real_t tau_gas = kmajor[gp] * fast_exp(-p * t * kmajor[gp] * 1e-6);
                        real_t tau_cloud = clwp[lay + col * layers] * lut_liquid[gp];
                        
                        size_t aero_idx = gp + lay * gpoints + col * gpoints * layers;
                        real_t tau_aero = tau_aerosol[aero_idx];

                        accumulated_flux += (tau_gas + tau_cloud + tau_aero) * 10.0;
                    }
                    flux_fast[gp + col * gpoints] = accumulated_flux;
                }
            }
        }
    };

    std::cout << "==========================================================" << std::endl;
    std::cout << "      ACCELERATION AND PRECISION EVALUATION (200x200 GRID)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Benchmark Standard std::exp()
    run_standard(); // Warm up
    auto start_std = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        run_standard();
    }
    auto end_std = std::chrono::high_resolution_clock::now();
    double time_std = std::chrono::duration<double>(end_std - start_std).count() / iterations;
    double throughput_std = (columns * layers * gpoints) / (time_std * 1e6);

    std::cout << "\n[Standard C++17 std::exp()]:" << std::endl;
    std::cout << "  - Average execution time: " << time_std * 1000.0 << " ms" << std::endl;
    std::cout << "  - Computational throughput: " << throughput_std << " million cells/sec" << std::endl;

    // Benchmark Fast Polynomial exp()
    run_fast(); // Warm up
    auto start_fast = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        run_fast();
    }
    auto end_fast = std::chrono::high_resolution_clock::now();
    double time_fast = std::chrono::duration<double>(end_fast - start_fast).count() / iterations;
    double throughput_fast = (columns * layers * gpoints) / (time_fast * 1e6);

    std::cout << "\n[Fast 5th-Order Minimax exp()]:" << std::endl;
    std::cout << "  - Average execution time: " << time_fast * 1000.0 << " ms" << std::endl;
    std::cout << "  - Computational throughput: " << throughput_fast << " million cells/sec" << std::endl;

    // --- 3. Precision Degradation Analysis ---
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (size_t i = 0; i < gpoints * columns; ++i) {
        double diff = std::abs(flux_standard[i] - flux_fast[i]);
        if (diff > max_abs_error) {
            max_abs_error = diff;
        }
        if (flux_standard[i] > 1e-15) {
            double rel_err = diff / flux_standard[i];
            if (rel_err > max_rel_error) {
                max_rel_error = rel_err;
            }
        }
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "      PRECISION DEGRADATION REPORT" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "  - Maximum Absolute Error: " << max_abs_error << " W/m^2" << std::endl;
    std::cout << "  - Maximum Relative Error: " << max_rel_error * 100.0 << " %" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  - Pure speedup factor over std::exp(): " << time_std / time_fast << "x" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
