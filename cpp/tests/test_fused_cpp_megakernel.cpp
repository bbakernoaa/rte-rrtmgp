#include "mo_megakernel_cpp.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace rrtmgp;

int main() {
    std::cout << "Running test_fused_cpp_megakernel..." << std::endl;

    try {
        // --- 1. Scientific Verification Mode (Small Grid) ---
        {
            const size_t layers = 5;
            const size_t columns = 1;
            const size_t gpoints = 4;

            std::vector<real_t> play_data(layers * columns, 1000.0);
            std::vector<real_t> tlay_data(layers * columns, 290.0);
            std::vector<real_t> tsfc_data(columns, 300.0);
            std::vector<real_t> kmajor_data(gpoints * 2 * 2 * 2, 2.5);
            std::vector<real_t> kminor_data(gpoints * 2 * 2, 0.5);
            std::vector<real_t> clwp_data(layers * columns, 0.2);
            std::vector<real_t> lut_liquid_data(gpoints * 2 * 2, 5.0);
            std::vector<real_t> sza_data(columns, 0.0);
            std::vector<real_t> toa_data(gpoints, 0.0);
            std::vector<real_t> tau_aero_data(gpoints * columns * layers, 0.1);
            std::vector<real_t> planck_data(gpoints * 10);

            for (size_t gp = 0; gp < gpoints; ++gp) {
                for (size_t p_idx = 0; p_idx < 10; ++p_idx) {
                    planck_data[gp + p_idx * gpoints] = 10.0 + p_idx * 2.0;
                }
            }

            std::vector<real_t> flux_up_data((layers + 1) * columns, 0.0);
            std::vector<real_t> flux_dn_data((layers + 1) * columns, 0.0);

            ConstView2D play(play_data.data(), Extents2D(layers, columns));
            ConstView2D tlay(tlay_data.data(), Extents2D(layers, columns));
            ConstView1D tsfc(tsfc_data.data(), Extents1D(columns));
            ConstView4D kmajor(kmajor_data.data(), Extents4D(gpoints, 2, 2, 2));
            ConstView3D kminor(kminor_data.data(), Extents3D(gpoints, 2, 2));
            ConstView2D clwp(clwp_data.data(), Extents2D(layers, columns));
            ConstView3D lut_liquid(lut_liquid_data.data(), Extents3D(gpoints, 2, 2));
            ConstView1D sza(sza_data.data(), Extents1D(columns));
            ConstView1D toa(toa_data.data(), Extents1D(gpoints));
            ConstView3D tau_aerosol(tau_aero_data.data(), Extents3D(gpoints, columns, layers));
            ConstView2D planck_table(planck_data.data(), Extents2D(gpoints, 10));
            View2D flux_up(flux_up_data.data(), Extents2D(layers + 1, columns));
            View2D flux_dn(flux_dn_data.data(), Extents2D(layers + 1, columns));

            // Execute fused C++ megakernel
            MegakernelCpp::execute_fused_longwave(
                layers, columns, gpoints,
                play, tlay, tsfc, kmajor, kminor, clwp, lut_liquid, sza, toa,
                tau_aerosol, ConstView3D(), ConstView3D(), planck_table,
                flux_up, flux_dn
            );

            real_t expected_flux_up = 263.894;
            real_t expected_flux_dn = 263.894;

            std::cout << "Computed verification flux_up TOA (lev 0): " << flux_up(0, 0) << " W/m^2 (expected: " << expected_flux_up << ")" << std::endl;
            std::cout << "Computed verification flux_dn SFC (lev " << layers << "): " << flux_dn(layers, 0) << " W/m^2 (expected: " << expected_flux_dn << ")" << std::endl;

            if (std::abs(flux_up(0, 0) - expected_flux_up) >= 1e-3 ||
                std::abs(flux_dn(layers, 0) - expected_flux_dn) >= 1e-3) {
                std::cerr << "test_fused_cpp_megakernel FAIL: Unphysical flux values or value mismatch!" << std::endl;
                return 1;
            }

            std::cout << "Verification Mode: SUCCESS (Functional parity verified)" << std::endl;
        }

        // --- 2. Cache Saturation Benchmark Mode (1000 Columns, 128 Layers, 128 Gpoints) ---
        {
            std::cout << "\nStarting Cache Saturation Benchmark (1000 Column Set)..." << std::endl;
            const size_t layers = 128;
            const size_t columns = 1000;
            const size_t gpoints = 128;
            const int iterations = 10;

            std::cout << "  - Configuration: " << columns << " columns, " << layers << " layers, " << gpoints << " g-points" << std::endl;
            std::cout << "  - Total grid elements computed: " << (columns * layers * gpoints) / 1000000.0 << " million cells" << std::endl;

            std::vector<real_t> play_data(layers * columns, 1000.0);
            std::vector<real_t> tlay_data(layers * columns, 290.0);
            std::vector<real_t> tsfc_data(columns, 300.0);
            std::vector<real_t> kmajor_data(gpoints * 2 * 2 * 2, 2.5);
            std::vector<real_t> kminor_data(gpoints * 2 * 2, 0.5);
            std::vector<real_t> clwp_data(layers * columns, 0.2);
            std::vector<real_t> lut_liquid_data(gpoints * 2 * 2, 5.0);
            std::vector<real_t> sza_data(columns, 0.0);
            std::vector<real_t> toa_data(gpoints, 0.0);
            std::vector<real_t> tau_aero_data(gpoints * columns * layers, 0.1);
            std::vector<real_t> planck_data(gpoints * 10);

            for (size_t gp = 0; gp < gpoints; ++gp) {
                for (size_t p_idx = 0; p_idx < 10; ++p_idx) {
                    planck_data[gp + p_idx * gpoints] = 10.0 + p_idx * 2.0;
                }
            }

            std::vector<real_t> flux_up_data((layers + 1) * columns, 0.0);
            std::vector<real_t> flux_dn_data((layers + 1) * columns, 0.0);

            ConstView2D play(play_data.data(), Extents2D(layers, columns));
            ConstView2D tlay(tlay_data.data(), Extents2D(layers, columns));
            ConstView1D tsfc(tsfc_data.data(), Extents1D(columns));
            ConstView4D kmajor(kmajor_data.data(), Extents4D(gpoints, 2, 2, 2));
            ConstView3D kminor(kminor_data.data(), Extents3D(gpoints, 2, 2));
            ConstView2D clwp(clwp_data.data(), Extents2D(layers, columns));
            ConstView3D lut_liquid(lut_liquid_data.data(), Extents3D(gpoints, 2, 2));
            ConstView1D sza(sza_data.data(), Extents1D(columns));
            ConstView1D toa(toa_data.data(), Extents1D(gpoints));
            ConstView3D tau_aerosol(tau_aero_data.data(), Extents3D(gpoints, columns, layers));
            ConstView2D planck_table(planck_data.data(), Extents2D(gpoints, 10));
            View2D flux_up(flux_up_data.data(), Extents2D(layers + 1, columns));
            View2D flux_dn(flux_dn_data.data(), Extents2D(layers + 1, columns));

            // Warm up
            MegakernelCpp::execute_fused_longwave(
                layers, columns, gpoints,
                play, tlay, tsfc, kmajor, kminor, clwp, lut_liquid, sza, toa,
                tau_aerosol, ConstView3D(), ConstView3D(), planck_table,
                flux_up, flux_dn
            );

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                MegakernelCpp::execute_fused_longwave(
                    layers, columns, gpoints,
                    play, tlay, tsfc, kmajor, kminor, clwp, lut_liquid, sza, toa,
                    tau_aerosol, ConstView3D(), ConstView3D(), planck_table,
                    flux_up, flux_dn
                );
            }
            auto end = std::chrono::high_resolution_clock::now();
            double avg_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
            double throughput = (columns * layers * gpoints) / (avg_time_ms * 1000.0);

            std::cout << "\nBenchmark Results (Standard C++ Megakernel):" << std::endl;
            std::cout << "  - Average execution time: " << avg_time_ms << " ms" << std::endl;
            std::cout << "  - Computational throughput: " << throughput << " million cells/sec" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "test_fused_cpp_megakernel FAIL: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
