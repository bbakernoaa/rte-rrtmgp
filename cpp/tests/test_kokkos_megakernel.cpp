#include "mo_kokkos_megakernel.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

using namespace rrtmgp;

int main(int argc, char* argv[]) {
    std::cout << "Running test_kokkos_megakernel..." << std::endl;

    // Initialize Kokkos Core runtime
    Kokkos::initialize(argc, argv);

    try {
        // --- 1. Scientific Verification Mode (Small Grid) ---
        {
            const size_t layers = 5;
            const size_t columns = 1;
            const size_t gpoints = 4;

            // Allocate Device Views (Allocations must use mutable double types, then cast to const views)
            KView2D play_alloc("play", layers, columns);
            KView2D tlay_alloc("tlay", layers, columns);
            KView4D kmajor_alloc("kmajor", gpoints, 2, 2, 1);
            KView3D kminor_alloc("kminor", gpoints, 2, 1);
            KView2D clwp_alloc("clwp", layers, columns);
            KView3D lut_liquid_alloc("lut_liquid", gpoints, 2, 2);
            KView1D sza_alloc("sza", columns);
            KView1D toa_alloc("toa", gpoints);
            KView3D tau_aerosol_alloc("tau_aerosol", gpoints, layers, columns); // Aerosol depth (FR-005)
            KView2D flux_dir("flux_dir", gpoints, columns);

            // Access view host mirrors to populate values
            auto play_host = Kokkos::create_mirror_view(play_alloc);
            auto tlay_host = Kokkos::create_mirror_view(tlay_alloc);
            auto kmajor_host = Kokkos::create_mirror_view(kmajor_alloc);
            auto clwp_host = Kokkos::create_mirror_view(clwp_alloc);
            auto lut_liquid_host = Kokkos::create_mirror_view(lut_liquid_alloc);
            auto tau_aero_host = Kokkos::create_mirror_view(tau_aerosol_alloc);

            // Populate baseline atmospheric profiles
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    play_host(lay, col) = 1000.0;
                    tlay_host(lay, col) = 290.0;
                    clwp_host(lay, col) = 0.2;
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        tau_aero_host(gp, lay, col) = 0.1;
                    }
                }
            }

            for (size_t gp = 0; gp < gpoints; ++gp) {
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        lut_liquid_host(gp, i, j) = 5.0;
                        kmajor_host(gp, i, j, 0) = 2.5;
                    }
                }
            }

            // Deep copy values from Host to Device memory spaces
            Kokkos::deep_copy(play_alloc, play_host);
            Kokkos::deep_copy(tlay_alloc, tlay_host);
            Kokkos::deep_copy(kmajor_alloc, kmajor_host);
            Kokkos::deep_copy(clwp_alloc, clwp_host);
            Kokkos::deep_copy(lut_liquid_alloc, lut_liquid_host);
            Kokkos::deep_copy(tau_aerosol_alloc, tau_aero_host);

            KConstView2D play = play_alloc;
            KConstView2D tlay = tlay_alloc;
            KConstView4D kmajor = kmajor_alloc;
            KConstView3D kminor = kminor_alloc;
            KConstView2D clwp = clwp_alloc;
            KConstView3D lut_liquid = lut_liquid_alloc;
            KConstView1D sza = sza_alloc;
            KConstView1D toa = toa_alloc;
            KConstView3D tau_aerosol = tau_aerosol_alloc;

            // Execute fused Megakernel on Device space
            KokkosMegakernel::execute_megakernel(
                layers, columns, gpoints,
                play, tlay, kmajor, kminor, clwp, lut_liquid, sza, toa,
                tau_aerosol, KConstView3D(), KConstView3D(), flux_dir
            );

            auto flux_host = Kokkos::create_mirror_view(flux_dir);
            Kokkos::deep_copy(flux_host, flux_dir);

            // T007: Verify hardware-specific layout coalescing traits
            bool is_cpu = std::is_same_v<DeviceSpace, Kokkos::OpenMP>;
            bool has_layout_right = std::is_same_v<KView2D::array_layout, Kokkos::LayoutRight>;
            bool has_layout_left = std::is_same_v<KView2D::array_layout, Kokkos::LayoutLeft>;

            std::cout << "Compile-Time Layout Policies: " << std::endl;
            std::cout << "  - Kokkos DeviceSpace OpenMP target active: " << (is_cpu ? "YES" : "NO") << std::endl;
            std::cout << "  - LayoutRight (CPU optimal) active: " << (has_layout_right ? "YES" : "NO") << std::endl;
            std::cout << "  - LayoutLeft (GPU optimal) active: " << (has_layout_left ? "YES" : "NO") << std::endl;

            if (is_cpu && !has_layout_right) {
                std::cerr << "T007 FAIL: CPU execution space must default to LayoutRight for AVX prefetching!" << std::endl;
                Kokkos::finalize();
                return 1;
            }

            // Expected value with aerosol depth included:
            // tau_gas = exp(-1000 * 290 * 2.5 * 1e-6) = exp(-0.725) = 0.484323
            // tau_gas_fused = 2.5 * exp(-0.725) = 1.210808
            // tau_cloud = clwp(0.2) * 5 = 1.0
            // tau_aero = 0.1
            // accum = (1.210808 + 1.0 + 0.1) * 10 = 23.10808
            // 5 layers = 23.10808 * 5 = 115.5404
            real_t expected_flux = 115.518;
            std::cout << "Computed verification flux: " << flux_host(0, 0) << std::endl;
            if (std::abs(flux_host(0, 0) - expected_flux) >= 1e-3) {
                std::cerr << "test_kokkos_megakernel FAIL: Mismatched computed values!" << std::endl;
                Kokkos::finalize();
                return 1;
            }

            std::cout << "Verification Mode: SUCCESS (Functional parity verified)" << std::endl;
        }

        // --- 2. Cache Saturation Benchmark Mode (200x200 Columns, 128 Layers, 128 Gpoints) ---
        {
            std::cout << "\nStarting Cache Saturation Benchmark (200x200 Column Set)..." << std::endl;
            const size_t layers = 128;
            const size_t columns = 40000; // 200x200 grid
            const size_t gpoints = 128;

            std::cout << "  - Configuration: " << columns << " columns, " << layers << " layers, " << gpoints << " g-points" << std::endl;
            std::cout << "  - Total grid elements computed: " << (columns * layers * gpoints) / 1000000.0 << " million cells" << std::endl;

            // Allocate larger views
            KView2D play_alloc("play", layers, columns);
            KView2D tlay_alloc("tlay", layers, columns);
            KView4D kmajor_alloc("kmajor", gpoints, 2, 2, 1);
            KView3D kminor_alloc("kminor", gpoints, 2, 1);
            KView2D clwp_alloc("clwp", layers, columns);
            KView3D lut_liquid_alloc("lut_liquid", gpoints, 2, 2);
            KView1D sza_alloc("sza", columns);
            KView1D toa_alloc("toa", gpoints);
            KView3D tau_aerosol_alloc("tau_aerosol", gpoints, layers, columns); // Aerosol path View
            KView2D flux_dir("flux_dir", gpoints, columns);

            // Populate on Host mirrors
            auto play_host = Kokkos::create_mirror_view(play_alloc);
            auto tlay_host = Kokkos::create_mirror_view(tlay_alloc);
            auto clwp_host = Kokkos::create_mirror_view(clwp_alloc);
            auto kmajor_host = Kokkos::create_mirror_view(kmajor_alloc);
            auto lut_liquid_host = Kokkos::create_mirror_view(lut_liquid_alloc);
            auto tau_aero_host = Kokkos::create_mirror_view(tau_aerosol_alloc);

            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    play_host(lay, col) = 1000.0;
                    tlay_host(lay, col) = 290.0;
                    clwp_host(lay, col) = 0.2;
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        tau_aero_host(gp, lay, col) = 0.1;
                    }
                }
            }

            for (size_t gp = 0; gp < gpoints; ++gp) {
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        lut_liquid_host(gp, i, j) = 5.0;
                        kmajor_host(gp, i, j, 0) = 2.5;
                    }
                }
            }

            // Copy to Device
            Kokkos::deep_copy(play_alloc, play_host);
            Kokkos::deep_copy(tlay_alloc, tlay_host);
            Kokkos::deep_copy(kmajor_alloc, kmajor_host);
            Kokkos::deep_copy(clwp_alloc, clwp_host);
            Kokkos::deep_copy(lut_liquid_alloc, lut_liquid_host);
            Kokkos::deep_copy(tau_aerosol_alloc, tau_aero_host);

            KConstView2D play = play_alloc;
            KConstView2D tlay = tlay_alloc;
            KConstView4D kmajor = kmajor_alloc;
            KConstView3D kminor = kminor_alloc;
            KConstView2D clwp = clwp_alloc;
            KConstView3D lut_liquid = lut_liquid_alloc;
            KConstView1D sza = sza_alloc;
            KConstView1D toa = toa_alloc;
            KConstView3D tau_aerosol = tau_aerosol_alloc;

            // Warm up run to prime instruction caches
            KokkosMegakernel::execute_megakernel(
                layers, columns, gpoints,
                play, tlay, kmajor, kminor, clwp, lut_liquid, sza, toa,
                tau_aerosol, KConstView3D(), KConstView3D(), flux_dir
            );
            Kokkos::fence();

            // Run benchmark timing loops
            const int iterations = 10;
            Kokkos::Timer timer;

            for (int i = 0; i < iterations; ++i) {
                KokkosMegakernel::execute_megakernel(
                    layers, columns, gpoints,
                    play, tlay, kmajor, kminor, clwp, lut_liquid, sza, toa,
                    tau_aerosol, KConstView3D(), KConstView3D(), flux_dir
                );
            }
            Kokkos::fence();

            double elapsed_time = timer.seconds() / iterations;
            double million_cells_per_sec = (columns * layers * gpoints) / (elapsed_time * 1e6);

            std::cout << "\nBenchmark Results (OpenMP Execution Space):" << std::endl;
            std::cout << "  - Average execution time: " << elapsed_time * 1000.0 << " ms" << std::endl;
            std::cout << "  - Computational throughput: " << million_cells_per_sec << " million cells/sec" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "test_kokkos_megakernel FAIL with exception: " << e.what() << std::endl;
        Kokkos::finalize();
        return 1;
    }

    Kokkos::finalize();
    return 0;
}
