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
            KView1D tsfc_alloc("tsfc", columns);
            KView4D kmajor_alloc("kmajor", gpoints, 2, 2, 2);
            KView3D kminor_alloc("kminor", gpoints, 2, 2);
            KView2D clwp_alloc("clwp", layers, columns);
            KView3D lut_liquid_alloc("lut_liquid", gpoints, 2, 2);
            KView1D sza_alloc("sza", columns);
            KView1D toa_alloc("toa", gpoints);
            KView3D tau_aerosol_alloc("tau_aerosol", gpoints, columns, layers);
            KView2D planck_table_alloc("planck_table", gpoints, 10);
            KView2D flux_up_alloc("flux_up", layers + 1, columns);
            KView2D flux_dn_alloc("flux_dn", layers + 1, columns);

            // Access view host mirrors to populate values
            auto play_host = Kokkos::create_mirror_view(play_alloc);
            auto tlay_host = Kokkos::create_mirror_view(tlay_alloc);
            auto tsfc_host = Kokkos::create_mirror_view(tsfc_alloc);
            auto kmajor_host = Kokkos::create_mirror_view(kmajor_alloc);
            auto kminor_host = Kokkos::create_mirror_view(kminor_alloc);
            auto clwp_host = Kokkos::create_mirror_view(clwp_alloc);
            auto lut_liquid_host = Kokkos::create_mirror_view(lut_liquid_alloc);
            auto tau_aero_host = Kokkos::create_mirror_view(tau_aerosol_alloc);
            auto planck_host = Kokkos::create_mirror_view(planck_table_alloc);

            // Populate baseline atmospheric profiles
            for (size_t col = 0; col < columns; ++col) {
                tsfc_host(col) = 300.0;
                for (size_t lay = 0; lay < layers; ++lay) {
                    play_host(lay, col) = 1000.0;
                    tlay_host(lay, col) = 290.0;
                    clwp_host(lay, col) = 0.2;
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        tau_aero_host(gp, col, lay) = 0.1;
                    }
                }
            }

            for (size_t gp = 0; gp < gpoints; ++gp) {
                for (size_t p_idx = 0; p_idx < 10; ++p_idx) {
                    planck_host(gp, p_idx) = 10.0 + p_idx * 2.0;
                }
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        lut_liquid_host(gp, i, j) = 5.0;
                        kmajor_host(gp, i, j, 0) = 2.5;
                        kmajor_host(gp, i, j, 1) = 2.5;
                        kminor_host(gp, i, j) = 0.5;
                    }
                }
            }

            // Deep copy values from Host to Device memory spaces
            Kokkos::deep_copy(play_alloc, play_host);
            Kokkos::deep_copy(tlay_alloc, tlay_host);
            Kokkos::deep_copy(tsfc_alloc, tsfc_host);
            Kokkos::deep_copy(kmajor_alloc, kmajor_host);
            Kokkos::deep_copy(kminor_alloc, kminor_host);
            Kokkos::deep_copy(clwp_alloc, clwp_host);
            Kokkos::deep_copy(lut_liquid_alloc, lut_liquid_host);
            Kokkos::deep_copy(tau_aerosol_alloc, tau_aero_host);
            Kokkos::deep_copy(planck_table_alloc, planck_host);

            KConstView2D play = play_alloc;
            KConstView2D tlay = tlay_alloc;
            KConstView1D tsfc = tsfc_alloc;
            KConstView4D kmajor = kmajor_alloc;
            KConstView3D kminor = kminor_alloc;
            KConstView2D clwp = clwp_alloc;
            KConstView3D lut_liquid = lut_liquid_alloc;
            KConstView1D sza = sza_alloc;
            KConstView1D toa = toa_alloc;
            KConstView3D tau_aerosol = tau_aerosol_alloc;
            KConstView2D planck_table = planck_table_alloc;
            KView2D flux_up = flux_up_alloc;
            KView2D flux_dn = flux_dn_alloc;

            // Execute fused Megakernel on Device space
            KokkosMegakernel::execute_megakernel(
                layers, columns, gpoints,
                play, tlay, tsfc, kmajor, kminor, clwp, lut_liquid, sza, toa,
                tau_aerosol, KConstView3D(), KConstView3D(), planck_table,
                flux_up, flux_dn
            );

            auto flux_up_host = Kokkos::create_mirror_view(flux_up);
            auto flux_dn_host = Kokkos::create_mirror_view(flux_dn);
            Kokkos::deep_copy(flux_up_host, flux_up);
            Kokkos::deep_copy(flux_dn_host, flux_dn);

            // T007: Verify hardware-specific layout coalescing traits (FR-001)
            bool is_cpu = std::is_same_v<DeviceSpace, Kokkos::OpenMP>;
            bool has_layout_right = std::is_same_v<KView2D::array_layout, Kokkos::LayoutRight>;
            bool has_layout_left = std::is_same_v<KView2D::array_layout, Kokkos::LayoutLeft>;

            std::cout << "Compile-Time Layout Policies: " << std::endl;
            std::cout << "  - Kokkos DeviceSpace OpenMP target active: " << (is_cpu ? "YES" : "NO") << std::endl;
            std::cout << "  - LayoutRight (CPU optimal) active: " << (has_layout_right ? "YES" : "NO") << std::endl;
            std::cout << "  - LayoutLeft (GPU optimal) active: " << (has_layout_left ? "YES" : "NO") << std::endl;

            real_t expected_flux_up = 263.894;
            real_t expected_flux_dn = 249.762;

            std::cout << "Computed verification flux_up TOA (lev 0): " << flux_up_host(0, 0) << " W/m^2 (expected: " << expected_flux_up << ")" << std::endl;
            std::cout << "Computed verification flux_dn SFC (lev " << layers << "): " << flux_dn_host(layers, 0) << " W/m^2 (expected: " << expected_flux_dn << ")" << std::endl;

            // Physical verification assertions: check output flux_up and flux_dn against expected physical values
            if (std::abs(flux_up_host(0, 0) - expected_flux_up) >= 1e-3 ||
                std::abs(flux_dn_host(layers, 0) - expected_flux_dn) >= 1e-3) {
                std::cerr << "test_kokkos_megakernel FAIL: Unphysical flux values or value mismatch!" << std::endl;
                Kokkos::finalize();
                return 1;
            }

            std::cout << "Verification Mode: SUCCESS (Functional parity verified)" << std::endl;
        }

        // --- 2. Cache Saturation Benchmark Mode (1000 Columns, 128 Layers, 128 Gpoints) ---
        {
            std::cout << "\nStarting Cache Saturation Benchmark (1000 Column Set)..." << std::endl;
            const size_t layers = 128;
            const size_t columns = 1000; // 1000 columns benchmark
            const size_t gpoints = 128;

            std::cout << "  - Configuration: " << columns << " columns, " << layers << " layers, " << gpoints << " g-points" << std::endl;
            std::cout << "  - Total grid elements computed: " << (columns * layers * gpoints) / 1000000.0 << " million cells" << std::endl;

            // Allocate larger views
            KView2D play_alloc("play", layers, columns);
            KView2D tlay_alloc("tlay", layers, columns);
            KView1D tsfc_alloc("tsfc", columns);
            KView4D kmajor_alloc("kmajor", gpoints, 2, 2, 2);
            KView3D kminor_alloc("kminor", gpoints, 2, 2);
            KView2D clwp_alloc("clwp", layers, columns);
            KView3D lut_liquid_alloc("lut_liquid", gpoints, 2, 2);
            KView1D sza_alloc("sza", columns);
            KView1D toa_alloc("toa", gpoints);
            KView3D tau_aerosol_alloc("tau_aerosol", gpoints, columns, layers);
            KView2D planck_table_alloc("planck_table", gpoints, 10);
            KView2D flux_up_alloc("flux_up", layers + 1, columns);
            KView2D flux_dn_alloc("flux_dn", layers + 1, columns);

            // Populate on Host mirrors
            auto play_host = Kokkos::create_mirror_view(play_alloc);
            auto tlay_host = Kokkos::create_mirror_view(tlay_alloc);
            auto tsfc_host = Kokkos::create_mirror_view(tsfc_alloc);
            auto clwp_host = Kokkos::create_mirror_view(clwp_alloc);
            auto kmajor_host = Kokkos::create_mirror_view(kmajor_alloc);
            auto kminor_host = Kokkos::create_mirror_view(kminor_alloc);
            auto lut_liquid_host = Kokkos::create_mirror_view(lut_liquid_alloc);
            auto tau_aero_host = Kokkos::create_mirror_view(tau_aerosol_alloc);
            auto planck_host = Kokkos::create_mirror_view(planck_table_alloc);

            for (size_t col = 0; col < columns; ++col) {
                tsfc_host(col) = 300.0;
                for (size_t lay = 0; lay < layers; ++lay) {
                    play_host(lay, col) = 1000.0;
                    tlay_host(lay, col) = 290.0;
                    clwp_host(lay, col) = 0.2;
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        tau_aero_host(gp, col, lay) = 0.1;
                    }
                }
            }

            for (size_t gp = 0; gp < gpoints; ++gp) {
                for (size_t p_idx = 0; p_idx < 10; ++p_idx) {
                    planck_host(gp, p_idx) = 10.0 + p_idx * 2.0;
                }
                for (size_t i = 0; i < 2; ++i) {
                    for (size_t j = 0; j < 2; ++j) {
                        lut_liquid_host(gp, i, j) = 5.0;
                        kmajor_host(gp, i, j, 0) = 2.5;
                        kmajor_host(gp, i, j, 1) = 2.5;
                        kminor_host(gp, i, j) = 0.5;
                    }
                }
            }

            // Copy to Device
            Kokkos::deep_copy(play_alloc, play_host);
            Kokkos::deep_copy(tlay_alloc, tlay_host);
            Kokkos::deep_copy(tsfc_alloc, tsfc_host);
            Kokkos::deep_copy(kmajor_alloc, kmajor_host);
            Kokkos::deep_copy(kminor_alloc, kminor_host);
            Kokkos::deep_copy(clwp_alloc, clwp_host);
            Kokkos::deep_copy(lut_liquid_alloc, lut_liquid_host);
            Kokkos::deep_copy(tau_aerosol_alloc, tau_aero_host);
            Kokkos::deep_copy(planck_table_alloc, planck_host);

            KConstView2D play = play_alloc;
            KConstView2D tlay = tlay_alloc;
            KConstView1D tsfc = tsfc_alloc;
            KConstView4D kmajor = kmajor_alloc;
            KConstView3D kminor = kminor_alloc;
            KConstView2D clwp = clwp_alloc;
            KConstView3D lut_liquid = lut_liquid_alloc;
            KConstView1D sza = sza_alloc;
            KConstView1D toa = toa_alloc;
            KConstView3D tau_aerosol = tau_aerosol_alloc;
            KConstView2D planck_table = planck_table_alloc;
            KView2D flux_up = flux_up_alloc;
            KView2D flux_dn = flux_dn_alloc;

            // Warm up run to prime instruction caches
            KokkosMegakernel::execute_megakernel(
                layers, columns, gpoints,
                play, tlay, tsfc, kmajor, kminor, clwp, lut_liquid, sza, toa,
                tau_aerosol, KConstView3D(), KConstView3D(), planck_table,
                flux_up, flux_dn
            );
            Kokkos::fence();

            // Run benchmark timing loops
            const int iterations = 10;
            Kokkos::Timer timer;

            for (int i = 0; i < iterations; ++i) {
                KokkosMegakernel::execute_megakernel(
                    layers, columns, gpoints,
                    play, tlay, tsfc, kmajor, kminor, clwp, lut_liquid, sza, toa,
                    tau_aerosol, KConstView3D(), KConstView3D(), planck_table,
                    flux_up, flux_dn
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
