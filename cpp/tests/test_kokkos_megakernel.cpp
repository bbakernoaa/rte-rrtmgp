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
        KView2D flux_dir("flux_dir", gpoints, columns);

        // Access view host mirrors to populate values (FR-006)
        auto play_host = Kokkos::create_mirror_view(play_alloc);
        auto tlay_host = Kokkos::create_mirror_view(tlay_alloc);
        auto kmajor_host = Kokkos::create_mirror_view(kmajor_alloc);
        auto clwp_host = Kokkos::create_mirror_view(clwp_alloc);
        auto lut_liquid_host = Kokkos::create_mirror_view(lut_liquid_alloc);

        // Populate baseline atmospheric profiles
        for (size_t col = 0; col < columns; ++col) {
            for (size_t lay = 0; lay < layers; ++lay) {
                play_host(lay, col) = 1000.0;
                tlay_host(lay, col) = 290.0;
                clwp_host(lay, col) = 0.2;
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

        // Implicitly cast to read-only views for execution contract interfaces
        KConstView2D play = play_alloc;
        KConstView2D tlay = tlay_alloc;
        KConstView4D kmajor = kmajor_alloc;
        KConstView3D kminor = kminor_alloc;
        KConstView2D clwp = clwp_alloc;
        KConstView3D lut_liquid = lut_liquid_alloc;
        KConstView1D sza = sza_alloc;
        KConstView1D toa = toa_alloc;

        std::cout << "T003: Successfully allocated and deep-copied View vectors." << std::endl;

        // Execute fused Megakernel on Device space
        KokkosMegakernel::execute_megakernel(
            layers, columns, gpoints,
            play, tlay, kmajor, kminor, clwp, lut_liquid, sza, toa, flux_dir
        );

        // Pull output fluxes back to Host to verify numerical parity
        auto flux_host = Kokkos::create_mirror_view(flux_dir);
        Kokkos::deep_copy(flux_host, flux_dir);

        // T007: Verify hardware-specific layout coalescing compile-time traits (FR-001)
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

        // Expected output:
        // For each layer:
        // tau_gas = kmajor(2.5) * exp(-1000 * 290 * 1e-6) = 2.5 * exp(-0.29) = 1.8706589
        // tau_cloud = clwp(0.2) * lut_liquid(5) = 1.0
        // accumulated_flux = (1.8706589 + 1.0) * 10 = 28.706589
        // 5 layers = 28.706589 * 5 = 143.532946
        real_t expected_flux = 143.532946;

        std::cout << "Computed output flux at gp0 col0: " << flux_host(0, 0) << std::endl;
        std::cout << "Expected output flux: " << expected_flux << std::endl;

        if (std::abs(flux_host(0, 0) - expected_flux) >= 1e-5) {
            std::cerr << "test_kokkos_megakernel FAIL: Mismatched computed values!" << std::endl;
            Kokkos::finalize();
            return 1;
        }

        std::cout << "test_kokkos_megakernel: SUCCESS (Numerical parity verified)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_kokkos_megakernel FAIL with exception: " << e.what() << std::endl;
        Kokkos::finalize();
        return 1;
    }

    Kokkos::finalize();
    return 0;
}
