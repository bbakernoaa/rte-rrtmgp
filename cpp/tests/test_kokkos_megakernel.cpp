#include "mo_kokkos_megakernel.h"
#include <iostream>
#include <stdexcept>

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

        // Implicitly cast to read-only views for execution contract interfaces
        KConstView2D play = play_alloc;
        KConstView2D tlay = tlay_alloc;
        KConstView4D kmajor = kmajor_alloc;
        KConstView3D kminor = kminor_alloc;
        KConstView2D clwp = clwp_alloc;
        KConstView3D lut_liquid = lut_liquid_alloc;
        KConstView1D sza = sza_alloc;
        KConstView1D toa = toa_alloc;

        std::cout << "T003: Successfully allocated Kokkos Views on Host/Device space." << std::endl;

        // TDD RED Phase: Expected to throw runtime_error since execute_megakernel is unimplemented
        KokkosMegakernel::execute_megakernel(
            layers, columns, gpoints,
            play, tlay, kmajor, kminor, clwp, lut_liquid, sza, toa, flux_dir
        );

        std::cerr << "test_kokkos_megakernel FAIL: Expected exception not thrown!" << std::endl;
        Kokkos::finalize();
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "T003 RED: Successfully caught expected runtime_error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_kokkos_megakernel FAIL with unexpected exception: " << e.what() << std::endl;
        Kokkos::finalize();
        return 1;
    }

    Kokkos::finalize();
    std::cout << "test_kokkos_megakernel: SUCCESS (RED phase verified)" << std::endl;
    return 0;
}
