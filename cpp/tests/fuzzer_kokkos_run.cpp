#include "mo_kokkos_megakernel.h"
#include <vector>
#include <stdexcept>

using namespace rrtmgp;

void init_kokkos_fuzzer() {
    Kokkos::initialize();
}

void finalize_kokkos_fuzzer() {
    Kokkos::finalize();
}

// Kokkos Megakernel wrapper (compiled in Kokkos translation unit, zero standard C++ mdspan headers)
void run_kokkos_solvers(
    size_t layers, size_t columns, size_t gpoints,
    const std::vector<real_t>& play,
    const std::vector<real_t>& tlay,
    const std::vector<real_t>& clwp,
    const std::vector<real_t>& rel,
    std::vector<real_t>& flux_kokkos
) {
    (void)rel;
    std::vector<real_t> kmajor_data(gpoints * 2 * 2 * 1, 2.5);
    std::vector<real_t> lut_liquid_data(gpoints * 2 * 2, 5.0);

    KView2D play_k("play", layers, columns);
    KView2D tlay_k("tlay", layers, columns);
    KView4D kmajor_k("kmajor", gpoints, 2, 2, 1);
    KView3D kminor_k("kminor", gpoints, 2, 1);
    KView2D clwp_k("clwp", layers, columns);
    KView3D lut_liquid_k("lut_liquid", gpoints, 2, 2);
    KView1D sza_k("sza", columns);
    KView1D toa_k("toa", gpoints);
    KView2D flux_dir_k("flux_dir", gpoints, columns);

    auto play_host = Kokkos::create_mirror_view(play_k);
    auto tlay_host = Kokkos::create_mirror_view(tlay_k);
    auto kmajor_host = Kokkos::create_mirror_view(kmajor_k);
    auto clwp_host = Kokkos::create_mirror_view(clwp_k);
    auto lut_liquid_host = Kokkos::create_mirror_view(lut_liquid_k);

    // Copy to Host mirrors
    std::copy(play.begin(), play.end(), play_host.data());
    std::copy(tlay.begin(), tlay.end(), tlay_host.data());
    std::copy(kmajor_data.begin(), kmajor_data.end(), kmajor_host.data());
    std::copy(clwp.begin(), clwp.end(), clwp_host.data());
    std::copy(lut_liquid_data.begin(), lut_liquid_data.end(), lut_liquid_host.data());

    // Deep copy to Device
    Kokkos::deep_copy(play_k, play_host);
    Kokkos::deep_copy(tlay_k, tlay_host);
    Kokkos::deep_copy(kmajor_k, kmajor_host);
    Kokkos::deep_copy(clwp_k, clwp_host);
    Kokkos::deep_copy(lut_liquid_k, lut_liquid_host);

    KokkosMegakernel::execute_megakernel(
        layers, columns, gpoints,
        play_k, tlay_k, kmajor_k, kminor_k, clwp_k, lut_liquid_k, sza_k, toa_k, flux_dir_k
    );

    auto flux_host_k = Kokkos::create_mirror_view(flux_dir_k);
    Kokkos::deep_copy(flux_host_k, flux_dir_k);

    std::copy(flux_host_k.data(), flux_host_k.data() + gpoints * columns, flux_kokkos.begin());
}
