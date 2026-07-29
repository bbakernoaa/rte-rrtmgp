#pragma once

#include <Kokkos_Core.hpp>

namespace rrtmgp {

// Decoupled precision parameter
using real_t = double;

// Kokkos Device and Host Execution Spaces
using DeviceSpace = Kokkos::DefaultExecutionSpace;
using HostSpace   = Kokkos::DefaultHostExecutionSpace;

// Dynamic N-Dimensional Views on Device Memory
using KView1D = Kokkos::View<real_t*, DeviceSpace>;
using KView2D = Kokkos::View<real_t**, DeviceSpace>;
using KView3D = Kokkos::View<real_t***, DeviceSpace>;
using KView4D = Kokkos::View<real_t****, DeviceSpace>;

using KConstView1D = Kokkos::View<const real_t*, DeviceSpace>;
using KConstView2D = Kokkos::View<const real_t**, DeviceSpace>;
using KConstView3D = Kokkos::View<const real_t***, DeviceSpace>;
using KConstView4D = Kokkos::View<const real_t****, DeviceSpace>;

class KokkosMegakernel {
public:
    // Fused Megakernel executing GasOptics, CloudOptics, and SolverSw in a single parallel sweep
    static void execute_megakernel(
        size_t layers,
        size_t columns,
        size_t gpoints,
        KConstView2D play,
        KConstView2D tlay,
        KConstView4D kmajor,
        KConstView3D kminor,
        KConstView2D clwp,
        KConstView3D lut_liquid,
        KConstView1D solar_zenith_angle,
        KConstView1D toa_flux,
        KView2D flux_dir
    );
};

} // namespace rrtmgp
