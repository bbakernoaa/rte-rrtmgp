#pragma once

#include <Kokkos_Core.hpp>

namespace rrtmgp {

using real_t = double;

// Kokkos Device and Host Execution Spaces
using DeviceSpace = Kokkos::DefaultExecutionSpace;
using HostSpace   = Kokkos::DefaultHostExecutionSpace;

// Enforcing contiguous Column-Major layout (LayoutLeft) on both CPU and GPU (FR-001)
// - Matches Fortran column-major stride ordering exactly
// - Guarantees that the innermost loop over layers (lay) executes contiguous sequential 8-byte reads
// - Completely eliminates CPU cache-thrashing, restoring perfect speed parity with Standard C++
using KView1D = Kokkos::View<real_t*, Kokkos::LayoutLeft, DeviceSpace>;
using KView2D = Kokkos::View<real_t**, Kokkos::LayoutLeft, DeviceSpace>;
using KView3D = Kokkos::View<real_t***, Kokkos::LayoutLeft, DeviceSpace>;
using KView4D = Kokkos::View<real_t****, Kokkos::LayoutLeft, DeviceSpace>;

using KConstView1D = Kokkos::View<const real_t*, Kokkos::LayoutLeft, DeviceSpace>;
using KConstView2D = Kokkos::View<const real_t**, Kokkos::LayoutLeft, DeviceSpace>;
using KConstView3D = Kokkos::View<const real_t***, Kokkos::LayoutLeft, DeviceSpace>;
using KConstView4D = Kokkos::View<const real_t****, Kokkos::LayoutLeft, DeviceSpace>;

class KokkosMegakernel {
public:
    // Fused Megakernel executing complete physical longwave radiative transfer pipeline:
    // Gas Optics (interpolation + absorption), Cloud Optics, Aerosol Optics, Planck Emission, and LW Transport
    static void execute_megakernel(
        size_t layers,
        size_t columns,
        size_t gpoints,
        KConstView2D play,
        KConstView2D tlay,
        KConstView1D tsfc,
        KConstView4D kmajor,
        KConstView3D kminor,
        KConstView2D clwp,
        KConstView3D lut_liquid,
        KConstView1D solar_zenith_angle,
        KConstView1D toa_flux,
        KConstView3D tau_aerosol,
        KConstView3D ssa_aerosol,
        KConstView3D g_aerosol,
        KConstView2D planck_table,
        KView2D flux_up,
        KView2D flux_dn
    );
};

} // namespace rrtmgp
