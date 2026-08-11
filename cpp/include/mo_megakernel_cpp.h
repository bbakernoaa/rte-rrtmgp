#pragma once

#include "mo_rte_kind.h"

namespace rrtmgp {

class MegakernelCpp {
public:
    // Single-pass fused longwave physics solver using standard C++17 mdspan and OpenMP
    static void execute_fused_longwave(
        size_t layers,
        size_t columns,
        size_t gpoints,
        ConstView2D play,
        ConstView2D tlay,
        ConstView1D tsfc,
        ConstView4D kmajor,
        ConstView3D kminor,
        ConstView2D clwp,
        ConstView3D lut_liquid,
        ConstView1D solar_zenith_angle,
        ConstView1D toa_flux,
        ConstView3D tau_aerosol,
        ConstView3D ssa_aerosol,
        ConstView3D g_aerosol,
        ConstView2D planck_table,
        View2D flux_up,
        View2D flux_dn
    );
};

} // namespace rrtmgp
