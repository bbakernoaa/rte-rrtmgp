#pragma once

#include "mo_rte_kind.h"

namespace rte {

using namespace rrtmgp;

class SolverSw {
public:
    static void solve_sw_noscat(
        ConstView3D tau,
        ConstView1D solar_zenith_angle,
        ConstView1D toa_flux,
        View2D flux_dir
    );

    static void solve_sw_2stream(
        ConstView3D tau,
        ConstView3D ssa,
        ConstView3D g,
        ConstView1D solar_zenith_angle,
        ConstView2D sfc_albedo,
        ConstView1D toa_flux,
        View2D flux_up,
        View2D flux_dn,
        View2D flux_dir
    );
};

} // namespace rte

namespace rte::kernels {

using namespace rrtmgp;

void sw_dif_and_source(
    ConstView3D tau,
    ConstView3D ssa,
    ConstView3D g,
    ConstView1D mu0,
    View3D R_dir,
    View3D T_dir,
    View3D R_dif,
    View3D T_dif,
    View3D source_up,
    View3D source_dn
);

void adding_doubling(
    ConstView3D R_dif,
    ConstView3D T_dif,
    ConstView3D source_up,
    ConstView3D source_dn,
    ConstView2D sfc_albedo,
    View2D flux_up,
    View2D flux_dn
);

} // namespace rte::kernels
