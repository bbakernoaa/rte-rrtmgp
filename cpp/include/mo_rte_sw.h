#pragma once

#include "mo_rte_kind.h"

namespace rte {

using namespace rrtmgp;

class SolverSw {
public:
    static void solve_sw_noscat(
        ConstView3D tau,
        ConstView2D solar_zenith_angle, // [nlay, ncol]
        ConstView2D inc_flux_dir,       // [ngpt, ncol]
        View3D flux_dir                 // [ngpt, nlay+1, ncol]
    );

    static void solve_sw_2stream(
        ConstView3D tau,                // [ngpt, nlay, ncol]
        ConstView3D ssa,                // [ngpt, nlay, ncol]
        ConstView3D g,                  // [ngpt, nlay, ncol]
        ConstView2D solar_zenith_angle, // [nlay, ncol]
        ConstView2D sfc_alb_dir,        // [ngpt, ncol]
        ConstView2D sfc_alb_dif,        // [ngpt, ncol]
        ConstView2D inc_flux_dir,       // [ngpt, ncol]
        View3D flux_up,                 // [ngpt, nlay+1, ncol]
        View3D flux_dn,                 // [ngpt, nlay+1, ncol]
        View3D flux_dir                 // [ngpt, nlay+1, ncol]
    );
};

} // namespace rte
