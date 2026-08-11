#pragma once

#include "mo_rte_kind.h"

namespace rte {

using namespace rrtmgp;

class SolverLw {
public:
    static void solve_lw_noscat(
        ConstView3D tau,               // [ngpt, nlay, ncol]
        ConstView3D lay_source,        // [ngpt, nlay, ncol]
        ConstView3D lev_source,        // [ngpt, nlay+1, ncol]
        ConstView2D sfc_emis,          // [ngpt, ncol]
        ConstView2D sfc_src,           // [ngpt, ncol]
        ConstView2D incident_flux,     // [ngpt, ncol]
        View3D flux_up,                // [ngpt, nlay+1, ncol]
        View3D flux_dn                 // [ngpt, nlay+1, ncol]
    );
};

} // namespace rte
