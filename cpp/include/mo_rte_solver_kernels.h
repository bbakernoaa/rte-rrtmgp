#pragma once

#include "mo_rte_kind.h"

namespace rte::kernels {

using namespace rrtmgp;

void lw_source_noscat(
    int ncol, int nlay, int gp,
    ConstView3D lay_source, ConstView3D lev_source, ConstView2D tau, ConstView2D trans,
    View2D source_dn, View2D source_up
);

void lw_transport_noscat_dn(
    int ncol, int nlay,
    ConstView2D trans, ConstView2D source_dn, View2D radn_dn
);

void lw_transport_noscat_up(
    int ncol, int nlay,
    ConstView2D trans, ConstView2D source_up, View2D radn_up
);

void sw_dif_and_source(
    int ncol, int nlay, int gp,
    ConstView2D mu0, ConstView2D sfc_alb_dir,
    ConstView3D tau, ConstView3D ssa, ConstView3D g,
    View2D Rdif, View2D Tdif,
    View2D source_dn, View2D source_up, View2D source_srf,
    View2D flux_dir
);

void adding(
    int ncol, int nlay, int gp,
    ConstView2D sfc_alb_dif,
    ConstView2D Rdif, ConstView2D Tdif,
    ConstView2D source_dn, ConstView2D source_up, ConstView2D source_srf,
    View2D flux_up, View2D flux_dn
);

} // namespace rte::kernels
