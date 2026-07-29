#pragma once

#include "mo_rte_kind.h"

namespace rte {

using namespace rrtmgp;

class SolverLw {
public:
    static void solve_lw_noscat(
        ConstView3D tau,
        ConstView3D lay_source,
        View2D flux_up,
        View2D flux_dn
    );
};

} // namespace rte
