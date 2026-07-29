#pragma once

#include "mo_rte_kind.h"
#include <vector>

namespace rrtmgp {

// Optional compilation helper providing static mock/reference gas optics coefficients.
// Allows compilation and execution of RRTMGP C++ port completely file-less in closed environments.
struct StaticCoefficients {
    static inline const std::vector<real_t> kmajor_ref = {
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5
    };
    static inline const std::vector<real_t> kminor_ref = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    };
};

} // namespace rrtmgp
