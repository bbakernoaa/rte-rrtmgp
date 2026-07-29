#pragma once

#include "mo_rte_kind.h"

namespace rrtmgp::kernels {

struct InterpolationWeights {
    size_t jpress[2];
    size_t jtemp[2];
    size_t jeta[2];
    real_t fpress;
    real_t ftemp;
    real_t feta;
};

// Procedural physics kernels isolated inside standalone namespace
// Replicates the exact mathematical subroutines from mo_gas_optics_rrtmgp_kernels.F90

void interpolation(
    ConstView1D ref_press,
    ConstView2D ref_temp,
    real_t play,
    real_t tlay,
    InterpolationWeights& weights
);

void compute_tau_absorption(
    const InterpolationWeights& weights,
    ConstView4D kmajor,
    ConstView3D kminor,
    View1D tau
);

void compute_tau_rayleigh(
    const InterpolationWeights& weights,
    ConstView3D krayleigh,
    View1D tau_rayleigh
);

void compute_Planck_source(
    const InterpolationWeights& weights,
    ConstView3D planck_fraction,
    View1D planck_source
);

} // namespace rrtmgp::kernels
