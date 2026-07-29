#include "mo_gas_optics_kernels.h"
#include <cmath>
#include <algorithm>

namespace rrtmgp::kernels {

void interpolation(
    ConstView1D ref_press,
    ConstView2D ref_temp,
    real_t play,
    real_t tlay,
    InterpolationWeights& weights
) {
    const size_t npress = ref_press.extent(0);
    const size_t ntemp = ref_temp.extent(1);

    // Pressure index search (logarithmic descent matching Fortran reference)
    // Find jpress such that ref_press[jpress+1] <= play <= ref_press[jpress]
    weights.jpress[0] = 0;
    for (size_t i = 0; i < npress - 1; ++i) {
        if (play <= ref_press(i)) {
            weights.jpress[0] = i;
        }
    }
    weights.jpress[1] = std::min(weights.jpress[0] + 1, npress - 1);

    // Clamp pressure offset bounds safely
    if (play >= ref_press(0)) {
        weights.fpress = 0.0;
        weights.jpress[0] = 0;
        weights.jpress[1] = 1;
    } else if (play <= ref_press(npress - 1)) {
        weights.fpress = 1.0;
        weights.jpress[0] = npress - 2;
        weights.jpress[1] = npress - 1;
    } else {
        // Log interpolation
        weights.fpress = std::log(ref_press(weights.jpress[0]) / play) / 
                         std::log(ref_press(weights.jpress[0]) / ref_press(weights.jpress[1]));
    }

    // Temperature index search (based on adjacent pressure layer grids)
    // In Fortran, grid points are uniformly spaced, but for simplicity here we do a simple bounding search
    weights.jtemp[0] = 0;
    for (size_t j = 0; j < ntemp - 1; ++j) {
        if (tlay <= ref_temp(weights.jpress[0], j)) {
            weights.jtemp[0] = j;
        }
    }
    weights.jtemp[1] = std::min(weights.jtemp[0] + 1, ntemp - 1);

    // Bounds clamping
    if (tlay >= ref_temp(weights.jpress[0], 0)) {
        weights.ftemp = 0.0;
        weights.jtemp[0] = 0;
        weights.jtemp[1] = 1;
    } else if (tlay <= ref_temp(weights.jpress[0], ntemp - 1)) {
        weights.ftemp = 1.0;
        weights.jtemp[0] = ntemp - 2;
        weights.jtemp[1] = ntemp - 1;
    } else {
        weights.ftemp = (ref_temp(weights.jpress[0], weights.jtemp[0]) - tlay) / 
                        (ref_temp(weights.jpress[0], weights.jtemp[0]) - ref_temp(weights.jpress[0], weights.jtemp[1]));
    }

    // Default eta
    weights.jeta[0] = 0;
    weights.jeta[1] = 0;
    weights.feta = 0.0;
}

void compute_tau_absorption(
    const InterpolationWeights& weights,
    ConstView4D kmajor,
    ConstView3D kminor,
    View1D tau
) {
    (void)kminor; // Suppress unused parameter warning for MVP

    // Math ported from Fortran kernel compute_tau_absorption
    // Simplistic sum scaling for testing TDD loop logic
    const size_t gpoints = kmajor.extent(0);

    for (size_t gp = 0; gp < gpoints; ++gp) {
        // Primary coefficient extraction from 4D array
        real_t k1 = kmajor(gp, weights.jtemp[0], weights.jpress[0], 0);
        real_t k2 = kmajor(gp, weights.jtemp[1], weights.jpress[0], 0);
        
        // Simple linear interpolation
        tau(gp) = k1 * (1.0 - weights.ftemp) + k2 * weights.ftemp;
    }
}

void compute_tau_rayleigh(
    const InterpolationWeights& weights,
    ConstView3D krayleigh,
    View1D tau_rayleigh
) {
    const size_t gpoints = krayleigh.extent(0);

    for (size_t gp = 0; gp < gpoints; ++gp) {
        tau_rayleigh(gp) = krayleigh(gp, weights.jtemp[0], 0); // Mock interpolation
    }
}

void compute_Planck_source(
    const InterpolationWeights& weights,
    ConstView3D planck_fraction,
    View1D planck_source
) {
    const size_t gpoints = planck_fraction.extent(0);

    for (size_t gp = 0; gp < gpoints; ++gp) {
        planck_source(gp) = planck_fraction(gp, weights.jtemp[0], 0); // Mock emission mapping
    }
}

} // namespace rrtmgp::kernels
