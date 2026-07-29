#pragma once

#include "mo_rte_kind.h"

namespace rte::extensions {

using namespace rrtmgp;

// Standard physical constants
constexpr real_t gravity = 9.80665;    // m/s^2
constexpr real_t cp_dry_air = 1004.6;  // J/kg/K
constexpr real_t seconds_per_day = 86400.0;

// Computes temperature trends (heating rates in K/day) based on flux divergence
void compute_heating_rates(
    ConstView2D flux_up, // Spectrally-summed level upward fluxes (layers + 1, columns)
    ConstView2D flux_dn, // Spectrally-summed level downward fluxes (layers + 1, columns)
    ConstView2D p_level, // Pressure boundary levels (layers + 1, columns) in Pa
    View2D heating_rate  // Output layer heating rates (layers, columns) in K/day
);

// Group g-point resolved fluxes (gpoints, columns) into coarse spectral bands
void reduce_byband(
    ConstView2D gpoint_flux,
    IntView1D gpoint_to_band,
    View2D band_flux
);

} // namespace rte::extensions
