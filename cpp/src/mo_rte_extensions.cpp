#include "mo_rte_extensions.h"
#include "mo_rte_util.h"
#include <cmath>
#include <stdexcept>

namespace rte::extensions {

void compute_heating_rates(
    ConstView2D flux_up,
    ConstView2D flux_dn,
    ConstView2D p_level,
    View2D heating_rate
) {
    const size_t layers = heating_rate.extent(0);
    const size_t columns = heating_rate.extent(1);

    // Validate boundaries
    validate_extent(flux_up, 0, layers + 1, "flux_up");
    validate_extent(flux_up, 1, columns, "flux_up");
    validate_extent(flux_dn, 0, layers + 1, "flux_dn");
    validate_extent(flux_dn, 1, columns, "flux_dn");
    validate_extent(p_level, 0, layers + 1, "p_level");
    validate_extent(p_level, 1, columns, "p_level");

    // Compute vertical heating trend sweeps
    for (size_t col = 0; col < columns; ++col) {
        for (size_t lay = 0; lay < layers; ++lay) {
            // Net flux at lower boundary (level lay)
            real_t net_low = flux_up(lay, col) - flux_dn(lay, col);
            // Net flux at upper boundary (level lay + 1)
            real_t net_high = flux_up(lay + 1, col) - flux_dn(lay + 1, col);

            // Pressure thickness delta p (Pa)
            real_t dp = p_level(lay, col) - p_level(lay + 1, col);
            if (std::abs(dp) < 1e-10) {
                throw std::invalid_argument("Pressure layer thickness delta p must be non-zero");
            }

            // Flux divergence delta F (W/m^2)
            real_t dF = net_low - net_high;

            // dT/dt (K/day) = (g / Cp) * (dF / dp) * 86400 seconds/day
            heating_rate(lay, col) = (gravity / cp_dry_air) * (dF / dp) * seconds_per_day;
        }
    }
}

void reduce_byband(
    ConstView2D gpoint_flux,
    IntView1D gpoint_to_band,
    View2D band_flux
) {
    const size_t gp_cnt = gpoint_flux.extent(0);
    const size_t columns = gpoint_flux.extent(1);
    const size_t bands = band_flux.extent(0);

    validate_extent(gpoint_to_band, 0, gp_cnt, "gpoint_to_band");
    validate_extent(band_flux, 1, columns, "band_flux");

    // Reset output band flux buffers
    for (size_t col = 0; col < columns; ++col) {
        for (size_t b = 0; b < bands; ++b) {
            band_flux(b, col) = 0.0;
        }
    }

    // Accumulate g-point spectral fluxes into coarse bands
    for (size_t col = 0; col < columns; ++col) {
        for (size_t gp = 0; gp < gp_cnt; ++gp) {
            int b_idx = gpoint_to_band(gp);
            if (b_idx < 0 || b_idx >= static_cast<int>(bands)) {
                throw std::out_of_range("gpoint_to_band maps to band index out of range");
            }
            band_flux(b_idx, col) += gpoint_flux(gp, col);
        }
    }
}

} // namespace rte::extensions
