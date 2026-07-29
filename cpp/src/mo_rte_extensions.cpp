#include "mo_rte_extensions.h"
#include "mo_rte_util.h"
#include <cmath>
#include <stdexcept>
#include <random>

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

void zenith_angle_spherical_correction(
    ConstView1D solar_zenith_angle,
    ConstView2D pressure_layers,
    View1D corrected_zenith_angle
) {
    const size_t columns = solar_zenith_angle.extent(0);

    // Validate boundaries
    validate_extent(pressure_layers, 1, columns, "pressure_layers");
    validate_extent(corrected_zenith_angle, 0, columns, "corrected_zenith_angle");

    for (size_t col = 0; col < columns; ++col) {
        real_t angle = solar_zenith_angle(col);
        // Correct the zenith angle cosine factor near twilight (refraction / spherical curvature)
        real_t mu = std::cos(angle);
        real_t p_bottom = pressure_layers(0, col); // bottom level pressure
        
        // Spherical correction adjustment formula
        real_t correction = 0.05 * (1.0 - p_bottom / 101325.0);
        corrected_zenith_angle(col) = std::min(1.0, std::max(0.01, mu + correction));
    }
}

void execute_mcica_sampling(
    ConstView2D cloud_fraction,
    size_t sub_columns_cnt,
    IntViewMut2D output_column_mask,
    unsigned int random_seed
) {
    const size_t layers = cloud_fraction.extent(0);
    const size_t columns = cloud_fraction.extent(1);

    // Validate boundaries
    validate_extent(output_column_mask, 0, layers, "output_column_mask");
    validate_extent(output_column_mask, 1, columns * sub_columns_cnt, "output_column_mask");

    // Initialize local seedable Mersenne Twister engine to guarantee 100% thread safety
    std::mt19937 generator(random_seed);
    std::uniform_real_distribution<real_t> distribution(0.0, 1.0);

    // Perform Monte Carlo cloud sub-column overlap mappings
    for (size_t col = 0; col < columns; ++col) {
        for (size_t sub = 0; sub < sub_columns_cnt; ++sub) {
            for (size_t lay = 0; lay < layers; ++lay) {
                real_t c_frac = cloud_fraction(lay, col);
                real_t r = distribution(generator);
                
                // If random draw is below layer cloud fraction, sub-column has a cloud
                if (r <= c_frac) {
                    output_column_mask(lay, col * sub_columns_cnt + sub) = 1; // cloud
                } else {
                    output_column_mask(lay, col * sub_columns_cnt + sub) = 0; // clear
                }
            }
        }
    }
}

} // namespace rte::extensions
