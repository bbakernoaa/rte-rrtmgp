#include "mo_rte_lw.h"
#include "mo_rte_util.h"
#include <cmath>

namespace rte {

void SolverLw::solve_lw_noscat(
    ConstView3D tau,
    ConstView3D lay_source,
    View2D flux_up,
    View2D flux_dn
) {
    const size_t gp_cnt = tau.extent(0);
    const size_t layers = tau.extent(1);
    const size_t columns = tau.extent(2);

    // Validate boundaries match strictly
    validate_extent(lay_source, 0, gp_cnt, "lay_source");
    validate_extent(lay_source, 1, layers, "lay_source");
    validate_extent(lay_source, 2, columns, "lay_source");
    validate_extent(flux_up, 0, gp_cnt, "flux_up");
    validate_extent(flux_up, 1, columns, "flux_up");
    validate_extent(flux_dn, 0, gp_cnt, "flux_dn");
    validate_extent(flux_dn, 1, columns, "flux_dn");

    // Clear output flux buffers
    for (size_t col = 0; col < columns; ++col) {
        for (size_t gp = 0; gp < gp_cnt; ++gp) {
            flux_up(gp, col) = 0.0;
            flux_dn(gp, col) = 0.0;
        }
    }

    // Mathematical loop nest executing radiative transfer integration (decoding layers)
    // Decoupled from GCM state, following column-major cache-friendly ordering
    for (size_t col = 0; col < columns; ++col) {
        for (size_t gp = 0; gp < gp_cnt; ++gp) {
            
            // 1. Downwelling propagation (Top-of-Atmosphere [TOA] to Surface)
            real_t current_dn = 0.0; // Zero downwelling at TOA
            for (int lay = static_cast<int>(layers) - 1; lay >= 0; --lay) {
                real_t t = tau(gp, lay, col);
                real_t trans = std::exp(-t);
                real_t emit = lay_source(gp, lay, col) * (1.0 - trans);
                current_dn = current_dn * trans + emit;
            }
            flux_dn(gp, col) = current_dn;

            // 2. Upwelling propagation (Surface to TOA)
            real_t current_up = 100.0 * 1e-6; // Surface emission boundary mock value
            for (size_t lay = 0; lay < layers; ++lay) {
                real_t t = tau(gp, lay, col);
                real_t trans = std::exp(-t);
                real_t emit = lay_source(gp, lay, col) * (1.0 - trans);
                current_up = current_up * trans + emit;
            }
            flux_up(gp, col) = current_up;
        }
    }
}

} // namespace rte
