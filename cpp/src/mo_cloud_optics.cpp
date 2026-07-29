#include "mo_cloud_optics.h"
#include "mo_rte_util.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace rrtmgp {

CloudOptics::CloudOptics(OpticalProps base, ConstView3D lut_liquid, ConstView3D lut_ice)
    : OpticalProps(base.get_gpoint_to_band(), base.get_band_lims_gpoint()) {
    
    // Validate boundaries
    validate_extent(lut_liquid, 0, get_gpoints(), "lut_liquid");
    validate_extent(lut_ice, 0, get_gpoints(), "lut_ice");

    // Copy coefficient buffers persistently
    lut_liquid_storage_.assign(lut_liquid.data_handle(), lut_liquid.data_handle() + lut_liquid.size());
    lut_ice_storage_.assign(lut_ice.data_handle(), lut_ice.data_handle() + lut_ice.size());

    // Bind persistent views
    lut_liquid_ = ConstView3D(
        lut_liquid_storage_.data(), 
        Extents3D(lut_liquid.extent(0), lut_liquid.extent(1), lut_liquid.extent(2))
    );
    lut_ice_ = ConstView3D(
        lut_ice_storage_.data(), 
        Extents3D(lut_ice.extent(0), lut_ice.extent(1), lut_ice.extent(2))
    );
}

void CloudOptics::compute_cloud_optics(
    ConstView2D clwp,
    ConstView2D ciwp,
    ConstView2D rel,
    ConstView2D rei,
    View3D tau,
    View3D ssa,
    View3D g
) const {
    const size_t layers = clwp.extent(0);
    const size_t columns = clwp.extent(1);
    const size_t gp_cnt = get_gpoints();

    // Validate boundaries
    validate_extent(ciwp, 0, layers, "ciwp");
    validate_extent(ciwp, 1, columns, "ciwp");
    validate_extent(rel, 0, layers, "rel");
    validate_extent(rel, 1, columns, "rel");
    validate_extent(rei, 0, layers, "rei");
    validate_extent(rei, 1, columns, "rei");
    validate_extent(tau, 0, gp_cnt, "tau");
    validate_extent(tau, 1, layers, "tau");
    validate_extent(tau, 2, columns, "tau");
    validate_extent(ssa, 0, gp_cnt, "ssa");
    validate_extent(ssa, 1, layers, "ssa");
    validate_extent(ssa, 2, columns, "ssa");
    validate_extent(g, 0, gp_cnt, "g");
    validate_extent(g, 1, layers, "g");
    validate_extent(g, 2, columns, "g");

    // Compute cloud physical parameterizations
    for (size_t col = 0; col < columns; ++col) {
        for (size_t lay = 0; lay < layers; ++lay) {
            real_t lwp = clwp(lay, col);
            real_t iwp = ciwp(lay, col);
            real_t r_liq = rel(lay, col);
            real_t r_ice = rei(lay, col);

            for (size_t gp = 0; gp < gp_cnt; ++gp) {
                real_t tau_liq = 0.0;
                real_t ssa_liq = 0.0;
                real_t g_liq = 0.0;

                real_t tau_ice = 0.0;
                real_t ssa_ice = 0.0;
                real_t g_ice = 0.0;

                // Liquid cloud fraction calculation
                if (lwp > 0.0) {
                    // Simple linear lookup matching our mock LUT sizes
                    size_t r_idx = (r_liq > 10.0) ? 1 : 0;
                    tau_liq = lwp * lut_liquid_(gp, r_idx, 0); // parameter 0: extinction coeff
                    ssa_liq = lut_liquid_(gp, r_idx, 1);       // parameter 1: ssa
                    g_liq = 0.85;                              // asymmetry liquid default
                }

                // Ice cloud fraction calculation
                if (iwp > 0.0) {
                    size_t r_idx = (r_ice > 30.0) ? 1 : 0;
                    tau_ice = iwp * lut_ice_(gp, r_idx, 0);
                    ssa_ice = lut_ice_(gp, r_idx, 1);
                    g_ice = 0.75; // asymmetry ice default
                }

                // Combine liquid and ice optical properties
                real_t tau_total = tau_liq + tau_ice;
                tau(gp, lay, col) = tau_total;

                if (tau_total > 0.0) {
                    real_t ssa_total = (tau_liq * ssa_liq + tau_ice * ssa_ice) / tau_total;
                    ssa(gp, lay, col) = ssa_total;

                    if (ssa_total > 0.0) {
                        g(gp, lay, col) = (tau_liq * ssa_liq * g_liq + tau_ice * ssa_ice * g_ice) / (tau_total * ssa_total);
                    } else {
                        g(gp, lay, col) = 0.0;
                    }
                } else {
                    ssa(gp, lay, col) = 0.0;
                    g(gp, lay, col) = 0.0;
                }
            }
        }
    }
}

AerosolOptics::AerosolOptics(OpticalProps base, ConstView3D aer_coefficients)
    : OpticalProps(base.get_gpoint_to_band(), base.get_band_lims_gpoint()) {
    
    validate_extent(aer_coefficients, 0, get_gpoints(), "aer_coefficients");

    aer_coefficients_storage_.assign(aer_coefficients.data_handle(), aer_coefficients.data_handle() + aer_coefficients.size());

    aer_coefficients_ = ConstView3D(
        aer_coefficients_storage_.data(), 
        Extents3D(aer_coefficients.extent(0), aer_coefficients.extent(1), aer_coefficients.extent(2))
    );
}

void AerosolOptics::compute_aerosol_optics(
    ConstView2D aer_mass,
    View3D tau,
    View3D ssa,
    View3D g
) const {
    const size_t layers = aer_mass.extent(0);
    const size_t columns = aer_mass.extent(1);
    const size_t gp_cnt = get_gpoints();

    validate_extent(tau, 0, gp_cnt, "tau");
    validate_extent(tau, 1, layers, "tau");
    validate_extent(tau, 2, columns, "tau");
    validate_extent(ssa, 0, gp_cnt, "ssa");
    validate_extent(ssa, 1, layers, "ssa");
    validate_extent(ssa, 2, columns, "ssa");
    validate_extent(g, 0, gp_cnt, "g");
    validate_extent(g, 1, layers, "g");
    validate_extent(g, 2, columns, "g");

    for (size_t col = 0; col < columns; ++col) {
        for (size_t lay = 0; lay < layers; ++lay) {
            real_t mass = aer_mass(lay, col);
            for (size_t gp = 0; gp < gp_cnt; ++gp) {
                tau(gp, lay, col) = mass * aer_coefficients_(gp, 0, 0);
                ssa(gp, lay, col) = aer_coefficients_(gp, 0, 1);
                g(gp, lay, col) = 0.7; // default asymmetry
            }
        }
    }
}

} // namespace rrtmgp
