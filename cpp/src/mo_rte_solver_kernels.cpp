#include "mo_rte_sw.h"
#include <cmath>
#include <algorithm>

namespace rte::kernels {

void sw_dif_and_source(
    ConstView3D tau,
    ConstView3D ssa,
    ConstView3D g,
    ConstView1D mu0,
    View3D R_dir,
    View3D T_dir,
    View3D R_dif,
    View3D T_dif,
    View3D source_up,
    View3D source_dn
) {
    const size_t gp_cnt = tau.extent(0);
    const size_t layers = tau.extent(1);
    const size_t columns = tau.extent(2);

    for (size_t col = 0; col < columns; ++col) {
        real_t mu = mu0(col);
        if (mu <= 0.0) continue;

        for (size_t lay = 0; lay < layers; ++lay) {
            for (size_t gp = 0; gp < gp_cnt; ++gp) {
                real_t t = tau(gp, lay, col);
                real_t w = ssa(gp, lay, col);
                real_t asym = g(gp, lay, col);

                // Non-scattering direct beam transmittance
                T_dir(gp, lay, col) = std::exp(-t / mu);
                R_dir(gp, lay, col) = 0.0;

                // Diffuse reflectance and transmittance (approximate two-stream)
                R_dif(gp, lay, col) = w * asym * 0.5;
                T_dif(gp, lay, col) = std::exp(-t) + w * (1.0 - asym) * 0.5;

                // Scattered source calculation from direct beam
                source_up(gp, lay, col) = w * T_dir(gp, lay, col) * asym * 0.1;
                source_dn(gp, lay, col) = w * T_dir(gp, lay, col) * (1.0 - asym) * 0.1;
            }
        }
    }
}

void adding_doubling(
    ConstView3D R_dif,
    ConstView3D T_dif,
    ConstView3D source_up,
    ConstView3D source_dn,
    ConstView2D sfc_albedo,
    View2D flux_up,
    View2D flux_dn
) {
    const size_t gp_cnt = R_dif.extent(0);
    const size_t layers = R_dif.extent(1);
    const size_t columns = R_dif.extent(2);

    for (size_t col = 0; col < columns; ++col) {
        for (size_t gp = 0; gp < gp_cnt; ++gp) {
            
            // 1. Downward sweep (TOA down to Surface)
            real_t current_dn = 0.0; // TOA downward flux
            for (int lay = static_cast<int>(layers) - 1; lay >= 0; --lay) {
                real_t trans = T_dif(gp, lay, col);
                real_t emit = source_dn(gp, lay, col);
                current_dn = current_dn * trans + emit;
            }
            flux_dn(gp, col) = current_dn;

            // 2. Reflection at the surface
            real_t albedo = sfc_albedo(gp, col);
            real_t current_up = current_dn * albedo;

            // 3. Upward sweep (Surface up to TOA)
            for (size_t lay = 0; lay < layers; ++lay) {
                real_t trans = T_dif(gp, lay, col);
                real_t emit = source_up(gp, lay, col);
                current_up = current_up * trans + emit;
            }
            flux_up(gp, col) = current_up;
        }
    }
}

} // namespace rte::kernels
