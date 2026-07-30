#include "mo_gas_optics.h"
#include "mo_cloud_optics.h"
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rrtmgp;

// Standard C++ solvers wrapper (compiled in standard translation unit, zero Kokkos headers)
void run_standard_solvers(
    size_t layers, size_t columns, size_t gpoints,
    const std::vector<real_t>& play,
    const std::vector<real_t>& tlay,
    const std::vector<real_t>& clwp,
    const std::vector<real_t>& rel,
    std::vector<real_t>& flux_std
) {
    const size_t bands = 4;

    std::vector<int> gpt_band_data(gpoints, 0);
    std::vector<int> band_lims_data(2 * bands, 0);
    auto gpt_view = IntView1D(gpt_band_data.data(), Extents1D(gpoints));
    auto blm_view = IntView2D(band_lims_data.data(), Extents2D(2, bands));
    OpticalProps spectral_props(gpt_view, blm_view);

    std::vector<real_t> kmajor_data(gpoints * 2 * 2 * 1, 2.5);
    std::vector<real_t> kminor_data(gpoints * 2 * 1, 0.8);
    std::vector<real_t> lut_liquid_data(gpoints * 2 * 2, 5.0);
    std::vector<real_t> lut_ice_data(gpoints * 2 * 2, 2.0);
    auto kmajor_std = ConstView4D(kmajor_data.data(), Extents4D(gpoints, 2, 2, 1));
    auto kminor_std = ConstView3D(kminor_data.data(), Extents3D(gpoints, 2, 1));
    auto lut_liq_std = ConstView3D(lut_liquid_data.data(), Extents3D(gpoints, 2, 2));
    auto lut_ice_std = ConstView3D(lut_ice_data.data(), Extents3D(gpoints, 2, 2));

    GasOptics gas_optics_std(spectral_props, kmajor_std, kminor_std);
    CloudOptics cloud_optics_std(spectral_props, lut_liq_std, lut_ice_std);

    auto play_std = ConstView2D(play.data(), Extents2D(layers, columns));
    auto tlay_std = ConstView2D(tlay.data(), Extents2D(layers, columns));
    auto clwp_std = ConstView2D(clwp.data(), Extents2D(layers, columns));
    auto rel_std = ConstView2D(rel.data(), Extents2D(layers, columns));

    // 1. Gas optics computation
    std::vector<real_t> tau_gas_std(gpoints * layers * columns, 0.0);
    auto tau_gas_view = View3D(tau_gas_std.data(), Extents3D(gpoints, layers, columns));
    gas_optics_std.compute_optical_properties(play_std, tlay_std, tau_gas_view);

    // 2. Cloud parameterization computation
    std::vector<real_t> tau_cloud_std(gpoints * layers * columns, 0.0);
    std::vector<real_t> ssa_cloud_std(gpoints * layers * columns, 0.0);
    std::vector<real_t> g_cloud_std(gpoints * layers * columns, 0.0);

    auto tau_cloud_view = View3D(tau_cloud_std.data(), Extents3D(gpoints, layers, columns));
    auto ssa_cloud_view = View3D(ssa_cloud_std.data(), Extents3D(gpoints, layers, columns));
    auto g_cloud_view = View3D(g_cloud_std.data(), Extents3D(gpoints, layers, columns));

    cloud_optics_std.compute_cloud_optics(clwp_std, clwp_std, rel_std, rel_std, tau_cloud_view, ssa_cloud_view, g_cloud_view);

    // 3. Simple solver loops matching minimax formula
    for (size_t col = 0; col < columns; ++col) {
        for (size_t gp = 0; gp < gpoints; ++gp) {
            real_t accum = 0.0;
            for (size_t lay = 0; lay < layers; ++lay) {
                real_t tau_g = kmajor_data[gp] * std::exp(-play_std(lay, col) * tlay_std(lay, col) * kmajor_data[gp] * 1e-6);
                real_t tau_c = clwp_std(lay, col) * lut_liquid_data[gp];
                accum += (tau_g + tau_c) * 10.0;
            }
            flux_std[gp + col * gpoints] = accum;
        }
    }
}
