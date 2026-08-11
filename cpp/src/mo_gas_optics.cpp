#include "mo_gas_optics.h"
#include "mo_rte_util.h"
#include "mo_gas_optics_kernels.h"
#include <algorithm>
#include <cstdint>

namespace rrtmgp {

// OpticalProps implementation
OpticalProps::OpticalProps(IntView1D gpoint_to_band, IntView2D band_lims_gpoint) {
    if (band_lims_gpoint.extent(0) != 2) {
        throw std::invalid_argument("band_lims_gpoint must have first dimension size 2");
    }

    gpoints_ = static_cast<int>(gpoint_to_band.extent(0));
    bands_ = static_cast<int>(band_lims_gpoint.extent(1));

    gpoint_to_band_storage_.assign(gpoint_to_band.data_handle(), gpoint_to_band.data_handle() + gpoint_to_band.size());
    band_lims_gpoint_storage_.assign(band_lims_gpoint.data_handle(), band_lims_gpoint.data_handle() + band_lims_gpoint.size());

    gpoint_to_band_ = IntView1D(gpoint_to_band_storage_.data(), Extents1D(gpoint_to_band_storage_.size()));
    band_lims_gpoint_ = IntView2D(band_lims_gpoint_storage_.data(), Extents2D(2, bands_));

    for (size_t i = 0; i < gpoint_to_band_storage_.size(); ++i) {
        if (gpoint_to_band_storage_[i] < 0 || gpoint_to_band_storage_[i] >= bands_) {
            throw std::out_of_range("gpoint_to_band value at index " + std::to_string(i) + " is out of range of [0, bands - 1]");
        }
    }
}

// GasOptics implementation
GasOptics::GasOptics(
    OpticalProps spectral_props,
    ConstView4D kmajor,
    ConstView3D kminor
) : OpticalProps(spectral_props.get_gpoint_to_band(), spectral_props.get_band_lims_gpoint()) {

    validate_extent(kmajor, 0, get_gpoints(), "kmajor");

    kmajor_storage_.assign(kmajor.data_handle(), kmajor.data_handle() + kmajor.size());
    kmajor_ = ConstView4D(
        kmajor_storage_.data(),
        Extents4D(kmajor.extent(0), kmajor.extent(1), kmajor.extent(2), kmajor.extent(3))
    );
    (void)kminor; // Legacy mock support: kminor handled gracefully or ignored if empty
}

void GasOptics::compute_optical_properties(
    ConstView2D play, ConstView2D plev,
    ConstView2D tlay, ConstView1D tsfc,
    const GasConcentrations& gas_concs,
    View3D tau,
    View3D tau_rayleigh,
    View2D planck_source
) const {
    (void)plev; (void)tsfc; (void)gas_concs; (void)tau_rayleigh; (void)planck_source;
    const int layers = play.extent(0);
    const int columns = play.extent(1);
    const int gp_cnt = get_gpoints();

    validate_extent(tlay, 0, layers, "tlay");
    validate_extent(tlay, 1, columns, "tlay");
    validate_extent(tau, 0, gp_cnt, "tau");
    validate_extent(tau, 1, layers, "tau");
    validate_extent(tau, 2, columns, "tau");

    // Perform physical boundary audits
    for (int col = 0; col < columns; ++col) {
        for (int lay = 0; lay < layers; ++lay) {
            if (play(lay, col) < 0.0) {
                throw std::invalid_argument("play cannot be negative");
            }
            if (tlay(lay, col) < 0.0) {
                throw std::invalid_argument("tlay cannot be below absolute zero (0 K)");
            }
        }
    }

    // Reusable thread-local workspace buffers to eliminate heap allocations per step
    thread_local std::vector<int> jtemp_data;
    thread_local std::vector<int> jpress_data;
    thread_local std::vector<uint8_t> tropo_data;
    thread_local std::vector<int> jeta_data;
    thread_local std::vector<real_t> col_mix_data;
    thread_local std::vector<real_t> fmajor_data;
    thread_local std::vector<real_t> fminor_data;
    thread_local std::vector<real_t> col_gas_data;
    thread_local std::vector<real_t> totplnk_data;
    thread_local std::vector<int> gpt_bnd_data;
    thread_local std::vector<real_t> lay_src_data;
    thread_local std::vector<real_t> col_dry_data;

    size_t sz_lc = static_cast<size_t>(layers) * columns;
    if (jtemp_data.size() < sz_lc) jtemp_data.resize(sz_lc);
    auto jtemp_view = View2D_Int(jtemp_data.data(), Extents2D(layers, columns));

    if (jpress_data.size() < sz_lc) jpress_data.resize(sz_lc);
    auto jpress_view = View2D_Int(jpress_data.data(), Extents2D(layers, columns));

    if (tropo_data.size() < sz_lc) tropo_data.resize(sz_lc);
    auto tropo_view = View2D_Bool(reinterpret_cast<bool*>(tropo_data.data()), Extents2D(layers, columns));

    int num_flav = nflav();
    size_t sz_flav = 2 * static_cast<size_t>(layers) * columns * std::max(1, num_flav);
    if (jeta_data.size() < sz_flav) jeta_data.resize(sz_flav);
    std::fill_n(jeta_data.data(), sz_flav, 0);
    auto jeta_view = View4D_Int(jeta_data.data(), Extents4D(2, layers, columns, num_flav));

    if (col_mix_data.size() < sz_flav) col_mix_data.resize(sz_flav);
    std::fill_n(col_mix_data.data(), sz_flav, 0.0);
    auto col_mix_view = View4D(col_mix_data.data(), Extents4D(2, layers, columns, num_flav));

    size_t sz_fmajor = 2 * 2 * 2 * static_cast<size_t>(layers) * columns * std::max(1, num_flav);
    if (fmajor_data.size() < sz_fmajor) fmajor_data.resize(sz_fmajor);
    std::fill_n(fmajor_data.data(), sz_fmajor, 0.0);
    auto fmajor_view = View6D(fmajor_data.data(), Extents6D(2, 2, 2, layers, columns, num_flav));

    size_t sz_fminor = 2 * 2 * static_cast<size_t>(layers) * columns * std::max(1, num_flav);
    if (fminor_data.size() < sz_fminor) fminor_data.resize(sz_fminor);
    std::fill_n(fminor_data.data(), sz_fminor, 0.0);
    auto fminor_view = View5D(fminor_data.data(), Extents5D(2, 2, layers, columns, num_flav));

    int num_gas = ngas();
    size_t sz_col_gas = static_cast<size_t>(std::max(1, num_gas)) * layers * columns;
    if (col_gas_data.size() < sz_col_gas) col_gas_data.resize(sz_col_gas);
    std::fill_n(col_gas_data.data(), sz_col_gas, 0.0);
    auto col_gas_view = View3D(col_gas_data.data(), Extents3D(std::max(1, num_gas), layers, columns));

    // Copy active gas concentrations into contiguous 3D view
    for (int igas = 0; igas < num_gas; ++igas) {
        try {
            auto vmr_2d = gas_concs.get_vmr(gas_names[igas]);
            #pragma omp parallel for collapse(2)
            for (int col = 0; col < columns; ++col) {
                for (int lay = 0; lay < layers; ++lay) {
                    col_gas_view(igas, lay, col) = vmr_2d(lay, col);
                }
            }
        } catch (const std::out_of_range&) {
            // Gas not provided in current atmosphere
        }
    }

    auto ref_press = ConstView1D(press_ref_log.data(), Extents1D(press_ref_log.size()));
    auto ref_temp = ConstView1D(temp_ref.data(), Extents1D(temp_ref.size()));
    auto vmr_ref_view = ConstView3D(vmr_ref.data(), Extents3D(2, num_gas, ntemp()));
    auto flavor_view = ConstView2D_Int(flavor.data(), Extents2D(2, num_flav));

    if (num_flav > 0 && npres() > 0 && ntemp() > 0) {
        kernels::interpolation(
            columns, layers, num_gas, num_flav, neta(), npres(), ntemp(),
            flavor_view,
            ref_press, ref_temp,
            press_ref_log_delta, temp_ref_min, temp_ref_delta, press_ref_trop_log,
            vmr_ref_view,
            play, tlay,
            col_gas_view,
            jtemp_view,
            fmajor_view,
            fminor_view,
            col_mix_view,
            tropo_view,
            jeta_view,
            jpress_view
        );

        // Setup metadata views for tau absorption
        auto gpoint_flavor_view = ConstView2D_Int(gpoint_flavor.data(), Extents2D(2, gp_cnt));
        auto band_lims_view = get_band_lims_gpoint();

        auto kminor_lower = ConstView3D(kminor_lower_storage_.data(), Extents3D(minor_limits_gpt_lower.size(), neta(), ntemp()));
        auto kminor_upper = ConstView3D(kminor_upper_storage_.data(), Extents3D(minor_limits_gpt_upper.size(), neta(), ntemp()));

        auto min_lim_lower = ConstView2D_Int(minor_limits_gpt_lower.data(), Extents2D(2, minor_limits_gpt_lower.size() / 2));
        auto min_lim_upper = ConstView2D_Int(minor_limits_gpt_upper.data(), Extents2D(2, minor_limits_gpt_upper.size() / 2));

        auto min_scale_lower = ConstView1D_Bool(reinterpret_cast<const bool*>(minor_scales_with_density_lower.data()), Extents1D(minor_scales_with_density_lower.size()));
        auto min_scale_upper = ConstView1D_Bool(reinterpret_cast<const bool*>(minor_scales_with_density_upper.data()), Extents1D(minor_scales_with_density_upper.size()));

        auto comp_lower = ConstView1D_Bool(reinterpret_cast<const bool*>(scale_by_complement_lower.data()), Extents1D(scale_by_complement_lower.size()));
        auto comp_upper = ConstView1D_Bool(reinterpret_cast<const bool*>(scale_by_complement_upper.data()), Extents1D(scale_by_complement_upper.size()));

        auto idx_min_lower = ConstView1D_Int(idx_minor_lower.data(), Extents1D(idx_minor_lower.size()));
        auto idx_min_upper = ConstView1D_Int(idx_minor_upper.data(), Extents1D(idx_minor_upper.size()));

        auto idx_min_scale_lower = ConstView1D_Int(idx_minor_scaling_lower.data(), Extents1D(idx_minor_scaling_lower.size()));
        auto idx_min_scale_upper = ConstView1D_Int(idx_minor_scaling_upper.data(), Extents1D(idx_minor_scaling_upper.size()));

        auto kmin_start_lower = ConstView1D_Int(kminor_start_lower.data(), Extents1D(kminor_start_lower.size()));
        auto kmin_start_upper = ConstView1D_Int(kminor_start_upper.data(), Extents1D(kminor_start_upper.size()));

        // Call the ported absorption kernel
        kernels::compute_tau_absorption(
            columns, layers, get_bands(), gp_cnt,
            num_gas, num_flav, neta(), npres(), ntemp(),
            minor_limits_gpt_lower.size() / 2, kminor_lower.extent(0),
            minor_limits_gpt_upper.size() / 2, kminor_upper.extent(0),
            1 /* idx_h2o mock */,
            gpoint_flavor_view, band_lims_view, kmajor_,
            kminor_lower, kminor_upper,
            min_lim_lower, min_lim_upper,
            min_scale_lower, min_scale_upper,
            comp_lower, comp_upper,
            idx_min_lower, idx_min_upper,
            idx_min_scale_lower, idx_min_scale_upper,
            kmin_start_lower, kmin_start_upper,
            tropo_view, col_mix_view, fmajor_view, fminor_view,
            play, tlay, col_gas_view,
            jeta_view, jtemp_view, jpress_view,
            tau
        );

        // If we're using a longwave optics config (has planck_frac)
        if (planck_frac.size() > 0) {
            auto planck_frac_view = ConstView4D(planck_frac.data(), Extents4D(gp_cnt, neta(), npres(), ntemp()));
            // Mock empty totplnk for now
            size_t sz_totplnk = static_cast<size_t>(get_bands()) * 10;
            if (totplnk_data.size() < sz_totplnk) totplnk_data.resize(sz_totplnk);
            std::fill_n(totplnk_data.data(), sz_totplnk, 0.0);
            auto totplnk_view = ConstView2D(totplnk_data.data(), Extents2D(get_bands(), 10));

            size_t sz_gpt_bnd = static_cast<size_t>(gp_cnt);
            if (gpt_bnd_data.size() < sz_gpt_bnd) gpt_bnd_data.resize(sz_gpt_bnd);
            std::fill_n(gpt_bnd_data.data(), sz_gpt_bnd, 1);
            auto gpt_bnd_view = ConstView1D_Int(gpt_bnd_data.data(), Extents1D(gp_cnt));

            // Mock lay_source view for API consistency
            size_t sz_lay_src = static_cast<size_t>(gp_cnt) * layers * columns;
            if (lay_src_data.size() < sz_lay_src) lay_src_data.resize(sz_lay_src);
            std::fill_n(lay_src_data.data(), sz_lay_src, 0.0);
            auto lay_src_view = View3D(lay_src_data.data(), Extents3D(gp_cnt, layers, columns));

            kernels::compute_Planck_source(
                columns, layers, get_bands(), gp_cnt, num_flav, neta(), npres(), ntemp(), 10,
                tlay, tsfc, 1, fmajor_view, jeta_view, tropo_view, jtemp_view, jpress_view,
                gpt_bnd_view, band_lims_view, planck_frac_view, ref_temp,
                1.0, 100.0, totplnk_view, gpoint_flavor_view, planck_source, lay_src_view
            );
        } else if (rayl_lower.size() > 0) {
            // Shortwave config
            auto krayl_view = ConstView4D(rayl_lower.data(), Extents4D(gp_cnt, neta(), ntemp(), 2));
            size_t sz_col_dry = static_cast<size_t>(layers) * columns;
            if (col_dry_data.size() < sz_col_dry) col_dry_data.resize(sz_col_dry);
            std::fill_n(col_dry_data.data(), sz_col_dry, 0.0);
            auto col_dry_view = ConstView2D(col_dry_data.data(), Extents2D(layers, columns));

            kernels::compute_tau_rayleigh(
                columns, layers, get_bands(), gp_cnt, num_gas, num_flav, neta(), npres(), ntemp(),
                gpoint_flavor_view, band_lims_view, krayl_view, 1, col_dry_view, col_gas_view,
                fminor_view, jeta_view, tropo_view, jtemp_view, tau_rayleigh
            );
        }
    }
}

} // namespace rrtmgp
