#include "mo_gas_optics_kernels.h"
#include <cmath>
#include <algorithm>

namespace rrtmgp::kernels {

void interpolation(
    int ncol, int nlay, int ngas, int nflav, int neta, int npres, int ntemp,
    ConstView2D_Int flavor,
    ConstView1D press_ref_log, ConstView1D temp_ref,
    real_t press_ref_log_delta, real_t temp_ref_min, real_t temp_ref_delta, real_t press_ref_trop_log,
    ConstView3D vmr_ref,
    ConstView2D play, ConstView2D tlay,
    ConstView3D col_gas,
    View2D_Int jtemp,
    View6D fmajor,
    View5D fminor,
    View4D col_mix,
    View2D_Bool tropo,
    View4D_Int jeta,
    View2D_Int jpress
) {
    (void)ngas;
    real_t temp_ref_delta_inv = 1.0 / temp_ref_delta;
    real_t press_ref_log_1 = press_ref_log(0);
    real_t press_ref_log_delta_inv = 1.0 / press_ref_log_delta;

    #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
        for (int lay = 0; lay < nlay; ++lay) {
            real_t t_diff = tlay(lay, col) - (temp_ref_min - temp_ref_delta);
            int jtemp_ = static_cast<int>(t_diff * temp_ref_delta_inv);
            jtemp(lay, col) = std::min(ntemp - 2, std::max(0, jtemp_ - 1));
            real_t ftemp = (tlay(lay, col) - temp_ref(jtemp(lay, col))) * temp_ref_delta_inv;

            real_t locpress = 1.0 + (std::log(play(lay, col)) - press_ref_log_1) * press_ref_log_delta_inv;
            int jpress_aint = std::min(npres - 2, std::max(0, static_cast<int>(locpress) - 1));
            jpress(lay, col) = jpress_aint;
            real_t fpress = locpress - static_cast<real_t>(jpress_aint + 1);

            tropo(lay, col) = play(lay, col) > std::exp(press_ref_trop_log);

            for (int iflav = 0; iflav < nflav; ++iflav) {
                int igas_1 = flavor(0, iflav) - 1;
                int igas_2 = flavor(1, iflav) - 1;
                int itropo = tropo(lay, col) ? 0 : 1;

                for (int itemp = 0; itemp < 2; ++itemp) {
                    real_t ratio_eta_half = vmr_ref(itropo, igas_1, jtemp(lay, col) + itemp) /
                                            vmr_ref(itropo, igas_2, jtemp(lay, col) + itemp);
                    col_mix(itemp, lay, col, iflav) = col_gas(igas_1, lay, col) + ratio_eta_half * col_gas(igas_2, lay, col);

                    real_t eta = 0.5;
                    if (col_mix(itemp, lay, col, iflav) > 2.0 * 1e-30) {
                        eta = col_gas(igas_1, lay, col) / col_mix(itemp, lay, col, iflav);
                    }
                    real_t loceta = eta * static_cast<real_t>(neta - 1);
                    jeta(itemp, lay, col, iflav) = std::min(static_cast<int>(loceta), neta - 2);
                    real_t feta = loceta - std::floor(loceta);

                    real_t ftemp_term = (itemp == 0) ? (1.0 - ftemp) : ftemp;

                    fminor(0, itemp, lay, col, iflav) = (1.0 - feta) * ftemp_term;
                    fminor(1, itemp, lay, col, iflav) = feta * ftemp_term;

                    fmajor(0, 0, itemp, lay, col, iflav) = (1.0 - fpress) * fminor(0, itemp, lay, col, iflav);
                    fmajor(1, 0, itemp, lay, col, iflav) = (1.0 - fpress) * fminor(1, itemp, lay, col, iflav);
                    fmajor(0, 1, itemp, lay, col, iflav) = fpress * fminor(0, itemp, lay, col, iflav);
                    fmajor(1, 1, itemp, lay, col, iflav) = fpress * fminor(1, itemp, lay, col, iflav);
                }
            }
        }
    }
}

void compute_tau_absorption(
    int ncol, int nlay, int nbnd, int ngpt,
    int ngas, int nflav, int neta, int npres, int ntemp,
    int nminorlower, int nminorklower,
    int nminorupper, int nminorkupper,
    int idx_h2o,
    ConstView2D_Int gpoint_flavor,
    ConstView2D_Int band_lims_gpt,
    ConstView4D kmajor,
    ConstView3D kminor_lower,
    ConstView3D kminor_upper,
    ConstView2D_Int minor_limits_gpt_lower, ConstView2D_Int minor_limits_gpt_upper,
    ConstView1D_Bool minor_scales_with_density_lower, ConstView1D_Bool minor_scales_with_density_upper,
    ConstView1D_Bool scale_by_complement_lower, ConstView1D_Bool scale_by_complement_upper,
    ConstView1D_Int idx_minor_lower, ConstView1D_Int idx_minor_upper,
    ConstView1D_Int idx_minor_scaling_lower, ConstView1D_Int idx_minor_scaling_upper,
    ConstView1D_Int kminor_start_lower, ConstView1D_Int kminor_start_upper,
    ConstView2D_Bool tropo,
    ConstView4D col_mix,
    ConstView6D fmajor,
    ConstView5D fminor,
    ConstView2D play, ConstView2D tlay, ConstView3D col_gas,
    ConstView4D_Int jeta, ConstView2D_Int jtemp, ConstView2D_Int jpress,
    View3D tau
) {
    (void)ncol; (void)nlay; (void)nbnd; (void)ngpt; (void)ngas; (void)nflav; (void)neta; (void)npres; (void)ntemp;
    (void)nminorlower; (void)nminorklower; (void)nminorupper; (void)nminorkupper; (void)idx_h2o;
    (void)gpoint_flavor; (void)band_lims_gpt; (void)kmajor; (void)kminor_lower; (void)kminor_upper;
    (void)minor_limits_gpt_lower; (void)minor_limits_gpt_upper;
    (void)minor_scales_with_density_lower; (void)minor_scales_with_density_upper;
    (void)scale_by_complement_lower; (void)scale_by_complement_upper;
    (void)idx_minor_lower; (void)idx_minor_upper;
    (void)idx_minor_scaling_lower; (void)idx_minor_scaling_upper;
    (void)kminor_start_lower; (void)kminor_start_upper;
    (void)tropo; (void)col_mix; (void)fmajor; (void)fminor;
    (void)play; (void)tlay; (void)col_gas;
    (void)jeta; (void)jtemp; (void)jpress; (void)tau;
    // Stub implementation
}

void compute_tau_rayleigh(
    int ncol, int nlay, int nbnd, int ngpt,
    int ngas, int nflav, int neta, int npres, int ntemp,
    ConstView2D_Int gpoint_flavor,
    ConstView2D_Int band_lims_gpt,
    ConstView4D krayl,
    int idx_h2o, ConstView2D col_dry, ConstView3D col_gas,
    ConstView5D fminor, ConstView4D_Int jeta, ConstView2D_Bool tropo, ConstView2D_Int jtemp,
    View3D tau_rayleigh
) {
    (void)ncol; (void)nlay; (void)nbnd; (void)ngpt; (void)ngas; (void)nflav; (void)neta; (void)npres; (void)ntemp;
    (void)gpoint_flavor; (void)band_lims_gpt; (void)krayl; (void)idx_h2o; (void)col_dry; (void)col_gas;
    (void)fminor; (void)jeta; (void)tropo; (void)jtemp; (void)tau_rayleigh;
    // Stub
}

void compute_Planck_source(
    int ncol, int nlay, int nbnd, int ngpt,
    int nflav, int neta, int npres, int ntemp, int nPlanckTemp,
    ConstView2D tlay, ConstView1D tsfc,
    int sfc_lay,
    ConstView6D fmajor, ConstView4D_Int jeta, ConstView2D_Bool tropo, ConstView2D_Int jtemp, ConstView2D_Int jpress,
    ConstView1D_Int gpoint_bands, ConstView2D_Int band_lims_gpt,
    ConstView4D planck_frac,
    ConstView1D temp_ref,
    real_t totplnk_delta, real_t totplnk_min,
    ConstView2D totplnk,
    ConstView2D_Int gpoint_flavor,
    View2D sfc_source,
    View3D lay_source
) {
    (void)ncol; (void)nlay; (void)nbnd; (void)ngpt; (void)nflav; (void)neta; (void)npres; (void)ntemp; (void)nPlanckTemp;
    (void)tlay; (void)tsfc; (void)sfc_lay; (void)fmajor; (void)jeta; (void)tropo; (void)jtemp; (void)jpress;
    (void)gpoint_bands; (void)band_lims_gpt; (void)planck_frac; (void)temp_ref; (void)totplnk_delta; (void)totplnk_min;
    (void)totplnk; (void)gpoint_flavor; (void)sfc_source; (void)lay_source;
    // Stub
}

} // namespace rrtmgp::kernels
