#include "mo_gas_optics_kernels.h"
#include <cmath>
#include <algorithm>

namespace rrtmgp::kernels {

// Ported interpolate3D_byflav inline helper
template<typename KMajorView, typename FMajorView, typename JetaView, typename ColMixView, typename TauView>
inline void interpolate3D_byflav(
    int jtemp, int jpress, int gptS, int gptE,
    const ColMixView& col_mix, const FMajorView& fmajor, const KMajorView& kmajor,
    const JetaView& jeta, TauView& tau, int col, int lay, int iflav) {

    int jeta1 = jeta(0, lay, col, iflav);
    int jeta2 = jeta(1, lay, col, iflav);
    real_t scaling1 = col_mix(0, lay, col, iflav);
    real_t scaling2 = col_mix(1, lay, col, iflav);

    real_t f000 = fmajor(0, 0, 0, lay, col, iflav);
    real_t f100 = fmajor(1, 0, 0, lay, col, iflav);
    real_t f010 = fmajor(0, 1, 0, lay, col, iflav);
    real_t f110 = fmajor(1, 1, 0, lay, col, iflav);

    real_t f001 = fmajor(0, 0, 1, lay, col, iflav);
    real_t f101 = fmajor(1, 0, 1, lay, col, iflav);
    real_t f011 = fmajor(0, 1, 1, lay, col, iflav);
    real_t f111 = fmajor(1, 1, 1, lay, col, iflav);

    int jp_m1 = jpress - 1;

    #pragma omp simd
    for (int igpt = gptS; igpt <= gptE; ++igpt) {
        real_t t1 = f000 * kmajor(igpt, jeta1, jp_m1, jtemp) +
                    f100 * kmajor(igpt, jeta1 + 1, jp_m1, jtemp) +
                    f010 * kmajor(igpt, jeta1, jpress, jtemp) +
                    f110 * kmajor(igpt, jeta1 + 1, jpress, jtemp);

        real_t t2 = f001 * kmajor(igpt, jeta2, jp_m1, jtemp + 1) +
                    f101 * kmajor(igpt, jeta2 + 1, jp_m1, jtemp + 1) +
                    f011 * kmajor(igpt, jeta2, jpress, jtemp + 1) +
                    f111 * kmajor(igpt, jeta2 + 1, jpress, jtemp + 1);

        tau(igpt, lay, col) += scaling1 * t1 + scaling2 * t2;
    }
}

template<typename KMinorView, typename FMinorView, typename JetaView, typename TauView>
inline void interpolate2D_byflav(
    int jtemp, int gptS, int gptE, int kminor_start,
    const FMinorView& fminor, const KMinorView& kminor,
    const JetaView& jeta, TauView& tau, int col, int lay, int iflav, real_t scaling) {

    real_t fm00 = fminor(0, 0, lay, col, iflav);
    real_t fm10 = fminor(1, 0, lay, col, iflav);
    real_t fm01 = fminor(0, 1, lay, col, iflav);
    real_t fm11 = fminor(1, 1, lay, col, iflav);

    int jeta0 = jeta(0, lay, col, iflav);
    int jeta1 = jeta(1, lay, col, iflav);

    #pragma omp simd
    for (int igpt = gptS; igpt <= gptE; ++igpt) {
        int idx = kminor_start + (igpt - gptS);
        real_t t = fm00 * kminor(idx, jeta0, jtemp) +
                   fm10 * kminor(idx, jeta0 + 1, jtemp) +
                   fm01 * kminor(idx, jeta1, jtemp + 1) +
                   fm11 * kminor(idx, jeta1 + 1, jtemp + 1);
        tau(igpt, lay, col) += scaling * t;
    }
}

template<typename KMinorView, typename MinorLimitsView, typename MinorScalesView, typename ScaleCompView, typename IdxMinorView, typename IdxScaleView, typename KMinorStartView>
void gas_optical_depths_minor(
    int ncol, int nlay, int ngpt, int ngas, int nflav, int ntemp, int neta, int nminor,
    int idx_h2o,
    ConstView1D_Int gpt_flv,
    const KMinorView& kminor,
    const MinorLimitsView& minor_limits_gpt,
    const MinorScalesView& minor_scales_with_density,
    const ScaleCompView& scale_by_complement,
    const IdxMinorView& idx_minor, const IdxScaleView& idx_minor_scaling,
    const KMinorStartView& kminor_start,
    ConstView2D play, ConstView2D tlay, ConstView3D col_gas,
    ConstView5D fminor, ConstView4D_Int jeta,
    ConstView2D_Int layer_limits, ConstView2D_Int jtemp,
    View3D tau
) {
    const real_t PaTohPa = 0.01;

    for (int imnr = 0; imnr < nminor; ++imnr) {
        #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
            int lay_start = layer_limits(0, col);
            int lay_end = layer_limits(1, col);

            if (lay_start >= 0 && lay_end >= 0 && lay_start <= lay_end) {
                for (int lay = lay_start; lay <= lay_end; ++lay) {
                    real_t scaling = col_gas(idx_minor(imnr) - 1, lay, col);

                    if (minor_scales_with_density(imnr)) {
                        scaling *= (PaTohPa * play(lay, col) / tlay(lay, col));
                        int idx_scale = idx_minor_scaling(imnr) - 1;
                        if (idx_scale >= 0) {
                            real_t vmr_fact = 1.0 / col_gas(0, lay, col);
                            real_t dry_fact = 1.0 / (1.0 + col_gas(idx_h2o - 1, lay, col) * vmr_fact);

                            if (scale_by_complement(imnr)) {
                                scaling *= (1.0 - col_gas(idx_scale, lay, col) * vmr_fact * dry_fact);
                            } else {
                                scaling *= (col_gas(idx_scale, lay, col) * vmr_fact * dry_fact);
                            }
                        }
                    }

                    int gptS = minor_limits_gpt(0, imnr) - 1;
                    int gptE = minor_limits_gpt(1, imnr) - 1;
                    int iflav = gpt_flv(gptS) - 1;

                    interpolate2D_byflav(
                        jtemp(lay, col), gptS, gptE, kminor_start(imnr) - 1,
                        fminor, kminor, jeta, tau, col, lay, iflav, scaling
                    );
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
    // 1. Layer limits
    std::vector<int> itropo_lower_data(ncol * 2, -1);
    std::vector<int> itropo_upper_data(ncol * 2, -1);
    auto itropo_lower = View2D_Int(itropo_lower_data.data(), Extents2D(2, ncol));
    auto itropo_upper = View2D_Int(itropo_upper_data.data(), Extents2D(2, ncol));

    bool top_at_1 = play(0, 0) < play(nlay - 1, 0);

    #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
        int lower_start = -1, lower_end = -1;
        int upper_start = -1, upper_end = -1;

        if (top_at_1) {
            upper_start = 0;
            for (int lay = 0; lay < nlay; ++lay) {
                if (!tropo(lay, col)) upper_end = lay;
            }
            for (int lay = 0; lay < nlay; ++lay) {
                if (tropo(lay, col)) { lower_start = lay; break; }
            }
            lower_end = nlay - 1;
        } else {
            lower_start = 0;
            for (int lay = 0; lay < nlay; ++lay) {
                if (tropo(lay, col)) lower_end = lay;
            }
            for (int lay = 0; lay < nlay; ++lay) {
                if (!tropo(lay, col)) { upper_start = lay; break; }
            }
            upper_end = nlay - 1;
        }

        itropo_lower(0, col) = lower_start;
        itropo_lower(1, col) = lower_end;
        itropo_upper(0, col) = upper_start;
        itropo_upper(1, col) = upper_end;
    }

    // 2. Major Species
    for (int ibnd = 0; ibnd < nbnd; ++ibnd) {
        int gptS = band_lims_gpt(0, ibnd) - 1;
        int gptE = band_lims_gpt(1, ibnd) - 1;

        #pragma omp parallel for collapse(2)
        for (int lay = 0; lay < nlay; ++lay) {
            for (int col = 0; col < ncol; ++col) {
                int itropo = tropo(lay, col) ? 0 : 1;
                int iflav = gpoint_flavor(itropo, gptS) - 1;
                interpolate3D_byflav(
                    jtemp(lay, col), jpress(lay, col) + itropo, gptS, gptE,
                    col_mix, fmajor, kmajor, jeta, tau, col, lay, iflav
                );
            }
        }
    }

    // 3. Minor lower
    std::vector<int> gpt_flv_lower_data(ngpt);
    for (int i = 0; i < ngpt; ++i) gpt_flv_lower_data[i] = gpoint_flavor(0, i);
    auto gpt_flv_lower = ConstView1D_Int(gpt_flv_lower_data.data(), Extents1D(ngpt));

    if (nminorlower > 0) {
        gas_optical_depths_minor(
            ncol, nlay, ngpt, ngas, nflav, ntemp, neta, nminorlower, idx_h2o,
            gpt_flv_lower, kminor_lower, minor_limits_gpt_lower, minor_scales_with_density_lower,
            scale_by_complement_lower, idx_minor_lower, idx_minor_scaling_lower, kminor_start_lower,
            play, tlay, col_gas, fminor, jeta, itropo_lower, jtemp, tau
        );
    }

    // 4. Minor upper
    std::vector<int> gpt_flv_upper_data(ngpt);
    for (int i = 0; i < ngpt; ++i) gpt_flv_upper_data[i] = gpoint_flavor(1, i);
    auto gpt_flv_upper = ConstView1D_Int(gpt_flv_upper_data.data(), Extents1D(ngpt));

    if (nminorupper > 0) {
        gas_optical_depths_minor(
            ncol, nlay, ngpt, ngas, nflav, ntemp, neta, nminorupper, idx_h2o,
            gpt_flv_upper, kminor_upper, minor_limits_gpt_upper, minor_scales_with_density_upper,
            scale_by_complement_upper, idx_minor_upper, idx_minor_scaling_upper, kminor_start_upper,
            play, tlay, col_gas, fminor, jeta, itropo_upper, jtemp, tau
        );
    }
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
    for (int ibnd = 0; ibnd < nbnd; ++ibnd) {
        int gptS = band_lims_gpt(0, ibnd) - 1;
        int gptE = band_lims_gpt(1, ibnd) - 1;

        #pragma omp parallel for collapse(2)
        for (int lay = 0; lay < nlay; ++lay) {
            for (int col = 0; col < ncol; ++col) {
                int itropo = tropo(lay, col) ? 0 : 1;
                int iflav = gpoint_flavor(itropo, gptS) - 1;

                real_t scaling = col_gas(idx_h2o - 1, lay, col) + col_dry(lay, col);

                // Inline interpolate2D_byflav specifically for krayl because krayl is [ngpt, neta, ntemp, 2]
                for (int igpt = gptS; igpt <= gptE; ++igpt) {
                    real_t t = fminor(0, 0, lay, col, iflav) * krayl(igpt, jeta(0, lay, col, iflav), jtemp(lay, col), itropo) +
                               fminor(1, 0, lay, col, iflav) * krayl(igpt, jeta(0, lay, col, iflav) + 1, jtemp(lay, col), itropo) +
                               fminor(0, 1, lay, col, iflav) * krayl(igpt, jeta(1, lay, col, iflav), jtemp(lay, col) + 1, itropo) +
                               fminor(1, 1, lay, col, iflav) * krayl(igpt, jeta(1, lay, col, iflav) + 1, jtemp(lay, col) + 1, itropo);
                    tau_rayleigh(igpt, lay, col) = scaling * t;
                }
            }
        }
    }
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
    std::vector<real_t> pfrac_data(ncol * nlay * ngpt);
    auto pfrac = View3D(pfrac_data.data(), Extents3D(ngpt, nlay, ncol));

    for (int ibnd = 0; ibnd < nbnd; ++ibnd) {
        int gptS = band_lims_gpt(0, ibnd) - 1;
        int gptE = band_lims_gpt(1, ibnd) - 1;

        #pragma omp parallel for collapse(2)
        for (int lay = 0; lay < nlay; ++lay) {
            for (int col = 0; col < ncol; ++col) {
                int itropo = tropo(lay, col) ? 0 : 1;
                int iflav = gpoint_flavor(itropo, gptS) - 1;

                int jeta1 = jeta(0, lay, col, iflav);
                int jeta2 = jeta(1, lay, col, iflav);

                for (int igpt = gptS; igpt <= gptE; ++igpt) {
                    real_t t1 = fmajor(0, 0, 0, lay, col, iflav) * planck_frac(igpt, jeta1, jpress(lay, col) + itropo - 1, jtemp(lay, col)) +
                                fmajor(1, 0, 0, lay, col, iflav) * planck_frac(igpt, jeta1 + 1, jpress(lay, col) + itropo - 1, jtemp(lay, col)) +
                                fmajor(0, 1, 0, lay, col, iflav) * planck_frac(igpt, jeta1, jpress(lay, col) + itropo, jtemp(lay, col)) +
                                fmajor(1, 1, 0, lay, col, iflav) * planck_frac(igpt, jeta1 + 1, jpress(lay, col) + itropo, jtemp(lay, col));

                    real_t t2 = fmajor(0, 0, 1, lay, col, iflav) * planck_frac(igpt, jeta2, jpress(lay, col) + itropo - 1, jtemp(lay, col) + 1) +
                                fmajor(1, 0, 1, lay, col, iflav) * planck_frac(igpt, jeta2 + 1, jpress(lay, col) + itropo - 1, jtemp(lay, col) + 1) +
                                fmajor(0, 1, 1, lay, col, iflav) * planck_frac(igpt, jeta2, jpress(lay, col) + itropo, jtemp(lay, col) + 1) +
                                fmajor(1, 1, 1, lay, col, iflav) * planck_frac(igpt, jeta2 + 1, jpress(lay, col) + itropo, jtemp(lay, col) + 1);

                    pfrac(igpt, lay, col) = t1 + t2;
                }
            }
        }
    }

    real_t totplnk_delta_inv = 1.0 / totplnk_delta;

    #pragma omp parallel for
    for (int col = 0; col < ncol; ++col) {
        real_t t_sfc = tsfc(col);
        real_t fac_sfc = (t_sfc - totplnk_min) * totplnk_delta_inv;
        int j_sfc = std::min(nPlanckTemp - 2, std::max(0, static_cast<int>(fac_sfc)));
        real_t f_sfc = fac_sfc - static_cast<real_t>(j_sfc);

        for (int lay = 0; lay < nlay; ++lay) {
            real_t t_lay = tlay(lay, col);
            real_t fac_lay = (t_lay - totplnk_min) * totplnk_delta_inv;
            int j_lay = std::min(nPlanckTemp - 2, std::max(0, static_cast<int>(fac_lay)));
            real_t f_lay = fac_lay - static_cast<real_t>(j_lay);

            for (int igpt = 0; igpt < ngpt; ++igpt) {
                int ibnd = gpoint_bands(igpt) - 1;

                real_t planck_sfc = totplnk(ibnd, j_sfc) * (1.0 - f_sfc) + totplnk(ibnd, j_sfc + 1) * f_sfc;
                real_t planck_lay = totplnk(ibnd, j_lay) * (1.0 - f_lay) + totplnk(ibnd, j_lay + 1) * f_lay;

                if (lay == sfc_lay - 1) { // 0-indexed surface layer
                    sfc_source(igpt, col) = pfrac(igpt, lay, col) * planck_sfc;
                }
                lay_source(igpt, lay, col) = pfrac(igpt, lay, col) * planck_lay;
            }
        }
    }
}

} // namespace rrtmgp::kernels
