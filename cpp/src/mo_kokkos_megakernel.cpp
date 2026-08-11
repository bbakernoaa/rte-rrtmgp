#include "mo_kokkos_megakernel.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>

namespace rrtmgp {

// Fast exponential for radiative transport using range scaling and repeated squaring
KOKKOS_INLINE_FUNCTION real_t fast_exp(real_t x) {
#ifdef ENABLE_FAST_EXP
    if (x < real_t(-16.0)) return real_t(0.0);
    if (x > real_t(0.0)) return std::exp(x);

    real_t y = x * real_t(0.125);
    real_t p = real_t(1.0) + y * (real_t(1.0) + y * (real_t(0.5) + y * (real_t(0.16666666666666667) + y * (real_t(0.041666666666666664) + y * real_t(0.008333333333333333)))));
    p *= p; // p^2
    p *= p; // p^4
    return p * p; // p^8
#else
    return std::exp(x);
#endif
}

void KokkosMegakernel::execute_megakernel(
    size_t layers,
    size_t columns,
    size_t gpoints,
    KConstView2D play,
    KConstView2D tlay,
    KConstView1D tsfc,
    KConstView4D kmajor,
    KConstView3D kminor,
    KConstView2D clwp,
    KConstView3D lut_liquid,
    KConstView1D solar_zenith_angle,
    KConstView1D toa_flux,
    KConstView3D tau_aerosol,
    KConstView3D ssa_aerosol,
    KConstView3D g_aerosol,
    KConstView2D planck_table,
    KView2D flux_up,
    KView2D flux_dn
) {
    (void)solar_zenith_angle;
    (void)ssa_aerosol;
    (void)g_aerosol;

    if (layers > 256) {
        Kokkos::abort("KokkosMegakernel Error: layers exceed maximum stack buffer size (256)");
    }

    const size_t tile_size = 16;
    const size_t num_tiles = (columns + tile_size - 1) / tile_size;

    bool has_aerosol = (tau_aerosol.size() > 0);
    bool has_kminor = (kminor.size() > 0);
    bool has_tsfc = (tsfc.size() > 0);
    bool has_planck = (planck_table.size() > 0);
    bool has_toa = (toa_flux.size() > 0);
    bool has_flux_up = (flux_up.size() > 0);
    bool has_flux_dn = (flux_dn.size() > 0);

    const real_t* play_ptr = play.data();
    const real_t* tlay_ptr = tlay.data();
    const real_t* tsfc_ptr = has_tsfc ? tsfc.data() : nullptr;
    const real_t* kmajor_ptr = kmajor.data();
    const real_t* kminor_ptr = has_kminor ? kminor.data() : nullptr;
    const real_t* clwp_ptr = clwp.data();
    const real_t* lut_liquid_ptr = lut_liquid.data();
    const real_t* tau_aerosol_ptr = has_aerosol ? tau_aerosol.data() : nullptr;
    const real_t* planck_ptr = has_planck ? planck_table.data() : nullptr;
    const real_t* toa_ptr = has_toa ? toa_flux.data() : nullptr;

    real_t* flux_up_ptr = has_flux_up ? flux_up.data() : nullptr;
    real_t* flux_dn_ptr = has_flux_dn ? flux_dn.data() : nullptr;

    constexpr bool is_layout_right = std::is_same_v<KView2D::array_layout, Kokkos::LayoutRight>;

    bool up_is_level = has_flux_up && (flux_up.extent(0) == layers + 1);
    bool dn_is_level = has_flux_dn && (flux_dn.extent(0) == layers + 1);

    Kokkos::parallel_for("fused_rrtmgp_tiled_megakernel", num_tiles,
        KOKKOS_LAMBDA(size_t tile_idx) {
            size_t col_start = tile_idx * tile_size;
            size_t col_end = (col_start + tile_size < columns) ? (col_start + tile_size) : columns;

            for (size_t col = col_start; col < col_end; ++col) {
                real_t t_sfc_val = tsfc_ptr ? tsfc_ptr[col] : 290.0;

                real_t col_flux_dn[257] = {0.0};
                real_t col_flux_up[257] = {0.0};

                size_t col_offset_2d = is_layout_right ? (col) : (col * layers);
                size_t col_stride_2d = is_layout_right ? (columns) : (1);

                size_t col_offset_3d = is_layout_right ? (col) : (col * layers * gpoints);
                size_t col_stride_3d = is_layout_right ? (columns) : (1);
                size_t gp_stride_3d = is_layout_right ? (columns * layers) : (layers);

                real_t fmajor_col[256][2][2][2];
                real_t fminor_col[256][2][2];
                real_t pt_fact_col[256];

                real_t lay_src_2d[256][128];
                real_t lev_src_2d[257][128];
                real_t tau_tot_2d[256][128];
                real_t g_flux_dn_2d[257][128];
                real_t g_flux_up_2d[257][128];

                for (size_t lay = 0; lay < layers; ++lay) {
                    size_t lay_idx_2d = is_layout_right ? (lay * col_stride_2d + col_offset_2d) : (lay + col_offset_2d);
                    real_t p = play_ptr[lay_idx_2d];
                    real_t t = tlay_ptr[lay_idx_2d];

                    pt_fact_col[lay] = -p * t * 1e-6;

                    real_t temp_ref_min = 160.0;
                    real_t temp_ref_delta = 15.0;
                    real_t t_diff = t - temp_ref_min;
                    int jtemp = static_cast<int>(t_diff / temp_ref_delta);
                    if (jtemp < 0) jtemp = 0;
                    real_t ftemp = (t - (temp_ref_min + jtemp * temp_ref_delta)) / temp_ref_delta;

                    real_t locpress = 1.0 + (std::log(p > 1.0 ? p : 1.0) - std::log(10.0)) / std::log(10.0);
                    int jpress = static_cast<int>(locpress);
                    if (jpress < 0) jpress = 0;
                    real_t fpress = locpress - static_cast<real_t>(jpress);

                    real_t feta = 0.5;
                    fminor_col[lay][0][0] = (1.0 - feta) * (1.0 - ftemp);
                    fminor_col[lay][1][0] = feta * (1.0 - ftemp);
                    fminor_col[lay][0][1] = (1.0 - feta) * ftemp;
                    fminor_col[lay][1][1] = feta * ftemp;

                    fmajor_col[lay][0][0][0] = (1.0 - fpress) * fminor_col[lay][0][0];
                    fmajor_col[lay][1][0][0] = (1.0 - fpress) * fminor_col[lay][1][0];
                    fmajor_col[lay][0][1][0] = (1.0 - fpress) * fminor_col[lay][0][1];
                    fmajor_col[lay][1][1][0] = (1.0 - fpress) * fminor_col[lay][1][1];
                    fmajor_col[lay][0][0][1] = fpress * fminor_col[lay][0][0];
                    fmajor_col[lay][1][0][1] = fpress * fminor_col[lay][1][0];
                    fmajor_col[lay][0][1][1] = fpress * fminor_col[lay][0][1];
                    fmajor_col[lay][1][1][1] = fpress * fminor_col[lay][1][1];
                }

                // 2. Optical depth & Planck sources per layer SIMD-vectorized over g-points
                for (size_t lay = 0; lay < layers; ++lay) {
                    size_t lay_idx_2d = is_layout_right ? (lay * col_stride_2d + col_offset_2d) : (lay + col_offset_2d);
                    real_t t = tlay_ptr[lay_idx_2d];
                    real_t lay_planck = 1.0 + 0.001 * (t - 250.0);
                    real_t fac = (t - 160.0) / 15.0;
                    int jp = static_cast<int>(fac);
                    if (jp < 0) jp = 0;
                    if (jp > 8) jp = 8;
                    real_t fp = fac - static_cast<real_t>(jp);

                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        real_t k_val = 0.0;
                        if (kmajor_ptr) {
                            k_val += fmajor_col[lay][0][0][0] * kmajor_ptr[gp * 8 + 0] +
                                     fmajor_col[lay][1][0][0] * kmajor_ptr[gp * 8 + 4] +
                                     fmajor_col[lay][0][1][0] * kmajor_ptr[gp * 8 + 2] +
                                     fmajor_col[lay][1][1][0] * kmajor_ptr[gp * 8 + 6] +
                                     fmajor_col[lay][0][0][1] * kmajor_ptr[gp * 8 + 1] +
                                     fmajor_col[lay][1][0][1] * kmajor_ptr[gp * 8 + 5] +
                                     fmajor_col[lay][0][1][1] * kmajor_ptr[gp * 8 + 3] +
                                     fmajor_col[lay][1][1][1] * kmajor_ptr[gp * 8 + 7];
                        }
                        if (kminor_ptr) {
                            k_val += fminor_col[lay][0][0] * kminor_ptr[gp * 4 + 0] +
                                     fminor_col[lay][1][0] * kminor_ptr[gp * 4 + 2] +
                                     fminor_col[lay][0][1] * kminor_ptr[gp * 4 + 1] +
                                     fminor_col[lay][1][1] * kminor_ptr[gp * 4 + 3];
                        }

                        real_t tau_gas = k_val * fast_exp(pt_fact_col[lay] * k_val);
                        real_t lut_val = lut_liquid_ptr ? lut_liquid_ptr[gp * 4] : 5.0;
                        real_t tau_cloud = clwp_ptr[lay_idx_2d] * lut_val;
                        real_t tau_aero = 0.0;
                        if (has_aerosol) {
                            size_t aero_idx = is_layout_right ? (gp * gp_stride_3d + col_offset_3d + lay * col_stride_3d) : (lay + gp * gp_stride_3d + col_offset_3d);
                            tau_aero = tau_aerosol_ptr[aero_idx];
                        }

                        tau_tot_2d[lay][gp] = tau_gas + tau_cloud + tau_aero;

                        if (planck_ptr) {
                            lay_src_2d[lay][gp] = planck_ptr[gp * 10 + jp] * (1.0 - fp) + planck_ptr[gp * 10 + jp + 1] * fp;
                        } else {
                            lay_src_2d[lay][gp] = lay_planck;
                        }
                    }
                }

                // 3. Level interface sources
                #pragma omp simd
                for (size_t gp = 0; gp < gpoints; ++gp) {
                    lev_src_2d[0][gp] = lay_src_2d[0][gp];
                }
                for (size_t lay = 1; lay < layers; ++lay) {
                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        lev_src_2d[lay][gp] = 0.5 * (lay_src_2d[lay - 1][gp] + lay_src_2d[lay][gp]);
                    }
                }
                if (planck_ptr) {
                    real_t fac = (t_sfc_val - 160.0) / 15.0;
                    int jp = static_cast<int>(fac);
                    if (jp < 0) jp = 0;
                    if (jp > 8) jp = 8;
                    real_t fp = fac - static_cast<real_t>(jp);
                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        lev_src_2d[layers][gp] = planck_ptr[gp * 10 + jp] * (1.0 - fp) + planck_ptr[gp * 10 + jp + 1] * fp;
                    }
                } else {
                    real_t lev_sfc = 1.0 + 0.001 * (t_sfc_val - 250.0);
                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        lev_src_2d[layers][gp] = lev_sfc;
                    }
                }

                // 4. Precompute transmissivity and sources for both downward and upward transport
                real_t trans_2d[256][128];
                real_t src_dn_2d[256][128];
                real_t src_up_2d[256][128];

                const real_t secant = 1.66;
                const real_t tau_thresh = 0.0186;

                for (size_t lay = 0; lay < layers; ++lay) {
                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        real_t tau_loc = tau_tot_2d[lay][gp] * secant;
                        real_t trans = fast_exp(-tau_loc);
                        real_t fact;
                        if (tau_loc > tau_thresh) {
                            fact = (1.0 - trans) / tau_loc - trans;
                        } else {
                            fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * 1.0 / 8.0));
                        }
                        trans_2d[lay][gp] = trans;
                        src_dn_2d[lay][gp] = (1.0 - trans) * lev_src_2d[lay + 1][gp] + 2.0 * fact * (lay_src_2d[lay][gp] - lev_src_2d[lay + 1][gp]);
                        src_up_2d[lay][gp] = (1.0 - trans) * lev_src_2d[lay][gp] + 2.0 * fact * (lay_src_2d[lay][gp] - lev_src_2d[lay][gp]);
                    }
                }

                // 5. Downward transport
                #pragma omp simd
                for (size_t gp = 0; gp < gpoints; ++gp) {
                    g_flux_dn_2d[0][gp] = toa_ptr ? toa_ptr[gp] : 0.0;
                }

                for (size_t lay = 0; lay < layers; ++lay) {
                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        g_flux_dn_2d[lay + 1][gp] = trans_2d[lay][gp] * g_flux_dn_2d[lay][gp] + src_dn_2d[lay][gp];
                    }
                }

                // 6. Upward transport
                real_t sfc_emis = 0.98;
                real_t sfc_albedo = 1.0 - sfc_emis;

                #pragma omp simd
                for (size_t gp = 0; gp < gpoints; ++gp) {
                    g_flux_up_2d[layers][gp] = g_flux_dn_2d[layers][gp] * sfc_albedo + sfc_emis * lev_src_2d[layers][gp];
                }

                for (int lay = static_cast<int>(layers) - 1; lay >= 0; --lay) {
                    #pragma omp simd
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        g_flux_up_2d[lay][gp] = trans_2d[lay][gp] * g_flux_up_2d[lay + 1][gp] + src_up_2d[lay][gp];
                    }
                }

                // 6. Spectral integration over g-points
                constexpr real_t pi_val = 3.14159265358979323846;
                for (size_t lev = 0; lev <= layers; ++lev) {
                    real_t sum_dn = 0.0;
                    real_t sum_up = 0.0;
                    #pragma omp simd reduction(+:sum_dn, sum_up)
                    for (size_t gp = 0; gp < gpoints; ++gp) {
                        sum_dn += g_flux_dn_2d[lev][gp];
                        sum_up += g_flux_up_2d[lev][gp];
                    }
                    col_flux_dn[lev] = sum_dn * pi_val;
                    col_flux_up[lev] = sum_up * pi_val;
                }

                if (has_flux_up && flux_up_ptr) {
                    if (up_is_level) {
                        for (size_t lev = 0; lev <= layers; ++lev) {
                            size_t idx = is_layout_right ? (lev * columns + col) : (lev + col * (layers + 1));
                            flux_up_ptr[idx] = col_flux_up[lev];
                        }
                    } else {
                        size_t out_idx = is_layout_right ? (col) : (col * gpoints);
                        flux_up_ptr[out_idx] = col_flux_up[0];
                    }
                }
                if (has_flux_dn && flux_dn_ptr) {
                    if (dn_is_level) {
                        for (size_t lev = 0; lev <= layers; ++lev) {
                            size_t idx = is_layout_right ? (lev * columns + col) : (lev + col * (layers + 1));
                            flux_dn_ptr[idx] = col_flux_dn[lev];
                        }
                    } else {
                        size_t out_idx = is_layout_right ? (col) : (col * gpoints);
                        flux_dn_ptr[out_idx] = col_flux_dn[layers];
                    }
                }
            }
        }
    );
}

} // namespace rrtmgp
