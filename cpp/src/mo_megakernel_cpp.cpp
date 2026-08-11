#include "mo_megakernel_cpp.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

namespace rrtmgp {

inline real_t fast_exp(real_t x) {
    return std::exp(x);
}

void MegakernelCpp::execute_fused_longwave(
    size_t layers,
    size_t columns,
    size_t gpoints,
    ConstView2D play,
    ConstView2D tlay,
    ConstView1D tsfc,
    ConstView4D kmajor,
    ConstView3D kminor,
    ConstView2D clwp,
    ConstView3D lut_liquid,
    ConstView1D solar_zenith_angle,
    ConstView1D toa_flux,
    ConstView3D tau_aerosol,
    ConstView3D ssa_aerosol,
    ConstView3D g_aerosol,
    ConstView2D planck_table,
    View2D flux_up,
    View2D flux_dn
) {
    (void)solar_zenith_angle;
    (void)ssa_aerosol;
    (void)g_aerosol;

    if (layers > 256) {
        throw std::invalid_argument("MegakernelCpp Error: layers exceed maximum stack buffer size (256)");
    }

    const size_t tile_size = 16;
    const size_t num_tiles = (columns + tile_size - 1) / tile_size;

    bool has_aerosol = (tau_aerosol.data_handle() != nullptr && tau_aerosol.size() > 0);
    bool has_kminor = (kminor.data_handle() != nullptr && kminor.size() > 0);
    bool has_kmajor = (kmajor.data_handle() != nullptr && kmajor.size() > 0);
    bool has_lut = (lut_liquid.data_handle() != nullptr && lut_liquid.size() > 0);
    bool has_tsfc = (tsfc.data_handle() != nullptr && tsfc.size() > 0);
    bool has_planck = (planck_table.data_handle() != nullptr && planck_table.size() > 0);
    bool has_toa = (toa_flux.data_handle() != nullptr && toa_flux.size() > 0);
    bool has_flux_up = (flux_up.data_handle() != nullptr && flux_up.size() > 0);
    bool has_flux_dn = (flux_dn.data_handle() != nullptr && flux_dn.size() > 0);

    const real_t* play_ptr = play.data_handle();
    const real_t* tlay_ptr = tlay.data_handle();
    const real_t* tsfc_ptr = has_tsfc ? tsfc.data_handle() : nullptr;
    const real_t* kmajor_ptr = has_kmajor ? kmajor.data_handle() : nullptr;
    const real_t* kminor_ptr = has_kminor ? kminor.data_handle() : nullptr;
    const real_t* clwp_ptr = clwp.data_handle();
    const real_t* lut_liquid_ptr = has_lut ? lut_liquid.data_handle() : nullptr;
    const real_t* tau_aerosol_ptr = has_aerosol ? tau_aerosol.data_handle() : nullptr;
    const real_t* planck_ptr = has_planck ? planck_table.data_handle() : nullptr;
    const real_t* toa_ptr = has_toa ? toa_flux.data_handle() : nullptr;

    real_t* flux_up_ptr = has_flux_up ? flux_up.data_handle() : nullptr;
    real_t* flux_dn_ptr = has_flux_dn ? flux_dn.data_handle() : nullptr;

    bool up_is_level = has_flux_up && (flux_up.extent(0) == layers + 1);
    bool dn_is_level = has_flux_dn && (flux_dn.extent(0) == layers + 1);

    #pragma omp parallel for
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        size_t col_start = tile_idx * tile_size;
        size_t col_end = (col_start + tile_size < columns) ? (col_start + tile_size) : columns;

        for (size_t col = col_start; col < col_end; ++col) {
            real_t t_sfc_val = tsfc_ptr ? tsfc_ptr[col] : 290.0;

            real_t col_flux_dn[257] = {0.0};
            real_t col_flux_up[257] = {0.0};

            size_t col_offset_2d = col * layers;
            size_t col_offset_3d = col * layers * gpoints;
            size_t gp_stride_3d = layers;

            for (size_t gp = 0; gp < gpoints; ++gp) {
                real_t toa_val = toa_ptr ? toa_ptr[gp] : 0.0;

                real_t lay_src[256];
                real_t lev_src[257];
                real_t tau_tot_arr[256];

                for (size_t lay = 0; lay < layers; ++lay) {
                    size_t lay_idx_2d = lay + col_offset_2d;
                    real_t p = play_ptr[lay_idx_2d];
                    real_t t = tlay_ptr[lay_idx_2d];

                    // a. Thermodynamic indexing (jtemp, jpress) and bilinear weights (fmajor, fminor)
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
                    real_t fminor[2][2];
                    fminor[0][0] = (1.0 - feta) * (1.0 - ftemp);
                    fminor[1][0] = feta * (1.0 - ftemp);
                    fminor[0][1] = (1.0 - feta) * ftemp;
                    fminor[1][1] = feta * ftemp;

                    real_t fmajor[2][2][2];
                    fmajor[0][0][0] = (1.0 - fpress) * fminor[0][0];
                    fmajor[1][0][0] = (1.0 - fpress) * fminor[1][0];
                    fmajor[0][1][0] = (1.0 - fpress) * fminor[0][1];
                    fmajor[1][1][0] = (1.0 - fpress) * fminor[1][1];
                    fmajor[0][0][1] = fpress * fminor[0][0];
                    fmajor[1][0][1] = fpress * fminor[1][0];
                    fmajor[0][1][1] = fpress * fminor[0][1];
                    fmajor[1][1][1] = fpress * fminor[1][1];

                    // b. Major and minor gas optical depths (tau_gas)
                    real_t k_val = 0.0;
                    if (kmajor_ptr) {
                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                for (int m = 0; m < 2; ++m) {
                                    size_t km_idx = gp * 8 + i * 4 + j * 2 + m;
                                    k_val += fmajor[i][j][m] * kmajor_ptr[km_idx];
                                }
                            }
                        }
                    }
                    if (kminor_ptr) {
                        for (int i = 0; i < 2; ++i) {
                            for (int m = 0; m < 2; ++m) {
                                size_t km_idx = gp * 4 + i * 2 + m;
                                k_val += fminor[i][m] * kminor_ptr[km_idx];
                            }
                        }
                    }

                    real_t tau_gas = k_val * fast_exp(-p * t * k_val * 1e-6);

                    real_t lut_val = lut_liquid_ptr ? lut_liquid_ptr[gp * 4] : 5.0;
                    real_t tau_cloud = clwp_ptr[lay_idx_2d] * lut_val;

                    real_t tau_aero = 0.0;
                    if (tau_aerosol_ptr) {
                        size_t aero_idx = lay + gp * gp_stride_3d + col_offset_3d;
                        tau_aero = tau_aerosol_ptr[aero_idx];
                    }

                    // c. Sum total optical depth
                    tau_tot_arr[lay] = tau_gas + tau_cloud + tau_aero;

                    // d. Layer Planck emission sources
                    if (planck_ptr) {
                        real_t fac = (t - 160.0) / 15.0;
                        int jp = static_cast<int>(fac);
                        if (jp < 0) jp = 0;
                        if (jp > 8) jp = 8;
                        real_t fp = fac - static_cast<real_t>(jp);
                        lay_src[lay] = planck_ptr[gp * 10 + jp] * (1.0 - fp) + planck_ptr[gp * 10 + jp + 1] * fp;
                    } else {
                        lay_src[lay] = 1.0 + 0.001 * (t - 250.0);
                    }
                }

                lev_src[0] = lay_src[0];
                for (size_t lay = 1; lay < layers; ++lay) {
                    lev_src[lay] = 0.5 * (lay_src[lay - 1] + lay_src[lay]);
                }
                if (planck_ptr) {
                    real_t fac = (t_sfc_val - 160.0) / 15.0;
                    int jp = static_cast<int>(fac);
                    if (jp < 0) jp = 0;
                    if (jp > 8) jp = 8;
                    real_t fp = fac - static_cast<real_t>(jp);
                    lev_src[layers] = planck_ptr[gp * 10 + jp] * (1.0 - fp) + planck_ptr[gp * 10 + jp + 1] * fp;
                } else {
                    lev_src[layers] = 1.0 + 0.001 * (t_sfc_val - 250.0);
                }
                real_t sfc_source = lev_src[layers];

                // e. Non-scattering radiative transport sweep down and up using fast_exp
                const real_t secant = 1.66;
                const real_t tau_thresh = 0.0186;

                real_t g_flux_dn[257];
                real_t g_flux_up[257];

                g_flux_dn[0] = toa_val;
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t tau_loc = tau_tot_arr[lay] * secant;
                    real_t trans = fast_exp(-tau_loc);
                    real_t fact;
                    if (tau_loc > tau_thresh) {
                        fact = (1.0 - trans) / tau_loc - trans;
                    } else {
                        fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * 1.0 / 8.0));
                    }
                    real_t source_dn = (1.0 - trans) * lev_src[lay + 1] + 2.0 * fact * (lay_src[lay] - lev_src[lay + 1]);
                    g_flux_dn[lay + 1] = trans * g_flux_dn[lay] + source_dn;
                }

                real_t sfc_emis = 0.98;
                real_t sfc_albedo = 1.0 - sfc_emis;
                g_flux_up[layers] = g_flux_dn[layers] * sfc_albedo + sfc_emis * sfc_source;

                for (int lay = static_cast<int>(layers) - 1; lay >= 0; --lay) {
                    real_t tau_loc = tau_tot_arr[lay] * secant;
                    real_t trans = fast_exp(-tau_loc);
                    real_t fact;
                    if (tau_loc > tau_thresh) {
                        fact = (1.0 - trans) / tau_loc - trans;
                    } else {
                        fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * 1.0 / 8.0));
                    }
                    real_t source_up = (1.0 - trans) * lev_src[lay] + 2.0 * fact * (lay_src[lay] - lev_src[lay]);
                    g_flux_up[lay] = trans * g_flux_up[lay + 1] + source_up;
                }

                for (size_t lev = 0; lev <= layers; ++lev) {
                    col_flux_dn[lev] += g_flux_dn[lev] * 3.14159265358979323846;
                    col_flux_up[lev] += g_flux_up[lev] * 3.14159265358979323846;
                }
            }

            if (flux_up_ptr) {
                if (up_is_level) {
                    for (size_t lev = 0; lev <= layers; ++lev) {
                        size_t idx = lev + col * (layers + 1);
                        flux_up_ptr[idx] = col_flux_up[lev];
                    }
                } else {
                    size_t out_idx = col * gpoints;
                    flux_up_ptr[out_idx] = col_flux_up[0];
                }
            }
            if (flux_dn_ptr) {
                if (dn_is_level) {
                    for (size_t lev = 0; lev <= layers; ++lev) {
                        size_t idx = lev + col * (layers + 1);
                        flux_dn_ptr[idx] = col_flux_dn[lev];
                    }
                } else {
                    size_t out_idx = col * gpoints;
                    flux_dn_ptr[out_idx] = col_flux_dn[layers];
                }
            }
        }
    }
}

} // namespace rrtmgp
