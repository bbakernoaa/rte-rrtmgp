#include "rrtmgp_c_api.h"
#include "mo_gas_optics.h"
#include "mo_gas_concentrations.h"
#include "mo_rte_lw.h"
#include "mo_rte_sw.h"

using namespace rrtmgp;
using namespace rte;

extern "C" {

rrtmgp_gas_optics_t rrtmgp_gas_optics_create(
    const int* gpoint_to_band, int n_gpoints,
    const int* band_lims, int n_bands,
    const double* kmajor, int kmajor_dim1, int kmajor_dim2, int kmajor_dim3, int kmajor_dim4,
    const double* kminor, int kminor_dim1, int kminor_dim2, int kminor_dim3
) {
    auto gpoint_to_band_view = IntView1D(const_cast<int*>(gpoint_to_band), Extents1D(n_gpoints));
    auto band_lims_view = IntView2D(const_cast<int*>(band_lims), Extents2D(2, n_bands));
    OpticalProps props(gpoint_to_band_view, band_lims_view);

    auto kmajor_view = ConstView4D(kmajor, Extents4D(kmajor_dim1, kmajor_dim2, kmajor_dim3, kmajor_dim4));
    auto kminor_view = ConstView3D(kminor, Extents3D(kminor_dim1, kminor_dim2, kminor_dim3));

    GasOptics* optics = new GasOptics(props, kmajor_view, kminor_view);
    return static_cast<rrtmgp_gas_optics_t>(optics);
}

void rrtmgp_gas_optics_delete(rrtmgp_gas_optics_t handle) {
    if (handle) {
        delete static_cast<GasOptics*>(handle);
    }
}

void rrtmgp_lw_run_cpp(
    rrtmgp_gas_optics_t gas_optics_handle,
    int ncol, int nlay, int ngpt,
    const double* p_lay, const double* t_lay,
    const double* p_lev, const double* t_lev,
    const double* tsfg,
    const double* vmr_h2o, const double* vmr_co2, const double* vmr_o3,
    const double* vmr_n2o, const double* vmr_ch4, const double* vmr_o2,
    const double* tau_clouds,
    double* flux_up, double* flux_dn
) {
    GasOptics* optics = static_cast<GasOptics*>(gas_optics_handle);
    (void)p_lev; (void)t_lev; (void)tsfg;

    // Create LayoutLeft mdspan views over the raw flat Fortran arrays
    auto play_view = ConstView2D(p_lay, Extents2D(nlay, ncol));
    auto tlay_view = ConstView2D(t_lay, Extents2D(nlay, ncol));

    // We populate a GasConcentrations object even if the current C++ optics doesn't use it yet,
    // to maintain the C++ API bridge architecture.
    GasConcentrations gas_concs(nlay, ncol);
    if (vmr_h2o) gas_concs.set_vmr("H2O", ConstView2D(vmr_h2o, Extents2D(nlay, ncol)));
    if (vmr_co2) gas_concs.set_vmr("CO2", ConstView2D(vmr_co2, Extents2D(nlay, ncol)));
    if (vmr_o3)  gas_concs.set_vmr("O3",  ConstView2D(vmr_o3,  Extents2D(nlay, ncol)));
    if (vmr_n2o) gas_concs.set_vmr("N2O", ConstView2D(vmr_n2o, Extents2D(nlay, ncol)));
    if (vmr_ch4) gas_concs.set_vmr("CH4", ConstView2D(vmr_ch4, Extents2D(nlay, ncol)));
    if (vmr_o2)  gas_concs.set_vmr("O2",  ConstView2D(vmr_o2,  Extents2D(nlay, ncol)));

    // Allocate temporary outputs (could be avoided if caller provides them)
    std::vector<real_t> tau_gas_data(ngpt * nlay * ncol, 0.0);
    auto tau_gas_view = View3D(tau_gas_data.data(), Extents3D(ngpt, nlay, ncol));

    std::vector<real_t> tau_rayl_data(ngpt * nlay * ncol, 0.0);
    auto tau_rayl_view = View3D(tau_rayl_data.data(), Extents3D(ngpt, nlay, ncol));
    
    std::vector<real_t> planck_source_data(ngpt * nlay * ncol, 0.0);
    auto planck_source_view = View2D(planck_source_data.data(), Extents2D(ngpt, ncol)); // Wait, planck source in Fortran is (ncol, nlay, ngpt)? No, (ncol, ngpt) for surface, plus (ncol, nlay, ngpt) for layers!

    auto plev_view = ConstView2D(p_lev, Extents2D(nlay + 1, ncol));
    auto tlev_view = ConstView2D(t_lev, Extents2D(nlay + 1, ncol));
    auto tsfc_view = ConstView1D(tsfg, Extents1D(ncol));

    // Compute gas optics
    optics->compute_optical_properties(play_view, plev_view, tlay_view, tsfc_view, gas_concs, tau_gas_view, tau_rayl_view, planck_source_view);

    // Combine with clouds if provided
    if (tau_clouds) {
        auto tau_cloud_view = ConstView3D(tau_clouds, Extents3D(ngpt, nlay, ncol));
        for (int c = 0; c < ncol; ++c) {
            for (int l = 0; l < nlay; ++l) {
                for (int g = 0; g < ngpt; ++g) {
                    tau_gas_view(g, l, c) += tau_cloud_view(g, l, c);
                }
            }
        }
    }

    // Prepare solver inputs
    std::vector<real_t> lay_source_data(ngpt * nlay * ncol, 1.0); 
    auto lay_source_view = ConstView3D(lay_source_data.data(), Extents3D(ngpt, nlay, ncol));
    
    std::vector<real_t> lev_source_data(ngpt * (nlay + 1) * ncol, 1.0);
    auto lev_source_view = ConstView3D(lev_source_data.data(), Extents3D(ngpt, nlay + 1, ncol));
    
    std::vector<real_t> sfc_emis_data(ngpt * ncol, 1.0);
    auto sfc_emis_view = ConstView2D(sfc_emis_data.data(), Extents2D(ngpt, ncol));
    
    auto sfc_src_view = ConstView2D(planck_source_data.data(), Extents2D(ngpt, ncol));
    
    std::vector<real_t> inc_flux_data(ngpt * ncol, 0.0);
    auto inc_flux_view = ConstView2D(inc_flux_data.data(), Extents2D(ngpt, ncol));

    std::vector<real_t> flux_up_data(ngpt * (nlay + 1) * ncol, 0.0);
    std::vector<real_t> flux_dn_data(ngpt * (nlay + 1) * ncol, 0.0);
    auto flux_up_view_3d = View3D(flux_up_data.data(), Extents3D(ngpt, nlay + 1, ncol));
    auto flux_dn_view_3d = View3D(flux_dn_data.data(), Extents3D(ngpt, nlay + 1, ncol));

    // Call Longwave Solver
    rte::SolverLw::solve_lw_noscat(
        tau_gas_view, lay_source_view, lev_source_view,
        sfc_emis_view, sfc_src_view, inc_flux_view,
        flux_up_view_3d, flux_dn_view_3d
    );
    
    // Copy back to 2D
    auto flux_up_view = View2D(flux_up, Extents2D(ncol, ngpt));
    auto flux_dn_view = View2D(flux_dn, Extents2D(ncol, ngpt));
    for (int c = 0; c < ncol; ++c) {
        for (int g = 0; g < ngpt; ++g) {
            flux_up_view(c, g) = flux_up_view_3d(g, nlay, c);
            flux_dn_view(c, g) = flux_dn_view_3d(g, 0, c);
        }
    }
}

void rrtmgp_sw_run_cpp(
    rrtmgp_gas_optics_t gas_optics_handle,
    int ncol, int nlay, int ngpt,
    const double* p_lay, const double* t_lay,
    const double* p_lev, const double* t_lev,
    const double* sza, const double* toa_flux, const double* sfc_alb_dir, const double* sfc_alb_dif,
    const double* vmr_h2o, const double* vmr_co2, const double* vmr_o3,
    const double* vmr_n2o, const double* vmr_ch4, const double* vmr_o2,
    const double* tau_clouds, const double* ssa_clouds, const double* g_clouds,
    double* flux_up, double* flux_dn, double* flux_dir
) {
    GasOptics* optics = static_cast<GasOptics*>(gas_optics_handle);
    (void)p_lev; (void)t_lev; (void)sfc_alb_dif; (void)vmr_h2o; (void)vmr_co2; (void)vmr_o3;
    (void)vmr_n2o; (void)vmr_ch4; (void)vmr_o2;

    auto play_view = ConstView2D(p_lay, Extents2D(nlay, ncol));
    auto tlay_view = ConstView2D(t_lay, Extents2D(nlay, ncol));

    std::vector<real_t> tau_gas_data(ngpt * nlay * ncol, 0.0);
    auto tau_gas_view = View3D(tau_gas_data.data(), Extents3D(ngpt, nlay, ncol));

    std::vector<real_t> tau_rayl_data(ngpt * nlay * ncol, 0.0);
    auto tau_rayl_view = View3D(tau_rayl_data.data(), Extents3D(ngpt, nlay, ncol));

    // SW doesn't compute planck source, so we'll just pass a dummy 2D view
    std::vector<real_t> planck_source_data(ngpt * ncol, 0.0);
    auto planck_source_view = View2D(planck_source_data.data(), Extents2D(ngpt, ncol));

    auto plev_view = ConstView2D(p_lev, Extents2D(nlay + 1, ncol));
    auto tlev_view = ConstView2D(t_lev, Extents2D(nlay + 1, ncol));
    auto tsfc_view = ConstView1D(nullptr, Extents1D(0)); // SW has no surface temp array

    GasConcentrations gas_concs(nlay, ncol);
    if (vmr_h2o) gas_concs.set_vmr("H2O", ConstView2D(vmr_h2o, Extents2D(nlay, ncol)));
    if (vmr_co2) gas_concs.set_vmr("CO2", ConstView2D(vmr_co2, Extents2D(nlay, ncol)));
    if (vmr_o3)  gas_concs.set_vmr("O3",  ConstView2D(vmr_o3,  Extents2D(nlay, ncol)));
    if (vmr_n2o) gas_concs.set_vmr("N2O", ConstView2D(vmr_n2o, Extents2D(nlay, ncol)));
    if (vmr_ch4) gas_concs.set_vmr("CH4", ConstView2D(vmr_ch4, Extents2D(nlay, ncol)));
    if (vmr_o2)  gas_concs.set_vmr("O2",  ConstView2D(vmr_o2,  Extents2D(nlay, ncol)));

    optics->compute_optical_properties(play_view, plev_view, tlay_view, tsfc_view, gas_concs, tau_gas_view, tau_rayl_view, planck_source_view);

    // Default arrays for solver
    std::vector<real_t> ssa_data(ngpt * nlay * ncol, 0.0);
    std::vector<real_t> g_data(ngpt * nlay * ncol, 0.0);
    auto ssa_view = View3D(ssa_data.data(), Extents3D(ngpt, nlay, ncol));
    auto g_view = View3D(g_data.data(), Extents3D(ngpt, nlay, ncol));

    if (tau_clouds && ssa_clouds && g_clouds) {
        auto cld_tau = ConstView3D(tau_clouds, Extents3D(ngpt, nlay, ncol));
        auto cld_ssa = ConstView3D(ssa_clouds, Extents3D(ngpt, nlay, ncol));
        auto cld_g = ConstView3D(g_clouds, Extents3D(ngpt, nlay, ncol));

        for (int c = 0; c < ncol; ++c) {
            for (int l = 0; l < nlay; ++l) {
                for (int gp = 0; gp < ngpt; ++gp) {
                    double t_gas = tau_gas_view(gp, l, c);
                    double t_cld = cld_tau(gp, l, c);
                    double t_tot = t_gas + t_cld;
                    tau_gas_view(gp, l, c) = t_tot;
                    if (t_tot > 0) {
                        ssa_view(gp, l, c) = (t_cld * cld_ssa(gp, l, c)) / t_tot;
                        if (ssa_view(gp, l, c) > 0) {
                            g_view(gp, l, c) = cld_g(gp, l, c);
                        }
                    }
                }
            }
        }
    }

    std::vector<real_t> sza_data(nlay * ncol, 0.0);
    for (int col = 0; col < ncol; ++col) {
        for (int lay = 0; lay < nlay; ++lay) {
            sza_data[lay + col * nlay] = sza[col];
        }
    }
    auto sza_view = ConstView2D(sza_data.data(), Extents2D(nlay, ncol));
    
    auto sfc_alb_dir_view = ConstView2D(sfc_alb_dir, Extents2D(ngpt, ncol));
    auto sfc_alb_dif_view = ConstView2D(sfc_alb_dif, Extents2D(ngpt, ncol));
    
    std::vector<real_t> inc_flux_data(ngpt * ncol, 0.0);
    for (int col = 0; col < ncol; ++col) {
        for (int gp = 0; gp < ngpt; ++gp) {
            inc_flux_data[gp + col * ngpt] = toa_flux[gp]; // mock mapping
        }
    }
    auto inc_flux_view = ConstView2D(inc_flux_data.data(), Extents2D(ngpt, ncol));

    std::vector<real_t> flux_up_data(ngpt * (nlay + 1) * ncol, 0.0);
    std::vector<real_t> flux_dn_data(ngpt * (nlay + 1) * ncol, 0.0);
    std::vector<real_t> flux_dir_data(ngpt * (nlay + 1) * ncol, 0.0);
    
    auto flux_up_view_3d = View3D(flux_up_data.data(), Extents3D(ngpt, nlay + 1, ncol));
    auto flux_dn_view_3d = View3D(flux_dn_data.data(), Extents3D(ngpt, nlay + 1, ncol));
    auto flux_dir_view_3d = View3D(flux_dir_data.data(), Extents3D(ngpt, nlay + 1, ncol));

    rte::SolverSw::solve_sw_2stream(
        tau_gas_view, ssa_view, g_view,
        sza_view, sfc_alb_dir_view, sfc_alb_dif_view, inc_flux_view,
        flux_up_view_3d, flux_dn_view_3d, flux_dir_view_3d
    );
    
    // Copy back
    auto flux_up_view = View2D(flux_up, Extents2D(ncol, ngpt));
    auto flux_dn_view = View2D(flux_dn, Extents2D(ncol, ngpt));
    auto flux_dir_view = View2D(flux_dir, Extents2D(ncol, ngpt));
    for (int c = 0; c < ncol; ++c) {
        for (int g = 0; g < ngpt; ++g) {
            flux_up_view(c, g) = flux_up_view_3d(g, nlay, c);
            flux_dn_view(c, g) = flux_dn_view_3d(g, 0, c);
            flux_dir_view(c, g) = flux_dir_view_3d(g, 0, c);
        }
    }
}

}
