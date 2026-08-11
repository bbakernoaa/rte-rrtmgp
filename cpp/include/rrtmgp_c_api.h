#ifndef RRTMGP_C_API_H
#define RRTMGP_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles for persistent C++ objects
typedef void* rrtmgp_gas_optics_t;
typedef void* rrtmgp_cloud_optics_t;
typedef void* rrtmgp_aerosol_optics_t;

// LW Init: In a complete implementation, this would either take the raw coefficient arrays
// extracted from Fortran, or a filename if C++ handles NetCDF directly.
// For the zero-copy bridge where Fortran loads NetCDF, we pass the coefficients.
// However, to mimic CCPP init strictly, if we pass filename, C++ needs to read it.
// Since CCPP passes the filename to `rrtmgp_lw_main_init`, we'll declare an init that takes arrays for now,
// or we can just mock the initialization if we assume Fortran does the loading.
// Let's pass the arrays required by GasOptics constructor.
rrtmgp_gas_optics_t rrtmgp_gas_optics_create(
    const int* gpoint_to_band, int n_gpoints,
    const int* band_lims, int n_bands,
    const double* kmajor, int kmajor_dim1, int kmajor_dim2, int kmajor_dim3, int kmajor_dim4,
    const double* kminor, int kminor_dim1, int kminor_dim2, int kminor_dim3
);

void rrtmgp_gas_optics_delete(rrtmgp_gas_optics_t handle);

// LW Run function mimicking the CCPP run signature
// Flat arrays, zero copy
void rrtmgp_lw_run_cpp(
    rrtmgp_gas_optics_t gas_optics_handle,
    int ncol, int nlay, int ngpt,
    const double* p_lay, const double* t_lay,
    const double* p_lev, const double* t_lev,
    const double* tsfg,
    const double* vmr_h2o, const double* vmr_co2, const double* vmr_o3,
    const double* vmr_n2o, const double* vmr_ch4, const double* vmr_o2,
    const double* tau_clouds, // pre-sampled from McICA
    double* flux_up, double* flux_dn
);

// SW Run function mimicking the CCPP run signature
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
);

#ifdef __cplusplus
}
#endif

#endif // RRTMGP_C_API_H
