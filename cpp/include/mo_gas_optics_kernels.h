#pragma once

#include "mo_rte_kind.h"

namespace rrtmgp::kernels {

// Procedural physics kernels isolated inside standalone namespace
// Replicates the exact mathematical subroutines from mo_gas_optics_rrtmgp_kernels.F90

void interpolation(
    int ncol, int nlay, int ngas, int nflav, int neta, int npres, int ntemp,
    ConstView2D_Int flavor, // [2, nflav]
    ConstView1D press_ref_log, ConstView1D temp_ref,
    real_t press_ref_log_delta, real_t temp_ref_min, real_t temp_ref_delta, real_t press_ref_trop_log,
    ConstView3D vmr_ref, // [2, ngas, ntemp]
    ConstView2D play, ConstView2D tlay,
    ConstView3D col_gas, // [ncol, nlay, ngas]
    View2D_Int jtemp,        // [ncol, nlay]
    View6D fmajor,       // [2, 2, 2, ncol, nlay, nflav]
    View5D fminor,       // [2, 2, ncol, nlay, nflav]
    View4D col_mix,      // [2, ncol, nlay, nflav]
    View2D_Bool tropo,   // [ncol, nlay]
    View4D_Int jeta,     // [2, ncol, nlay, nflav]
    View2D_Int jpress    // [ncol, nlay]
);

void compute_tau_absorption(
    int ncol, int nlay, int nbnd, int ngpt,
    int ngas, int nflav, int neta, int npres, int ntemp,
    int nminorlower, int nminorklower,
    int nminorupper, int nminorkupper,
    int idx_h2o,
    ConstView2D_Int gpoint_flavor, // [2, ngpt]
    ConstView2D_Int band_lims_gpt, // [2, nbnd]
    ConstView4D kmajor,        // [ngpt, neta, npres, ntemp]
    ConstView3D kminor_lower,  // [nminorklower, neta, ntemp]
    ConstView3D kminor_upper,  // [nminorkupper, neta, ntemp]
    ConstView2D_Int minor_limits_gpt_lower, ConstView2D_Int minor_limits_gpt_upper, // [2, nminor]
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
    View3D tau                 // [ngpt, nlay, ncol]
);

void compute_tau_rayleigh(
    int ncol, int nlay, int nbnd, int ngpt,
    int ngas, int nflav, int neta, int npres, int ntemp,
    ConstView2D_Int gpoint_flavor,
    ConstView2D_Int band_lims_gpt,
    ConstView4D krayl,         // [ngpt, neta, ntemp, 2]
    int idx_h2o, ConstView2D col_dry, ConstView3D col_gas,
    ConstView5D fminor, ConstView4D_Int jeta, ConstView2D_Bool tropo, ConstView2D_Int jtemp,
    View3D tau_rayleigh
);

void compute_Planck_source(
    int ncol, int nlay, int nbnd, int ngpt,
    int nflav, int neta, int npres, int ntemp, int nPlanckTemp,
    ConstView2D tlay, ConstView1D tsfc,
    int sfc_lay,
    ConstView6D fmajor, ConstView4D_Int jeta, ConstView2D_Bool tropo, ConstView2D_Int jtemp, ConstView2D_Int jpress,
    ConstView1D_Int gpoint_bands, ConstView2D_Int band_lims_gpt,
    ConstView4D planck_frac,   // [ngpt, neta, npres, ntemp]
    ConstView1D temp_ref,
    real_t totplnk_delta, real_t totplnk_min,
    ConstView2D totplnk,       // [nPlanckTemp, nbnd]
    ConstView2D_Int gpoint_flavor,
    View2D sfc_source,         // [ngpt, ncol]
    View3D lay_source          // [ngpt, nlay, ncol]
);

} // namespace rrtmgp::kernels
