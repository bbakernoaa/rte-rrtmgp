#pragma once

#include "mo_rte_kind.h"
#include "mo_gas_concentrations.h"
#include <vector>

namespace rrtmgp {

class OpticalProps {
public:
    OpticalProps(IntView1D gpoint_to_band, IntView2D band_lims_gpoint);

    int get_gpoints() const noexcept { return gpoints_; }
    int get_bands() const noexcept { return bands_; }

    IntView1D get_gpoint_to_band() const noexcept { return gpoint_to_band_; }
    IntView2D get_band_lims_gpoint() const noexcept { return band_lims_gpoint_; }

private:
    int gpoints_;
    int bands_;

    // Persistent backing storage for views to ensure we don't have dangling views
    std::vector<int> gpoint_to_band_storage_;
    std::vector<int> band_lims_gpoint_storage_;

    IntView1D gpoint_to_band_;
    IntView2D band_lims_gpoint_;
};

class GasOptics : public OpticalProps {
public:
    // We will expand the constructor or use a dedicated loader
    GasOptics(
        OpticalProps spectral_props,
        ConstView4D kmajor,
        ConstView3D kminor
    );

    // Metadata & grids
    std::vector<real_t> press_ref;
    std::vector<real_t> press_ref_log;
    std::vector<real_t> temp_ref;

    real_t press_ref_min, press_ref_max;
    real_t temp_ref_min,  temp_ref_max;
    real_t press_ref_log_delta, temp_ref_delta, press_ref_trop_log;

    std::vector<std::string> gas_names;
    std::vector<real_t> vmr_ref; // [2, ngas, ntemp]

    std::vector<int> flavor; // [2, nflav]
    std::vector<int> gpoint_flavor; // [2, ngpt]

    // Minor gases
    std::vector<int> minor_limits_gpt_lower, minor_limits_gpt_upper;
    std::vector<int> minor_scales_with_density_lower, minor_scales_with_density_upper;
    std::vector<int> scale_by_complement_lower, scale_by_complement_upper;
    std::vector<int> idx_minor_lower, idx_minor_upper;
    std::vector<int> idx_minor_scaling_lower, idx_minor_scaling_upper;
    std::vector<int> kminor_start_lower, kminor_start_upper;

    // Coefficients
    std::vector<real_t> kmajor_storage_;
    std::vector<real_t> kminor_lower_storage_;
    std::vector<real_t> kminor_upper_storage_;

    std::vector<real_t> planck_frac;
    std::vector<real_t> rayl_lower, rayl_upper;
    std::vector<real_t> solar_source;

    // Extents properties
    int ngas() const { return gas_names.size(); }
    int nflav() const { return flavor.size() / 2; }
    int neta() const { return 2; } // RRTMGP hardcoded binary species parameter
    int npres() const { return press_ref.size(); }
    int ntemp() const { return temp_ref.size(); }

    void compute_optical_properties(
        ConstView2D play, ConstView2D plev,
        ConstView2D tlay, ConstView1D tsfc,
        const GasConcentrations& gas_concs,
        View3D tau,
        View3D tau_rayleigh,
        View2D planck_source
    ) const;

    ConstView4D kmajor_;
    ConstView3D kminor_;
};

} // namespace rrtmgp
