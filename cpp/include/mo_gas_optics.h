#pragma once

#include "mo_rte_kind.h"
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
    GasOptics(
        OpticalProps spectral_props,
        ConstView4D kmajor,
        ConstView3D kminor
    );

    void compute_optical_properties(
        ConstView2D play,
        ConstView2D tlay,
        View3D tau
    ) const;

private:
    std::vector<real_t> kmajor_storage_;
    std::vector<real_t> kminor_storage_;

    ConstView4D kmajor_;
    ConstView3D kminor_;
};

} // namespace rrtmgp
