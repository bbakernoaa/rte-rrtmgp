#include "mo_gas_optics.h"
#include "mo_rte_util.h"
#include "mo_gas_optics_kernels.h"
#include <algorithm>

namespace rrtmgp {

// OpticalProps implementation
OpticalProps::OpticalProps(IntView1D gpoint_to_band, IntView2D band_lims_gpoint) {
    // Validate boundary sizes first
    if (band_lims_gpoint.extent(0) != 2) {
        throw std::invalid_argument("band_lims_gpoint must have first dimension size 2");
    }

    gpoints_ = static_cast<int>(gpoint_to_band.extent(0));
    bands_ = static_cast<int>(band_lims_gpoint.extent(1));

    // Allocate storage and copy elements
    gpoint_to_band_storage_.assign(gpoint_to_band.data_handle(), gpoint_to_band.data_handle() + gpoint_to_band.size());
    band_lims_gpoint_storage_.assign(band_lims_gpoint.data_handle(), band_lims_gpoint.data_handle() + band_lims_gpoint.size());

    // Create backing persistent mdspan views mapping to the copied storage
    gpoint_to_band_ = IntView1D(gpoint_to_band_storage_.data(), Extents1D(gpoint_to_band_storage_.size()));
    band_lims_gpoint_ = IntView2D(band_lims_gpoint_storage_.data(), Extents2D(2, bands_));

    // Validate range values of the mapping array
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
    
    // Validate size alignment
    validate_extent(kmajor, 0, get_gpoints(), "kmajor");
    validate_extent(kminor, 0, get_gpoints(), "kminor");

    // Copy coefficient memory buffers
    kmajor_storage_.assign(kmajor.data_handle(), kmajor.data_handle() + kmajor.size());
    kminor_storage_.assign(kminor.data_handle(), kminor.data_handle() + kminor.size());

    // Construct persistent views matching layout_left
    kmajor_ = ConstView4D(
        kmajor_storage_.data(), 
        Extents4D(kmajor.extent(0), kmajor.extent(1), kmajor.extent(2), kmajor.extent(3))
    );
    kminor_ = ConstView3D(
        kminor_storage_.data(), 
        Extents3D(kminor.extent(0), kminor.extent(1), kminor.extent(2))
    );
}

void GasOptics::compute_optical_properties(
    ConstView2D play,
    ConstView2D tlay,
    View3D tau
) const {
    const size_t layers = play.extent(0);
    const size_t columns = play.extent(1);
    const size_t gp_cnt = get_gpoints();

    // Verify all input arrays match climate grid dimensions
    validate_extent(tlay, 0, layers, "tlay");
    validate_extent(tlay, 1, columns, "tlay");
    validate_extent(tau, 0, gp_cnt, "tau");
    validate_extent(tau, 1, layers, "tau");
    validate_extent(tau, 2, columns, "tau");

    // Perform physical boundary audits
    for (size_t col = 0; col < columns; ++col) {
        for (size_t lay = 0; lay < layers; ++lay) {
            if (play(lay, col) < 0.0) {
                throw std::invalid_argument("play cannot be negative");
            }
            if (tlay(lay, col) < 0.0) {
                throw std::invalid_argument("tlay cannot be below absolute zero (0 K)");
            }
        }
    }

    // Mock Reference pressure and temperature grids for the kernels
    std::vector<real_t> ref_press_data = {1000.0, 500.0, 100.0};
    std::vector<real_t> ref_temp_data = {
        300.0, 250.0, 
        280.0, 200.0, 
        250.0, 180.0  
    };
    auto ref_press = ConstView1D(ref_press_data.data(), Extents1D(3));
    auto ref_temp = ConstView2D(ref_temp_data.data(), Extents2D(3, 2));

    // Core computational kernel - compute optical properties
    for (size_t col = 0; col < columns; ++col) {
        for (size_t lay = 0; lay < layers; ++lay) {
            
            // Fetch physical state
            real_t p = play(lay, col);
            real_t t = tlay(lay, col);

            // Create temporary 1D view slicing directly into the contiguous 3D optical depth tau array
            auto tau_slice = View1D(&tau(0, lay, col), Extents1D(gp_cnt));

            // Execute 3D coordinate interpolation safely with bounds clamping
            kernels::InterpolationWeights weights;
            kernels::interpolation(ref_press, ref_temp, p, t, weights);

            // Execute gaseous absorption kernel mapping fractional weights to k-distribution major/minor coefficients
            kernels::compute_tau_absorption(weights, kmajor_, kminor_, tau_slice);
        }
    }
}

} // namespace rrtmgp
