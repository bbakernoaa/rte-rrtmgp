#pragma once

#include "mo_rte_kind.h"
#include "mo_gas_optics.h" // For OpticalProps base class

namespace rrtmgp {

class CloudOptics : public OpticalProps {
public:
    // Initializes cloud LUTs (Liquid and Ice lookup tables)
    CloudOptics(OpticalProps base, ConstView3D lut_liquid, ConstView3D lut_ice);

    // Computes macroscopic optical properties given water path and crystal dimensions
    void compute_cloud_optics(
        ConstView2D clwp, // Cloud liquid water path
        ConstView2D ciwp, // Cloud ice water path
        ConstView2D rel,  // Effective radius liquid
        ConstView2D rei,  // Effective radius ice
        View3D tau,       // Output absorption/scattering optical depth
        View3D ssa,       // Output single-scattering albedo
        View3D g          // Output asymmetry parameter
    ) const;

private:
    std::vector<real_t> lut_liquid_storage_;
    std::vector<real_t> lut_ice_storage_;

    ConstView3D lut_liquid_;
    ConstView3D lut_ice_;
};

class AerosolOptics : public OpticalProps {
public:
    AerosolOptics(OpticalProps base, ConstView3D aer_coefficients);

    void compute_aerosol_optics(
        ConstView2D aer_mass,
        View3D tau,
        View3D ssa,
        View3D g
    ) const;

private:
    std::vector<real_t> aer_coefficients_storage_;
    ConstView3D aer_coefficients_;
};

} // namespace rrtmgp
