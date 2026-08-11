#pragma once

#include "mo_rte_kind.h"
#include "mo_gas_optics.h" // For OpticalProps base class

namespace rrtmgp {

class CloudOptics : public OpticalProps {
public:
    // Initializes cloud LUTs (Liquid and Ice lookup tables)
    CloudOptics(
        OpticalProps base,
        real_t radliq_lwr, real_t radliq_upr, real_t radliq_fac,
        real_t radice_lwr, real_t radice_upr, real_t radice_fac,
        ConstView2D extliq, ConstView2D ssaliq, ConstView2D asyliq,
        ConstView3D extice, ConstView3D ssaice, ConstView3D asyice
    );

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
    real_t radliq_lwr_, radliq_fac_;
    real_t radice_lwr_, radice_fac_;

    std::vector<real_t> extliq_storage_;
    std::vector<real_t> ssaliq_storage_;
    std::vector<real_t> asyliq_storage_;

    std::vector<real_t> extice_storage_;
    std::vector<real_t> ssaice_storage_;
    std::vector<real_t> asyice_storage_;

    ConstView2D extliq_;
    ConstView2D ssaliq_;
    ConstView2D asyliq_;

    ConstView3D extice_;
    ConstView3D ssaice_;
    ConstView3D asyice_;
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
