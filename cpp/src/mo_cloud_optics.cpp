#include "mo_cloud_optics.h"
#include <stdexcept>

namespace rrtmgp {

CloudOptics::CloudOptics(OpticalProps base, ConstView3D lut_liquid, ConstView3D lut_ice)
    : OpticalProps(base.get_gpoint_to_band(), base.get_band_lims_gpoint()) {
    (void)lut_liquid;
    (void)lut_ice;
    throw std::runtime_error("CloudOptics is not implemented yet");
}

void CloudOptics::compute_cloud_optics(
    ConstView2D clwp,
    ConstView2D ciwp,
    ConstView2D rel,
    ConstView2D rei,
    View3D tau,
    View3D ssa,
    View3D g
) const {
    (void)clwp;
    (void)ciwp;
    (void)rel;
    (void)rei;
    (void)tau;
    (void)ssa;
    (void)g;
    throw std::runtime_error("CloudOptics::compute_cloud_optics is not implemented yet");
}

AerosolOptics::AerosolOptics(OpticalProps base, ConstView3D aer_coefficients)
    : OpticalProps(base.get_gpoint_to_band(), base.get_band_lims_gpoint()) {
    (void)aer_coefficients;
    throw std::runtime_error("AerosolOptics is not implemented yet");
}

void AerosolOptics::compute_aerosol_optics(
    ConstView2D aer_mass,
    View3D tau,
    View3D ssa,
    View3D g
) const {
    (void)aer_mass;
    (void)tau;
    (void)ssa;
    (void)g;
    throw std::runtime_error("AerosolOptics::compute_aerosol_optics is not implemented yet");
}

} // namespace rrtmgp
