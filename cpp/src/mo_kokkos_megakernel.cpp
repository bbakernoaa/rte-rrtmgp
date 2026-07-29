#include "mo_kokkos_megakernel.h"
#include <stdexcept>

namespace rrtmgp {

void KokkosMegakernel::execute_megakernel(
    size_t layers,
    size_t columns,
    size_t gpoints,
    KConstView2D play,
    KConstView2D tlay,
    KConstView4D kmajor,
    KConstView3D kminor,
    KConstView2D clwp,
    KConstView3D lut_liquid,
    KConstView1D solar_zenith_angle,
    KConstView1D toa_flux,
    KView2D flux_dir
) {
    (void)layers;
    (void)columns;
    (void)gpoints;
    (void)play;
    (void)tlay;
    (void)kmajor;
    (void)kminor;
    (void)clwp;
    (void)lut_liquid;
    (void)solar_zenith_angle;
    (void)toa_flux;
    (void)flux_dir;
    throw std::runtime_error("execute_megakernel is not implemented yet");
}

} // namespace rrtmgp
