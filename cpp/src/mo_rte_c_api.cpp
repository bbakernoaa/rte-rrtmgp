#include "mo_rte_c_api.h"
#include "mo_kokkos_megakernel.h"

using namespace rrtmgp;

extern "C" {

void c_execute_megakernel(
    size_t layers,
    size_t columns,
    size_t gpoints,
    const double* play,
    const double* tlay,
    const double* kmajor,
    const double* clwp,
    const double* lut_liquid,
    const double* sza,
    const double* toa,
    double* flux_dir
) {
    // Kokkos allows shallow-wrapping unmanaged raw C pointers using native Views!
    // This completely eliminates any memory copy overhead, running at absolute peak speeds (FR-003)
    KConstView2D play_view(play, layers, columns);
    KConstView2D tlay_view(tlay, layers, columns);
    KConstView4D kmajor_view(kmajor, gpoints, 2, 2, 1);
    KConstView2D clwp_view(clwp, layers, columns);
    KConstView3D lut_liquid_view(lut_liquid, gpoints, 2, 2);
    KConstView1D sza_view(sza, columns);
    KConstView1D toa_view(toa, gpoints);
    KView2D flux_dir_view(flux_dir, gpoints, columns);

    // Delegate to parallel KokkosMegakernel execution
    KokkosMegakernel::execute_megakernel(
        layers, columns, gpoints,
        play_view, tlay_view, kmajor_view, KConstView3D(), clwp_view, lut_liquid_view,
        sza_view, toa_view, flux_dir_view
    );
    
    // Ensure all parallel computations complete before returning to caller
    DeviceSpace().fence();
}

}
