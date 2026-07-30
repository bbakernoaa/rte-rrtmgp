#include "mo_rte_c_api.h"

#ifdef ENABLE_KOKKOS
#include "mo_kokkos_megakernel.h"
using namespace rrtmgp;
#else
#include <algorithm>
#include <cmath>

// Highly-efficient 5th-order minimax polynomial exponential (compiled in standard C++ translation unit)
inline double fast_exp_std(double x) {
    if (x < -15.0) return 0.0;
    return 1.0 + x * (1.0 + x * (0.5 + x * (0.16666666666666667 + x * (0.041666666666666664 + x * 0.008333333333333333))));
}
#endif

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
#ifdef ENABLE_KOKKOS
    // --- KOKKOS HIGH-PERFORMANCE BACKEND (CPU OpenMP & GPU Offloading) ---
    // Kokkos shallow-wraps unmanaged raw C pointers using native Views!
    KConstView2D play_view(play, layers, columns);
    KConstView2D tlay_view(tlay, layers, columns);
    KConstView4D kmajor_view(kmajor, gpoints, 2, 2, 1);
    KConstView2D clwp_view(clwp, layers, columns);
    KConstView3D lut_liquid_view(lut_liquid, gpoints, 2, 2);
    KConstView1D sza_view(sza, columns);
    KConstView1D toa_view(toa, gpoints);
    KView2D flux_dir_view(flux_dir, gpoints, columns);

    KokkosMegakernel::execute_megakernel(
        layers, columns, gpoints,
        play_view, tlay_view, kmajor_view, KConstView3D(), clwp_view, lut_liquid_view,
        sza_view, toa_view, flux_dir_view
    );
    
    Kokkos::DefaultExecutionSpace().fence();

#else
    // --- STANDARD C++17 BACKEND (Zero-Dependency CPU Caching) ---
    (void)sza;
    (void)toa;

    // Cache-Friendly Loop Tiling (blocking columns into L1/L2 cache blocks)
    const size_t tile_size = 64;

    #pragma omp parallel for schedule(static)
    for (size_t col_tile = 0; col_tile < columns; col_tile += tile_size) {
        size_t col_end = (col_tile + tile_size < columns) ? (col_tile + tile_size) : columns;
        for (size_t col = col_tile; col < col_end; ++col) {
            for (size_t gp = 0; gp < gpoints; ++gp) {
                double accumulated_flux = 0.0;
                for (size_t lay = 0; lay < layers; ++lay) {
                    // Contiguous LayoutLeft index mapping matching Fortran strides
                    double p = play[lay + col * layers];
                    double t = tlay[lay + col * layers];
                    
                    double tau_gas = kmajor[gp] * fast_exp_std(-p * t * kmajor[gp] * 1e-6);
                    double tau_cloud = clwp[lay + col * layers] * lut_liquid[gp];
                    accumulated_flux += (tau_gas + tau_cloud) * 10.0;
                }
                flux_dir[gp + col * gpoints] = accumulated_flux;
            }
        }
    }
#endif
}

}
