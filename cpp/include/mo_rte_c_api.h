#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Zero-overhead C API wrapping our high-performance parallel Kokkos Megakernel
// Now supporting complete physical feature parity with aerosols (FR-005)
void c_execute_megakernel(
    size_t layers,
    size_t columns,
    size_t gpoints,
    const double* play,         // Contiguous 2D array (layers, columns)
    const double* tlay,         // Contiguous 2D array (layers, columns)
    const double* kmajor,       // Contiguous 4D array (gpoints, 2, 2, 1)
    const double* clwp,         // Contiguous 2D array (layers, columns)
    const double* lut_liquid,   // Contiguous 3D array (gpoints, 2, 2)
    const double* sza,          // Contiguous 1D array (columns)
    const double* toa,          // Contiguous 1D array (gpoints)
    const double* tau_aerosol,   // Contiguous 3D array (gpoints, layers, columns) - aerosol optical depth
    const double* ssa_aerosol,   // Contiguous 3D array (gpoints, layers, columns) - aerosol single-scattering albedo
    const double* g_aerosol,     // Contiguous 3D array (gpoints, layers, columns) - aerosol asymmetry factor
    double* flux_dir            // Contiguous 2D output array (gpoints, columns)
);

#ifdef __cplusplus
}
#endif
