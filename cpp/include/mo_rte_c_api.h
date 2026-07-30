#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Zero-overhead C API wrapping our high-performance parallel Kokkos Megakernel
void c_execute_megakernel(
    size_t layers,
    size_t columns,
    size_t gpoints,
    const double* play,         // Contiguous column-major 2D array (layers, columns)
    const double* tlay,         // Contiguous column-major 2D array (layers, columns)
    const double* kmajor,       // Contiguous column-major 4D array (gpoints, 2, 2, 1)
    const double* clwp,         // Contiguous column-major 2D array (layers, columns)
    const double* lut_liquid,   // Contiguous column-major 3D array (gpoints, 2, 2)
    const double* sza,          // Contiguous 1D array (columns)
    const double* toa,          // Contiguous 1D array (gpoints)
    double* flux_dir            // Contiguous column-major 2D output array (gpoints, columns)
);

#ifdef __cplusplus
}
#endif
