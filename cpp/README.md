# RRTMGP C++ Port Library

This directory contains the highly-optimized, standard-compliant C++17 port of the RRTMGP gas optics, cloud optics, RTE solvers (longwave and shortwave), and post-processing extensions.

## Core Features

1. **Zero External Dependencies**: Direct numerical calculation using standard libraries and the Kokkos `mdspan` C++17 backport.
1. **Standard C++17 Compliance**: Compiles flawlessly on standard compiler suites (GCC, Clang, MSVC).
1. **Array Layout Protection**: Uses column-major contiguous layouts (`layout_left`) to mirror Fortran array strides exactly, preventing multi-dimensional mapping/translation bugs.
1. **Precision Agnostic**: Uses compile-time `real_t` type definitions to configure single or double precision builds globally.
1. **Separation of Solvers & Optics**: Keeps physical solvers (`SolverLw`, `SolverSw`) completely decoupled from spectral parameterization databases (`GasOptics`, `CloudOptics`, `AerosolOptics`).
1. **Thread-Safe Registry**: Stores concentrations dynamically under an uppercase normalized hash registry (`GasConcentrations`).
1. **Exascale-Grade GPU/CPU Parallelism**: Support compiling with the full Kokkos Core C++ framework to fuse optics and solvers into unified high-performance hardware-coalesced parallel loop sweeps.

## Directory Structure

- `include/`: API headers, `mdspan.hpp` backport, kind types, solver kernels, extensions, and `mo_kokkos_megakernel.h`.
- `src/`: Optical properties, dynamic registries, solvers, scattering kernels, extensions, and `mo_kokkos_megakernel.cpp`.
- `tests/`: GTest/CTest compatible unit tests and parallel benchmark suites.

## Compilation & Verification

### Standard Build (Zero-Dependency)

To compile the standard baseline sequential library using CMake:

```bash
# Configure the build directory
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release

# Build targets
cmake --build cpp/build --parallel

# Execute the entire CTest suite (12 registered tests)
ctest --test-dir cpp/build --output-on-failure
```

### High-Performance Kokkos Build (GPU/CPU Team Policies)

To compile with the Kokkos Megakernel enabled:

```bash
# Configure enabling Kokkos
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DENABLE_KOKKOS=ON

# On macOS, supply standard Homebrew OpenMP compiler/linker flags:
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DENABLE_KOKKOS=ON \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"

# Build targets
cmake --build cpp/build --parallel

# Execute the entire CTest suite including test_kokkos_megakernel
ctest --test-dir cpp/build --output-on-failure
```

Execute the full regression simulation runner:

```bash
./cpp/build/val_runner
```

Outputs are securely saved to `cpp_fluxes.txt`.
