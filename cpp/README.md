# RRTMGP C++ Port Library

This directory contains the highly-optimized, standard-compliant C++17 port of the RRTMGP gas optics, cloud optics, RTE solvers (longwave and shortwave), and post-processing extensions.

## Core Features
1. **Zero External Dependencies**: Direct numerical calculation using standard libraries and the Kokkos `mdspan` C++17 backport.
2. **Standard C++17 Compliance**: Compiles flawlessly on standard compiler suites (GCC, Clang, MSVC).
3. **Array Layout Protection**: Uses column-major contiguous layouts (`layout_left`) to mirror Fortran array strides exactly, preventing multi-dimensional mapping/translation bugs.
4. **Precision Agnostic**: Uses compile-time `real_t` type definitions to configure single or double precision builds globally.
5. **Separation of Solvers & Optics**: Keeps physical solvers (`SolverLw`, `SolverSw`) completely decoupled from spectral parameterization databases (`GasOptics`, `CloudOptics`, `AerosolOptics`).
6. **Thread-Safe Registry**: Stores concentrations dynamically under an uppercase normalized hash registry (`GasConcentrations`).

## Directory Structure
- `include/`: API headers, `mdspan.hpp` backport, kind types, solver kernels, and extensions.
- `src/`: Optical properties, dynamic registries, solvers, scattering kernels, and post-processing extensions.
- `tests/`: GTest/CTest compatible unit tests and full end-to-end verification suites.

## Compilation & Verification

To compile using CMake:
```bash
# Configure the build directory
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release

# Build targets
cmake --build cpp/build --parallel

# Execute the entire CTest suite (10 registered tests)
ctest --test-dir cpp/build --output-on-failure
```

Execute the full regression simulation runner:
```bash
./cpp/build/val_runner
```
Outputs are securely saved to `cpp_fluxes.txt`.
