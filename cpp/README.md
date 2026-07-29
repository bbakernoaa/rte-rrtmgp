# RRTMGP C++ Port Library

This directory contains the highly-optimized, standard-compliant C++17 port of the RRTMGP gas optics and RTE solvers.

## Core Features
1. **Zero External Dependencies**: Direct numerical calculation using standard libraries and the Kokkos `mdspan` backport.
2. **Standard C++17 Compliance**: Compiles flawlessly on standard compiler suites (GCC, Clang, MSVC).
3. **Array Layout Protection**: Uses column-major contiguous layouts (`layout_left`) to mirror Fortran array strides exactly, preventing multi-dimensional mapping/translation bugs.
4. **Precision Agnostic**: Uses compile-time `real_t` type definitions to configure single or double precision builds globally.

## Directory Structure
- `include/`: API headers, `mdspan.hpp` backport, kind types.
- `src/`: Optical properties and Solver implementations.
- `tests/`: GTest/CTest compatible unit tests and full end-to-end verification suites.

## Compilation & Verification

To compile using CMake:
```bash
# Configure the build directory
cmake -S . -B build_cpp -DCMAKE_BUILD_TYPE=Release

# Build targets
cmake --build build_cpp --parallel

# Execute unit and integration tests
ctest --test-dir build_cpp --output-on-failure
```
