# Fused & Modular Standard C++ Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a zero-dependency, single-pass Fused Standard C++ Megakernel (`MegakernelCpp`) using `mdspan` and integrate it alongside the Modular C++ interface into the scaling benchmark suite.

**Architecture:** Create `MegakernelCpp` in `cpp/include/mo_megakernel_cpp.h` and `cpp/src/mo_megakernel_cpp.cpp` to execute fused longwave optics and solver physics in a single OpenMP loop nest. Add benchmark target `test_fused_cpp_megakernel` and update `run_scaling_benchmark.sh` to track all four solver variants across thread sweeps.

**Tech Stack:** C++17, OpenMP, `mdspan`, CMake.

## Global Constraints

- Preserve 100% scientific precision (relative flux error $\le 0.02\%$ vs Fortran reference).
- Fused Standard C++ Megakernel throughput must match Kokkos Megakernel speed on CPU ($\le 160$ ms on 1,000 columns).
- All CTest targets must pass cleanly.

---

### Task 1: Create Fused Standard C++ Megakernel (`MegakernelCpp`)

**Files:**
- Create: `cpp/include/mo_megakernel_cpp.h`
- Create: `cpp/src/mo_megakernel_cpp.cpp`
- Modify: `cpp/CMakeLists.txt`
- Test: `cpp/tests/test_fused_cpp_megakernel.cpp`

**Interfaces:**
- Consumes: C++17 `mdspan` Views (`ConstView2D`, `ConstView3D`, `View2D`) for atmospheric profiles, lookup tables, and output fluxes.
- Produces: `MegakernelCpp::execute_fused_longwave` computing longwave level fluxes `flux_up` and `flux_dn`.

- [ ] **Step 1: Write `mo_megakernel_cpp.h` and `mo_megakernel_cpp.cpp`**
  Implement `MegakernelCpp::execute_fused_longwave` fusing thermodynamic lookup, gas absorption, cloud/aerosol summation, Planck emission, and radiative transport into a single tiled OpenMP loop nest.

- [ ] **Step 2: Add test executable `test_fused_cpp_megakernel.cpp` and update `CMakeLists.txt`**
  Add `test_fused_cpp_megakernel` target in `cpp/CMakeLists.txt` and verify functional parity in `test_fused_cpp_megakernel.cpp`.

- [ ] **Step 3: Build and test `test_fused_cpp_megakernel`**
  Run `cmake --build cpp/build --target test_fused_cpp_megakernel` and execute `./cpp/build/test_fused_cpp_megakernel`.

- [ ] **Step 4: Commit changes**
  Commit new fused standard C++ megakernel implementation.

---

### Task 2: Integrate Fused Standard C++ into Benchmark Sweep & Report

**Files:**
- Modify: `cpp/run_scaling_benchmark.sh`
- Test: `cpp/run_scaling_benchmark.sh`

- [ ] **Step 1: Update `run_scaling_benchmark.sh`**
  Add `./cpp/build/test_fused_cpp_megakernel` to the thread scaling sweep loop alongside Fortran, Modular C++, and Kokkos Megakernel.

- [ ] **Step 2: Execute full scaling benchmark suite**
  Run `bash cpp/build/run_scaling_benchmark.sh` across 1, 2, 4, 6, and 10 threads and record results.

- [ ] **Step 3: Run full CTest suite**
  Run `ctest --test-dir cpp/build --output-on-failure` to ensure 100% test pass rate.

- [ ] **Step 4: Final commit**
  Commit benchmark script and final verified codebase.
