# C++ Solver Optimization & Full Physical Parity Kokkos Megakernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate Standard C++ OpenMP barrier overhead and heap allocations, and implement full physical longwave radiation solver parity in the Kokkos Megakernel.

**Architecture:** Standard C++ solver functions are refactored to use coarse-grained outer-loop OpenMP parallel regions (`collapse(2)`) and reusable pre-allocated workspace buffers. The Kokkos Megakernel is extended to execute the complete single-pass physical longwave pipeline (interpolation, major/minor optical depths, Planck emission, and non-scattering transport).

**Tech Stack:** C++17, Kokkos Core, OpenMP, CMake, Fortran 2008.

## Global Constraints

- Preserve 100% scientific precision (relative flux error $\le 0.02\%$ vs Fortran reference).
- Standard C++ solver throughput must scale linearly across OpenMP threads and execute in $\le 160$ ms on 1,000 columns.
- All CTest targets must pass cleanly.

---

### Task 1: Standard C++ OpenMP Coarse-Graining & Workspace Pre-allocation

**Files:**
- Modify: `cpp/src/mo_gas_optics.cpp`
- Modify: `cpp/src/mo_rte_lw.cpp`
- Modify: `cpp/src/mo_rte_solver_kernels.cpp`
- Test: `cpp/tests/test_cpp_baseline.cpp`

**Interfaces:**
- Consumes: Existing `GasOptics::compute_optical_properties` and `SolverLw::solve_lw_noscat` signatures.
- Produces: High-throughput Standard C++ execution without inner OpenMP fork/joins or per-step heap allocations.

- [ ] **Step 1: Refactor `GasOptics` workspace allocation**
  Pre-allocate thread-local or static workspace buffers in `GasOptics::compute_optical_properties` for `fmajor`, `fminor`, `col_mix`, `jeta`, `col_gas`, eliminating heap `malloc`/`free` calls per step.

- [ ] **Step 2: Hoist `#pragma omp parallel` in `SolverLw::solve_lw_noscat` and `kernels`**
  Remove `#pragma omp parallel for` from inner layer loops inside `lw_source_noscat`, `lw_transport_noscat_dn`, and `lw_transport_noscat_up`. Apply `#pragma omp parallel for collapse(2)` at the outer `(gp, col)` loop in `solve_lw_noscat`.

- [ ] **Step 3: Compile and run baseline test**
  Run `cmake --build cpp/build --target test_cpp_baseline` and verify execution speed on 1, 2, 4, 6, and 10 threads.

- [ ] **Step 4: Verify test passes and commit**
  Confirm test execution without thread stalls and commit changes.

---

### Task 2: Full Physical Parity Kokkos Megakernel Implementation

**Files:**
- Modify: `cpp/include/mo_kokkos_megakernel.h`
- Modify: `cpp/src/mo_kokkos_megakernel.cpp`
- Test: `cpp/tests/test_kokkos_megakernel.cpp`

**Interfaces:**
- Consumes: Kokkos `View` types for atmospheric state (`play`, `tlay`, `clwp`, `tau_aerosol`) and spectral lookup tables (`kmajor`, `kminor`, `lut_liquid`).
- Produces: Physical upward and downward longwave fluxes (`flux_up`, `flux_dn`).

- [ ] **Step 1: Write failing/extended verification assertion in `test_kokkos_megakernel.cpp`**
  Add physical longwave flux assertions comparing `flux_up` and `flux_dn` against expected physical values.

- [ ] **Step 2: Extend `execute_megakernel` interface and implementation**
  In `cpp/src/mo_kokkos_megakernel.cpp`, implement the full single-pass tile loop:
  1. Thermodynamic index lookup (`jtemp`, `jpress`) and bilinear interpolation factors (`fmajor`, `fminor`).
  2. Major/minor gas optical depth calculation ($\tau_{\text{gas}}$) and total optical depth summation ($\tau_{\text{total}} = \tau_{\text{gas}} + \tau_{\text{cloud}} + \tau_{\text{aero}}$).
  3. Planck emission source function evaluation (`lay_source`, `lev_source`, `sfc_source`).
  4. Non-scattering radiative transport sweep down and up.

- [ ] **Step 3: Compile and run Kokkos Megakernel test**
  Run `cmake --build cpp/build --target test_kokkos_megakernel` and execute `./cpp/build/test_kokkos_megakernel`.

- [ ] **Step 4: Verify parity and commit**
  Ensure functional parity checks pass and commit changes.

---

### Task 3: Full Suite Benchmarking & Report Verification

**Files:**
- Modify: `cpp/build/run_scaling_benchmark.sh`
- Test: `cpp/build/run_scaling_benchmark.sh`

- [ ] **Step 1: Run full scaling benchmark sweep**
  Execute `bash cpp/build/run_scaling_benchmark.sh` across 1, 2, 4, 6, and 10 OpenMP threads.

- [ ] **Step 2: Validate performance metrics**
  Verify that Standard C++ executes in $\le 160$ ms and Kokkos Megakernel runs at peak throughput across all thread counts.

- [ ] **Step 3: Run full CTest regression suite**
  Execute `ctest --test-dir cpp/build --output-on-failure` to ensure all 12 C++ tests pass.

- [ ] **Step 4: Final commit**
  Commit all benchmark script updates and final verified code.
