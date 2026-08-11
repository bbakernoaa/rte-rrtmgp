#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORTRAN_BIN="${SCRIPT_DIR}/test_fortran_megakernel"
CPP_BIN="${SCRIPT_DIR}/test_cpp_baseline"
FUSED_CPP_BIN="${SCRIPT_DIR}/test_fused_cpp_megakernel"
KOKKOS_BIN="${SCRIPT_DIR}/test_kokkos_megakernel"

if [[ ! -x "$FORTRAN_BIN" || ! -x "$CPP_BIN" || ! -x "$FUSED_CPP_BIN" || ! -x "$KOKKOS_BIN" ]]; then
    echo "Error: One or more benchmark executables not found in ${SCRIPT_DIR}"
    echo "Please build the project first."
    exit 1
fi

echo "=========================================================="
echo "      RRTMGP PORT SCALING SWEEP (40,000 COLUMNS GRID)"
echo "      Total computed cells per iteration: 655.36 million"
echo "=========================================================="
echo ""

# Dynamic thread sweep
THREADS=(1 2 4 6 10)

if [[ "$(uname)" == "Darwin" ]]; then
    export OMP_PROC_BIND=false
else
    export OMP_PROC_BIND=true
    export OMP_PLACES=threads
fi

for t in "${THREADS[@]}"; do
    echo "----------------------------------------------------------"
    echo "  Configuring OMP_NUM_THREADS=$t"
    echo "----------------------------------------------------------"
    export OMP_NUM_THREADS=$t
    
    echo "[1. Reference Fortran OpenMP]"
    "$FORTRAN_BIN"
    
    echo "[2. Modular Standard C++]"
    "$CPP_BIN"
    
    echo "[3. Fused Standard C++ Megakernel]"
    "$FUSED_CPP_BIN" | grep -A 3 "Benchmark Results"
    
    echo "[4. Kokkos Parallel Megakernel]"
    "$KOKKOS_BIN" | grep -A 3 "Benchmark Results"
    
    echo ""
done
