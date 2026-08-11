cmake_minimum_required(VERSION 3.15)
project(rrtmgp_cpp LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_KOKKOS "Enable Kokkos high-performance megakernel" OFF)
option(
  ENABLE_FAST_EXP
  "Enable 5th-order minimax polynomial exponential approximation"
  ON
)

find_package(OpenMP)

if(ENABLE_KOKKOS)
  find_package(Kokkos REQUIRED)
  add_compile_definitions(ENABLE_KOKKOS)
endif()

if(ENABLE_FAST_EXP)
  add_compile_definitions(ENABLE_FAST_EXP)
endif()

# Compile options for robustness and performance
if(MSVC)
  add_compile_options(/W4 /WX /O2)
else()
  add_compile_options(-Wall -Wextra -Werror -O3)
endif()

# Core library
add_library(
  rrtmgp_cpp
  src/mo_cloud_optics.cpp
  src/mo_gas_concentrations.cpp
  src/mo_gas_optics.cpp
  src/mo_gas_optics_kernels.cpp
  src/mo_megakernel_cpp.cpp
  src/mo_rte_c_api.cpp
  src/mo_rte_extensions.cpp
  src/mo_rte_lw.cpp
  src/mo_rte_solver_kernels.cpp
  src/mo_rte_sw.cpp
  src/rrtmgp_c_api.cpp
)

if(ENABLE_KOKKOS)
  target_sources(rrtmgp_cpp PRIVATE src/mo_kokkos_megakernel.cpp)
  target_link_libraries(rrtmgp_cpp PUBLIC Kokkos::kokkos)
endif()

target_include_directories(
  rrtmgp_cpp PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Fortran CCPP Interface bridge library
add_library(
  rrtmgp_ccpp_bridge
  src/mo_rrtmgp_c_bindings.F90 src/rrtmgp_lw_main.F90 src/rrtmgp_sw_main.F90
)
target_link_libraries(rrtmgp_ccpp_bridge PRIVATE rrtmgp_cpp)

# Enable testing
enable_testing()

# Helper macro for adding tests
function(add_rrtmgp_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE rrtmgp_cpp)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Test targets
add_rrtmgp_test(test_optical_props)
add_rrtmgp_test(test_gas_optics)
add_rrtmgp_test(test_solver_lw)
add_rrtmgp_test(test_solver_sw)
add_rrtmgp_test(test_cloud_optics)
add_rrtmgp_test(test_heating_rates)
add_rrtmgp_test(test_gas_concentrations)
add_rrtmgp_test(test_zenith_spherical)
add_rrtmgp_test(test_cloud_sampling)
add_rrtmgp_test(test_fused_cpp_megakernel)

if(ENABLE_KOKKOS)
  add_rrtmgp_test(test_kokkos_megakernel)

  add_executable(
    test_profile_fuzzer
    tests/fuzzer_kokkos_run.cpp
    tests/fuzzer_std_run.cpp
    tests/test_profile_fuzzer.cpp
  )
  target_link_libraries(test_profile_fuzzer PRIVATE rrtmgp_cpp)
  add_test(NAME test_profile_fuzzer COMMAND test_profile_fuzzer)
endif()

# Validation runner executable
add_executable(val_runner tests/val_runner.cpp)
target_link_libraries(val_runner PRIVATE rrtmgp_cpp)

# Locate NetCDF library and headers portably (supporting macOS Homebrew and
# standard Linux)
find_path(
  NETCDF_INC netcdf.h
  PATHS /opt/homebrew/include /usr/local/include /usr/include
)
find_library(NETCDF_LIB netcdf PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)

# NetCDF Loader Library (Optional wrapper so core remains zero-dependency)
add_library(rrtmgp_nc_loader src/mo_rrtmgp_netcdf.cpp)
target_include_directories(rrtmgp_nc_loader PRIVATE ${NETCDF_INC} include)
target_link_libraries(rrtmgp_nc_loader PRIVATE rrtmgp_cpp ${NETCDF_LIB})

# Add the loader to the CCPP bridge
target_link_libraries(rrtmgp_ccpp_bridge PRIVATE rrtmgp_nc_loader)

# End-to-End All-Sky Simulation Driver linked privately to NetCDF
add_executable(cpp_allsky tests/cpp_allsky.cpp)
target_include_directories(cpp_allsky PRIVATE ${NETCDF_INC})
target_link_libraries(cpp_allsky PRIVATE rrtmgp_cpp ${NETCDF_LIB})

# Benchmark targets
add_executable(test_cpp_baseline tests/test_cpp_baseline.cpp)
target_link_libraries(test_cpp_baseline PRIVATE rrtmgp_cpp)

add_executable(
  test_fortran_megakernel
  tests/test_fortran_megakernel.f90
  ${CMAKE_CURRENT_SOURCE_DIR}/../rte/kernels/mo_rte_kind.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/../rte/kernels/mo_rte_util_array.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/../rte/kernels/mo_rte_solver_kernels.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/../rrtmgp/kernels/mo_gas_optics_rrtmgp_kernels.F90
)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(test_fortran_megakernel PRIVATE -Wno-c-binding-type -Wno-error)
endif()

if(OpenMP_FOUND)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(rrtmgp_cpp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test_cpp_baseline PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(test_fortran_megakernel PRIVATE OpenMP::OpenMP_Fortran)
  endif()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_scaling_benchmark.sh
  ${CMAKE_CURRENT_BINARY_DIR}/run_scaling_benchmark.sh
  COPYONLY
)
