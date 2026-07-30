cmake_minimum_required(VERSION 3.15)
project(rrtmgp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_KOKKOS "Enable Kokkos high-performance megakernel" OFF)

if(ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)
    add_compile_definitions(ENABLE_KOKKOS)
endif()

# Compile options for robustness and performance
if(MSVC)
    add_compile_options(/W4 /WX /O2)
else()
    add_compile_options(-Wall -Wextra -Werror -O3)
endif()

# Core library
add_library(rrtmgp_cpp
    src/mo_gas_optics_kernels.cpp
    src/mo_gas_optics.cpp
    src/mo_rte_lw.cpp
    src/mo_gas_concentrations.cpp
    src/mo_rte_solver_kernels.cpp
    src/mo_cloud_optics.cpp
    src/mo_rte_extensions.cpp
    src/mo_rte_sw.cpp
)

if(ENABLE_KOKKOS)
    target_sources(rrtmgp_cpp PRIVATE src/mo_kokkos_megakernel.cpp src/mo_rte_c_api.cpp)
    target_link_libraries(rrtmgp_cpp PUBLIC Kokkos::kokkos)
endif()

target_include_directories(rrtmgp_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

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
add_rrtmgp_test(test_kernel_interpolation)
add_rrtmgp_test(test_kernel_absorption)
add_rrtmgp_test(test_kernel_optics_thermal)
add_rrtmgp_test(test_gas_concentrations)
add_rrtmgp_test(test_zenith_spherical)
add_rrtmgp_test(test_cloud_sampling)

if(ENABLE_KOKKOS)
    add_rrtmgp_test(test_kokkos_megakernel)
    
    add_executable(test_profile_fuzzer 
        tests/test_profile_fuzzer.cpp 
        tests/fuzzer_std_run.cpp 
        tests/fuzzer_kokkos_run.cpp
    )
    target_link_libraries(test_profile_fuzzer PRIVATE rrtmgp_cpp)
    add_test(NAME test_profile_fuzzer COMMAND test_profile_fuzzer)
endif()

# Validation runner executable
add_executable(val_runner tests/val_runner.cpp)
target_link_libraries(val_runner PRIVATE rrtmgp_cpp)

# Locate NetCDF library and headers portably (supporting macOS Homebrew and standard Linux)
find_path(NETCDF_INC netcdf.h PATHS /opt/homebrew/include /usr/local/include /usr/include)
find_library(NETCDF_LIB netcdf PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)

# End-to-End All-Sky Simulation Driver linked privately to NetCDF
add_executable(cpp_allsky tests/cpp_allsky.cpp)
target_include_directories(cpp_allsky PRIVATE ${NETCDF_INC})
target_link_libraries(cpp_allsky PRIVATE rrtmgp_cpp ${NETCDF_LIB})


