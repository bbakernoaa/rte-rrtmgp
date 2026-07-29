cmake_minimum_required(VERSION 3.15)
project(rrtmgp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Validation runner executable
add_executable(val_runner tests/val_runner.cpp)
target_link_libraries(val_runner PRIVATE rrtmgp_cpp)
