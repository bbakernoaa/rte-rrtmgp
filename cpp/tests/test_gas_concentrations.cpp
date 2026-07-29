#include "mo_gas_concentrations.h"
#include "mo_rte_util.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace rrtmgp;

int main() {
    std::cout << "Running test_gas_concentrations..." << std::endl;

    const size_t layers = 5;
    const size_t columns = 2;

    GasConcentrations registry(layers, columns);

    // TDD Test 1: Querying empty registry should throw std::out_of_range
    try {
        registry.get_vmr("co2");
        std::cerr << "test_gas_concentrations FAIL: Querying unregistered gas did not throw out_of_range" << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cout << "T003/T007: Caught expected out_of_range exception for unregistered gas: " << e.what() << std::endl;
    }

    // Prepare valid mixing profile (layers=5, columns=2)
    std::vector<real_t> vmr_data(layers * columns, 350.0); // 350 ppm
    auto vmr_view = ConstView2D(vmr_data.data(), Extents2D(layers, columns));

    // TDD Test 2: Valid Registration and Retrieval
    try {
        registry.set_vmr("CO2", vmr_view);
        auto retrieved = registry.get_vmr("CO2");
        
        // Verify dimensions and values safely without assert()
        if (retrieved.extent(0) != layers || retrieved.extent(1) != columns) {
            std::cerr << "test_gas_concentrations FAIL: Retrived dimensions mismatched layers/columns grid" << std::endl;
            return 1;
        }
        if (std::abs(retrieved(0, 0) - 350.0) >= 1e-10 || std::abs(retrieved(4, 1) - 350.0) >= 1e-10) {
            std::cerr << "test_gas_concentrations FAIL: Retrived VMR values mismatch registered parameters" << std::endl;
            return 1;
        }
        std::cout << "T004: Valid registration & retrieved values match perfectly." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_gas_concentrations FAIL: " << e.what() << std::endl;
        return 1;
    }

    // TDD Test 3: Case Insensitivity
    try {
        // Querying with lowercase 'co2' or mixed 'Co2' should return the same CO2 data
        auto retrieved_lower = registry.get_vmr("co2");
        auto retrieved_mixed = registry.get_vmr("Co2");
        if (std::abs(retrieved_lower(0, 0) - 350.0) >= 1e-10 || std::abs(retrieved_mixed(4, 1) - 350.0) >= 1e-10) {
            std::cerr << "test_gas_concentrations FAIL: Case insensitivity lookup values mismatch" << std::endl;
            return 1;
        }
        std::cout << "T004: Case-insensitivity lookups (co2, Co2) matched successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_gas_concentrations FAIL on case-insensitivity: " << e.what() << std::endl;
        return 1;
    }

    // TDD Test 4: Boundary mismatch validation (US2 / T008)
    std::vector<real_t> invalid_vmr_data(4 * columns, 100.0); // 4 layers (expected 5)
    auto invalid_vmr_view = ConstView2D(invalid_vmr_data.data(), Extents2D(4, columns));
    try {
        registry.set_vmr("CH4", invalid_vmr_view);
        std::cerr << "test_gas_concentrations FAIL: Expected invalid_argument exception on dimension mismatch, but none was thrown!" << std::endl;
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "T008: Caught expected invalid_argument exception on boundary size mismatch: " << e.what() << std::endl;
    }

    std::cout << "test_gas_concentrations: SUCCESS" << std::endl;
    return 0;
}
