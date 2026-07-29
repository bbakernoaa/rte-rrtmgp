#include "mo_gas_concentrations.h"
#include "mo_rte_util.h"
#include <algorithm>
#include <cctype>

namespace rrtmgp {

GasConcentrations::GasConcentrations(size_t layers, size_t columns)
    : layers_(layers), columns_(columns) {}

std::string GasConcentrations::normalize_name(std::string_view name) const {
    std::string uppercase_name(name);
    // Convert ASCII characters to uppercase for case-insensitive indexing
    std::transform(uppercase_name.begin(), uppercase_name.end(), uppercase_name.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return uppercase_name;
}

void GasConcentrations::set_vmr(std::string_view gas_name, ConstView2D vmr_array) {
    // T009: Verify boundary extents match constructor dimensions strictly
    validate_extent(vmr_array, 0, layers_, "vmr_array layers");
    validate_extent(vmr_array, 1, columns_, "vmr_array columns");

    std::string key = normalize_name(gas_name);

    // T005: Copy the input view values to a persistent backing memory buffer
    std::vector<real_t> data(vmr_array.data_handle(), vmr_array.data_handle() + vmr_array.size());
    
    // Insert or overwrite the entry
    gas_storage_[key] = std::move(data);
}

ConstView2D GasConcentrations::get_vmr(std::string_view gas_name) const {
    std::string key = normalize_name(gas_name);

    // T007: Audit lookups for missing gases and throw out_of_range
    auto it = gas_storage_.find(key);
    if (it == gas_storage_.end()) {
        throw std::out_of_range("Gas species '" + std::string(gas_name) + "' is not registered in the concentrations registry");
    }

    // T006: Return direct ConstView2D mapping to internal contiguous storage safely without allocations
    return ConstView2D(it->second.data(), Extents2D(layers_, columns_));
}

} // namespace rrtmgp
