#pragma once

#include "mo_rte_kind.h"
#include <string_view>
#include <string>
#include <unordered_map>
#include <vector>

namespace rrtmgp {

class GasConcentrations {
public:
    // Initialize the registry with standard grid dimensions
    GasConcentrations(size_t layers, size_t columns);

    // Register a gas concentration profile (copies values)
    void set_vmr(std::string_view gas_name, ConstView2D vmr_array);

    // Query a registered gas profile (zero-overhead direct view mapping to internal storage)
    ConstView2D get_vmr(std::string_view gas_name) const;

private:
    size_t layers_;
    size_t columns_;

    // Case-insensitive backing storage mapping normalized gas name strings
    std::unordered_map<std::string, std::vector<real_t>> gas_storage_;

    std::string normalize_name(std::string_view name) const;
};

} // namespace rrtmgp
