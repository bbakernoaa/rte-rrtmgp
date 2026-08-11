#pragma once

#include <stdexcept>
#include <string>
#include "mdspan.hpp"

namespace rrtmgp {

// Inline helper utilities to validate mdspan array dimensions at API boundaries
template <typename ViewType1, typename ViewType2>
inline void validate_dimensions_match(const ViewType1& v1, const ViewType2& v2, const std::string& name1, const std::string& name2) {
    if (v1.rank() != v2.rank()) {
        throw std::invalid_argument("Dimension mismatch: " + name1 + " rank (" + std::to_string(v1.rank()) +
                                    ") does not match " + name2 + " rank (" + std::to_string(v2.rank()) + ")");
    }
    for (size_t r = 0; r < v1.rank(); ++r) {
        if (v1.extent(r) != v2.extent(r)) {
            throw std::invalid_argument("Dimension mismatch: " + name1 + " extent at rank " + std::to_string(r) +
                                        " (" + std::to_string(v1.extent(r)) + ") does not match " +
                                        name2 + " (" + std::to_string(v2.extent(r)) + ")");
        }
    }
}

template <typename ViewType>
inline void validate_extent(const ViewType& view, size_t dim, size_t expected_extent, const std::string& name) {
    if (dim >= view.rank()) {
        throw std::out_of_range("Dimension out of range: requested rank " + std::to_string(dim) +
                                " on view " + name + " of rank " + std::to_string(view.rank()));
    }
    if (view.extent(dim) != expected_extent) {
        throw std::invalid_argument("Dimension mismatch: " + name + " extent at rank " + std::to_string(dim) +
                                    " (" + std::to_string(view.extent(dim)) + ") expected to be " +
                                    std::to_string(expected_extent));
    }
}

} // namespace rrtmgp
