#pragma once

#include "mdspan.hpp"
#include <cstddef>

namespace stdex = std::experimental;

namespace rrtmgp {

// Configurable working precision matching Fortran 'wp' kind parameter
using real_t = double;

// Dynamic 1D, 2D, 3D, and 4D Extents using the proper std::dynamic_extent
using Extents1D = stdex::extents<size_t, std::dynamic_extent>;
using Extents2D = stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
using Extents3D = stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
using Extents4D = stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

// Multidimensional Views
using View1D = stdex::mdspan<real_t, Extents1D, stdex::layout_left>;
using View2D = stdex::mdspan<real_t, Extents2D, stdex::layout_left>;
using View3D = stdex::mdspan<real_t, Extents3D, stdex::layout_left>;
using View4D = stdex::mdspan<real_t, Extents4D, stdex::layout_left>;

using ConstView1D = stdex::mdspan<const real_t, Extents1D, stdex::layout_left>;
using ConstView2D = stdex::mdspan<const real_t, Extents2D, stdex::layout_left>;
using ConstView3D = stdex::mdspan<const real_t, Extents3D, stdex::layout_left>;
using ConstView4D = stdex::mdspan<const real_t, Extents4D, stdex::layout_left>;

using IntView1D  = stdex::mdspan<const int, Extents1D, stdex::layout_left>;
using IntView2D  = stdex::mdspan<const int, Extents2D, stdex::layout_left>;
using IntViewMut2D = stdex::mdspan<int, Extents2D, stdex::layout_left>;

} // namespace rrtmgp
