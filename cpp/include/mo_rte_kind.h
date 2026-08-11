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
using Extents5D = stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
using Extents6D = stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

// Multidimensional Views
using View1D = stdex::mdspan<real_t, Extents1D, stdex::layout_left>;
using View2D = stdex::mdspan<real_t, Extents2D, stdex::layout_left>;
using View3D = stdex::mdspan<real_t, Extents3D, stdex::layout_left>;
using View4D = stdex::mdspan<real_t, Extents4D, stdex::layout_left>;
using View5D = stdex::mdspan<real_t, Extents5D, stdex::layout_left>;
using View6D = stdex::mdspan<real_t, Extents6D, stdex::layout_left>;

using ConstView1D = stdex::mdspan<const real_t, Extents1D, stdex::layout_left>;
using ConstView2D = stdex::mdspan<const real_t, Extents2D, stdex::layout_left>;
using ConstView3D = stdex::mdspan<const real_t, Extents3D, stdex::layout_left>;
using ConstView4D = stdex::mdspan<const real_t, Extents4D, stdex::layout_left>;
using ConstView5D = stdex::mdspan<const real_t, Extents5D, stdex::layout_left>;
using ConstView6D = stdex::mdspan<const real_t, Extents6D, stdex::layout_left>;

using View1D_Int = stdex::mdspan<int, Extents1D, stdex::layout_left>;
using View2D_Int = stdex::mdspan<int, Extents2D, stdex::layout_left>;
using View3D_Int = stdex::mdspan<int, Extents3D, stdex::layout_left>;
using View4D_Int = stdex::mdspan<int, Extents4D, stdex::layout_left>;

using ConstView1D_Int = stdex::mdspan<const int, Extents1D, stdex::layout_left>;
using ConstView2D_Int = stdex::mdspan<const int, Extents2D, stdex::layout_left>;
using ConstView3D_Int = stdex::mdspan<const int, Extents3D, stdex::layout_left>;
using ConstView4D_Int = stdex::mdspan<const int, Extents4D, stdex::layout_left>;

using View1D_Bool = stdex::mdspan<bool, Extents1D, stdex::layout_left>;
using View2D_Bool = stdex::mdspan<bool, Extents2D, stdex::layout_left>;
using ConstView1D_Bool = stdex::mdspan<const bool, Extents1D, stdex::layout_left>;
using ConstView2D_Bool = stdex::mdspan<const bool, Extents2D, stdex::layout_left>;

using IntView1D  = stdex::mdspan<const int, Extents1D, stdex::layout_left>;
using IntView2D  = stdex::mdspan<const int, Extents2D, stdex::layout_left>;
using IntViewMut2D = stdex::mdspan<int, Extents2D, stdex::layout_left>;

} // namespace rrtmgp
