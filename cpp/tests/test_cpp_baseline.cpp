#include "mo_gas_optics.h"
#include "mo_rte_lw.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>

using namespace rrtmgp;
using namespace rte;

int main() {
    const size_t layers = 128;
    const size_t columns = 1000;
    const size_t gpoints = 128;
    const int iterations = 10;

    // Simulate memory layout setup using our new true mathematical wrappers
    // --------------------------------------------------------------------
    std::cout << "==========================================================" << std::endl;
    std::cout << "      FULL MATHEMATICAL PORT BENCHMARK (" << columns << " cols)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::vector<int> gpoint_to_band_data(gpoints, 0);
    std::vector<int> band_lims_data = {0, static_cast<int>(gpoints)};
    auto gpoint_to_band_view = IntView1D(gpoint_to_band_data.data(), Extents1D(gpoints));
    auto band_lims_view = IntView2D(band_lims_data.data(), Extents2D(2, 1));
    OpticalProps props(gpoint_to_band_view, band_lims_view);

    std::vector<real_t> kmajor_data(gpoints * 2 * 2 * 1, 2.5);
    std::vector<real_t> kminor_data(gpoints * 2 * 1, 0.8);
    auto kmajor_std = ConstView4D(kmajor_data.data(), Extents4D(gpoints, 2, 2, 1));
    auto kminor_std = ConstView3D(kminor_data.data(), Extents3D(gpoints, 2, 1));

    GasOptics gas_optics_std(props, kmajor_std, kminor_std);
    // Mock metadata for Gas Optics
    gas_optics_std.press_ref_log = {std::log(1000.0), std::log(500.0), std::log(10.0)};
    gas_optics_std.temp_ref = {200.0, 300.0};
    gas_optics_std.temp_ref_min = 180.0;
    gas_optics_std.temp_ref_delta = 10.0;
    gas_optics_std.press_ref_log_delta = std::log(500.0) - std::log(1000.0);
    gas_optics_std.press_ref_trop_log = std::log(100.0);
    gas_optics_std.gas_names = {"H2O"};
    gas_optics_std.vmr_ref = {0.1, 0.1, 0.1, 0.1};
    gas_optics_std.flavor = {1, 1};
    gas_optics_std.gpoint_flavor = std::vector<int>(2 * gpoints, 1);
    
    // Arrays
    std::vector<real_t> play(layers * columns, 1000.0);
    std::vector<real_t> plev((layers + 1) * columns, 1000.0);
    std::vector<real_t> tlay(layers * columns, 290.0);
    std::vector<real_t> tsfc(columns, 300.0);
    
    auto play_std = ConstView2D(play.data(), Extents2D(layers, columns));
    auto plev_std = ConstView2D(plev.data(), Extents2D(layers + 1, columns));
    auto tlay_std = ConstView2D(tlay.data(), Extents2D(layers, columns));
    auto tsfc_std = ConstView1D(tsfc.data(), Extents1D(columns));

    GasConcentrations gas_concs(layers, columns);
    
    std::vector<real_t> tau_gas_std(gpoints * layers * columns, 0.0);
    auto tau_gas_view = View3D(tau_gas_std.data(), Extents3D(gpoints, layers, columns));
    
    std::vector<real_t> tau_rayl_std(gpoints * layers * columns, 0.0);
    auto tau_rayl_view = View3D(tau_rayl_std.data(), Extents3D(gpoints, layers, columns));
    
    std::vector<real_t> planck_src_std(gpoints * columns, 0.0);
    auto planck_src_view = View2D(planck_src_std.data(), Extents2D(gpoints, columns));

    // LW Solver Arrays
    std::vector<real_t> lay_source_data(gpoints * layers * columns, 1.0); 
    auto lay_source_view = ConstView3D(lay_source_data.data(), Extents3D(gpoints, layers, columns));
    
    std::vector<real_t> lev_source_data(gpoints * (layers + 1) * columns, 1.0);
    auto lev_source_view = ConstView3D(lev_source_data.data(), Extents3D(gpoints, layers + 1, columns));
    
    std::vector<real_t> sfc_emis_data(gpoints * columns, 1.0);
    auto sfc_emis_view = ConstView2D(sfc_emis_data.data(), Extents2D(gpoints, columns));
    
    std::vector<real_t> inc_flux_data(gpoints * columns, 0.0);
    auto inc_flux_view = ConstView2D(inc_flux_data.data(), Extents2D(gpoints, columns));

    std::vector<real_t> flux_up_data(gpoints * (layers + 1) * columns, 0.0);
    std::vector<real_t> flux_dn_data(gpoints * (layers + 1) * columns, 0.0);
    auto flux_up_view_3d = View3D(flux_up_data.data(), Extents3D(gpoints, layers + 1, columns));
    auto flux_dn_view_3d = View3D(flux_dn_data.data(), Extents3D(gpoints, layers + 1, columns));

    omp_set_num_threads(omp_get_max_threads());

    auto run_full_pipeline = [&]() {
        gas_optics_std.compute_optical_properties(play_std, plev_std, tlay_std, tsfc_std, gas_concs, tau_gas_view, tau_rayl_view, planck_src_view);

        rte::SolverLw::solve_lw_noscat(
            tau_gas_view, lay_source_view, lev_source_view,
            sfc_emis_view, planck_src_view, inc_flux_view,
            flux_up_view_3d, flux_dn_view_3d
        );
    };

    run_full_pipeline(); // Warm up
    auto start_pipe = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        run_full_pipeline();
    }
    auto end_pipe = std::chrono::high_resolution_clock::now();
    double time_pipe = std::chrono::duration<double>(end_pipe - start_pipe).count() / iterations;
    double throughput_pipe = (columns * layers * gpoints) / (time_pipe * 1e6);

    std::cout << "\n[Full Physical Parity Execution]:" << std::endl;
    std::cout << "  - Average execution time: " << time_pipe * 1000.0 << " ms" << std::endl;
    std::cout << "  - Computational throughput: " << throughput_pipe << " million cells/sec" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
