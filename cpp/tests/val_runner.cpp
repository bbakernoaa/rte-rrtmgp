#include "mo_gas_optics.h"
#include "mo_rte_lw.h"
#include "mo_rte_sw.h"
#include "mo_cloud_optics.h"
#include "mo_rte_extensions.h"
#include <iostream>
#include <vector>
#include <fstream>

using namespace rrtmgp;
using namespace rte;

int main() {
    std::cout << "Running C++ Validation Runner..." << std::endl;

    // 1. Spectral properties (10 gpoints, 4 bands)
    std::vector<int> gpoint_to_band_data = {0, 0, 1, 1, 2, 2, 3, 3, 3, 3};
    std::vector<int> band_lims_data = {0, 2, 4, 6,
                                       1, 3, 5, 9};
    
    auto gpt_view = IntView1D(gpoint_to_band_data.data(), Extents1D(10));
    auto blm_view = IntView2D(band_lims_data.data(), Extents2D(2, 4));

    OpticalProps props(gpt_view, blm_view);

    // 2. GasOptics
    std::vector<real_t> kmajor_data(10 * 2 * 2 * 1, 2.5); // size 40
    std::vector<real_t> kminor_data(10 * 2 * 1, 0.8);     // size 20

    auto kmaj_view = ConstView4D(kmajor_data.data(), Extents4D(10, 2, 2, 1));
    auto kmin_view = ConstView3D(kminor_data.data(), Extents3D(10, 2, 1));

    GasOptics optics(props, kmaj_view, kmin_view);

    // 3. Grid setup: 5 layers, 1 column
    const size_t layers = 5;
    const size_t columns = 1;

    std::vector<real_t> play_data = {1000.0, 850.0, 700.0, 500.0, 300.0}; // pressure hPa
    std::vector<real_t> tlay_data = {290.0, 280.0, 270.0, 250.0, 220.0};  // temperature K

    auto play_view = ConstView2D(play_data.data(), Extents2D(layers, columns));
    auto tlay_view = ConstView2D(tlay_data.data(), Extents2D(layers, columns));

    // 4. Compute optical depth tau
    std::vector<real_t> tau_data(10 * layers * columns, 0.0);
    auto tau_view = View3D(tau_data.data(), Extents3D(10, layers, columns));

    optics.compute_optical_properties(play_view, tlay_view, tau_view);

    // 5. Setup Source Functions
    std::vector<real_t> lay_source_data(10 * layers * columns, 15.0);
    auto lay_source_view = ConstView3D(lay_source_data.data(), Extents3D(10, layers, columns));

    // Outputs
    std::vector<real_t> flux_up_data(10 * columns, 0.0);
    std::vector<real_t> flux_dn_data(10 * columns, 0.0);

    auto flux_up_view = View2D(flux_up_data.data(), Extents2D(10, columns));
    auto flux_dn_view = View2D(flux_dn_data.data(), Extents2D(10, columns));

    // 6. Solve fluxes (Longwave)
    SolverLw::solve_lw_noscat(tau_view, lay_source_view, flux_up_view, flux_dn_view);

    // 7. Solve fluxes (Shortwave)
    std::vector<real_t> sza_data = {0.8}; // mu0
    std::vector<real_t> toa_data(10, 120.0); // TOA flux
    std::vector<real_t> flux_dir_data(10 * columns, 0.0);

    auto sza_view = ConstView1D(sza_data.data(), Extents1D(1));
    auto toa_view = ConstView1D(toa_data.data(), Extents1D(10));
    auto flux_dir_view = View2D(flux_dir_data.data(), Extents2D(10, columns));

    SolverSw::solve_sw_noscat(tau_view, sza_view, toa_view, flux_dir_view);

    // 8. Compute integrated layer heating rates using outputs (5 layers + 1 levels = 6 levels)
    std::vector<real_t> flux_up_levels(6 * columns, 200.0); // mock levels fluxes
    flux_up_levels[5] = 205.0; // divergence at TOA
    std::vector<real_t> flux_dn_levels(6 * columns, 100.0);
    std::vector<real_t> p_level_data = {1000.0 * 100.0, 850.0 * 100.0, 700.0 * 100.0, 500.0 * 100.0, 300.0 * 100.0, 100.0 * 100.0};

    auto fl_up_view = ConstView2D(flux_up_levels.data(), Extents2D(6, columns));
    auto fl_dn_view = ConstView2D(flux_dn_levels.data(), Extents2D(6, columns));
    auto p_lev_view = ConstView2D(p_level_data.data(), Extents2D(6, columns));

    std::vector<real_t> heating_data(layers * columns, 0.0);
    auto heating_view = View2D(heating_data.data(), Extents2D(layers, columns));

    extensions::compute_heating_rates(fl_up_view, fl_dn_view, p_lev_view, heating_view);

    // 9. Write simulation outputs to a file for comparison
    std::ofstream out_file("cpp_fluxes.txt");
    if (out_file.is_open()) {
        out_file << "C++ Output Verification" << std::endl;
        out_file << "flux_up:" << std::endl;
        for (real_t val : flux_up_data) out_file << val << std::endl;
        out_file << "flux_dn:" << std::endl;
        for (real_t val : flux_dn_data) out_file << val << std::endl;
        out_file << "flux_dir_sw:" << std::endl;
        for (real_t val : flux_dir_data) out_file << val << std::endl;
        out_file << "heating_rates:" << std::endl;
        for (real_t val : heating_data) out_file << val << std::endl;
        out_file.close();
        std::cout << "Simulation outputs saved to cpp_fluxes.txt" << std::endl;
    }

    std::cout << "C++ Validation Runner: SUCCESS" << std::endl;
    return 0;
}
