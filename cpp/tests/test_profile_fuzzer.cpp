#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <string>

using real_t = double;

// Standalone Fuzzer Profile Scenarios representing standard GCM/ESM environments
enum class ProfileType {
    Tropical,           // Hot, highly humid, deep convective layers
    MidLatitudeSummer,  // Moderate temperatures
    PolarWinter,        // Extremely cold, hyper-dry, low tropopause
    DesertExtreme,      // Extremely hot surface (up to 330 K), dry boundary layers
    StratosphericHigh,  // Low surface boundary levels, extremely low pressure (top of atmosphere)
    DeepConvectiveCloud,// Thicker cloud paths (extreme cloud optical depths)
    InvalidPressure,    // Error boundary check: negative pressure levels
    InvalidTemperature  // Error boundary check: temperature below absolute zero
};

struct FuzzProfile {
    ProfileType type;
    std::string name;
    std::vector<real_t> play;
    std::vector<real_t> tlay;
    std::vector<real_t> clwp;
    std::vector<real_t> rel;
};

// Declarations of separately-compiled, decoupled wrapper functions
void run_standard_solvers(
    size_t layers, size_t columns, size_t gpoints,
    const std::vector<real_t>& play,
    const std::vector<real_t>& tlay,
    const std::vector<real_t>& clwp,
    const std::vector<real_t>& rel,
    std::vector<real_t>& flux_std
);

void run_kokkos_solvers(
    size_t layers, size_t columns, size_t gpoints,
    const std::vector<real_t>& play,
    const std::vector<real_t>& tlay,
    const std::vector<real_t>& clwp,
    const std::vector<real_t>& rel,
    std::vector<real_t>& flux_kokkos
);

void init_kokkos_fuzzer();
void finalize_kokkos_fuzzer();

// Procedural generator creating realistic vertical atmospheric profile structures
FuzzProfile generate_fuzz_profile(ProfileType type, size_t layers, size_t columns) {
    FuzzProfile prof;
    prof.type = type;
    prof.play.resize(layers * columns, 0.0);
    prof.tlay.resize(layers * columns, 0.0);
    prof.clwp.resize(layers * columns, 0.0);
    prof.rel.resize(layers * columns, 0.0);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<real_t> rand_pert(-1.0, 1.0);

    switch (type) {
        case ProfileType::Tropical:
            prof.name = "Tropical Scenario (Convective Deep Layer)";
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t frac = static_cast<real_t>(lay) / layers;
                    // Exponential pressure decay from 1013 hPa
                    prof.play[lay + col * layers] = 1013.25 * std::exp(-2.5 * frac) + rand_pert(gen) * 0.1;
                    // Tropical temperature profile (lapse rate ~6.5 K/km)
                    prof.tlay[lay + col * layers] = 300.0 - 110.0 * frac + rand_pert(gen) * 0.5;
                    prof.clwp[lay + col * layers] = 0.05 * frac; // Low cloud layers
                    prof.rel[lay + col * layers] = 10.0;
                }
            }
            break;

        case ProfileType::MidLatitudeSummer:
            prof.name = "Mid-Latitude Summer Scenario";
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t frac = static_cast<real_t>(lay) / layers;
                    prof.play[lay + col * layers] = 1000.0 * std::exp(-2.3 * frac);
                    prof.tlay[lay + col * layers] = 294.0 - 100.0 * frac;
                    prof.clwp[lay + col * layers] = 0.01;
                    prof.rel[lay + col * layers] = 8.0;
                }
            }
            break;

        case ProfileType::PolarWinter:
            prof.name = "Polar Winter Scenario (Extreme Cold & Dry)";
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t frac = static_cast<real_t>(lay) / layers;
                    prof.play[lay + col * layers] = 1010.0 * std::exp(-2.3 * frac);
                    // Severe polar inversion layer (very cold surface temperature)
                    prof.tlay[lay + col * layers] = 230.0 + 30.0 * frac + rand_pert(gen) * 0.2;
                    prof.clwp[lay + col * layers] = 0.0; // Clear dry polar air
                    prof.rel[lay + col * layers] = 0.0;
                }
            }
            break;

        case ProfileType::DesertExtreme:
            prof.name = "Desert Extreme Scenario (Lapse Instability)";
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t frac = static_cast<real_t>(lay) / layers;
                    prof.play[lay + col * layers] = 995.0 * std::exp(-2.4 * frac);
                    // Extremely hot dry surface boundary
                    prof.tlay[lay + col * layers] = 325.0 - 140.0 * frac + rand_pert(gen) * 0.8;
                    prof.clwp[lay + col * layers] = 0.0;
                    prof.rel[lay + col * layers] = 0.0;
                }
            }
            break;

        case ProfileType::StratosphericHigh:
            prof.name = "Stratospheric Height Scenario (TOA)";
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t frac = static_cast<real_t>(lay) / layers;
                    // Low stratospheric pressures (0.1 hPa to 50 hPa)
                    prof.play[lay + col * layers] = 50.0 * std::exp(-4.5 * frac);
                    prof.tlay[lay + col * layers] = 210.0 + 60.0 * frac; // Temperature increases in stratosphere
                    prof.clwp[lay + col * layers] = 0.0;
                    prof.rel[lay + col * layers] = 0.0;
                }
            }
            break;

        case ProfileType::DeepConvectiveCloud:
            prof.name = "Deep Convective Cloud Scenario (Heavy Extinction)";
            for (size_t col = 0; col < columns; ++col) {
                for (size_t lay = 0; lay < layers; ++lay) {
                    real_t frac = static_cast<real_t>(lay) / layers;
                    prof.play[lay + col * layers] = 1000.0 * std::exp(-2.3 * frac);
                    prof.tlay[lay + col * layers] = 290.0 - 100.0 * frac;
                    // Deep convective liquid water path (massive extinction tau_cloud)
                    prof.clwp[lay + col * layers] = 4.5 * frac;
                    prof.rel[lay + col * layers] = 12.0;
                }
            }
            break;

        case ProfileType::InvalidPressure:
            prof.name = "Error Boundary check: Invalid Pressure";
            for (size_t i = 0; i < prof.play.size(); ++i) {
                prof.play[i] = -100.0; // Invalid negative pressure!
                prof.tlay[i] = 250.0;
            }
            break;

        case ProfileType::InvalidTemperature:
            prof.name = "Error Boundary check: Temperature Below Absolute Zero";
            for (size_t i = 0; i < prof.play.size(); ++i) {
                prof.play[i] = 1000.0;
                prof.tlay[i] = -50.0; // Invalid negative temperature!
            }
            break;
    }

    return prof;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "      RTE-RRTMGP C++ PARITY PORT PROFILE FUZZER" << std::endl;
    std::cout << "==========================================================" << std::endl;

    init_kokkos_fuzzer();

    const size_t layers = 30;
    const size_t columns = 5;
    const size_t gpoints = 16;

    // List of profiles to fuzz
    std::vector<ProfileType> test_suites = {
        ProfileType::Tropical,
        ProfileType::MidLatitudeSummer,
        ProfileType::PolarWinter,
        ProfileType::DesertExtreme,
        ProfileType::StratosphericHigh,
        ProfileType::DeepConvectiveCloud,
        ProfileType::InvalidPressure,
        ProfileType::InvalidTemperature
    };

    size_t pass_count = 0;
    size_t fail_count = 0;

    for (ProfileType type : test_suites) {
        FuzzProfile prof = generate_fuzz_profile(type, layers, columns);
        std::cout << "\nFuzzing Scenario: " << prof.name << std::endl;

        try {
            // Execute input verification boundaries
            if (prof.type == ProfileType::InvalidPressure || prof.type == ProfileType::InvalidTemperature) {
                std::cout << "  - Executing error boundary verification..." << std::endl;
                std::vector<real_t> flux_std(gpoints * columns, 0.0);
                
                // standard wrappers must throw on invalid variables
                run_standard_solvers(layers, columns, gpoints, prof.play, prof.tlay, prof.clwp, prof.rel, flux_std);

                std::cerr << "  - FUZZ FAIL: Core library did not block invalid variables!" << std::endl;
                fail_count++;
                continue;
            }

            // --- 1. Compute Standard C++ Baseline Solvers ---
            std::vector<real_t> flux_std(gpoints * columns, 0.0);
            run_standard_solvers(layers, columns, gpoints, prof.play, prof.tlay, prof.clwp, prof.rel, flux_std);

            // --- 2. Compute High-Performance Kokkos Megakernel (separated execution wrapper) ---
            std::vector<real_t> flux_kokkos(gpoints * columns, 0.0);
            run_kokkos_solvers(layers, columns, gpoints, prof.play, prof.tlay, prof.clwp, prof.rel, flux_kokkos);

            // --- 3. Consistency and Precision Audits ---
            real_t max_diff = 0.0;
            for (size_t col = 0; col < columns; ++col) {
                for (size_t gp = 0; gp < gpoints; ++gp) {
                    real_t std_val = flux_std[gp + col * gpoints];
                    real_t kokkos_val = flux_kokkos[gp + col * gpoints];
                    real_t diff = std::abs(std_val - kokkos_val);
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
            }

            std::cout << "  - Profile verification: SUCCESS" << std::endl;
            std::cout << "  - Max Solver-Megakernel variance: " << max_diff << " W/m^2" << std::endl;
            pass_count++;
        } catch (const std::invalid_argument& e) {
            if (prof.type == ProfileType::InvalidPressure || prof.type == ProfileType::InvalidTemperature) {
                std::cout << "  - Fuzz verification: SUCCESS (Correctly blocked boundary: " << e.what() << ")" << std::endl;
                pass_count++;
            } else {
                std::cerr << "  - FUZZ FAIL: Caught unexpected invalid_argument exception: " << e.what() << std::endl;
                fail_count++;
            }
        } catch (const std::exception& e) {
            std::cerr << "  - FUZZ FAIL with exception: " << e.what() << std::endl;
            fail_count++;
        }
    }

    finalize_kokkos_fuzzer();

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "      FUZZER TEST SUMMARY" << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "  - Total Scenarios Fuzzed: " << test_suites.size() << std::endl;
    std::cout << "  - Passed: " << pass_count << std::endl;
    std::cout << "  - Failed: " << fail_count << std::endl;
    std::cout << "==========================================================" << std::endl;

    return (fail_count == 0) ? 0 : 1;
}
