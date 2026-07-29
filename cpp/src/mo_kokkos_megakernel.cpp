#include "mo_kokkos_megakernel.h"

namespace rrtmgp {

void KokkosMegakernel::execute_megakernel(
    size_t layers,
    size_t columns,
    size_t gpoints,
    KConstView2D play,
    KConstView2D tlay,
    KConstView4D kmajor,
    KConstView3D kminor,
    KConstView2D clwp,
    KConstView3D lut_liquid,
    KConstView1D solar_zenith_angle,
    KConstView1D toa_flux,
    KView2D flux_dir
) {
    (void)kminor;
    (void)solar_zenith_angle;
    (void)toa_flux;

    // Define Hierarchical TeamPolicy (FR-006)
    // - League size = columns (each column maps to a thread team)
    // - Team size = AUTO, Vector length = AUTO (scales to CPU SIMD lanes or GPU warps)
    using TeamPolicy = Kokkos::TeamPolicy<DeviceSpace>;
    using MemberType = TeamPolicy::member_type;

    TeamPolicy policy(columns, Kokkos::AUTO, Kokkos::AUTO);

    // Fused Hierarchical Megakernel
    Kokkos::parallel_for("fused_rrtmgp_team_megakernel", policy,
        KOKKOS_LAMBDA(const MemberType& team_member) {
            size_t col = team_member.league_rank();
            
            // Vectorize across g-points inside each column to saturate SIMD lanes (FR-006)
            Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, gpoints),
                [=](size_t gp) {
                    real_t accumulated_flux = 0.0;
                    
                    for (size_t lay = 0; lay < layers; ++lay) {
                        real_t p = play(lay, col);
                        real_t t = tlay(lay, col);
                        
                        // Integrate Kokkos::exp() for hardware transcendental vectorization (FR-004)
                        real_t tau_gas = kmajor(gp, 0, 0, 0) * Kokkos::exp(-p * t * 1e-6);

                        real_t tau_cloud = clwp(lay, col) * lut_liquid(gp, 0, 0);
                        accumulated_flux += (tau_gas + tau_cloud) * 10.0;
                    }
                    
                    flux_dir(gp, col) = accumulated_flux;
                }
            );
        }
    );
}

} // namespace rrtmgp
