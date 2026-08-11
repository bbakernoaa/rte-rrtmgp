#include "mo_cloud_optics.h"
#include "mo_rte_util.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace rrtmgp {

namespace kernels {
inline void compute_cld_from_table(
    int ncol, int nlay, int ngpt,
    ConstView2D lwp, ConstView2D re,
    int nsteps, real_t step_size, real_t offset,
    ConstView2D tau_table, ConstView2D ssa_table, ConstView2D asy_table,
    View3D tau, View3D taussa, View3D taussag
) {
    for (int igpt = 0; igpt < ngpt; ++igpt) {
        for (int ilay = 0; ilay < nlay; ++ilay) {
            for (int icol = 0; icol < ncol; ++icol) {
                if (lwp(ilay, icol) > 0.0) {
                    int index = std::min(static_cast<int>(std::floor((re(ilay, icol) - offset) / step_size)) + 1, nsteps - 1);
                    index = std::max(1, index);

                    real_t fint = (re(ilay, icol) - offset) / step_size - static_cast<real_t>(index - 1);
                    fint = std::max(0.0, std::min(fint, 1.0)); // Clamp fint to [0, 1] to prevent wild extrapolation

                    real_t t = lwp(ilay, icol) *
                        (tau_table(index - 1, igpt) + fint * (tau_table(index, igpt) - tau_table(index - 1, igpt)));

                    real_t ts = t *
                        (ssa_table(index - 1, igpt) + fint * (ssa_table(index, igpt) - ssa_table(index - 1, igpt)));

                    taussag(igpt, ilay, icol) += ts *
                        (asy_table(index - 1, igpt) + fint * (asy_table(index, igpt) - asy_table(index - 1, igpt)));

                    taussa(igpt, ilay, icol) += ts;
                    tau(igpt, ilay, icol) += t;
                }
            }
        }
    }
}
}

CloudOptics::CloudOptics(
    OpticalProps base,
    real_t radliq_lwr, real_t radliq_upr, real_t radliq_fac,
    real_t radice_lwr, real_t radice_upr, real_t radice_fac,
    ConstView2D extliq, ConstView2D ssaliq, ConstView2D asyliq,
    ConstView3D extice, ConstView3D ssaice, ConstView3D asyice
) : OpticalProps(base.get_gpoint_to_band(), base.get_band_lims_gpoint()),
    radliq_lwr_(radliq_lwr), radliq_fac_(radliq_fac),
    radice_lwr_(radice_lwr), radice_fac_(radice_fac) {
    (void)radliq_upr; (void)radice_upr;

    validate_extent(extliq, 1, get_gpoints(), "extliq");

    extliq_storage_.assign(extliq.data_handle(), extliq.data_handle() + extliq.size());
    ssaliq_storage_.assign(ssaliq.data_handle(), ssaliq.data_handle() + ssaliq.size());
    asyliq_storage_.assign(asyliq.data_handle(), asyliq.data_handle() + asyliq.size());

    extice_storage_.assign(extice.data_handle(), extice.data_handle() + extice.size());
    ssaice_storage_.assign(ssaice.data_handle(), ssaice.data_handle() + ssaice.size());
    asyice_storage_.assign(asyice.data_handle(), asyice.data_handle() + asyice.size());

    extliq_ = ConstView2D(extliq_storage_.data(), Extents2D(extliq.extent(0), extliq.extent(1)));
    ssaliq_ = ConstView2D(ssaliq_storage_.data(), Extents2D(ssaliq.extent(0), ssaliq.extent(1)));
    asyliq_ = ConstView2D(asyliq_storage_.data(), Extents2D(asyliq.extent(0), asyliq.extent(1)));

    extice_ = ConstView3D(extice_storage_.data(), Extents3D(extice.extent(0), extice.extent(1), extice.extent(2)));
    ssaice_ = ConstView3D(ssaice_storage_.data(), Extents3D(ssaice.extent(0), ssaice.extent(1), ssaice.extent(2)));
    asyice_ = ConstView3D(asyice_storage_.data(), Extents3D(asyice.extent(0), asyice.extent(1), asyice.extent(2)));
}

void CloudOptics::compute_cloud_optics(
    ConstView2D clwp, ConstView2D ciwp,
    ConstView2D rel, ConstView2D rei,
    View3D tau, View3D ssa, View3D g
) const {
    const int layers = clwp.extent(0);
    const int columns = clwp.extent(1);
    const int gp_cnt = get_gpoints();

    std::vector<real_t> taussa_data(gp_cnt * layers * columns, 0.0);
    std::vector<real_t> taussag_data(gp_cnt * layers * columns, 0.0);
    auto taussa = View3D(taussa_data.data(), Extents3D(gp_cnt, layers, columns));
    auto taussag = View3D(taussag_data.data(), Extents3D(gp_cnt, layers, columns));

    #pragma omp parallel for
    for (int col = 0; col < columns; ++col) {
        for (int lay = 0; lay < layers; ++lay) {
            for (int gp = 0; gp < gp_cnt; ++gp) {
                tau(gp, lay, col) = 0.0;
                ssa(gp, lay, col) = 0.0;
                g(gp, lay, col) = 0.0;
            }
        }
    }

    kernels::compute_cld_from_table(
        columns, layers, gp_cnt, clwp, rel,
        extliq_.extent(0), radliq_fac_, radliq_lwr_,
        extliq_, ssaliq_, asyliq_,
        tau, taussa, taussag
    );

    auto extice_slice = ConstView2D(&extice_(0, 0, 0), Extents2D(extice_.extent(0), gp_cnt));
    auto ssaice_slice = ConstView2D(&ssaice_(0, 0, 0), Extents2D(ssaice_.extent(0), gp_cnt));
    auto asyice_slice = ConstView2D(&asyice_(0, 0, 0), Extents2D(asyice_.extent(0), gp_cnt));

    kernels::compute_cld_from_table(
        columns, layers, gp_cnt, ciwp, rei,
        extice_.extent(0), radice_fac_, radice_lwr_,
        extice_slice, ssaice_slice, asyice_slice,
        tau, taussa, taussag
    );

    #pragma omp parallel for
    for (int col = 0; col < columns; ++col) {
        for (int lay = 0; lay < layers; ++lay) {
            for (int gp = 0; gp < gp_cnt; ++gp) {
                real_t t = tau(gp, lay, col);
                if (t > 0.0) {
                    ssa(gp, lay, col) = taussa(gp, lay, col) / t;
                    if (ssa(gp, lay, col) > 0.0) {
                        g(gp, lay, col) = taussag(gp, lay, col) / taussa(gp, lay, col);
                    }
                }
            }
        }
    }
}

AerosolOptics::AerosolOptics(OpticalProps base, ConstView3D aer_coefficients)
    : OpticalProps(base.get_gpoint_to_band(), base.get_band_lims_gpoint()) {
    validate_extent(aer_coefficients, 0, get_gpoints(), "aer_coefficients");
    aer_coefficients_storage_.assign(aer_coefficients.data_handle(), aer_coefficients.data_handle() + aer_coefficients.size());
    aer_coefficients_ = ConstView3D(
        aer_coefficients_storage_.data(),
        Extents3D(aer_coefficients.extent(0), aer_coefficients.extent(1), aer_coefficients.extent(2))
    );
}

void AerosolOptics::compute_aerosol_optics(
    ConstView2D aer_mass, View3D tau, View3D ssa, View3D g
) const {
    const int layers = aer_mass.extent(0);
    const int columns = aer_mass.extent(1);
    const int gp_cnt = get_gpoints();

    #pragma omp parallel for
    for (int col = 0; col < columns; ++col) {
        for (int lay = 0; lay < layers; ++lay) {
            real_t mass = aer_mass(lay, col);
            for (int gp = 0; gp < gp_cnt; ++gp) {
                tau(gp, lay, col) = mass * aer_coefficients_(gp, 0, 0);
                ssa(gp, lay, col) = aer_coefficients_(gp, 0, 1);
                g(gp, lay, col) = 0.7; // default asymmetry
            }
        }
    }
}

} // namespace rrtmgp
