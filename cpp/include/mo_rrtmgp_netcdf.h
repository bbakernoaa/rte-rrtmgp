#ifndef MO_RRTMGP_NETCDF_H
#define MO_RRTMGP_NETCDF_H

#include "mo_gas_optics.h"
#include <string>
#include <memory>

namespace rrtmgp {

// Factory function to create GasOptics directly from a NetCDF file
std::unique_ptr<GasOptics> load_gas_optics_from_nc(const std::string& filename);

}

#ifdef __cplusplus
extern "C" {
#endif

// C-API exposed to Fortran CCPP bridge
void* rrtmgp_gas_optics_create_from_nc(const char* filename);

#ifdef __cplusplus
}
#endif

#endif // MO_RRTMGP_NETCDF_H
