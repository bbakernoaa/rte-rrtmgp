#include "mo_rrtmgp_netcdf.h"
#include <netcdf.h>
#include <stdexcept>
#include <vector>

namespace rrtmgp {

// Error helper for NetCDF C API calls
inline void check_nc(int retval, const std::string& msg) {
    if (retval != NC_NOERR) {
        throw std::runtime_error("NetCDF Error: " + msg + " (" + nc_strerror(retval) + ")");
    }
}

// In-memory NetCDF profile reader utility
struct NetCDFReader {
    int ncid;

    NetCDFReader(const std::string& filename) {
        check_nc(nc_open(filename.c_str(), NC_NOWRITE, &ncid), "Opening " + filename);
    }

    ~NetCDFReader() {
        nc_close(ncid);
    }

    bool has_var(const std::string& name) const {
        int varid;
        return nc_inq_varid(ncid, name.c_str(), &varid) == NC_NOERR;
    }

    void read_double(const std::string& name, std::vector<real_t>& data) {
        int varid;
        check_nc(nc_inq_varid(ncid, name.c_str(), &varid), "Inquiring " + name);
        check_nc(nc_get_var_double(ncid, varid, data.data()), "Reading " + name);
    }

    void read_int(const std::string& name, std::vector<int>& data) {
        int varid;
        check_nc(nc_inq_varid(ncid, name.c_str(), &varid), "Inquiring " + name);
        check_nc(nc_get_var_int(ncid, varid, data.data()), "Reading " + name);
    }
    
    size_t get_dim(const std::string& name) const {
        int dimid;
        check_nc(nc_inq_dimid(ncid, name.c_str(), &dimid), "Inq dim " + name);
        size_t len;
        check_nc(nc_inq_dimlen(ncid, dimid, &len), "Len dim " + name);
        return len;
    }
};

std::unique_ptr<GasOptics> load_gas_optics_from_nc(const std::string& filename) {
    NetCDFReader reader(filename);
    
    size_t n_bands = reader.get_dim("bnd");
    size_t n_gpt = reader.get_dim("gpt");
    size_t n_press = reader.get_dim("press");
    size_t n_temp = reader.get_dim("temp");
    size_t n_eta = reader.get_dim("mix_frac");
    size_t n_flav = reader.get_dim("flavor");
    size_t n_absorbers = reader.get_dim("absorber");
    (void)n_flav; (void)n_absorbers;

    // Read optical properties basic metadata
    std::vector<int> band_lims_gpoint(2 * n_bands);
    reader.read_int("bnd_limits_gpt", band_lims_gpoint);
    // Note: NetCDF stores bnd_limits_gpt as (bnd, 2), we need to map to layout left?
    // In Fortran it's (2, n_bands), so reading contiguous C is fine if NetCDF array is (bnd, 2).
    // Let's assume standard layout conversion.

    std::vector<int> gpoint_to_band(n_gpt); // Fortran array gpoint_flavor etc...
    // The coefficients...
    std::vector<real_t> kmajor(n_gpt * n_eta * n_press * n_temp);
    reader.read_double("kmajor", kmajor);

    // Read reference coordinates
    std::vector<real_t> press_ref(n_press);
    reader.read_double("press_ref", press_ref);

    std::vector<real_t> temp_ref(n_temp);
    reader.read_double("temp_ref", temp_ref);

    std::vector<real_t> press_ref_trop(1);
    reader.read_double("press_ref_trop", press_ref_trop);

    // Minor gases
    std::vector<real_t> kminor_lower;
    if (reader.has_var("kminor_lower")) {
        size_t n_minor_lower = reader.get_dim("minor_absorber_intervals_lower");
        kminor_lower.resize(n_minor_lower * n_eta * n_temp);
        reader.read_double("kminor_lower", kminor_lower);
    }

    std::vector<real_t> kminor_upper;
    if (reader.has_var("kminor_upper")) {
        size_t n_minor_upper = reader.get_dim("minor_absorber_intervals_upper");
        kminor_upper.resize(n_minor_upper * n_eta * n_temp);
        reader.read_double("kminor_upper", kminor_upper);
    }

    // Planck / Rayleigh
    std::vector<real_t> planck_frac(n_gpt * n_eta * n_press * n_temp);
    if (reader.has_var("plank_fraction")) { // Note spelling in netcdf file
        reader.read_double("plank_fraction", planck_frac);
    }

    std::vector<real_t> krayl(n_gpt * n_eta * n_temp * 2);
    if (reader.has_var("rayl_lower")) {
        // Will need to combine lower/upper into krayl
    }

    // Return mock for now while we build the struct out
    // TODO: implement full population
    return nullptr;
}

}

extern "C" {
void* rrtmgp_gas_optics_create_from_nc(const char* filename) {
    auto optics = rrtmgp::load_gas_optics_from_nc(filename);
    return optics.release();
}
}
