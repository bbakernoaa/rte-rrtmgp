module mo_rrtmgp_c_bindings
  use iso_c_binding, only: c_ptr, c_int, c_double
  implicit none
  private

  public :: rrtmgp_gas_optics_create
  public :: rrtmgp_gas_optics_delete
  public :: rrtmgp_lw_run_cpp
  public :: rrtmgp_sw_run_cpp

  interface
    function rrtmgp_gas_optics_create( &
        gpoint_to_band, n_gpoints, &
        band_lims, n_bands, &
        kmajor, kmajor_dim1, kmajor_dim2, kmajor_dim3, kmajor_dim4, &
        kminor, kminor_dim1, kminor_dim2, kminor_dim3) bind(c, name="rrtmgp_gas_optics_create")
      use iso_c_binding
      type(c_ptr) :: rrtmgp_gas_optics_create
      integer(c_int), intent(in) :: gpoint_to_band(*)
      integer(c_int), value :: n_gpoints
      integer(c_int), intent(in) :: band_lims(*)
      integer(c_int), value :: n_bands
      real(c_double), intent(in) :: kmajor(*)
      integer(c_int), value :: kmajor_dim1, kmajor_dim2, kmajor_dim3, kmajor_dim4
      real(c_double), intent(in) :: kminor(*)
      integer(c_int), value :: kminor_dim1, kminor_dim2, kminor_dim3
    end function rrtmgp_gas_optics_create

    subroutine rrtmgp_gas_optics_delete(handle) bind(c, name="rrtmgp_gas_optics_delete")
      use iso_c_binding
      type(c_ptr), value :: handle
    end subroutine rrtmgp_gas_optics_delete

    subroutine rrtmgp_lw_run_cpp( &
        gas_optics_handle, &
        ncol, nlay, ngpt, &
        p_lay, t_lay, &
        p_lev, t_lev, &
        tsfg, &
        vmr_h2o, vmr_co2, vmr_o3, &
        vmr_n2o, vmr_ch4, vmr_o2, &
        tau_clouds, &
        flux_up, flux_dn) bind(c, name="rrtmgp_lw_run_cpp")
      use iso_c_binding
      type(c_ptr), value :: gas_optics_handle
      integer(c_int), value :: ncol, nlay, ngpt
      real(c_double), intent(in) :: p_lay(*), t_lay(*)
      real(c_double), intent(in) :: p_lev(*), t_lev(*)
      real(c_double), intent(in) :: tsfg(*)
      real(c_double), intent(in) :: vmr_h2o(*), vmr_co2(*), vmr_o3(*)
      real(c_double), intent(in) :: vmr_n2o(*), vmr_ch4(*), vmr_o2(*)
      real(c_double), intent(in) :: tau_clouds(*)
      real(c_double), intent(out) :: flux_up(*), flux_dn(*)
    end subroutine rrtmgp_lw_run_cpp

    subroutine rrtmgp_sw_run_cpp( &
        gas_optics_handle, &
        ncol, nlay, ngpt, &
        p_lay, t_lay, &
        p_lev, t_lev, &
        sza, toa_flux, sfc_alb_dir, sfc_alb_dif, &
        vmr_h2o, vmr_co2, vmr_o3, &
        vmr_n2o, vmr_ch4, vmr_o2, &
        tau_clouds, ssa_clouds, g_clouds, &
        flux_up, flux_dn, flux_dir) bind(c, name="rrtmgp_sw_run_cpp")
      use iso_c_binding
      type(c_ptr), value :: gas_optics_handle
      integer(c_int), value :: ncol, nlay, ngpt
      real(c_double), intent(in) :: p_lay(*), t_lay(*)
      real(c_double), intent(in) :: p_lev(*), t_lev(*)
      real(c_double), intent(in) :: sza(*), toa_flux(*), sfc_alb_dir(*), sfc_alb_dif(*)
      real(c_double), intent(in) :: vmr_h2o(*), vmr_co2(*), vmr_o3(*)
      real(c_double), intent(in) :: vmr_n2o(*), vmr_ch4(*), vmr_o2(*)
      real(c_double), intent(in) :: tau_clouds(*), ssa_clouds(*), g_clouds(*)
      real(c_double), intent(out) :: flux_up(*), flux_dn(*), flux_dir(*)
    end subroutine rrtmgp_sw_run_cpp
  end interface

end module mo_rrtmgp_c_bindings
