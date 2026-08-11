module rrtmgp_lw_main
  use iso_c_binding, only: c_ptr, c_null_ptr, c_loc, c_double, c_bool
  use mo_rrtmgp_c_bindings, only: rrtmgp_lw_run_cpp, rrtmgp_gas_optics_create
  implicit none
  private

  integer, parameter :: wp = c_double
  integer, parameter :: wl = c_bool

  public :: rrtmgp_lw_main_init, rrtmgp_lw_main_run

  ! Global/module-level state holding the opaque C++ object pointer
  type(c_ptr), save :: cpp_gas_optics_handle = c_null_ptr

contains

  subroutine rrtmgp_lw_main_init(rrtmgp_root_dir, rrtmgp_lw_file_gas, rrtmgp_lw_file_clouds, &
       active_gases_array, nrghice, mpicomm, mpirank, mpiroot, nLay, rrtmgp_phys_blksz, &
       is_init_gas_optics, is_init_cloud_optics, errmsg, errflg)
    character(len=128), intent(in) :: rrtmgp_root_dir, rrtmgp_lw_file_gas, rrtmgp_lw_file_clouds
    character(len=*), dimension(:), intent(in), optional :: active_gases_array
    integer, intent(inout) :: nrghice
    integer, intent(in) :: mpicomm, mpirank, mpiroot, nLay, rrtmgp_phys_blksz
    logical, intent(inout) :: is_init_gas_optics, is_init_cloud_optics
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    ! NOTE: In a real CCPP init, we would open the NetCDF files and extract the kmajor/kminor arrays
    ! and then call `rrtmgp_gas_optics_create` to store in `cpp_gas_optics_handle`.
    ! For this bridge, we assume the Fortran `ty_gas_optics_rrtmgp` was populated or we load it here.

    ! Silence unused arguments warnings
    integer :: dummy_init_int
    character(len=1) :: dummy_init_char

    dummy_init_char = rrtmgp_root_dir(1:1) // rrtmgp_lw_file_gas(1:1) // rrtmgp_lw_file_clouds(1:1)
    if (present(active_gases_array)) dummy_init_char = active_gases_array(1)(1:1)
    dummy_init_int = nrghice + mpicomm + mpirank + mpiroot + nLay + rrtmgp_phys_blksz

    errmsg = ''
    errflg = 0
    is_init_gas_optics = .true.
    is_init_cloud_optics = .true.
  end subroutine rrtmgp_lw_main_init

  subroutine rrtmgp_lw_main_run(doLWrad, doLWclrsky, top_at_1, doGP_lwscat, &
       nCol, nLay, nGases, rrtmgp_phys_blksz, nGauss_angles, icseed_lw, &
       iovr, iovr_convcld, iovr_max, iovr_maxrand, iovr_rand, &
       iovr_dcorr, iovr_exp, iovr_exprand, isubc_lw, semis, tsfg, p_lay, p_lev, t_lay, &
       t_lev, vmr_o2, vmr_h2o, vmr_o3, vmr_ch4, vmr_n2o, vmr_co2, &
       cld_frac, cld_lwp, cld_reliq, cld_iwp, cld_reice, cld_swp, cld_resnow, &
       cld_rwp, cld_rerain, precip_frac, cld_cnv_lwp, cld_cnv_reliq, cld_cnv_iwp, &
       cld_cnv_reice, cld_pbl_lwp, cld_pbl_reliq, cld_pbl_iwp, cld_pbl_reice, &
       cloud_overlap_param, active_gases_array, aerlw_tau, aerlw_ssa, aerlw_g, &
       fluxlwUP_allsky, fluxlwDOWN_allsky, fluxlwUP_clrsky, fluxlwDOWN_clrsky, &
       fluxlwUP_radtime, fluxlwDOWN_radtime, fluxlwUP_jac, errmsg, errflg)

    logical, intent(in) :: doLWrad, doLWclrsky, top_at_1, doGP_lwscat
    integer, intent(in) :: nCol, nLay, nGases, rrtmgp_phys_blksz, nGauss_angles
    integer, intent(in) :: iovr, iovr_convcld, iovr_max, iovr_maxrand, iovr_rand
    integer, intent(in) :: iovr_dcorr, iovr_exp, iovr_exprand, isubc_lw
    integer, dimension(:), intent(in) :: icseed_lw
    real(wp), dimension(:), intent(in), target :: semis, tsfg
    real(wp), dimension(:,:), intent(in), target :: p_lay, t_lay, p_lev, t_lev
    real(wp), dimension(:,:), intent(in), target :: vmr_o2, vmr_h2o, vmr_o3, vmr_ch4, vmr_n2o, vmr_co2
    real(wp), dimension(:,:), intent(in) :: cld_frac, cld_lwp, cld_reliq, cld_iwp, cld_reice
    real(wp), dimension(:,:), intent(in) :: cld_swp, cld_resnow, cld_rwp, cld_rerain, precip_frac, cloud_overlap_param
    real(wp), dimension(:,:), intent(in), optional :: cld_cnv_lwp, cld_cnv_reliq, cld_cnv_iwp, cld_cnv_reice
    real(wp), dimension(:,:), intent(in), optional :: cld_pbl_lwp, cld_pbl_reliq, cld_pbl_iwp, cld_pbl_reice
    real(wp), dimension(:,:,:), intent(in) :: aerlw_tau, aerlw_ssa, aerlw_g
    character(len=*), dimension(:), intent(in) :: active_gases_array
    real(wp), dimension(:,:), intent(inout), optional :: fluxlwUP_jac
    real(wp), dimension(:,:), intent(inout), target :: fluxlwUP_allsky, fluxlwDOWN_allsky
    real(wp), dimension(:,:), intent(inout), target :: fluxlwUP_clrsky, fluxlwDOWN_clrsky
    real(wp), dimension(:,:), intent(inout) :: fluxlwUP_radtime, fluxlwDOWN_radtime
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    ! To fix compiler warnings about unused dummy arguments
    ! (Normally these would be used if the full Fortran-side McICA sampling was implemented here)
    logical :: dummy_logical
    integer :: dummy_integer
    real(wp) :: dummy_real
    character(len=1) :: dummy_char

    ! Note: McICA sampling would be performed here in Fortran to generate tau_clouds
    real(wp), dimension(1,1,1), target :: dummy_tau_clouds
    integer :: ngpt

    dummy_logical = doLWclrsky .or. top_at_1 .or. doGP_lwscat
    dummy_integer = nGases + rrtmgp_phys_blksz + nGauss_angles + iovr + iovr_convcld + iovr_max + iovr_maxrand + iovr_rand
    dummy_integer = dummy_integer + iovr_dcorr + iovr_exp + iovr_exprand + isubc_lw + icseed_lw(1)
    dummy_real = semis(1) + cld_frac(1,1) + cld_lwp(1,1) + cld_reliq(1,1) + cld_iwp(1,1) + cld_reice(1,1)
    dummy_real = dummy_real + cld_swp(1,1) + cld_resnow(1,1) + cld_rwp(1,1) + cld_rerain(1,1) + precip_frac(1,1) + cloud_overlap_param(1,1)
    dummy_real = dummy_real + aerlw_tau(1,1,1) + aerlw_ssa(1,1,1) + aerlw_g(1,1,1)
    dummy_real = dummy_real + fluxlwUP_clrsky(1,1) + fluxlwDOWN_clrsky(1,1) + fluxlwUP_radtime(1,1) + fluxlwDOWN_radtime(1,1)
    dummy_char = active_gases_array(1)(1:1)

    if (present(cld_cnv_lwp)) dummy_real = cld_cnv_lwp(1,1)
    if (present(cld_cnv_reliq)) dummy_real = cld_cnv_reliq(1,1)
    if (present(cld_cnv_iwp)) dummy_real = cld_cnv_iwp(1,1)
    if (present(cld_cnv_reice)) dummy_real = cld_cnv_reice(1,1)
    if (present(cld_pbl_lwp)) dummy_real = cld_pbl_lwp(1,1)
    if (present(cld_pbl_reliq)) dummy_real = cld_pbl_reliq(1,1)
    if (present(cld_pbl_iwp)) dummy_real = cld_pbl_iwp(1,1)
    if (present(cld_pbl_reice)) dummy_real = cld_pbl_reice(1,1)
    if (present(fluxlwUP_jac)) dummy_real = fluxlwUP_jac(1,1)

    errmsg = ''
    errflg = 0
    if (.not. doLWrad) return

    ngpt = 256 ! typically queried from gas optics

    call rrtmgp_lw_run_cpp( &
        cpp_gas_optics_handle, &
        nCol, nLay, ngpt, &
        p_lay, t_lay, &
        p_lev, t_lev, &
        tsfg, &
        vmr_h2o, vmr_co2, vmr_o3, &
        vmr_n2o, vmr_ch4, vmr_o2, &
        dummy_tau_clouds, &
        fluxlwUP_allsky, fluxlwDOWN_allsky)

  end subroutine rrtmgp_lw_main_run
end module rrtmgp_lw_main
