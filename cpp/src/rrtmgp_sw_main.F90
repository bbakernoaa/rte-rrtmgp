module rrtmgp_sw_main
  use iso_c_binding, only: c_ptr, c_null_ptr, c_loc, c_double, c_bool
  use mo_rrtmgp_c_bindings, only: rrtmgp_sw_run_cpp, rrtmgp_gas_optics_create
  implicit none
  private

  integer, parameter :: wp = c_double
  integer, parameter :: wl = c_bool

  public :: rrtmgp_sw_main_init, rrtmgp_sw_main_run

  ! Global/module-level state holding the opaque C++ object pointer
  type(c_ptr), save :: cpp_gas_optics_handle = c_null_ptr

contains

  subroutine rrtmgp_sw_main_init(rrtmgp_root_dir, rrtmgp_sw_file_gas, rrtmgp_sw_file_clouds, &
       active_gases_array, nrghice, mpicomm, mpirank, mpiroot, nLay, rrtmgp_phys_blksz, &
       is_init_gas_optics, is_init_cloud_optics, errmsg, errflg)
    character(len=128), intent(in) :: rrtmgp_root_dir, rrtmgp_sw_file_gas, rrtmgp_sw_file_clouds
    character(len=*), dimension(:), intent(in), optional :: active_gases_array
    integer, intent(inout) :: nrghice
    integer, intent(in) :: mpicomm, mpirank, mpiroot, nLay, rrtmgp_phys_blksz
    logical, intent(inout) :: is_init_gas_optics, is_init_cloud_optics
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    integer :: dummy_init_int
    character(len=1) :: dummy_init_char

    dummy_init_char = rrtmgp_root_dir(1:1) // rrtmgp_sw_file_gas(1:1) // rrtmgp_sw_file_clouds(1:1)
    if (present(active_gases_array)) dummy_init_char = active_gases_array(1)(1:1)
    dummy_init_int = nrghice + mpicomm + mpirank + mpiroot + nLay + rrtmgp_phys_blksz

    errmsg = ''
    errflg = 0
    is_init_gas_optics = .true.
    is_init_cloud_optics = .true.
  end subroutine rrtmgp_sw_main_init

  subroutine rrtmgp_sw_main_run(doSWrad, doSWclrsky, top_at_1, doGP_swscat, &
       nCol, nLay, nGases, rrtmgp_phys_blksz, icseed_sw, &
       iovr, iovr_convcld, iovr_max, iovr_maxrand, iovr_rand, &
       iovr_dcorr, iovr_exp, iovr_exprand, isubc_sw, sza, toa_flux, &
       sfc_alb_dir, sfc_alb_dif, p_lay, p_lev, t_lay, t_lev, &
       vmr_o2, vmr_h2o, vmr_o3, vmr_ch4, vmr_n2o, vmr_co2, &
       cld_frac, cld_lwp, cld_reliq, cld_iwp, cld_reice, cld_swp, cld_resnow, &
       cld_rwp, cld_rerain, precip_frac, cld_cnv_lwp, cld_cnv_reliq, cld_cnv_iwp, &
       cld_cnv_reice, cld_pbl_lwp, cld_pbl_reliq, cld_pbl_iwp, cld_pbl_reice, &
       cloud_overlap_param, active_gases_array, aersw_tau, aersw_ssa, aersw_g, &
       fluxswUP_allsky, fluxswDOWN_allsky, fluxswDIR_allsky, &
       fluxswUP_clrsky, fluxswDOWN_clrsky, fluxswDIR_clrsky, &
       fluxswUP_radtime, fluxswDOWN_radtime, errmsg, errflg)

    logical, intent(in) :: doSWrad, doSWclrsky, top_at_1, doGP_swscat
    integer, intent(in) :: nCol, nLay, nGases, rrtmgp_phys_blksz
    integer, intent(in) :: iovr, iovr_convcld, iovr_max, iovr_maxrand, iovr_rand
    integer, intent(in) :: iovr_dcorr, iovr_exp, iovr_exprand, isubc_sw
    integer, dimension(:), intent(in) :: icseed_sw
    real(wp), dimension(:), intent(in), target :: sza, toa_flux, sfc_alb_dir, sfc_alb_dif
    real(wp), dimension(:,:), intent(in), target :: p_lay, t_lay, p_lev, t_lev
    real(wp), dimension(:,:), intent(in), target :: vmr_o2, vmr_h2o, vmr_o3, vmr_ch4, vmr_n2o, vmr_co2
    real(wp), dimension(:,:), intent(in) :: cld_frac, cld_lwp, cld_reliq, cld_iwp, cld_reice
    real(wp), dimension(:,:), intent(in) :: cld_swp, cld_resnow, cld_rwp, cld_rerain, precip_frac, cloud_overlap_param
    real(wp), dimension(:,:), intent(in), optional :: cld_cnv_lwp, cld_cnv_reliq, cld_cnv_iwp, cld_cnv_reice
    real(wp), dimension(:,:), intent(in), optional :: cld_pbl_lwp, cld_pbl_reliq, cld_pbl_iwp, cld_pbl_reice
    real(wp), dimension(:,:,:), intent(in) :: aersw_tau, aersw_ssa, aersw_g
    character(len=*), dimension(:), intent(in) :: active_gases_array
    real(wp), dimension(:,:), intent(inout), target :: fluxswUP_allsky, fluxswDOWN_allsky, fluxswDIR_allsky
    real(wp), dimension(:,:), intent(inout), target :: fluxswUP_clrsky, fluxswDOWN_clrsky, fluxswDIR_clrsky
    real(wp), dimension(:,:), intent(inout) :: fluxswUP_radtime, fluxswDOWN_radtime
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    logical :: dummy_logical
    integer :: dummy_integer
    real(wp) :: dummy_real
    character(len=1) :: dummy_char

    real(wp), dimension(1,1,1), target :: dummy_tau_clouds, dummy_ssa_clouds, dummy_g_clouds
    integer :: ngpt

    dummy_logical = doSWclrsky .or. top_at_1 .or. doGP_swscat
    dummy_integer = nGases + rrtmgp_phys_blksz + iovr + iovr_convcld + iovr_max + iovr_maxrand + iovr_rand
    dummy_integer = dummy_integer + iovr_dcorr + iovr_exp + iovr_exprand + isubc_sw + icseed_sw(1)
    dummy_real = cld_frac(1,1) + cld_lwp(1,1) + cld_reliq(1,1) + cld_iwp(1,1) + cld_reice(1,1)
    dummy_real = dummy_real + cld_swp(1,1) + cld_resnow(1,1) + cld_rwp(1,1) + cld_rerain(1,1) + precip_frac(1,1) + cloud_overlap_param(1,1)
    dummy_real = dummy_real + aersw_tau(1,1,1) + aersw_ssa(1,1,1) + aersw_g(1,1,1)
    dummy_real = dummy_real + fluxswUP_clrsky(1,1) + fluxswDOWN_clrsky(1,1) + fluxswDIR_clrsky(1,1) + fluxswUP_radtime(1,1) + fluxswDOWN_radtime(1,1)
    dummy_char = active_gases_array(1)(1:1)

    if (present(cld_cnv_lwp)) dummy_real = cld_cnv_lwp(1,1)
    if (present(cld_cnv_reliq)) dummy_real = cld_cnv_reliq(1,1)
    if (present(cld_cnv_iwp)) dummy_real = cld_cnv_iwp(1,1)
    if (present(cld_cnv_reice)) dummy_real = cld_cnv_reice(1,1)
    if (present(cld_pbl_lwp)) dummy_real = cld_pbl_lwp(1,1)
    if (present(cld_pbl_reliq)) dummy_real = cld_pbl_reliq(1,1)
    if (present(cld_pbl_iwp)) dummy_real = cld_pbl_iwp(1,1)
    if (present(cld_pbl_reice)) dummy_real = cld_pbl_reice(1,1)

    errmsg = ''
    errflg = 0
    if (.not. doSWrad) return

    ngpt = 224

    call rrtmgp_sw_run_cpp( &
        cpp_gas_optics_handle, &
        nCol, nLay, ngpt, &
        p_lay, t_lay, &
        p_lev, t_lev, &
        sza, toa_flux, sfc_alb_dir, sfc_alb_dif, &
        vmr_h2o, vmr_co2, vmr_o3, &
        vmr_n2o, vmr_ch4, vmr_o2, &
        dummy_tau_clouds, dummy_ssa_clouds, dummy_g_clouds, &
        fluxswUP_allsky, fluxswDOWN_allsky, fluxswDIR_allsky)

  end subroutine rrtmgp_sw_main_run
end module rrtmgp_sw_main
