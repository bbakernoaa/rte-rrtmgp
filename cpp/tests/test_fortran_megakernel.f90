program test_fortran_megakernel
  use omp_lib
  use mo_gas_optics_rrtmgp_kernels
  use mo_rte_solver_kernels
  use mo_rte_kind, only: wp, wl
  implicit none
  integer :: layers, columns, gpoints, nflav
  real(wp), allocatable :: play(:,:), plev(:,:), tlay(:,:), tsfc(:), col_gas(:,:,:)
  integer, allocatable :: jtemp(:,:), jpress(:,:), jeta(:,:,:,:)
  real(wp), allocatable :: fmajor(:,:,:,:,:,:), fminor(:,:,:,:,:)
  real(wp), allocatable :: kmajor(:,:,:,:), col_mix(:,:,:,:)
  logical(wl), allocatable :: tropo(:,:)
  real(wp), allocatable :: tau_gas(:,:,:), tau_rayl(:,:,:), planck_src(:,:)
  real(wp), allocatable :: lay_source(:,:,:), lev_source(:,:,:), sfc_emis(:,:), inc_flux(:,:)
  real(wp), allocatable :: flux_up(:,:,:), flux_dn(:,:,:)

  integer :: iter, iterations
  integer(8) :: count_rate, count_start, count_end
  real(8) :: elapsed_time, million_cells_per_sec

  ! Dummy variables for interpolation
  integer, allocatable :: flavor(:,:)
  real(wp), allocatable :: press_ref_log(:), temp_ref(:), vmr_ref(:,:,:)
  real(wp) :: press_ref_log_delta, temp_ref_min, temp_ref_delta, press_ref_trop_log

  ! Dummy variables for absorption
  integer :: nbnd, ngas, neta, npres, ntemp, nminorlower, nminorklower, nminorupper, nminorkupper, idx_h2o
  integer, allocatable :: gpoint_flavor(:,:), band_lims_gpt(:,:)
  real(wp), allocatable :: kminor_lower(:,:,:), kminor_upper(:,:,:)
  integer, allocatable :: minor_limits_gpt_lower(:,:), minor_limits_gpt_upper(:,:)
  logical(wl), allocatable :: minor_scales_with_density_lower(:), minor_scales_with_density_upper(:)
  logical(wl), allocatable :: scale_by_complement_lower(:), scale_by_complement_upper(:)
  integer, allocatable :: idx_minor_lower(:), idx_minor_upper(:)
  integer, allocatable :: idx_minor_scaling_lower(:), idx_minor_scaling_upper(:)
  integer, allocatable :: kminor_start_lower(:), kminor_start_upper(:)

  layers = 128
  columns = 40000
  gpoints = 128
  iterations = 10

  nflav = 1
  ngas = 1
  neta = 2
  npres = 3
  ntemp = 2
  nbnd = 1
  nminorlower = 0
  nminorklower = 0
  nminorupper = 0
  nminorkupper = 0
  idx_h2o = 1

  allocate(play(columns, layers))
  allocate(plev(columns, layers+1))
  allocate(tlay(columns, layers))
  allocate(tsfc(columns))
  allocate(col_gas(columns, layers, 0:ngas))

  allocate(jtemp(columns, layers))
  allocate(jpress(columns, layers))
  allocate(jeta(2, columns, layers, nflav))
  allocate(tropo(columns, layers))
  allocate(col_mix(2, columns, layers, nflav))

  allocate(fmajor(2, 2, 2, columns, layers, nflav))
  allocate(fminor(2, 2, columns, layers, nflav))
  allocate(kmajor(ntemp, neta, npres+1, gpoints))

  allocate(tau_gas(columns, layers, gpoints))
  allocate(tau_rayl(columns, layers, gpoints))
  allocate(planck_src(columns, gpoints))

  allocate(lay_source(columns, layers, gpoints))
  allocate(lev_source(columns, layers+1, gpoints))
  allocate(sfc_emis(columns, gpoints))
  allocate(inc_flux(columns, gpoints))

  allocate(flux_up(columns, layers+1, gpoints))
  allocate(flux_dn(columns, layers+1, gpoints))

  ! Alloc dummy variables
  allocate(flavor(2, nflav))
  allocate(press_ref_log(npres), temp_ref(ntemp), vmr_ref(2, 0:ngas, ntemp))
  allocate(gpoint_flavor(2, gpoints), band_lims_gpt(2, nbnd))
  allocate(kminor_lower(ntemp, neta, 1), kminor_upper(ntemp, neta, 1))
  allocate(minor_limits_gpt_lower(2, 1), minor_limits_gpt_upper(2, 1))
  allocate(minor_scales_with_density_lower(1), minor_scales_with_density_upper(1))
  allocate(scale_by_complement_lower(1), scale_by_complement_upper(1))
  allocate(idx_minor_lower(1), idx_minor_upper(1))
  allocate(idx_minor_scaling_lower(1), idx_minor_scaling_upper(1))
  allocate(kminor_start_lower(1), kminor_start_upper(1))

  flavor = 1
  press_ref_log = [log(1000.0_wp), log(500.0_wp), log(10.0_wp)]
  temp_ref = [200.0_wp, 300.0_wp]
  vmr_ref = 0.1_wp
  press_ref_log_delta = log(500.0_wp) - log(1000.0_wp)
  temp_ref_min = 180.0_wp
  temp_ref_delta = 10.0_wp
  press_ref_trop_log = log(100.0_wp)
  gpoint_flavor = 1
  band_lims_gpt(1, 1) = 1
  band_lims_gpt(2, 1) = gpoints

  play = 1000.0_wp
  plev = 1000.0_wp
  tlay = 290.0_wp
  tsfc = 300.0_wp
  kmajor = 2.5_wp
  col_gas = 0.1_wp

  lay_source = 1.0_wp
  lev_source = 1.0_wp
  sfc_emis = 1.0_wp
  inc_flux = 0.0_wp

  tau_gas = 0.0_wp
  tau_rayl = 0.0_wp
  planck_src = 0.0_wp

  ! Warm up
  call run_kernels()

  call system_clock(count_start, count_rate)
  do iter = 1, iterations
     call run_kernels()
  end do
  call system_clock(count_end, count_rate)

  elapsed_time = dble(count_end - count_start) / dble(count_rate) / dble(iterations)
  million_cells_per_sec = dble(columns * layers * gpoints) / (elapsed_time * 1e6_8)

  print '(A)', "=========================================================="
  print '(A,I0,A)', "      FORTRAN FULL PIPELINE BENCHMARK (", columns, " cols)"
  print '(A)', "=========================================================="
  print '(A,F10.3,A)', "  Average execution time: ", elapsed_time * 1000.0_8, " ms"
  print '(A,F10.2,A)', "  Computational throughput: ", million_cells_per_sec, " million cells/sec"
  print '(A)', "=========================================================="

contains

  subroutine run_kernels()
    integer :: g, nmus, b, nblocks, block_size, col_start, col_end, ncol_b
    real(wp), dimension(1) :: weight
    real(wp), dimension(columns, gpoints, 1) :: D_secant
    logical(wl) :: do_broadband, do_Jacobians, do_rescaling
    real(wp), dimension(columns, layers+1) :: broadband_up, broadband_dn, flux_upJac
    real(wp), dimension(columns, gpoints) :: sfc_srcJac
    real(wp), dimension(columns, layers, gpoints) :: ssa, g_param

    nmus = 1
    D_secant = 1.66_wp
    weight(1) = 1.0_wp
    do_broadband = .false.
    do_Jacobians = .false.
    do_rescaling = .false.

    block_size = 128
    nblocks = (columns + block_size - 1) / block_size

    !$omp parallel do private(b, col_start, col_end, ncol_b) schedule(static)
    do b = 1, nblocks
       col_start = (b - 1) * block_size + 1
       col_end = min(b * block_size, columns)
       ncol_b = col_end - col_start + 1

       call interpolation( &
                   ncol_b, layers, ngas, nflav, neta, npres, ntemp, &
                   flavor, &
                   press_ref_log, temp_ref, press_ref_log_delta, &
                   temp_ref_min, temp_ref_delta, press_ref_trop_log, &
                   vmr_ref, &
                   play(col_start, 1), tlay(col_start, 1), col_gas(col_start, 1, 0), &
                   jtemp(col_start, 1), fmajor(1, 1, 1, col_start, 1, 1), &
                   fminor(1, 1, col_start, 1, 1), col_mix(1, col_start, 1, 1), &
                   tropo(col_start, 1), jeta(1, col_start, 1, 1), jpress(col_start, 1))

       call compute_tau_absorption( &
                   ncol_b, layers, nbnd, gpoints, &
                   ngas, nflav, neta, npres, ntemp, &
                   nminorlower, nminorklower, nminorupper, nminorkupper, &
                   idx_h2o, gpoint_flavor, band_lims_gpt, &
                   kmajor, kminor_lower, kminor_upper, &
                   minor_limits_gpt_lower, minor_limits_gpt_upper, &
                   minor_scales_with_density_lower, minor_scales_with_density_upper, &
                   scale_by_complement_lower, scale_by_complement_upper, &
                   idx_minor_lower, idx_minor_upper, &
                   idx_minor_scaling_lower, idx_minor_scaling_upper, &
                   kminor_start_lower, kminor_start_upper, &
                   tropo(col_start, 1), col_mix(1, col_start, 1, 1), &
                   fmajor(1, 1, 1, col_start, 1, 1), fminor(1, 1, col_start, 1, 1), &
                   play(col_start, 1), tlay(col_start, 1), col_gas(col_start, 1, 0), &
                   jeta(1, col_start, 1, 1), jtemp(col_start, 1), jpress(col_start, 1), &
                   tau_gas(col_start, 1, 1))

       call lw_solver_noscat(ncol_b, layers, gpoints, .true._wl, nmus, D_secant(col_start, 1, 1), weight, &
                                 tau_gas(col_start, 1, 1), lay_source(col_start, 1, 1), &
                                 lev_source(col_start, 1, 1), sfc_emis(col_start, 1), &
                                 planck_src(col_start, 1), &
                                 inc_flux(col_start, 1), flux_up(col_start, 1, 1), &
                                 flux_dn(col_start, 1, 1), &
                                 do_broadband, broadband_up(col_start, 1), &
                                 broadband_dn(col_start, 1), &
                                 do_Jacobians, sfc_srcJac(col_start, 1), &
                                 flux_upJac(col_start, 1), &
                                 do_rescaling, ssa(col_start, 1, 1), g_param(col_start, 1, 1))
    end do
    !$omp end parallel do
  end subroutine run_kernels

end program test_fortran_megakernel
