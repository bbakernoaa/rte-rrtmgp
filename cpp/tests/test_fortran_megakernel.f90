program test_fortran_megakernel
  use omp_lib
  implicit none
  integer :: layers, columns, gpoints
  real(8), allocatable :: play(:,:), tlay(:,:), clwp(:,:)
  real(8), allocatable :: kmajor(:,:,:,:), lut_liquid(:,:,:)
  real(8), allocatable :: flux_dir(:,:)
  integer :: iter, iterations
  integer(8) :: count_rate, count_start, count_end
  real(8) :: elapsed_time, million_cells_per_sec

  layers = 128
  columns = 40000 ! 200x200 Grid
  gpoints = 128
  iterations = 10

  allocate(play(layers, columns))
  allocate(tlay(layers, columns))
  allocate(clwp(layers, columns))
  allocate(kmajor(gpoints, 2, 2, 1))
  allocate(lut_liquid(gpoints, 2, 2))
  allocate(flux_dir(gpoints, columns))

  play = 1000.0_8
  tlay = 290.0_8
  clwp = 0.2_8
  lut_liquid = 5.0_8
  kmajor = 2.5_8
  flux_dir = 0.0_8

  ! Warm up
  call run_kernels()

  call system_clock(count_start, count_rate)
  do iter = 1, iterations
     call run_kernels()
  end do
  call system_clock(count_end, count_rate)

  elapsed_time = dble(count_end - count_start) / dble(count_rate) / dble(iterations)
  million_cells_per_sec = dble(columns * layers * gpoints) / (elapsed_time * 1e6_8)

  print '(A,F10.3,A,F10.2,A)', "  Average execution time: ", elapsed_time * 1000.0_8, " ms | Throughput: ", million_cells_per_sec, " million cells/sec"

contains

  subroutine run_kernels()
    integer :: c, l, g
    real(8) :: tau_gas, tau_cloud, tau_total
    !$omp parallel do private(g, l, tau_gas, tau_cloud, tau_total)
    do c = 1, columns
       do g = 1, gpoints
          flux_dir(g, c) = 0.0_8
          do l = 1, layers
             ! Forcing exponential calculation to depend on g-point, preventing compiler hoisting (LICM)
             tau_gas = exp(-play(l, c) * tlay(l, c) * kmajor(g, 1, 1, 1) * 1e-6_8)
             tau_cloud = clwp(l, c) * lut_liquid(g, 1, 1)
             tau_total = tau_gas + tau_cloud
             flux_dir(g, c) = flux_dir(g, c) + tau_total * 10.0_8
          end do
       end do
    end do
  end subroutine run_kernels

end program test_fortran_megakernel
