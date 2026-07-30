module mo_rrtmgp_cpp_interface
  use iso_c_binding
  implicit none

  ! 100% standard Fortran 2003+ ISO_C_BINDING wrappers linking natively to the C++ core
  interface
     subroutine c_execute_megakernel(layers, columns, gpoints, play, tlay, kmajor, clwp, lut_liquid, sza, toa, flux_dir) &
          bind(C, name="c_execute_megakernel")
       use iso_c_binding
       integer(c_size_t), value :: layers, columns, gpoints
       real(c_double), intent(in) :: play(*), tlay(*)
       real(c_double), intent(in) :: kmajor(*), clwp(*), lut_liquid(*)
       real(c_double), intent(in) :: sza(*), toa(*)
       real(c_double), intent(out) :: flux_dir(*)
     end subroutine c_execute_megakernel
  end interface

end module mo_rrtmgp_cpp_interface
