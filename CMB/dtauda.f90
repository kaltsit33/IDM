! IDM background evolution
function dtauda(this,a)
    use constants
    use results
    implicit none
    class(CAMBdata) :: this
    real(dl), intent(in) :: a
    real(dl), dimension(:), allocatable :: zlist, zprime, alist
    real(dl) :: h0, o20, log_kc1
    real(dl) :: y_min(2), y_max(2), z_prime
    real(dl) :: t0, hstep, z(2)
    integer :: i, nstep
    real(dl) :: dtauda

    H0 = this%CP%H0
    O20 = this%CP%O20
    log_kC1 = this%CP%log_kC1

    nstep = 100000
    t0 = 1._dl / H0
    hstep = - t0 / nstep
    z(1) = 0._dl
    z(2) = -H0
    do i = 1, nstep
        call rk45(t0, z, hstep, z)
        zlist(i) = z(1)
        zprime(i) = z(2)
        alist(i) = 1._dl / (z(1) + 1._dl)
        t0 = t0 + hstep
    end do
    do i = 1, nstep
        if (a >= alist(i)) then
            y_min(1) = zlist(i)
            y_min(2) = zprime(i)
            y_max(1) = zlist(i+1)
            y_max(2) = zprime(i+1)
            exit
        end if
    end do
    z_prime = (1._dl / a - 1._dl - y_min(1)) / (y_max(1) - y_min(1)) & 
     * (y_max(2) - y_min(2)) + y_min(2)
    dtauda = -1._dl / (a**3._dl * z_prime) * c / 1000._dl

    ! define the rk45 method to solve the differential equation
    contains
    subroutine rk45(t, y, h, f)
        implicit none
        real(dl), intent(in) :: t, h
        real(dl), intent(inout) :: y(2)
        real(dl), intent(out) :: f(2)
        real(dl) :: k1(2), k2(2), k3(2), k4(2)
        ! Calculate the four Runge-Kutta coefficients
        call func(t, y, f)
        k1 = h * f
        call func(t + h / 2, y + k1 / 2, f)
        k2 = h * f
        call func(t + h / 2, y + k2 / 2, f)
        k3 = h * f
        call func(t + h, y + k3, f)
        k4 = h * f
        ! Update the solution
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    end subroutine rk45
    subroutine func(t, y, f)
        implicit none
        real(dl), intent(in) :: t, y(2)
        real(dl), intent(out) :: f(2)
        real(dl) :: O10, kC1
        real(dl) :: temp1, temp2, temp3, temp4
        O10 = 1._dl - O20
        kC1 = 10._dl**log_kC1
        ! Define the system of differential equations
        f(1) = y(2) ! y'
        temp1 = H0**4._dl*kC1*O10**2._dl*(y(1)**4._dl + 1._dl) &
            + 3._dl*H0**4._dl*O10**2._dl*y(1)**2._dl*(2._dl*kC1 - 3._dl*y(2))
        temp2 = H0**4._dl*O10**2._dl*y(1)**3._dl*(4._dl*kC1 - 3._dl*y(2)) &
            - 3._dl*H0**4._dl*O10**2._dl*y(2) + 5._dl*H0**2._dl*O10*y(2)**3._dl
        temp3 = kC1*y(2)**4._dl + H0**2._dl*O10*y(1)* &
            (4._dl*H0**2._dl*kC1*O10 - 9._dl*H0**2._dl*O10*y(2) + 5._dl*y(2)**3._dl)
        temp4 = 2._dl*H0**2._dl*O10*(1._dl + y(1))**2._dl*y(2)
        f(2) = (temp1 + temp2 - temp3) / temp4 ! y''
    end subroutine func

end function dtauda