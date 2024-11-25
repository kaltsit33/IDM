function dtauda(this,a)

    use results

    implicit none
    class(CAMBdata) :: this
    real(dl), intent(in) :: a
    real(dl) :: dtauda
    real(dl) :: list(100000,2)
    integer :: i, j, iostat

    open(10, file='temp.txt')
    do i=1,100000
        read(10, *, iostat=iostat) list(i,:)
        if (iostat /= 0) exit
    end do
    close(10)
    
    j = sum(minloc(abs(a-list(:,1))))
    if (j == 1) then
        dtauda = list(1,2)
    else
        dtauda = (list(j,2)-list(j-1,2))/(list(j,1)-list(j-1,1))*(a-list(j,1))+list(j,2)
    end if
    write(*,*) 'a=', a, 'dtauda=', dtauda

end function dtauda