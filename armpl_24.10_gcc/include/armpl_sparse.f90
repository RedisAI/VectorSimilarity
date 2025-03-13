!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!   SPDX-FileCopyrightText: Copyright 2015-2024 NAG
!
!   Interface blocks for Fortran sparse routines in ARMPL
!
module armpl_sparse

    use armpl_kinds
    use armpl_sparse_params

!   Sparse matrix utility subroutines

    interface armpl_spmat_create_csr

        subroutine armpl_spmat_create_csr_s(A, m, n, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csr_s

        subroutine armpl_spmat_create_csr_d(A, m, n, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csr_d

        subroutine armpl_spmat_create_csr_c(A, m, n, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csr_c

        subroutine armpl_spmat_create_csr_z(A, m, n, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_ptr(m+1)
            integer, intent(in) :: col_indx(*)
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csr_z

    end interface armpl_spmat_create_csr

    interface armpl_spmat_create_csc

        subroutine armpl_spmat_create_csc_s(A, m, n, row_indx, col_ptr, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csc_s

        subroutine armpl_spmat_create_csc_d(A, m, n, row_indx, col_ptr, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csc_d

        subroutine armpl_spmat_create_csc_c(A, m, n, row_indx, col_ptr, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csc_c

        subroutine armpl_spmat_create_csc_z(A, m, n, row_indx, col_ptr, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_ptr(n+1)
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_csc_z

    end interface armpl_spmat_create_csc

    interface armpl_spmat_create_coo

        subroutine armpl_spmat_create_coo_s(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_coo_s

        subroutine armpl_spmat_create_coo_d(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_coo_d

        subroutine armpl_spmat_create_coo_c(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_coo_c

        subroutine armpl_spmat_create_coo_z(A, m, n, nnz, row_indx, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer, intent(in) :: m, n, nnz
            integer, intent(in) :: row_indx(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_coo_z

    end interface armpl_spmat_create_coo

    interface
        subroutine armpl_spmat_destroy(A, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: A
            integer, intent(out) :: info
        end subroutine armpl_spmat_destroy
    end interface

    interface
        subroutine armpl_spmat_hint(A, sparse_hint_type, sparse_hint_value, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: sparse_hint_type, sparse_hint_value
            integer, intent(out) :: info
        end subroutine armpl_spmat_hint
    end interface

    interface
        subroutine armpl_spmv_optimize(A, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine armpl_spmv_optimize
    end interface

    interface armpl_spmat_update

        subroutine armpl_spmat_update_s(A, n_updates, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            real(kind=armpl_r32), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine armpl_spmat_update_s

        subroutine armpl_spmat_update_d(A, n_updates, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            real(kind=armpl_r64), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine armpl_spmat_update_d

        subroutine armpl_spmat_update_c(A, n_updates, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            complex(kind=armpl_r32), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine armpl_spmat_update_c

        subroutine armpl_spmat_update_z(A, n_updates, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: n_updates
            integer, intent(in) :: row_indx(n_updates)
            integer, intent(in) :: col_indx(n_updates)
            complex(kind=armpl_r64), intent(in) :: vals(n_updates)
            integer, intent(out) :: info
        end subroutine armpl_spmat_update_z

    end interface armpl_spmat_update

    interface armpl_spmat_create_dense

        subroutine armpl_spmat_create_dense_s(A, layout, m, n, lda, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_dense_s

        subroutine armpl_spmat_create_dense_d(A, layout, m, n, lda, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_dense_d

        subroutine armpl_spmat_create_dense_c(A, layout, m, n, lda, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_dense_c

        subroutine armpl_spmat_create_dense_z(A, layout, m, n, lda, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: lda
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_dense_z

    end interface armpl_spmat_create_dense

    interface armpl_spmat_create_bsr

        subroutine armpl_spmat_create_bsr_s(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_bsr_s

        subroutine armpl_spmat_create_bsr_d(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_bsr_d

        subroutine armpl_spmat_create_bsr_c(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_bsr_c

        subroutine armpl_spmat_create_bsr_z(A, block_layout, m, n, block_size, row_ptr, col_indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: block_size
            integer, intent(in) :: row_ptr(*)
            integer, intent(in) :: col_indx(*)
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spmat_create_bsr_z

    end interface armpl_spmat_create_bsr

    interface
        function armpl_spmat_create_null(m, n)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8) :: armpl_spmat_create_null
            integer, intent(in) :: m
            integer, intent(in) :: n
        end function armpl_spmat_create_null
    end interface

    interface
        function armpl_spmat_create_identity(n)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8) :: armpl_spmat_create_identity
            integer, intent(in) :: n
        end function armpl_spmat_create_identity
    end interface

    interface
        subroutine armpl_spmat_query(A, index_base, m, n, nnz, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine armpl_spmat_query
    end interface

    interface armpl_spvec_create
        subroutine armpl_spvec_create_s(x, index_base, n, nnz, indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_create_s

        subroutine armpl_spvec_create_d(x, index_base, n, nnz, indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_create_d

        subroutine armpl_spvec_create_c(x, index_base, n, nnz, indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_create_c

        subroutine armpl_spvec_create_z(x, index_base, n, nnz, indx, vals, flags, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(out) :: x
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer, intent(in) :: nnz
            integer, intent(in) :: indx(*)
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_create_z
    end interface armpl_spvec_create

    interface
        subroutine armpl_spvec_query(x, index_base, n, nnz, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: info
        end subroutine armpl_spvec_query
    end interface

    interface
        subroutine armpl_spvec_destroy(x, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            integer, intent(out) :: info
        end subroutine armpl_spvec_destroy
    end interface

    interface armpl_spvec_export
        subroutine armpl_spvec_export_s(x, index_base, n, nnz, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            real(kind=armpl_r32), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_export_s

        subroutine armpl_spvec_export_d(x, index_base, n, nnz, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            real(kind=armpl_r64), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_export_d

        subroutine armpl_spvec_export_c(x, index_base, n, nnz, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            complex(kind=armpl_r32), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_export_c

        subroutine armpl_spvec_export_z(x, index_base, n, nnz, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            integer, intent(out) :: index_base
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, intent(out) :: indx(*)
            complex(kind=armpl_r64), intent(out) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_export_z
    end interface armpl_spvec_export

    interface armpl_spvec_gather
        subroutine armpl_spvec_gather_s(x_d, index_base, n, x_s, flags, info)
            use armpl_kinds
            implicit none
            real(kind=armpl_r32), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=armpl_i8), intent(out) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_gather_s

        subroutine armpl_spvec_gather_d(x_d, index_base, n, x_s, flags, info)
            use armpl_kinds
            implicit none
            real(kind=armpl_r64), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=armpl_i8), intent(out) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_gather_d

        subroutine armpl_spvec_gather_c(x_d, index_base, n, x_s, flags, info)
            use armpl_kinds
            implicit none
            complex(kind=armpl_r32), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=armpl_i8), intent(out) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_gather_c

        subroutine armpl_spvec_gather_z(x_d, index_base, n, x_s, flags, info)
            use armpl_kinds
            implicit none
            complex(kind=armpl_r64), intent(in) :: x_d(*)
            integer, intent(in) :: index_base
            integer, intent(in) :: n
            integer(kind=armpl_i8), intent(out) :: x_s
            integer, intent(in) :: flags
            integer, intent(out) :: info
        end subroutine armpl_spvec_gather_z
    end interface armpl_spvec_gather

    interface armpl_spvec_scatter
        subroutine armpl_spvec_scatter_s(x_s, x_d, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x_s
            real(kind=armpl_r32), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_scatter_s

        subroutine armpl_spvec_scatter_d(x_s, x_d, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x_s
            real(kind=armpl_r64), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_scatter_d

        subroutine armpl_spvec_scatter_c(x_s, x_d, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x_s
            complex(kind=armpl_r32), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_scatter_c

        subroutine armpl_spvec_scatter_z(x_s, x_d, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x_s
            complex(kind=armpl_r64), intent(out) :: x_d(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_scatter_z
    end interface armpl_spvec_scatter

    interface armpl_spvec_update
        subroutine armpl_spvec_update_s(x, n_updates, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            real(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_update_s

        subroutine armpl_spvec_update_d(x, n_updates, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            real(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_update_d

        subroutine armpl_spvec_update_c(x, n_updates, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            complex(kind=armpl_r32), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_update_c

        subroutine armpl_spvec_update_z(x, n_updates, indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            integer, intent(in) :: n_updates
            integer, intent(in) :: indx(*)
            complex(kind=armpl_r64), intent(in) :: vals(*)
            integer, intent(out) :: info
        end subroutine armpl_spvec_update_z
    end interface armpl_spvec_update

    interface armpl_spdot_exec
        subroutine armpl_spdot_exec_s(x, y, result, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            real(kind=armpl_r32), intent(in) :: y(*)
            real(kind=armpl_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine armpl_spdot_exec_s

        subroutine armpl_spdot_exec_d(x, y, result, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            real(kind=armpl_r64), intent(in) :: y(*)
            real(kind=armpl_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine armpl_spdot_exec_d
    end interface armpl_spdot_exec

    interface armpl_spdotu_exec
        subroutine armpl_spdotu_exec_c(x, y, result, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r32), intent(in) :: y(*)
            complex(kind=armpl_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine armpl_spdotu_exec_c

        subroutine armpl_spdotu_exec_z(x, y, result, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r64), intent(in) :: y(*)
            complex(kind=armpl_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine armpl_spdotu_exec_z
    end interface armpl_spdotu_exec

    interface armpl_spdotc_exec
        subroutine armpl_spdotc_exec_c(x, y, result, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r32), intent(in) :: y(*)
            complex(kind=armpl_r32), intent(out) :: result
            integer, intent(out) :: info
        end subroutine armpl_spdotc_exec_c

        subroutine armpl_spdotc_exec_z(x, y, result, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r64), intent(in) :: y(*)
            complex(kind=armpl_r64), intent(out) :: result
            integer, intent(out) :: info
        end subroutine armpl_spdotc_exec_z
    end interface armpl_spdotc_exec

    interface armpl_spaxpby_exec
        subroutine armpl_spaxpby_exec_s(alpha, x, beta, y, info)
            use armpl_kinds
            implicit none
            real(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            real(kind=armpl_r32), intent(in) :: beta
            real(kind=armpl_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spaxpby_exec_s

        subroutine armpl_spaxpby_exec_d(alpha, x, beta, y, info)
            use armpl_kinds
            implicit none
            real(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            real(kind=armpl_r64), intent(in) :: beta
            real(kind=armpl_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spaxpby_exec_d

        subroutine armpl_spaxpby_exec_c(alpha, x, beta, y, info)
            use armpl_kinds
            implicit none
            complex(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r32), intent(in) :: beta
            complex(kind=armpl_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spaxpby_exec_c

        subroutine armpl_spaxpby_exec_z(alpha, x, beta, y, info)
            use armpl_kinds
            implicit none
            complex(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r64), intent(in) :: beta
            complex(kind=armpl_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spaxpby_exec_z
    end interface armpl_spaxpby_exec

    interface armpl_spwaxpby_exec
        subroutine armpl_spwaxpby_exec_s(alpha, x, beta, y, w, info)
            use armpl_kinds
            implicit none
            real(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            real(kind=armpl_r32), intent(in) :: beta
            real(kind=armpl_r32), intent(in) :: y(*)
            real(kind=armpl_r32), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine armpl_spwaxpby_exec_s

        subroutine armpl_spwaxpby_exec_d(alpha, x, beta, y, w, info)
            use armpl_kinds
            implicit none
            real(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            real(kind=armpl_r64), intent(in) :: beta
            real(kind=armpl_r64), intent(in) :: y(*)
            real(kind=armpl_r64), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine armpl_spwaxpby_exec_d

        subroutine armpl_spwaxpby_exec_c(alpha, x, beta, y, w, info)
            use armpl_kinds
            implicit none
            complex(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r32), intent(in) :: beta
            complex(kind=armpl_r32), intent(in) :: y(*)
            complex(kind=armpl_r32), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine armpl_spwaxpby_exec_c

        subroutine armpl_spwaxpby_exec_z(alpha, x, beta, y, w, info)
            use armpl_kinds
            implicit none
            complex(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: x
            complex(kind=armpl_r64), intent(in) :: beta
            complex(kind=armpl_r64), intent(in) :: y(*)
            complex(kind=armpl_r64), intent(out) :: w(*)
            integer, intent(out) :: info
        end subroutine armpl_spwaxpby_exec_z
    end interface armpl_spwaxpby_exec

    interface
        subroutine armpl_spmm_optimize(transA, transB, alpha, A, B, beta, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            integer(kind=armpl_i4), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i8), intent(in) :: B
            integer(kind=armpl_i4), intent(in) :: beta
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spmm_optimize
    end interface

    interface
        subroutine armpl_spadd_optimize(transA, transB, alpha, A, beta, B, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            integer(kind=armpl_i4), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: beta
            integer(kind=armpl_i8), intent(in) :: B
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spadd_optimize
    end interface

    interface
        subroutine armpl_spsv_optimize(A, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: info
        end subroutine armpl_spsv_optimize
    end interface

!   Sparse matrix--vector execution subroutines

    interface armpl_spmv_exec

        subroutine armpl_spmv_exec_s(trans, alpha, A, x, beta, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            real(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            real(kind=armpl_r32), intent(in) :: x(*)
            real(kind=armpl_r32), intent(in) :: beta
            real(kind=armpl_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spmv_exec_s

        subroutine armpl_spmv_exec_d(trans, alpha, A, x, beta, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            real(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            real(kind=armpl_r64), intent(in) :: x(*)
            real(kind=armpl_r64), intent(in) :: beta
            real(kind=armpl_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spmv_exec_d

        subroutine armpl_spmv_exec_c(trans, alpha, A, x, beta, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            complex(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            complex(kind=armpl_r32), intent(in) :: x(*)
            complex(kind=armpl_r32), intent(in) :: beta
            complex(kind=armpl_r32), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spmv_exec_c

        subroutine armpl_spmv_exec_z(trans, alpha, A, x, beta, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            complex(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            complex(kind=armpl_r64), intent(in) :: x(*)
            complex(kind=armpl_r64), intent(in) :: beta
            complex(kind=armpl_r64), intent(inout) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spmv_exec_z

    end interface armpl_spmv_exec

!   Sparse matrix--matrix execution subroutines

    interface armpl_spmm_exec

        subroutine armpl_spmm_exec_s(transA, transB, alpha, A, B, beta, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            real(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i8), intent(in) :: B
            real(kind=armpl_r32), intent(in) :: beta
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spmm_exec_s

        subroutine armpl_spmm_exec_d(transA, transB, alpha, A, B, beta, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            real(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i8), intent(in) :: B
            real(kind=armpl_r64), intent(in) :: beta
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spmm_exec_d

        subroutine armpl_spmm_exec_c(transA, transB, alpha, A, B, beta, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            complex(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i8), intent(in) :: B
            complex(kind=armpl_r32), intent(in) :: beta
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spmm_exec_c

        subroutine armpl_spmm_exec_z(transA, transB, alpha, A, B, beta, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            complex(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i8), intent(in) :: B
            complex(kind=armpl_r64), intent(in) :: beta
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spmm_exec_z

    end interface armpl_spmm_exec

!   Sparse matrix triangular solve execution subroutines

    interface armpl_spsv_exec

        subroutine armpl_spsv_exec_s(trans, A, x, alpha, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            integer(kind=armpl_i8), intent(in) :: A
            real(kind=armpl_r32), intent(out) :: x(*)
            real(kind=armpl_r32), intent(in) :: alpha
            real(kind=armpl_r32), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spsv_exec_s

        subroutine armpl_spsv_exec_d(trans, A, x, alpha, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            integer(kind=armpl_i8), intent(in) :: A
            real(kind=armpl_r64), intent(out) :: x(*)
            real(kind=armpl_r64), intent(in) :: alpha
            real(kind=armpl_r64), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spsv_exec_d

        subroutine armpl_spsv_exec_c(trans, A, x, alpha, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            integer(kind=armpl_i8), intent(in) :: A
            complex(kind=armpl_r32), intent(out) :: x(*)
            complex(kind=armpl_r32), intent(in) :: alpha
            complex(kind=armpl_r32), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spsv_exec_c

        subroutine armpl_spsv_exec_z(trans, A, x, alpha, y, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: trans
            integer(kind=armpl_i8), intent(in) :: A
            complex(kind=armpl_r64), intent(out) :: x(*)
            complex(kind=armpl_r64), intent(in) :: alpha
            complex(kind=armpl_r64), intent(in) :: y(*)
            integer, intent(out) :: info
        end subroutine armpl_spsv_exec_z

    end interface armpl_spsv_exec

    interface armpl_spadd_exec

        subroutine armpl_spadd_exec_s(transA, transB, alpha, A, beta, B, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            real(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            real(kind=armpl_r32), intent(in) :: beta
            integer(kind=armpl_i8), intent(in) :: B
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spadd_exec_s

        subroutine armpl_spadd_exec_d(transA, transB, alpha, A, beta, B, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            real(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            real(kind=armpl_r64), intent(in) :: beta
            integer(kind=armpl_i8), intent(in) :: B
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spadd_exec_d

        subroutine armpl_spadd_exec_c(transA, transB, alpha, A, beta, B, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            complex(kind=armpl_r32), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            complex(kind=armpl_r32), intent(in) :: beta
            integer(kind=armpl_i8), intent(in) :: B
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spadd_exec_c

        subroutine armpl_spadd_exec_z(transA, transB, alpha, A, beta, B, C, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i4), intent(in) :: transA
            integer(kind=armpl_i4), intent(in) :: transB
            complex(kind=armpl_r64), intent(in) :: alpha
            integer(kind=armpl_i8), intent(in) :: A
            complex(kind=armpl_r64), intent(in) :: beta
            integer(kind=armpl_i8), intent(in) :: B
            integer(kind=armpl_i8), intent(out) :: C
            integer, intent(out) :: info
        end subroutine armpl_spadd_exec_z

    end interface armpl_spadd_exec

!   Sparse matrix export subroutines

    interface armpl_spmat_export_csr

        subroutine armpl_spmat_export_csr_s(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csr_s

        subroutine armpl_spmat_export_csr_d(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csr_d

        subroutine armpl_spmat_export_csr_c(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csr_c

        subroutine armpl_spmat_export_csr_z(A, index_base, m, n, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csr_z

    end interface armpl_spmat_export_csr

    interface armpl_spmat_export_csc

        subroutine armpl_spmat_export_csc_s(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            real(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csc_s

        subroutine armpl_spmat_export_csc_d(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            real(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csc_d

        subroutine armpl_spmat_export_csc_c(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            complex(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csc_c

        subroutine armpl_spmat_export_csc_z(A, index_base, m, n, row_indx, col_ptr, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_ptr(:)
            complex(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_csc_z

    end interface armpl_spmat_export_csc

    interface armpl_spmat_export_coo

        subroutine armpl_spmat_export_coo_s(A, m, n, nnz, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_coo_s

        subroutine armpl_spmat_export_coo_d(A, m, n, nnz, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_coo_d

        subroutine armpl_spmat_export_coo_c(A, m, n, nnz, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_coo_c

        subroutine armpl_spmat_export_coo_z(A, m, n, nnz, row_indx, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: nnz
            integer, allocatable, intent(out) :: row_indx(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_coo_z

    end interface armpl_spmat_export_coo

    interface armpl_spmat_export_dense

        subroutine armpl_spmat_export_dense_s(A, layout, m, n, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            real(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_dense_s

        subroutine armpl_spmat_export_dense_d(A, layout, m, n, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            real(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_dense_d

        subroutine armpl_spmat_export_dense_c(A, layout, m, n, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            complex(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_dense_c

        subroutine armpl_spmat_export_dense_z(A, layout, m, n, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: layout
            integer, intent(out) :: m
            integer, intent(out) :: n
            complex(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_dense_z

    end interface armpl_spmat_export_dense

    interface armpl_spmat_export_bsr

        subroutine armpl_spmat_export_bsr_s(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_bsr_s

        subroutine armpl_spmat_export_bsr_d(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            real(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_bsr_d

        subroutine armpl_spmat_export_bsr_c(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=armpl_r32), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_bsr_c

        subroutine armpl_spmat_export_bsr_z(A, block_layout, index_base, m, n, block_size, row_ptr, col_indx, vals, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
            integer(kind=armpl_i4), intent(in) :: block_layout
            integer, intent(in) :: index_base
            integer, intent(out) :: m
            integer, intent(out) :: n
            integer, intent(out) :: block_size
            integer, allocatable, intent(out) :: row_ptr(:)
            integer, allocatable, intent(out) :: col_indx(:)
            complex(kind=armpl_r64), allocatable, intent(out) :: vals(:)
            integer, intent(out) :: info
        end subroutine armpl_spmat_export_bsr_z

    end interface armpl_spmat_export_bsr

    interface armpl_sprot_exec

        subroutine armpl_sprot_exec_s(x, y, c, s, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            real(kind=armpl_r32), intent(inout) :: y(*)
            real(kind=armpl_r32), intent(in) :: c
            real(kind=armpl_r32), intent(in) :: s
            integer, intent(out) :: info
        end subroutine armpl_sprot_exec_s

        subroutine armpl_sprot_exec_d(x, y, c, s, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            real(kind=armpl_r64), intent(inout) :: y(*)
            real(kind=armpl_r64), intent(in) :: c
            real(kind=armpl_r64), intent(in) :: s
            integer, intent(out) :: info
        end subroutine armpl_sprot_exec_d

        subroutine armpl_sprot_exec_c(x, y, c, s, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            complex(kind=armpl_r32), intent(inout) :: y(*)
            real(kind=armpl_r32), intent(in) :: c
            complex(kind=armpl_r32), intent(in) :: s
            integer, intent(out) :: info
        end subroutine armpl_sprot_exec_c

        subroutine armpl_sprot_exec_z(x, y, c, s, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            complex(kind=armpl_r64), intent(inout) :: y(*)
            real(kind=armpl_r64), intent(in) :: c
            complex(kind=armpl_r64), intent(in) :: s
            integer, intent(out) :: info
        end subroutine armpl_sprot_exec_z

        subroutine armpl_sprot_exec_cs(x, y, c, s, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            complex(kind=armpl_r32), intent(inout) :: y(*)
            real(kind=armpl_r32), intent(in) :: c
            real(kind=armpl_r32), intent(in) :: s
            integer, intent(out) :: info
        end subroutine armpl_sprot_exec_cs

        subroutine armpl_sprot_exec_zd(x, y, c, s, info)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(inout) :: x
            complex(kind=armpl_r64), intent(inout) :: y(*)
            real(kind=armpl_r64), intent(in) :: c
            real(kind=armpl_r64), intent(in) :: s
            integer, intent(out) :: info
        end subroutine armpl_sprot_exec_zd

    end interface armpl_sprot_exec

    interface
        subroutine armpl_spmat_print_err(A)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: A
        end subroutine armpl_spmat_print_err

        subroutine armpl_spvec_print_err(x)
            use armpl_kinds
            implicit none
            integer(kind=armpl_i8), intent(in) :: x
        end subroutine armpl_spvec_print_err
    end interface

end module armpl_sparse
