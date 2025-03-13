!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!   SPDX-FileCopyrightText: Copyright 2015-2024 NAG
!
!   Parameters for Fortran sparse routines in ARMPL
!
module armpl_sparse_params
    use armpl_kinds
    implicit none

!   Fortran versions of C enums

    integer(kind=armpl_i4), parameter :: armpl_status_success = 0
    integer(kind=armpl_i4), parameter :: armpl_status_input_parameter_error = 1
    integer(kind=armpl_i4), parameter :: armpl_status_execution_failure = 2
    integer(kind=armpl_i4), parameter :: armpl_sparse_create_nocopy = 3

    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_structure = 50
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spmv_operation = 60
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spmm_operation = 61
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spadd_operation = 62
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spsv_operation = 63
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spmm_strategy = 64
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_memory = 70
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spmv_invocations = 80
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spmm_invocations = 81
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spadd_invocations = 82
    integer(kind=armpl_i4), parameter :: armpl_sparse_hint_spsv_invocations = 83

    integer(kind=armpl_i4), parameter :: armpl_col_major = 90
    integer(kind=armpl_i4), parameter :: armpl_row_major = 91

    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_dense = 100
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_unstructured = 101
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_symmetric = 110
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_diagonal = 120
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_blockdiagonal = 130
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_banded = 140
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_triangular = 150
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_blocktriangular = 160
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_hermitian = 170
    integer(kind=armpl_i4), parameter :: armpl_sparse_structure_hpcg = 180

    integer(kind=armpl_i4), parameter :: armpl_sparse_memory_noallocs = 200
    integer(kind=armpl_i4), parameter :: armpl_sparse_memory_allocs = 201

    integer(kind=armpl_i4), parameter :: armpl_sparse_operation_notrans = 300
    integer(kind=armpl_i4), parameter :: armpl_sparse_operation_trans = 310
    integer(kind=armpl_i4), parameter :: armpl_sparse_operation_conjtrans = 320

    integer(kind=armpl_i4), parameter :: armpl_sparse_invocations_single = 400
    integer(kind=armpl_i4), parameter :: armpl_sparse_invocations_few = 410
    integer(kind=armpl_i4), parameter :: armpl_sparse_invocations_many = 420

    integer(kind=armpl_i4), parameter :: armpl_sparse_scalar_one = 500
    integer(kind=armpl_i4), parameter :: armpl_sparse_scalar_zero = 501
    integer(kind=armpl_i4), parameter :: armpl_sparse_scalar_any = 502

    integer(kind=armpl_i4), parameter :: armpl_sparse_spmm_strat_unset = 600
    integer(kind=armpl_i4), parameter :: armpl_sparse_spmm_strat_opt_no_struct = 601
    integer(kind=armpl_i4), parameter :: armpl_sparse_spmm_strat_opt_part_struct = 602
    integer(kind=armpl_i4), parameter :: armpl_sparse_spmm_strat_opt_full_struct = 603

end module armpl_sparse_params
