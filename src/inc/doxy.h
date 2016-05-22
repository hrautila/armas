
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXY_H
#define __DOXY_H 1

/* ---------------------------------------------------------------------------
 * Definitions for Doxygen 
 */

// internally available functions
#define __blk_add      __x_blk_add
#define __blk_scale    __x_blk_scale
#define __blk_print    __x_blk_print
#define __vec_print    __x_vec_print

// internal type dependent BLAS functions
#define __kernel_colblk_inner    __x_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __x_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __x_kernel_colwise_inner_scale_c
#define __kernel_inner           __x_kernel_inner
#define __kernel_params          __x_kernel_params
#define __gemv_recursive         __x_gemv_recursive
#define __rank_diag              __x_rank_diag
#define __rank2_blk              __x_rank2_blk
#define __trmm_unb               __x_trmm_unb
#define __trmm_blk               __x_trmm_blk
#define __trmm_recursive         __x_trmm_recursive
#define __trmm_blk_recursive     __x_trmm_blk_recursive
#define __trmv_recursive         __x_trmv_recursive
#define __trsv_recursive         __x_trsv_recursive
#define __solve_left_unb         __x_solve_left_unb
#define __solve_right_unb        __x_solve_right_unb
#define __solve_blocked          __x_solve_blocked
#define __solve_recursive        __x_solve_recursive
#define __solve_blk_recursive    __x_solve_blk_recursive
#define __update_trmv_unb        __x_update_trmv_unb
#define __update_trmv_recursive  __x_update_trmv_recursive
#define __update_trm_blk         __x_update_trm_blk
#define __update_trm_blocked     __x_update_trm_blocked
#define __update_ger_unb         __x_update_ger_unb
#define __update_ger_recursive   __x_update_ger_recursive

// extended precision functions
#define __vec_axpy_ext           __x_vec_axpy_ext
#define __vec_axpby_ext          __x_vec_axpby_ext
#define __vec_dot_ext            __x_vec_dot_ext
#define __vec_sum_ext            __x_vec_sum_ext
#define __vec_asum_ext           __x_vec_asum_ext
#define __gemv_ext_unb           __x_gemv_ext_unb
#define __gemv_update_ext_unb    __x_gemv_update_ext_unb
#define __symv_ext_unb           __x_symv_ext_unb
#define __trsv_ext_unb           __x_trsv_ext_unb
#define __trmv_ext_unb           __x_trmv_ext_unb
#define __update_ger_ext_unb     __x_update_ger_ext_unb
#define __update_trmv_ext_unb    __x_update_trmv_ext_unb
#define __update2_symv_ext_unb   __x_update2_symv_ext_unb
#define __kernel_ext_colblk_inner    __x_kernel_ext_colblk_inner
#define __kernel_ext_panel_inner     __x_kernel_ext_panel_inner
#define __kernel_ext_panel_inner_dA  __x_kernel_ext_panel_inner_dA
#define __kernel_ext_panel_inner_dB  __x_kernel_ext_panel_inner_dB
#define __kernel_ext_colwise_inner_no_scale  __x_kernel_ext_colwise_inner_no_scale
#define __kernel_ext_colwise_inner_scale_c   __x_kernel_ext_colwise_inner_scale_c
#define __kernel_ext_symm         __x_kernel_ext_symm
#define __kernel_inner_ext        __x_kernel_inner_ext
#define __trmm_ext_blk            __x_trmm_ext_blk
#define __solve_ext_blk           __x_solve_ext_blk
#define __solve_ext_unb           __x_solve_ext_unb
#define __solve_ext               __x_solve_ext
#define __update_ext_trm_blocked  __x_update_ext_trm_blocked
#define __update_ext_trm_naive    __x_update_ext_trm_naive
#define __blk_scale_ext           __x_blk_scale_ext
#define __rank_ext_diag           __x_rank_ext_diag
#define __rank_ext_blk            __x_rank_ext_blk

// internal matrix block routines
#define __partition_2x1           __x_partition_2x1
#define __repartition_2x1to3x1    __x_repartition_2x1to3x1
#define __continue_3x1to2x1       __x_continue_3x1to2x1
#define __partition_1x2           __x_partition_1x2
#define __repartition_1x2to1x3    __x_repartition_1x2to1x3
#define __continue_1x3to1x2       __x_continue_1x3to1x2
#define __partition_2x2           __x_partition_2x2
#define __repartition_2x2to3x3    __x_repartition_2x2to3x3
#define __continue_3x3to2x2       __x_continue_2x3to2x2
#define __merge2x1                __x_merge2x1
#define __merge1x2                __x_merge1x2


// ------------------------------------------------------------------------------
// public matrix functions
#define __armas_size          armas_x_size
#define __armas_init          armas_x_init
#define __armas_release       armas_x_release
#define __armas_alloc         armas_x_alloc
#define __armas_free          armas_x_free
#define __armas_make          armas_x_make
#define __armas_column        armas_x_column
#define __armas_row           armas_x_row
#define __armas_submatrix     armas_x_submatrix
#define __armas_subvector     armas_x_subvector
#define __armas_submatrix_ext armas_x_submatrix_ext
#define __armas_diag          armas_x_diag
#define __armas_get           armas_x_get
#define __armas_set           armas_x_set
#define __armas_set_at        armas_x_set_at
#define __armas_get_at        armas_x_get_at
#define __armas_index         armas_x_index
#define __armas_data          armas_x_data
#define __armas_col_as_row    armas_x_col_as_row
#define __armas_printf        armas_x_printf
#define __armas_print         armas_x_print
#define __armas_set_values    armas_x_set_values
#define __armas_newcopy       armas_x_newcopy
#define __armas_mcopy         armas_x_mcopy
#define __armas_mscale        armas_x_mscale
#define __armas_madd          armas_x_madd
#define __armas_mnorm         armas_x_mnorm
#define __armas_norm          armas_x_norm
#define __armas_transpose     armas_x_transpose
#define __armas_allclose      armas_x_allclose
#define __armas_intolerance   armas_x_intolerance
#define __armas_make_trm      armas_x_make_trm
#define __armas_scale_plus    armas_x_scale_plus
#define __armas_isvector      armas_x_isvector
// element-wise functions
#define __armas_add_elems     armas_x_add_elems
#define __armas_sub_elems     armas_x_sub_elems
#define __armas_mul_elems     armas_x_mul_elems
#define __armas_div_elems     armas_x_div_elems
#define __armas_apply         armas_x_apply
#define __armas_apply2        armas_x_apply2

#define __armas_submatrix_unsafe  armas_x_submatrix_unsafe
#define __armas_subvector_unsafe  armas_x_subvector_unsafe
#define __armas_diag_unsafe       armas_x_diag_unsafe
#define __armas_get_unsafe        armas_x_get_unsafe
#define __armas_set_unsafe        armas_x_set_unsafe
#define __armas_get_at_unsafe     armas_x_get_at_unsafe
#define __armas_set_at_unsafe     armas_x_set_at_unsafe

#define __armas_index_valid  __armas_x_index_valid
#define __armas_real_index   __armas_x_real_index

// public functions: blas level 3, matrix-matrix
#define __armas_mult        armas_x_mult
#define __armas_mult_sym    armas_x_mult_sym
#define __armas_mult_trm    armas_x_mult_trm
#define __armas_solve_trm   armas_x_solve_trm
#define __armas_update_sym  armas_x_update_sym
#define __armas_update2_sym armas_x_update2_sym
#define __armas_update_trm  armas_x_update_trm
// marker for blas level 3
#define __armas_blas3   1

// public functions: blas level 2, matrix-vector
#define __armas_mvmult        armas_x_mvmult
#define __armas_mvmult_sym    armas_x_mvmult_sym
#define __armas_mvmult_trm    armas_x_mvmult_trm
#define __armas_mvsolve_trm   armas_x_mvsolve_trm
#define __armas_mvupdate      armas_x_mvupdate
#define __armas_mvupdate_trm  armas_x_mvupdate_trm
#define __armas_mvupdate_sym  armas_x_mvupdate_sym
#define __armas_mvupdate2_sym armas_x_mvupdate2_sym
// marker for blas level 2
#define __armas_blas2   1

#define __armas_mvmult_diag   armas_x_mvmult_diag
#define __armas_mvsolve_diag  armas_x_mvsolve_diag

// public functions: blas level 1, vector-vector
#define __armas_nrm2    armas_x_nrm2
#define __armas_asum    armas_x_asum
#define __armas_scale   armas_x_scale
#define __armas_iamax   armas_x_iamax
#define __armas_dot     armas_x_dot
#define __armas_axpy    armas_x_axpy
#define __armas_axpby   armas_x_axpby
#define __armas_swap    armas_x_swap
#define __armas_copy    armas_x_copy
// marker for blas level 1
#define __armas_blas1   1
// 
#define __armas_invscale armas_x_invscale
#define __armas_sum      armas_x_sum
#define __armas_amax     armas_x_amax
#define __armas_add      armas_x_add
#define __armas_iamin    armas_x_iamin

// internal helpers
#define __swap_rows               __x_swap_rows
#define __swap_cols               __x_swap_cols
#define __apply_pivots            __x_apply_pivots
#define __apply_row_pivots        __x_apply_row_pivots
#define __pivot_index             __x_pivot_index
// marker for above function
#define __lapack_pivots  1
// public pivot functions
#define __armas_pivot_rows      armas_x_pivot_rows


// Bidiagonal reduction
#define __armas_bdreduce         armas_x_bdreduce
#define __armas_bdreduce_work    armas_x_bdreduce_work
#define __armas_bdmult           armas_x_bdmult
#define __armas_bdmult_work      armas_x_bdmult_work
#define __armas_bdbuild          armas_x_bdbuild
#define __armas_bdbuild_work     armas_x_bdbuild_work

// Symmetric LDL functions
#define __armas_bkfactor        armas_x_bkfactor
#define __armas_bkfactor_work   armas_x_bkfactor_work
#define __armas_bksolve         armas_x_bksolve
#define __armas_bksolve_work    armas_x_bksolve_work
#define __armas_ldlfactor        armas_x_ldlfactor
#define __armas_ldlsolve         armas_x_ldlsolve
// visible internal functions
#define __unblk_bkfactor_lower __x_unblk_bkfactor_lower
#define __unblk_bkfactor_upper __x_unblk_bkfactor_upper
#define __unblk_bksolve_lower  __x_unblk_bksolve_lower
#define __unblk_bksolve_upper  __x_unblk_bksolve_upper
#define __blk_bkfactor_lower   __x_blk_bkfactor_lower
#define __blk_bkfactor_upper   __x_blk_bkfactor_upper
// marker
#define __ldlbk 1

// Cholesky 
#define __armas_cholfactor       armas_x_cholfactor
#define __armas_cholupdate       armas_x_cholupdate
#define __armas_cholsolve        armas_x_cholsolve

// Hessenberg functions
#define __armas_hessreduce       armas_x_hessreduce
#define __armas_hessreduce_work  armas_x_hessreduce_work
#define __armas_hessmult         armas_x_hessmult
#define __armas_hessmult_work    armas_x_hessmult_work

// householder functions
#define __compute_householder      __x_compute_householder
#define __compute_householder_vec  __x_compute_householder_vec
#define __compute_householder_rev  __x_compute_householder_rev
#define __apply_householder2x1     __x_apply_householder2x1
#define __apply_householder1x1     __x_apply_householder1x1
// marker for householder functions
#define __householder              1

// LQ functions
#define __armas_lqbuild         armas_x_lqbuild
#define __armas_lqbuild_work    armas_x_lqbuild_work
#define __armas_lqfactor        armas_x_lqfactor
#define __armas_lqfactor_work   armas_x_lqfactor_work
#define __armas_lqmult          armas_x_lqmult
#define __armas_lqmult_work     armas_x_lqmult_work
#define __armas_lqreflector     armas_x_lqreflector
#define __armas_lqsolve         armas_x_lqsolve
#define __armas_lqsolve_work    armas_x_lqsolve_work
// internal LQ related functions available for others
#define __update_lq_left        __x_update_lq_left
#define __update_lq_right       __x_update_lq_right
#define __unblk_lq_reflector    __x_unblk_lq_reflector

// LU functions
#define __armas_lufactor        armas_x_lufactor
#define __armas_lusolve         armas_x_lusolve

// QL functions
#define __armas_qlbuild         armas_x_qlbuild
#define __armas_qlbuild_work    armas_x_qlbuild_work
#define __armas_qlfactor        armas_x_qlfactor
#define __armas_qlfactor_work   armas_x_qlfactor_work
#define __armas_qlmult          armas_x_qlmult
#define __armas_qlmult_work     armas_x_qlmult_work
#define __armas_qlreflector     armas_x_qlreflector
// internal QL related function available for others
#define __update_ql_left        __x_update_ql_left
#define __update_ql_right       __x_update_ql_right
#define __unblk_ql_reflector    __x_unblk_ql_reflector

// QR functions
#define __armas_qrfactor        armas_x_qrfactor
#define __armas_qrfactor_work   armas_x_qrfactor_work
#define __armas_qrmult          armas_x_qrmult
#define __armas_qrmult_work     armas_x_qrmult_work
#define __armas_qrbuild         armas_x_qrbuild
#define __armas_qrbuild_work    armas_x_qrbuild_work
#define __armas_qrreflector     armas_x_qrreflector
#define __armas_qrsolve         armas_x_qrsolve
#define __armas_qrsolve_work    armas_x_qrsolve_work
// internal QR related function available for others
#define __update_qr_left        __x_update_qr_left
#define __update_qr_right       __x_update_qr_right
#define __unblk_qr_reflector    __x_unblk_qr_reflector
#define __unblk_qrfactor        __x_unblk_qrfactor

// QRT functions
#define __armas_qrtfactor       armas_x_qrtfactor
#define __armas_qrtfactor_work  armas_x_qrtfactor_work
#define __armas_qrtmult         armas_x_qrtmult
#define __armas_qrtmult_work    armas_x_qrtmult_work

// RQ functions
#define __armas_rqbuild         armas_x_rqbuild
#define __armas_rqbuild_work    armas_x_rqbuild_work
#define __armas_rqfactor        armas_x_rqfactor
#define __armas_rqfactor_work   armas_x_rqfactor_work
#define __armas_rqmult          armas_x_rqmult
#define __armas_rqmult_work     armas_x_rqmult_work
#define __armas_rqreflector     armas_x_rqreflector
// internal RQ related function available for others
#define __update_rq_left        __x_update_rq_left
#define __update_rq_right       __x_update_rq_right
#define __unblk_rq_reflector    __x_unblk_rq_reflector

// Tridiagonalization functions
#define __armas_trdreduce       armas_x_trdreduce
#define __armas_trdreduce_work  armas_x_trdreduce_work
#define __armas_trdbuild        armas_x_trdbuild
#define __armas_trdbuild_work   armas_x_trdbuild_work
#define __armas_trdmult         armas_x_trdmult
#define __armas_trdmult_work    armas_x_trdmult_work
// Tridiagonal EVD
#define __armas_trdeigen        armas_x_trdeigen
#define __armas_trdbisect       armas_x_trdbisect
#define __armas_trdsec_solve    armas_x_trdsec_solve
#define __armas_trdsec_eigen    armas_x_trdsec_eigen
#define __armas_trdsec_solve_vec armas_x_trdsec_solve_vec

// Eigenvalue
#define __armas_eigen_sym           armas_x_eigen_sym
#define __armas_eigen_sym_selected  armas_x_eigen_sym_selected

// Givens
#define __armas_gvcompute       armas_x_gvcompute
#define __armas_gvrotate        armas_x_gvrotate
#define __armas_gvrot_vec       armas_x_gvrot_vec
#define __armas_gvleft          armas_x_gvleft
#define __armas_gvright         armas_x_gvright
#define __armas_gvupdate        armas_x_gvupdate

// Bidiagonal SVD
#define __armas_bdsvd           armas_x_bdsvd
#define __armas_bdsvd_work      armas_x_bdsvd_work
#define __armas_dqds            armas_x_dqds

// internal 
#define __bdsvd2x2              __x_bdsvd2x2
#define __bdsvd2x2_vec          __x_bdsvd2x2_vec
#define __bdsvd_golub		__x_bdsvd_golub
#define __bdsvd_demmel		__x_bdsvd_demmel

// SVD
#define __armas_svd		armas_x_svd
#define __armas_svd_work	armas_x_svd_work

// Butterfly
#define __armas_gen_rbt         armas_x_gen_rbt
#define __armas_size_rbt        armas_x_size_rbt
#define __armas_mult_rbt        armas_x_mult_rbt
#define __armas_update2_rbt     armas_x_update2_rbt
#define __armas_update2_sym_rbt armas_x_update2_sym_rbt
#define __update2_rbt_descend   __x_update2_rbt_descend

// internal
#define __sym_eigen2x2          __x_sym_eigen2x2
#define __sym_eigen2x2vec       __x_sym_eigen2x2vec
#define __trdevd_qr             __x_trdevd_qr

// Sorting vectors (internal)
#define __pivot_sort              __x_pivot_sort
#define __eigen_sort              __x_eigen_sort
#define __abs_sort_vec            __x_abs_sort_vec 
#define __sort_vec                __x_sort_vec 
#define __sort_eigenvec           __x_sort_eigenvec

// Additional
#define __armas_qdroots         armas_x_qdroots
#define __armas_discriminant    armas_x_discriminant
#define __armas_mult_diag       armas_x_mult_diag
#define __armas_solve_diag      armas_x_solve_diag
#define __armas_scale_to        armas_x_scale_to

// common 
#if defined(__update_lq_left) && defined(__update_lq_right) && defined(__unblk_lq_reflector)
#define __update_lq 1
#endif

#if defined(__update_ql_left) && defined(__update_ql_right) && defined(__unblk_ql_reflector)
#define __update_ql 1
#endif

#if defined(__update_qr_left) && defined(__update_qr_right) && defined(__unblk_qr_reflector)
#define __update_qr 1
#endif

#if defined(__update_rq_left) && defined(__update_rq_right) && defined(__unblk_rq_reflector)
#define __update_rq 1
#endif


#if defined(__armas_blas1) && defined(__armas_blas2) && defined(__armas_blas3)
#define __armas_blas 1
#endif

#endif  /* __DOXY_H */


