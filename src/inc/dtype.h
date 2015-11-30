
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DTYPE_H
#define __DTYPE_H 1

#if HAVE_CONFIG_H
  #include <config.h>
#endif

#include <math.h>
#include <complex.h>

#ifdef COMPLEX128
/* ---------------------------------------------------------------------------
 * Definitions for double precision complex numbers.
 */
typedef double complex DTYPE;
typedef double ABSTYPE;

#define __NEED_CONJUGATE 1

#define __ABSZERO 0.0
#define __ABSONE  1.0
#define __ZERO    0.0 + 0.0i
#define __ONE     1.0 + 0.0i

#define __DATA_FORMAT "(%8.1e,%8.1ei)"
#define __PRINTABLE(a) creal(a), cimag(a)

// functions from math.h
#define __ABS  cabs
#define __SQRT csqrt

// internally available functions
#define __blk_add      __z_blk_add
#define __blk_scale    __z_blk_scale
#define __blk_print    __z_blk_print
#define __vec_print    __z_vec_print

#define __kernel_colblk_inner    __z_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __z_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __z_kernel_colwise_inner_scale_c
#define __kernel_inner           __z_kernel_inner
#define __kernel_params          __z_kernel_params

// internal matrix block routines
#define __partition_2x1           __z_partition_2x1
#define __repartition_2x1to3x1    __z_repartition_2x1to3x1
#define __continue_3x1to2x1       __z_continue_3x1to2x1
#define __partition_1x2           __z_partition_1x2
#define __repartition_1x2to1x3    __z_repartition_1x2to1x3
#define __continue_1x3to1x2       __z_continue_1x3to1x2
#define __partition_2x2           __z_partition_2x2
#define __repartition_2x2to2x3    __z_repartition_2x2to2x3
#define __continue_3x3to2x2       __z_continue_2x3to2x2

// public matrix functions
#define __armas_dense_t       armas_z_dense_t
#define __armas_size          armas_z_size
#define __armas_init          armas_z_init
#define __armas_release       armas_z_release
#define __armas_alloc         armas_z_alloc
#define __armas_free          armas_z_free
#define __armas_make          armas_z_make
#define __armas_column        armas_z_column
#define __armas_row           armas_z_row
#define __armas_submatrix     armas_z_submatrix
#define __armas_submatrix_ext armas_z_submatrix_ext
#define __armas_diag          armas_z_diag
#define __armas_get           armas_z_get
#define __armas_set           armas_z_set
#define __armas_set_at        armas_z_set_at
#define __armas_get_at        armas_z_get_at
#define __armas_index         armas_z_index
#define __armas_data          armas_z_data
#define __armas_printf        armas_z_printf
#define __armas_print         armas_z_print
#define __armas_set_values    armas_z_set_values
#define __armas_mcopy         armas_z_mcopy
#define __armas_mscale        armas_z_mscale
#define __armas_madd          armas_z_madd
#define __armas_mnorm         armas_z_mnorm
#define __armas_transpose     armas_z_transpose
#define __armas_allclose      armas_z_allclose
#define __armas_intolerance   armas_z_intolerance
#define __armas_make_trm      armas_z_make_trm
#define __armas_scale_plus    armas_z_scale_plus
#define __armas_isvector      armas_z_isvector

#define __armas_submatrix_unsafe  armas_z_submatrix_unsafe
#define __armas_subvector_unsafe  armas_z_subvector_unsafe
#define __armas_diag_unsafe       armas_z_diag_unsafe
#define __armas_get_unsafe        armas_z_get_unsafe
#define __armas_set_unsafe        armas_z_set_unsafe
#define __armas_get_at_unsafe     armas_z_get_at_unsafe
#define __armas_set_at_unsafe     armas_z_set_at_unsafe

#define __armas_index_valid  __armas_z_index_valid
#define __armas_real_index   __armas_z_real_index


// public functions
#define __armas_mult       armas_z_mult
#define __armas_mult_sym   armas_z_mult_sym

// public functions: blas level 1, vector-vector
#define __armas_nrm2     armas_z_nrm2
#define __armas_asum     armas_z_asum
#define __armas_scale    armas_z_scale
#define __armas_iamax    armas_z_iamax
#define __armas_dot      armas_z_dot
#define __armas_axpy     armas_z_axpy
#define __armas_axpby    armas_z_axpby
#define __armas_swap     armas_z_swap
#define __armas_copy     armas_z_copy
// 
#define __armas_invscale armas_z_invscale
#define __armas_sum      armas_z_sum
#define __armas_amax     armas_z_amax
#define __armas_add      armas_z_add


#elif COMPLEX64
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */
typedef  float _Complex DTYPE;
typedef float ABSTYPE;

#define __ABSZERO 0.0
#define __ZERO    0.0 + 0.0i
#define __ONE     1.0 + 0.0i;

// functions from math.h
#define __ABS  cabsf

#define __SCALE  __C_SCALE
#define __MSCALE __C_MSCALE
#define __kernel_colblk_inner  __c_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __c_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __c_kernel_colwise_inner_scale_c
#define __kernel_inner  __c_kernel_inner
#define __kernel_params __c_kernel_params

#define __armas_dense_t  armas_c_dense_t
#define __armas_mult     armas_c_mult

#elif FLOAT32
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */
typedef float DTYPE;
typedef float ABSTYPE;

#define __ABSZERO (float)0.0
#define __ABSONE  (float)1.0
#define __ZERO    (float)0.0
#define __ONE     (float)1.0
#define __TWO     (float)2.0
#define __HALF    (float)0.5

#define __DATA_FORMAT "%8.1e"
#define __PRINTABLE(a) (a)

// functions from math.h
#define __ABS   fabsf
#define __SQRT  sqrtf
#define __HYPOT hypotf
#define __SIGN  signbit
#define __COPYSIGN  copysignf
#define __POW   powf
#define __EXP   expf

// internally available functions
#define __blk_add      __s_blk_add
#define __blk_scale    __s_blk_scale
#define __blk_print    __s_blk_print
#define __vec_print    __s_vec_print

// internal type dependent BLAS functions
#define __kernel_colblk_inner  __s_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __s_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __s_kernel_colwise_inner_scale_c
#define __kernel_inner           __s_kernel_inner
#define __kernel_params          __s_kernel_params
#define __gemv_recursive         __s_gemv_recursive
#define __rank_diag              __s_rank_diag
#define __rank2_blk              __s_rank2_blk
#define __trmm_unb               __s_trmm_unb
#define __trmm_blk               __s_trmm_blk
#define __trmm_recursive         __s_trmm_recursive
#define __trmm_blk_recursive     __s_trmm_blk_recursive
#define __trmv_recursive         __s_trmv_recursive
#define __trsv_recursive         __s_trsv_recursive
#define __solve_left_unb         __s_solve_left_unb
#define __solve_right_unb        __s_solve_right_unb
#define __solve_blocked          __s_solve_blocked
#define __solve_recursive        __s_solve_recursive
#define __solve_blk_recursive    __s_solve_blk_recursive
#define __update_trmv_unb        __s_update_trmv_unb
#define __update_trmv_recursive  __s_update_trmv_recursive
#define __update_trm_blk         __s_update_trm_blk
#define __update_trm_blocked     __s_update_trm_blocked
#define __update_ger_unb         __s_update_ger_unb
#define __update_ger_recursive   __s_update_ger_recursive

// extended precision functions
#define __vec_axpy_ext           __s_vec_axpy_ext
#define __vec_axpby_ext          __s_vec_axpby_ext
#define __vec_dot_ext            __s_vec_dot_ext
#define __vec_sum_ext            __s_vec_sum_ext
#define __vec_asum_ext           __s_vec_asum_ext
#define __gemv_ext_unb           __s_gemv_ext_unb
#define __gemv_update_ext_unb    __s_gemv_update_ext_unb
#define __symv_ext_unb           __s_symv_ext_unb
#define __trsv_ext_unb           __s_trsv_ext_unb
#define __trmv_ext_unb           __s_trmv_ext_unb
#define __update_ger_ext_unb     __s_update_ger_ext_unb
#define __update_trmv_ext_unb    __s_update_trmv_ext_unb
#define __update2_symv_ext_unb   __s_update2_symv_ext_unb
#define __kernel_ext_colblk_inner    __s_kernel_ext_colblk_inner
#define __kernel_ext_panel_inner     __s_kernel_ext_panel_inner
#define __kernel_ext_panel_inner_dA  __s_kernel_ext_panel_inner_dA
#define __kernel_ext_panel_inner_dB  __s_kernel_ext_panel_inner_dB
#define __kernel_ext_colwise_inner_no_scale  __s_kernel_ext_colwise_inner_no_scale
#define __kernel_ext_colwise_inner_scale_c   __s_kernel_ext_colwise_inner_scale_c
#define __kernel_ext_symm         __s_kernel_ext_symm
#define __kernel_inner_ext        __s_kernel_inner_ext
#define __trmm_ext_blk            __s_trmm_ext_blk
#define __solve_ext_blk           __s_solve_ext_blk
#define __solve_ext_unb           __s_solve_ext_unb
#define __solve_ext               __s_solve_ext
#define __update_ext_trm_blocked  __s_update_ext_trm_blocked
#define __update_ext_trm_naive    __s_update_ext_trm_naive
#define __blk_scale_ext           __s_blk_scale_ext
#define __rank_ext_diag           __s_rank_ext_diag

// internal matrix block routines
#define __partition_2x1           __s_partition_2x1
#define __repartition_2x1to3x1    __s_repartition_2x1to3x1
#define __continue_3x1to2x1       __s_continue_3x1to2x1
#define __partition_1x2           __s_partition_1x2
#define __repartition_1x2to1x3    __s_repartition_1x2to1x3
#define __continue_1x3to1x2       __s_continue_1x3to1x2
#define __partition_2x2           __s_partition_2x2
#define __repartition_2x2to3x3    __s_repartition_2x2to3x3
#define __continue_3x3to2x2       __s_continue_2x3to2x2
#define __merge2x1                __s_merge2x1
#define __merge1x2                __s_merge1x2


// ------------------------------------------------------------------------------
// public matrix type
#define __armas_dense_t  armas_s_dense_t
// nil matrix
#define __nil                 ((__armas_dense_t *)0)

// ------------------------------------------------------------------------------
// public matrix functions
#define __armas_size          armas_s_size
#define __armas_init          armas_s_init
#define __armas_release       armas_s_release
#define __armas_alloc         armas_s_alloc
#define __armas_free          armas_s_free
#define __armas_make          armas_s_make
#define __armas_column        armas_s_column
#define __armas_row           armas_s_row
#define __armas_submatrix     armas_s_submatrix
#define __armas_subvector     armas_s_subvector
#define __armas_submatrix_ext armas_s_submatrix_ext
#define __armas_diag          armas_s_diag
#define __armas_get           armas_s_get
#define __armas_set           armas_s_set
#define __armas_set_at        armas_s_set_at
#define __armas_get_at        armas_s_get_at
#define __armas_index         armas_s_index
#define __armas_data          armas_s_data
#define __armas_col_as_row    armas_s_col_as_row
#define __armas_printf        armas_s_printf
#define __armas_print         armas_s_print
#define __armas_set_values    armas_s_set_values
#define __armas_newcopy       armas_s_newcopy
#define __armas_mcopy         armas_s_mcopy
#define __armas_mscale        armas_s_mscale
#define __armas_madd          armas_s_madd
#define __armas_mnorm         armas_s_mnorm
#define __armas_norm          armas_s_norm
#define __armas_transpose     armas_s_transpose
#define __armas_allclose      armas_s_allclose
#define __armas_intolerance   armas_s_intolerance
#define __armas_make_trm      armas_s_make_trm
#define __armas_scale_plus    armas_s_scale_plus
#define __armas_isvector      armas_s_isvector
// element-wise functions
#define __armas_add_elem      armas_s_add_elem
#define __armas_sub_elem      armas_s_sub_elem
#define __armas_mul_elem      armas_s_mul_elem
#define __armas_div_elem      armas_s_div_elem
#define __armas_apply         armas_s_apply
#define __armas_apply2        armas_s_apply2


#define __armas_submatrix_unsafe  armas_s_submatrix_unsafe
#define __armas_subvector_unsafe  armas_s_subvector_unsafe
#define __armas_diag_unsafe       armas_s_diag_unsafe
#define __armas_get_unsafe        armas_s_get_unsafe
#define __armas_set_unsafe        armas_s_set_unsafe
#define __armas_get_at_unsafe     armas_s_get_at_unsafe
#define __armas_set_at_unsafe     armas_s_set_at_unsafe

#define __armas_index_valid  __armas_s_index_valid
#define __armas_real_index   __armas_s_real_index


#define __armas_mult        armas_s_mult
#define __armas_mult_sym    armas_s_mult_sym
#define __armas_mult_trm    armas_s_mult_trm
#define __armas_solve_trm   armas_s_solve_trm
#define __armas_update_sym  armas_s_update_sym
#define __armas_update2_sym armas_s_update2_sym
#define __armas_update_trm  armas_s_update_trm
// marker for blas level 3
#define __armas_blas3   1

// public functions: blas level 2, matrix-vector
#define __armas_mvmult        armas_s_mvmult
#define __armas_mvmult_sym    armas_s_mvmult_sym
#define __armas_mvmult_trm    armas_s_mvmult_trm
#define __armas_mvsolve_trm   armas_s_mvsolve_trm
#define __armas_mvupdate      armas_s_mvupdate
#define __armas_mvupdate_trm  armas_s_mvupdate_trm
#define __armas_mvupdate_sym  armas_s_mvupdate_sym
#define __armas_mvupdate2_sym armas_s_mvupdate2_sym
// marker for blas level 2
#define __armas_blas2   1

#define __armas_mvmult_diag   armas_s_mvmult_diag
#define __armas_mvsolve_diag  armas_s_mvsolve_diag

// public functions: blas level 1, vector-vector
#define __armas_nrm2    armas_s_nrm2
#define __armas_asum    armas_s_asum
#define __armas_scale   armas_s_scale
#define __armas_iamax   armas_s_iamax
#define __armas_dot     armas_s_dot
#define __armas_axpy    armas_s_axpy
#define __armas_axpby   armas_s_axpby
#define __armas_swap    armas_s_swap
#define __armas_copy    armas_s_copy
// marker for blas level 1
#define __armas_blas1   1
// 
#define __armas_invscale armas_s_invscale
#define __armas_sum      armas_s_sum
#define __armas_amax     armas_s_amax
#define __armas_add      armas_s_add
#define __armas_iamin    armas_s_iamin

#else  // default is double precision float (FLOAT64)
/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
typedef double DTYPE;
typedef double ABSTYPE;

#define __ABSZERO (double)0.0
#define __ABSONE  (double)1.0
#define __ZERO    (double)0.0
#define __ONE     (double)1.0
#define __TWO     (double)2.0
#define __HALF    0.5L

#define __DATA_FORMAT "%8.1e"
#define __PRINTABLE(a) (a)

// functions from math.h
#define __ABS  fabs
#define __SQRT sqrt
#define __HYPOT hypot
#define __SIGN  signbit
#define __COPYSIGN copysign
#define __POW   pow
#define __EXP   exp

// internally available functions
#define __blk_add      __d_blk_add
#define __blk_scale    __d_blk_scale
#define __blk_print    __d_blk_print
#define __vec_print    __d_vec_print

// internal type dependent BLAS functions
#define __kernel_colblk_inner    __d_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __d_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __d_kernel_colwise_inner_scale_c
#define __kernel_inner           __d_kernel_inner
#define __kernel_params          __d_kernel_params
#define __gemv_recursive         __d_gemv_recursive
#define __rank_diag              __d_rank_diag
#define __rank2_blk              __d_rank2_blk
#define __trmm_unb               __d_trmm_unb
#define __trmm_blk               __d_trmm_blk
#define __trmm_recursive         __d_trmm_recursive
#define __trmm_blk_recursive     __d_trmm_blk_recursive
#define __trmv_recursive         __d_trmv_recursive
#define __trsv_recursive         __d_trsv_recursive
#define __solve_left_unb         __d_solve_left_unb
#define __solve_right_unb        __d_solve_right_unb
#define __solve_blocked          __d_solve_blocked
#define __solve_recursive        __d_solve_recursive
#define __solve_blk_recursive    __d_solve_blk_recursive
#define __update_trmv_unb        __d_update_trmv_unb
#define __update_trmv_recursive  __d_update_trmv_recursive
#define __update_trm_blk         __d_update_trm_blk
#define __update_trm_blocked     __d_update_trm_blocked
#define __update_ger_unb         __d_update_ger_unb
#define __update_ger_recursive   __d_update_ger_recursive

// extended precision functions
#define __vec_axpy_ext           __d_vec_axpy_ext
#define __vec_axpby_ext          __d_vec_axpby_ext
#define __vec_dot_ext            __d_vec_dot_ext
#define __vec_sum_ext            __d_vec_sum_ext
#define __vec_asum_ext           __d_vec_asum_ext
#define __gemv_ext_unb           __d_gemv_ext_unb
#define __gemv_update_ext_unb    __d_gemv_update_ext_unb
#define __symv_ext_unb           __d_symv_ext_unb
#define __trsv_ext_unb           __d_trsv_ext_unb
#define __trmv_ext_unb           __d_trmv_ext_unb
#define __update_ger_ext_unb     __d_update_ger_ext_unb
#define __update_trmv_ext_unb    __d_update_trmv_ext_unb
#define __update2_symv_ext_unb   __d_update2_symv_ext_unb
#define __kernel_ext_colblk_inner    __d_kernel_ext_colblk_inner
#define __kernel_ext_panel_inner     __d_kernel_ext_panel_inner
#define __kernel_ext_panel_inner_dA  __d_kernel_ext_panel_inner_dA
#define __kernel_ext_panel_inner_dB  __d_kernel_ext_panel_inner_dB
#define __kernel_ext_colwise_inner_no_scale  __d_kernel_ext_colwise_inner_no_scale
#define __kernel_ext_colwise_inner_scale_c   __d_kernel_ext_colwise_inner_scale_c
#define __kernel_ext_symm         __d_kernel_ext_symm
#define __kernel_inner_ext        __d_kernel_inner_ext
#define __trmm_ext_blk            __d_trmm_ext_blk
#define __solve_ext_blk           __d_solve_ext_blk
#define __solve_ext_unb           __d_solve_ext_unb
#define __solve_ext               __d_solve_ext
#define __update_ext_trm_blocked  __d_update_ext_trm_blocked
#define __update_ext_trm_naive    __d_update_ext_trm_naive
#define __blk_scale_ext           __d_blk_scale_ext
#define __rank_ext_diag           __d_rank_ext_diag

// internal matrix block routines
#define __partition_2x1           __d_partition_2x1
#define __repartition_2x1to3x1    __d_repartition_2x1to3x1
#define __continue_3x1to2x1       __d_continue_3x1to2x1
#define __partition_1x2           __d_partition_1x2
#define __repartition_1x2to1x3    __d_repartition_1x2to1x3
#define __continue_1x3to1x2       __d_continue_1x3to1x2
#define __partition_2x2           __d_partition_2x2
#define __repartition_2x2to3x3    __d_repartition_2x2to3x3
#define __continue_3x3to2x2       __d_continue_2x3to2x2
#define __merge2x1                __d_merge2x1
#define __merge1x2                __d_merge1x2

// ------------------------------------------------------------------------------
// public matrix types
#define __armas_dense_t       armas_d_dense_t

// nil matrix
#define __nil                 ((__armas_dense_t *)0)

// ------------------------------------------------------------------------------
// public matrix functions
#define __armas_size          armas_d_size
#define __armas_init          armas_d_init
#define __armas_release       armas_d_release
#define __armas_alloc         armas_d_alloc
#define __armas_free          armas_d_free
#define __armas_make          armas_d_make
#define __armas_column        armas_d_column
#define __armas_row           armas_d_row
#define __armas_submatrix     armas_d_submatrix
#define __armas_subvector     armas_d_subvector
#define __armas_submatrix_ext armas_d_submatrix_ext
#define __armas_diag          armas_d_diag
#define __armas_get           armas_d_get
#define __armas_set           armas_d_set
#define __armas_set_at        armas_d_set_at
#define __armas_get_at        armas_d_get_at
#define __armas_index         armas_d_index
#define __armas_data          armas_d_data
#define __armas_col_as_row    armas_d_col_as_row
#define __armas_printf        armas_d_printf
#define __armas_print         armas_d_print
#define __armas_set_values    armas_d_set_values
#define __armas_newcopy       armas_d_newcopy
#define __armas_mcopy         armas_d_mcopy
#define __armas_mscale        armas_d_mscale
#define __armas_madd          armas_d_madd
#define __armas_mnorm         armas_d_mnorm
#define __armas_norm          armas_d_norm
#define __armas_transpose     armas_d_transpose
#define __armas_allclose      armas_d_allclose
#define __armas_intolerance   armas_d_intolerance
#define __armas_make_trm      armas_d_make_trm
#define __armas_scale_plus    armas_d_scale_plus
#define __armas_isvector      armas_d_isvector
// element-wise functions
#define __armas_add_elem      armas_d_add_elem
#define __armas_sub_elem      armas_d_sub_elem
#define __armas_mul_elem      armas_d_mul_elem
#define __armas_div_elem      armas_d_div_elem
#define __armas_apply         armas_d_apply
#define __armas_apply2        armas_d_apply2

#define __armas_submatrix_unsafe  armas_d_submatrix_unsafe
#define __armas_subvector_unsafe  armas_d_subvector_unsafe
#define __armas_diag_unsafe       armas_d_diag_unsafe
#define __armas_get_unsafe        armas_d_get_unsafe
#define __armas_set_unsafe        armas_d_set_unsafe
#define __armas_get_at_unsafe     armas_d_get_at_unsafe
#define __armas_set_at_unsafe     armas_d_set_at_unsafe

#define __armas_index_valid  __armas_d_index_valid
#define __armas_real_index   __armas_d_real_index

// public functions: blas level 3, matrix-matrix
#define __armas_mult        armas_d_mult
#define __armas_mult_sym    armas_d_mult_sym
#define __armas_mult_trm    armas_d_mult_trm
#define __armas_solve_trm   armas_d_solve_trm
#define __armas_update_sym  armas_d_update_sym
#define __armas_update2_sym armas_d_update2_sym
#define __armas_update_trm  armas_d_update_trm
// marker for blas level 3
#define __armas_blas3   1

// public functions: blas level 2, matrix-vector
#define __armas_mvmult        armas_d_mvmult
#define __armas_mvmult_sym    armas_d_mvmult_sym
#define __armas_mvmult_trm    armas_d_mvmult_trm
#define __armas_mvsolve_trm   armas_d_mvsolve_trm
#define __armas_mvupdate      armas_d_mvupdate
#define __armas_mvupdate_trm  armas_d_mvupdate_trm
#define __armas_mvupdate_sym  armas_d_mvupdate_sym
#define __armas_mvupdate2_sym armas_d_mvupdate2_sym
// marker for blas level 2
#define __armas_blas2   1

#define __armas_mvmult_diag   armas_d_mvmult_diag
#define __armas_mvsolve_diag  armas_d_mvsolve_diag

// public functions: blas level 1, vector-vector
#define __armas_nrm2    armas_d_nrm2
#define __armas_asum    armas_d_asum
#define __armas_scale   armas_d_scale
#define __armas_iamax   armas_d_iamax
#define __armas_dot     armas_d_dot
#define __armas_axpy    armas_d_axpy
#define __armas_axpby   armas_d_axpby
#define __armas_swap    armas_d_swap
#define __armas_copy    armas_d_copy
// marker for blas level 1
#define __armas_blas1   1
// 
#define __armas_invscale armas_d_invscale
#define __armas_sum      armas_d_sum
#define __armas_amax     armas_d_amax
#define __armas_add      armas_d_add
#define __armas_iamin    armas_d_iamin

#endif  /* FLOAT64 */

#if defined(__armas_blas1) && defined(__armas_blas2) && defined(__armas_blas3)
#define __armas_blas 1
#endif


#endif  /* DTYPE_H */
