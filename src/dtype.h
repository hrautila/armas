
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef __DTYPE_H
#define __DTYPE_H 1

#include <math.h>
#include <complex.h>

#ifdef COMPLEX128
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
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
#if 0
#define __MSCALE       __Z_MSCALE
#define __MPRINT       __Z_MPRINT
#define __VPRINT       __Z_VPRINT
#define __tile_print   __z_tile_print
#define __SCALE        __Z_SCALE
#endif

#define __blk_add      __z_blk_add
#define __blk_scale    __z_blk_scale
#define __blk_print    __z_blk_print
#define __vec_print    __z_vec_print

#define __kernel_colblk_inner    __z_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __z_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __z_kernel_colwise_inner_scale_c
#define __kernel_inner           __z_kernel_inner
#define __kernel_params          __z_kernel_params

// public matrix functions
#define __armas_dense_t     armas_z_dense_t
#define __armas_size        armas_z_size
#define __armas_init        armas_z_init
#define __armas_release     armas_z_release
#define __armas_alloc       armas_z_alloc
#define __armas_free        armas_z_free
#define __armas_make        armas_z_make
#define __armas_column      armas_z_column
#define __armas_row         armas_z_row
#define __armas_submatrix   armas_z_submatrix
#define __armas_diag        armas_z_diag
#define __armas_get         armas_z_get
#define __armas_set         armas_z_set
#define __armas_set_at      armas_z_set_at
#define __armas_get_at      armas_z_get_at
#define __armas_index       armas_z_index
#define __armas_data        armas_z_data
#define __armas_printf      armas_z_printf
#define __armas_print       armas_z_print
#define __armas_set_values  armas_z_set_values
#define __armas_mcopy       armas_z_mcopy
#define __armas_mscale      armas_z_mscale
#define __armas_madd        armas_z_madd
#define __armas_mnorm       armas_z_mnorm
#define __armas_transpose   armas_z_transpose
#define __armas_allclose    armas_z_allclose
#define __armas_intolerance armas_z_intolerance
#define __armas_mk_trm      armas_z_mk_trm
#define __armas_scale_plus  armas_z_scale_plus

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
#define __ZERO    (float)0.0
#define __ONE     (float)1.0

// functions from math.h
#define __ABS  fabsf
#define __SQRT sqrtf

#define __SCALE  __S_SCALE
#define __MSCALE __S_MSCALE
#define __kernel_colblk_inner  __s_kernel_colblk_inner
#define __kernel_colwise_inner_no_scale  __s_kernel_colwise_inner_no_scale
#define __kernel_colwise_inner_scale_c   __s_kernel_colwise_inner_scale_c
#define __kernel_inner  __s_kernel_inner
#define __kernel_params __s_kernel_params

#define __armas_dense_t  armas_s_dense_t
#define __armas_mult     armas_s_mult

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

#define __DATA_FORMAT "%8.1e"
#define __PRINTABLE(a) (a)

// functions from math.h
#define __ABS  fabs
#define __SQRT sqrt

// internally available functions
#if 0
#define __MSCALE       __D_MSCALE
#define __MPRINT       __D_MPRINT
#define __VPRINT       __D_VPRINT
#define __SCALE        __D_SCALE
#define __tile_print   __d_tile_print
#endif

#define __blk_add      __d_blk_add
#define __blk_scale    __d_blk_scale
#define __blk_print    __d_blk_print
#define __vec_print    __d_vec_print

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
#define __update_ger_unb         __d_update_ger_unb
#define __update_ger_recursive   __d_update_ger_recursive

// public matrix functions
#define __armas_dense_t     armas_d_dense_t
#define __armas_size        armas_d_size
#define __armas_init        armas_d_init
#define __armas_release     armas_d_release
#define __armas_alloc       armas_d_alloc
#define __armas_free        armas_d_free
#define __armas_make        armas_d_make
#define __armas_column      armas_d_column
#define __armas_row         armas_d_row
#define __armas_submatrix   armas_d_submatrix
#define __armas_diag        armas_d_diag
#define __armas_get         armas_d_get
#define __armas_set         armas_d_set
#define __armas_set_at      armas_d_set_at
#define __armas_get_at      armas_d_get_at
#define __armas_index       armas_d_index
#define __armas_data        armas_d_data
#define __armas_printf      armas_d_printf
#define __armas_print       armas_d_print
#define __armas_set_values  armas_d_set_values
#define __armas_mcopy       armas_d_mcopy
#define __armas_mscale      armas_d_mscale
#define __armas_madd        armas_d_madd
#define __armas_mnorm       armas_d_mnorm
#define __armas_transpose   armas_d_transpose
#define __armas_allclose    armas_d_allclose
#define __armas_intolerance armas_d_intolerance
#define __armas_mk_trm      armas_d_mk_trm
#define __armas_scale_plus  armas_d_scale_plus

#define __armas_index_valid  __armas_d_index_valid
#define __armas_real_index   __armas_d_real_index

// public functions: blas level 3, matrix-matrix
#define __armas_mult        armas_d_mult
#define __armas_mult_sym    armas_d_mult_sym
#define __armas_mult_trm    armas_d_mult_trm
#define __armas_solve_trm   armas_d_solve_trm
#define __armas_update_sym  armas_d_update_sym
#define __armas_2update_sym armas_d_2update_sym
#define __armas_update_trm  armas_d_update_trm

// public functions: blas level 2, matrix-vector
#define __armas_mvmult        armas_d_mvmult
#define __armas_mvmult_sym    armas_d_mvmult_sym
#define __armas_mvmult_trm    armas_d_mvmult_trm
#define __armas_mvsolve_trm   armas_d_mvsolve_trm
#define __armas_mvupdate      armas_d_mvupdate
#define __armas_mvupdate_trm  armas_d_mvupdate_trm
#define __armas_mvupdate_sym  armas_d_mvupdate_sym
#define __armas_mv2update_sym armas_d_mv2update_sym

#define __armas_mvmult_diag   armas_d_mvmult_diag
#define __armas_mvsolve_diag  armas_d_mvsolve_diag

// public functions: blas level 1, vector-vector
#define __armas_nrm2    armas_d_nrm2
#define __armas_asum    armas_d_asum
#define __armas_scale   armas_d_scale
#define __armas_iamax   armas_d_iamax
#define __armas_dot     armas_d_dot
#define __armas_axpy    armas_d_axpy
#define __armas_swap    armas_d_swap
#define __armas_copy    armas_d_copy
// 
#define __armas_invscale armas_d_invscale
#define __armas_sum      armas_d_sum
#define __armas_amax     armas_d_amax
#define __armas_add      armas_d_add

#endif

#endif
