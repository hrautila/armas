
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN
#ifndef __ARMAS_DLPACK_H
#define __ARMAS_DLPACK_H 1

#include <float.h>

#ifdef COMPLEX128
/* ---------------------------------------------------------------------------
 * Definitions for double precision complex numbers.
 */


#elif COMPLEX64
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */


#elif FLOAT32
/* ---------------------------------------------------------------------------
 * Definitions for single precision real numbers.
 */
#define __SAFEMIN     FLT_MIN
// machine accurancy as in LAPACK library 
#define __EPS         (FLT_EPSILON/2.0)


// internal helpers
#define __swap_rows               __s_swap_rows
#define __swap_cols               __s_swap_cols
#define __apply_pivots            __s_apply_pivots
#define __apply_row_pivots        __s_apply_row_pivots
#define __pivot_index             __s_pivot_index
// marker for above function
#define __lapack_pivots  1
// public pivot functions
#define __armas_pivot_rows      armas_s_pivot_rows

// Additional
#define __armas_qdroots         armas_s_qdroots
#define __armas_discriminant    armas_s_discriminant
#define __armas_mult_diag       armas_s_mult_diag
#define __armas_solve_diag      armas_s_solve_diag
#define __armas_scale_to        armas_s_scale_to

// Bidiagonal reduction
#define __armas_bdreduce         armas_s_bdreduce
#define __armas_bdreduce_work    armas_s_bdreduce_work
#define __armas_bdmult           armas_s_bdmult
#define __armas_bdmult_work      armas_s_bdmult_work
#define __armas_bdbuild          armas_s_bdbuild
#define __armas_bdbuild_work     armas_s_bdbuild_work

// Symmetric LDL functions
#define __armas_bkfactor        armas_s_bkfactor
#define __armas_bkfactor_work   armas_s_bkfactor_work
#define __armas_bksolve         armas_s_bksolve
#define __armas_bksolve_work    armas_s_bksolve_work
// visible internal functions
#define __unblk_bkfactor_lower __s_unblk_bkfactor_lower
#define __unblk_bkfactor_upper __s_unblk_bkfactor_upper
#define __unblk_bksolve_lower  __s_unblk_bksolve_lower
#define __unblk_bksolve_upper  __s_unblk_bksolve_upper
#define __blk_bkfactor_lower   __s_blk_bkfactor_lower
#define __blk_bkfactor_upper   __s_blk_bkfactor_upper
// marker
#define __ldlbk 1

// Cholesky 
#define __armas_cholfactor       armas_s_cholfactor
#define __armas_cholsolve        armas_s_cholsolve

// Givens
#define __armas_gvcompute       armas_s_gvcompute
#define __armas_gvrotate        armas_s_gvrotate
#define __armas_gvleft          armas_s_gvleft
#define __armas_gvright         armas_s_gvright
#define __armas_gvupdate        armas_s_gvupdate

// Hessenberg functions
#define __armas_hessreduce       armas_s_hessreduce
#define __armas_hessreduce_work  armas_s_hessreduce_work
#define __armas_hessmult         armas_s_hessmult
#define __armas_hessmult_work    armas_s_hessmult_work

// householder functions
#define __compute_householder      __s_compute_householder
#define __compute_householder_vec  __s_compute_householder_vec
#define __compute_householder_rev  __s_compute_householder_rev
#define __apply_householder2x1     __s_apply_householder2x1
#define __apply_householder1x1     __s_apply_householder1x1
// marker for householder functions
#define __householder              1

// LQ functions
#define __armas_lqbuild         armas_s_lqbuild
#define __armas_lqbuild_work    armas_s_lqbuild_work
#define __armas_lqfactor        armas_s_lqfactor
#define __armas_lqfactor_work   armas_s_lqfactor_work
#define __armas_lqmult          armas_s_lqmult
#define __armas_lqmult_work     armas_s_lqmult_work
#define __armas_lqreflector     armas_s_lqreflector
#define __armas_lqsolve         armas_s_lqsolve
#define __armas_lqsolve_work    armas_s_lqsolve_work
// internal LQ related functions available for others
#define __update_lq_left        __s_update_lq_left
#define __update_lq_right       __s_update_lq_right
#define __unblk_lq_reflector    __s_unblk_lq_reflector

// LU functions
#define __armas_lufactor        armas_s_lufactor
#define __armas_lusolve         armas_s_lusolve

// QL functions
#define __armas_qlbuild         armas_s_qlbuild
#define __armas_qlbuild_work    armas_s_qlbuild_work
#define __armas_qlfactor        armas_s_qlfactor
#define __armas_qlfactor_work   armas_s_qlfactor_work
#define __armas_qlmult          armas_s_qlmult
#define __armas_qlmult_work     armas_s_qlmult_work
#define __armas_qlreflector     armas_s_qlreflector
// internal QL related function available for others
#define __update_ql_left        __s_update_ql_left
#define __update_ql_right       __s_update_ql_right
#define __unblk_ql_reflector    __s_unblk_ql_reflector


// QR functions
#define __armas_qrfactor        armas_s_qrfactor
#define __armas_qrfactor_work   armas_s_qrfactor_work
#define __armas_qrmult          armas_s_qrmult
#define __armas_qrmult_work     armas_s_qrmult_work
#define __armas_qrbuild         armas_s_qrbuild
#define __armas_qrbuild_work    armas_s_qrbuild_work
#define __armas_qrreflector     armas_s_qrreflector
#define __armas_qrsolve         armas_s_qrsolve
#define __armas_qrsolve_work    armas_s_qrsolve_work
// internal QR related function available for others
#define __update_qr_left        __s_update_qr_left
#define __update_qr_right       __s_update_qr_right
#define __unblk_qr_reflector    __s_unblk_qr_reflector
#define __unblk_qrfactor        __s_unblk_qrfactor

// QRT functions
#define __armas_qrtfactor       armas_s_qrtfactor
#define __armas_qrtfactor_work  armas_s_qrtfactor_work
#define __armas_qrtmult         armas_s_qrtmult
#define __armas_qrtmult_work    armas_s_qrtmult_work

// RQ functions
#define __armas_rqbuild         armas_s_rqbuild
#define __armas_rqbuild_work    armas_s_rqbuild_work
#define __armas_rqfactor        armas_s_rqfactor
#define __armas_rqfactor_work   armas_s_rqfactor_work
#define __armas_rqmult          armas_s_rqmult
#define __armas_rqmult_work     armas_s_rqmult_work
#define __armas_rqreflector     armas_s_rqreflector
// internal RQ related function available for others
#define __update_rq_left        __s_update_rq_left
#define __update_rq_right       __s_update_rq_right
#define __unblk_rq_reflector    __s_unblk_rq_reflector

// Tridiagonalization functions
#define __armas_trdreduce       armas_s_trdreduce
#define __armas_trdreduce_work  armas_s_trdreduce_work
#define __armas_trdbuild        armas_s_trdbuild
#define __armas_trdbuild_work   armas_s_trdbuild_work
#define __armas_trdmult         armas_s_trdmult
#define __armas_trdmult_work    armas_s_trdmult_work
// Tridiagonal EVD
#define __armas_trdeigen        armas_s_trdeigen
#define __armas_trdsec_solve    armas_s_trdsec_solve
#define __armas_trdsec_eigen    armas_s_trdsec_eigen
#define __armas_trdsec_solve_vec armas_s_trdsec_solve_vec

// Eigenvalue
#define __armas_eigen_sym       armas_s_eigen_sym

// Bidiagonal SVD
#define __armas_bdsvd           armas_s_bdsvd
#define __armas_bdsvd_work      armas_s_bdsvd_work
#define __armas_dqds            armas_s_dqds

// internal 
#define __bdsvd2x2              __s_bdsvd2x2
#define __bdsvd2x2_vec          __s_bdsvd2x2_vec
#define __bdsvd_golub		__s_bdsvd_golub
#define __bdsvd_demmel		__s_bdsvd_demmel

// SVD
#define __armas_svd		armas_s_svd
#define __armas_svd_work	armas_s_svd_work

// Butterfly
#define __armas_gen_rbt         armas_s_gen_rbt
#define __armas_size_rbt        armas_s_size_rbt
#define __armas_mult_rbt        armas_s_mult_rbt
#define __armas_update2_rbt     armas_s_update2_rbt
#define __armas_update2_sym_rbt armas_s_update2_sym_rbt
#define __update2_rbt_descend   __s_update2_rbt_descend

// internal
#define __sym_eigen2x2          __s_sym_eigen2x2
#define __sym_eigen2x2vec       __s_sym_eigen2x2vec
#define __trdevd_qr             __s_trdevd_qr

// Sorting vectors (internal)
#define __pivot_sort              __s_pivot_sort
#define __eigen_sort              __s_eigen_sort
#define __abs_sort_vec            __s_abs_sort_vec 
#define __sort_vec                __s_sort_vec 
#define __sort_eigenvec           __s_sort_eigenvec


/* End of FLOAT32
 * ---------------------------------------------------------------------------
 */
#else
/* ---------------------------------------------------------------------------
 * Definitions for double precision real numbers. (FLOAT64)
 */

#define __SAFEMIN     DBL_MIN
// machine accurancy as in LAPACK library 
#define __EPS         (DBL_EPSILON/2.0)

// internal helpers
#define __swap_rows               __d_swap_rows
#define __swap_cols               __d_swap_cols
#define __apply_pivots            __d_apply_pivots
#define __apply_row_pivots        __d_apply_row_pivots
#define __pivot_index             __d_pivot_index
// marker for above function
#define __lapack_pivots  1
// public pivot functions
#define __armas_pivot_rows      armas_d_pivot_rows


// Bidiagonal reduction
#define __armas_bdreduce         armas_d_bdreduce
#define __armas_bdreduce_work    armas_d_bdreduce_work
#define __armas_bdmult           armas_d_bdmult
#define __armas_bdmult_work      armas_d_bdmult_work
#define __armas_bdbuild          armas_d_bdbuild
#define __armas_bdbuild_work     armas_d_bdbuild_work

// Symmetric LDL functions
#define __armas_bkfactor        armas_d_bkfactor
#define __armas_bkfactor_work   armas_d_bkfactor_work
#define __armas_bksolve         armas_d_bksolve
#define __armas_bksolve_work    armas_d_bksolve_work
// visible internal functions
#define __unblk_bkfactor_lower __d_unblk_bkfactor_lower
#define __unblk_bkfactor_upper __d_unblk_bkfactor_upper
#define __unblk_bksolve_lower  __d_unblk_bksolve_lower
#define __unblk_bksolve_upper  __d_unblk_bksolve_upper
#define __blk_bkfactor_lower   __d_blk_bkfactor_lower
#define __blk_bkfactor_upper   __d_blk_bkfactor_upper
// marker
#define __ldlbk 1

// Cholesky 
#define __armas_cholfactor       armas_d_cholfactor
#define __armas_cholsolve        armas_d_cholsolve

// Hessenberg functions
#define __armas_hessreduce       armas_d_hessreduce
#define __armas_hessreduce_work  armas_d_hessreduce_work
#define __armas_hessmult         armas_d_hessmult
#define __armas_hessmult_work    armas_d_hessmult_work

// householder functions
#define __compute_householder      __d_compute_householder
#define __compute_householder_vec  __d_compute_householder_vec
#define __compute_householder_rev  __d_compute_householder_rev
#define __apply_householder2x1     __d_apply_householder2x1
#define __apply_householder1x1     __d_apply_householder1x1
// marker for householder functions
#define __householder              1

// LQ functions
#define __armas_lqbuild         armas_d_lqbuild
#define __armas_lqbuild_work    armas_d_lqbuild_work
#define __armas_lqfactor        armas_d_lqfactor
#define __armas_lqfactor_work   armas_d_lqfactor_work
#define __armas_lqmult          armas_d_lqmult
#define __armas_lqmult_work     armas_d_lqmult_work
#define __armas_lqreflector     armas_d_lqreflector
#define __armas_lqsolve         armas_d_lqsolve
#define __armas_lqsolve_work    armas_d_lqsolve_work
// internal LQ related functions available for others
#define __update_lq_left        __d_update_lq_left
#define __update_lq_right       __d_update_lq_right
#define __unblk_lq_reflector    __d_unblk_lq_reflector

// LU functions
#define __armas_lufactor        armas_d_lufactor
#define __armas_lusolve         armas_d_lusolve

// QL functions
#define __armas_qlbuild         armas_d_qlbuild
#define __armas_qlbuild_work    armas_d_qlbuild_work
#define __armas_qlfactor        armas_d_qlfactor
#define __armas_qlfactor_work   armas_d_qlfactor_work
#define __armas_qlmult          armas_d_qlmult
#define __armas_qlmult_work     armas_d_qlmult_work
#define __armas_qlreflector     armas_d_qlreflector
// internal QL related function available for others
#define __update_ql_left        __d_update_ql_left
#define __update_ql_right       __d_update_ql_right
#define __unblk_ql_reflector    __d_unblk_ql_reflector

// QR functions
#define __armas_qrfactor        armas_d_qrfactor
#define __armas_qrfactor_work   armas_d_qrfactor_work
#define __armas_qrmult          armas_d_qrmult
#define __armas_qrmult_work     armas_d_qrmult_work
#define __armas_qrbuild         armas_d_qrbuild
#define __armas_qrbuild_work    armas_d_qrbuild_work
#define __armas_qrreflector     armas_d_qrreflector
#define __armas_qrsolve         armas_d_qrsolve
#define __armas_qrsolve_work    armas_d_qrsolve_work
// internal QR related function available for others
#define __update_qr_left        __d_update_qr_left
#define __update_qr_right       __d_update_qr_right
#define __unblk_qr_reflector    __d_unblk_qr_reflector
#define __unblk_qrfactor        __d_unblk_qrfactor

// QRT functions
#define __armas_qrtfactor       armas_d_qrtfactor
#define __armas_qrtfactor_work  armas_d_qrtfactor_work
#define __armas_qrtmult         armas_d_qrtmult
#define __armas_qrtmult_work    armas_d_qrtmult_work

// RQ functions
#define __armas_rqbuild         armas_d_rqbuild
#define __armas_rqbuild_work    armas_d_rqbuild_work
#define __armas_rqfactor        armas_d_rqfactor
#define __armas_rqfactor_work   armas_d_rqfactor_work
#define __armas_rqmult          armas_d_rqmult
#define __armas_rqmult_work     armas_d_rqmult_work
#define __armas_rqreflector     armas_d_rqreflector
// internal RQ related function available for others
#define __update_rq_left        __d_update_rq_left
#define __update_rq_right       __d_update_rq_right
#define __unblk_rq_reflector    __d_unblk_rq_reflector

// Tridiagonalization functions
#define __armas_trdreduce       armas_d_trdreduce
#define __armas_trdreduce_work  armas_d_trdreduce_work
#define __armas_trdbuild        armas_d_trdbuild
#define __armas_trdbuild_work   armas_d_trdbuild_work
#define __armas_trdmult         armas_d_trdmult
#define __armas_trdmult_work    armas_d_trdmult_work
// Tridiagonal EVD
#define __armas_trdeigen        armas_d_trdeigen
#define __armas_trdsec_solve    armas_d_trdsec_solve
#define __armas_trdsec_eigen    armas_d_trdsec_eigen
#define __armas_trdsec_solve_vec armas_d_trdsec_solve_vec

// Eigenvalue
#define __armas_eigen_sym       armas_d_eigen_sym

// Givens
#define __armas_gvcompute       armas_d_gvcompute
#define __armas_gvrotate        armas_d_gvrotate
#define __armas_gvleft          armas_d_gvleft
#define __armas_gvright         armas_d_gvright
#define __armas_gvupdate        armas_d_gvupdate

// Bidiagonal SVD
#define __armas_bdsvd           armas_d_bdsvd
#define __armas_bdsvd_work      armas_d_bdsvd_work
#define __armas_dqds            armas_d_dqds

// internal 
#define __bdsvd2x2              __d_bdsvd2x2
#define __bdsvd2x2_vec          __d_bdsvd2x2_vec
#define __bdsvd_golub		__d_bdsvd_golub
#define __bdsvd_demmel		__d_bdsvd_demmel

// SVD
#define __armas_svd		armas_d_svd
#define __armas_svd_work	armas_d_svd_work

// Butterfly
#define __armas_gen_rbt         armas_d_gen_rbt
#define __armas_size_rbt        armas_d_size_rbt
#define __armas_mult_rbt        armas_d_mult_rbt
#define __armas_update2_rbt     armas_d_update2_rbt
#define __armas_update2_sym_rbt armas_d_update2_sym_rbt
#define __update2_rbt_descend   __d_update2_rbt_descend

// internal
#define __sym_eigen2x2          __d_sym_eigen2x2
#define __sym_eigen2x2vec       __d_sym_eigen2x2vec
#define __trdevd_qr             __d_trdevd_qr

// Sorting vectors (internal)
#define __pivot_sort              __d_pivot_sort
#define __eigen_sort              __d_eigen_sort
#define __abs_sort_vec            __d_abs_sort_vec 
#define __sort_vec                __d_sort_vec 
#define __sort_eigenvec           __d_sort_eigenvec

// Additional
#define __armas_qdroots         armas_d_qdroots
#define __armas_discriminant    armas_d_discriminant
#define __armas_mult_diag       armas_d_mult_diag
#define __armas_solve_diag      armas_d_solve_diag
#define __armas_scale_to        armas_d_scale_to

#endif /* FLOAT64 */

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

static inline
DTYPE __MIN(DTYPE a, DTYPE b) {
  return a < b ? a : b;
}

static inline
DTYPE __MAX(DTYPE a, DTYPE b) {
  return a > b ? a : b;
}


#endif  /* __ARMAS_DLPACK_H */
#endif  /* __DOXYGEN */
