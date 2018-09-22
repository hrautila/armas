
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
#define __pivot_index             __s_pivot_index
// public pivot functions
#define armas_x_pivot_rows      armas_s_pivot_rows
#define armas_x_pivot_cols      armas_s_pivot_cols
#define armas_x_pivot           armas_s_pivot
// marker for above function
#define __lapack_pivots  1

// Additional
#define armas_x_qdroots         armas_s_qdroots
#define armas_x_discriminant    armas_s_discriminant
#define armas_x_mult_diag       armas_s_mult_diag
#define armas_x_solve_diag      armas_s_solve_diag
#define armas_x_scale_to        armas_s_scale_to

// Bidiagonal reduction
#define armas_x_bdreduce         armas_s_bdreduce
#define armas_x_bdmult           armas_s_bdmult
#define armas_x_bdbuild          armas_s_bdbuild

// Symmetric LDL functions
#define armas_x_bkfactor        armas_s_bkfactor
#define armas_x_bksolve         armas_s_bksolve
#define armas_x_ldlfactor       armas_s_ldlfactor
#define armas_x_ldlupdate       armas_s_ldlupdate
#define armas_x_ldlsolve        armas_s_ldlsolve
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
#define armas_x_cholfactor       armas_s_cholfactor
#define armas_x_cholupdate       armas_s_cholupdate
#define armas_x_cholsolve        armas_s_cholsolve
// internal pivoting versions
#define __cholfactor_pv          __s_cholfactor_pv
#define __cholsolve_pv           __s_cholsolve_pv

// Givens
#define armas_x_gvcompute       armas_s_gvcompute
#define armas_x_gvrotate        armas_s_gvrotate
#define armas_x_gvrot_vec       armas_s_gvrot_vec
#define armas_x_gvleft          armas_s_gvleft
#define armas_x_gvright         armas_s_gvright
#define armas_x_gvupdate        armas_s_gvupdate

// Hessenberg functions
#define armas_x_hessreduce       armas_s_hessreduce
#define armas_x_hessmult         armas_s_hessmult

// householder functions
#define __compute_householder      __s_compute_householder
#define __compute_householder_vec  __s_compute_householder_vec
#define __compute_householder_rev  __s_compute_householder_rev
#define __apply_householder2x1     __s_apply_householder2x1
#define __apply_householder1x1     __s_apply_householder1x1
#define armas_x_house              armas_s_house
#define armas_x_houseapply         armas_s_houseapply
#define armas_x_houseapply2x1      armas_s_houseapply2x1
#define armas_x_housemult          armas_s_housemult
#define armas_x_housemult_w        armas_s_housemult_w
#define armas_x_hhouse             armas_s_hhouse
#define armas_x_hhouse_apply       armas_s_hhouse_apply
// marker for householder functions
#define __householder              1

// LQ functions
#define armas_x_lqbuild         armas_s_lqbuild
#define armas_x_lqbuild_work    armas_s_lqbuild_work
#define armas_x_lqfactor        armas_s_lqfactor
#define armas_x_lqfactor_work   armas_s_lqfactor_work
#define armas_x_lqmult          armas_s_lqmult
#define armas_x_lqmult_work     armas_s_lqmult_work
#define armas_x_lqreflector     armas_s_lqreflector
#define armas_x_lqsolve         armas_s_lqsolve
#define armas_x_lqsolve_work    armas_s_lqsolve_work
// internal LQ related functions available for others
#define __update_lq_left        __s_update_lq_left
#define __update_lq_right       __s_update_lq_right
#define __unblk_lq_reflector    __s_unblk_lq_reflector

// LU functions
#define armas_x_lufactor        armas_s_lufactor
#define armas_x_lusolve         armas_s_lusolve

// QL functions
#define armas_x_qlbuild         armas_s_qlbuild
#define armas_x_qlbuild_work    armas_s_qlbuild_work
#define armas_x_qlfactor        armas_s_qlfactor
#define armas_x_qlfactor_work   armas_s_qlfactor_work
#define armas_x_qlmult          armas_s_qlmult
#define armas_x_qlmult_work     armas_s_qlmult_work
#define armas_x_qlreflector     armas_s_qlreflector
// internal QL related function available for others
#define __update_ql_left        __s_update_ql_left
#define __update_ql_right       __s_update_ql_right
#define __unblk_ql_reflector    __s_unblk_ql_reflector


// QR functions
#define armas_x_qrfactor        armas_s_qrfactor
#define armas_x_qrfactor_work   armas_s_qrfactor_work
#define armas_x_qrmult          armas_s_qrmult
#define armas_x_qrmult_work     armas_s_qrmult_work
#define armas_x_qrbuild         armas_s_qrbuild
#define armas_x_qrbuild_work    armas_s_qrbuild_work
#define armas_x_qrreflector     armas_s_qrreflector
#define armas_x_qrsolve         armas_s_qrsolve
#define armas_x_qrsolve_work    armas_s_qrsolve_work
// internal QR related function available for others
#define __update_qr_left        __s_update_qr_left
#define __update_qr_right       __s_update_qr_right
#define __unblk_qr_reflector    __s_unblk_qr_reflector
#define __unblk_qrfactor        __s_unblk_qrfactor

// QRT functions
#define armas_x_qrtfactor       armas_s_qrtfactor
#define armas_x_qrtfactor_work  armas_s_qrtfactor_work
#define armas_x_qrtmult         armas_s_qrtmult
#define armas_x_qrtmult_work    armas_s_qrtmult_work

// RQ functions
#define armas_x_rqbuild         armas_s_rqbuild
#define armas_x_rqbuild_work    armas_s_rqbuild_work
#define armas_x_rqfactor        armas_s_rqfactor
#define armas_x_rqfactor_work   armas_s_rqfactor_work
#define armas_x_rqmult          armas_s_rqmult
#define armas_x_rqmult_work     armas_s_rqmult_work
#define armas_x_rqreflector     armas_s_rqreflector
// internal RQ related function available for others
#define __update_rq_left        __s_update_rq_left
#define __update_rq_right       __s_update_rq_right
#define __unblk_rq_reflector    __s_unblk_rq_reflector

// Tridiagonalization functions
#define armas_x_trdreduce       armas_s_trdreduce
#define armas_x_trdreduce_work  armas_s_trdreduce_work
#define armas_x_trdbuild        armas_s_trdbuild
#define armas_x_trdbuild_work   armas_s_trdbuild_work
#define armas_x_trdmult         armas_s_trdmult
#define armas_x_trdmult_work    armas_s_trdmult_work
// Tridiagonal EVD
#define armas_x_trdeigen        armas_s_trdeigen
#define armas_x_trdbisect       armas_s_trdbisect
#define armas_x_trdsec_solve    armas_s_trdsec_solve
#define armas_x_trdsec_eigen    armas_s_trdsec_eigen
#define armas_x_trdsec_solve_vec armas_s_trdsec_solve_vec

// Eigenvalue
#define armas_x_eigen_sym           armas_s_eigen_sym
#define armas_x_eigen_sym_selected  armas_s_eigen_sym_selected

// Bidiagonal SVD
#define armas_x_bdsvd           armas_s_bdsvd
#define armas_x_bdsvd_work      armas_s_bdsvd_work
#define armas_x_dqds            armas_s_dqds

// internal 
#define __bdsvd2x2              __s_bdsvd2x2
#define __bdsvd2x2_vec          __s_bdsvd2x2_vec
#define __bdsvd_golub		__s_bdsvd_golub
#define __bdsvd_demmel		__s_bdsvd_demmel

// SVD
#define armas_x_svd		armas_s_svd
#define armas_x_svd_work	armas_s_svd_work

// Butterfly
#define armas_x_gen_rbt         armas_s_gen_rbt
#define armas_x_size_rbt        armas_s_size_rbt
#define armas_x_mult_rbt        armas_s_mult_rbt
#define armas_x_update2_rbt     armas_s_update2_rbt
#define armas_x_update2_sym_rbt armas_s_update2_sym_rbt
#define __update2_rbt_descend   __s_update2_rbt_descend

// inverse
#define armas_x_inverse_trm     armas_s_inverse_trm
#define armas_x_inverse         armas_s_inverse
#define armas_x_inverse_psd     armas_s_inverse_psd
#define armas_x_ldlinverse_sym  armas_s_ldlinverse_sym

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
#define __pivot_index             __d_pivot_index
// public pivot functions
#define armas_x_pivot_rows      armas_d_pivot_rows
#define armas_x_pivot_cols      armas_d_pivot_cols
#define armas_x_pivot           armas_d_pivot
// marker for above function
#define __lapack_pivots  1


// Bidiagonal reduction
#define armas_x_bdreduce         armas_d_bdreduce
#define armas_x_bdreduce_w       armas_d_bdreduce_w
#define armas_x_bdmult           armas_d_bdmult
#define armas_x_bdmult_w         armas_d_bdmult_w
#define armas_x_bdbuild          armas_d_bdbuild
#define armas_x_bdbuild_w        armas_d_bdbuild_w

// Symmetric LDL functions
#define armas_x_bkfactor        armas_d_bkfactor
#define armas_x_bkfactor_w      armas_d_bkfactor_w
#define armas_x_bksolve         armas_d_bksolve
#define armas_x_bksolve_w       armas_d_bksolve_w
#define armas_x_ldlfactor       armas_d_ldlfactor
#define armas_x_ldlfactor_w     armas_d_ldlfactor_w
#define armas_x_ldlupdate       armas_d_ldlupdate
#define armas_x_ldlsolve        armas_d_ldlsolve

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
#define armas_x_cholesky         armas_d_cholesky
#define armas_x_cholfactor       armas_d_cholfactor
#define armas_x_cholfactor_w     armas_d_cholfactor_w
#define armas_x_cholupdate       armas_d_cholupdate
#define armas_x_cholsolve        armas_d_cholsolve
// internal pivoting versions
#define __cholfactor_pv          __d_cholfactor_pv
#define __cholsolve_pv           __d_cholsolve_pv

// Hessenberg functions
#define armas_x_hessreduce       armas_d_hessreduce
#define armas_x_hessreduce_w     armas_d_hessreduce_w
#define armas_x_hessmult         armas_d_hessmult
#define armas_x_hessmult_w       armas_d_hessmult_w

// householder functions
#define __compute_householder      __d_compute_householder
#define __compute_householder_vec  __d_compute_householder_vec
#define __compute_householder_rev  __d_compute_householder_rev
#define __apply_householder2x1     __d_apply_householder2x1
#define __apply_householder1x1     __d_apply_householder1x1
#define armas_x_house              armas_d_house
#define armas_x_houseapply         armas_d_houseapply
#define armas_x_houseapply2x1      armas_d_houseapply2x1
#define armas_x_housemult          armas_d_housemult
#define armas_x_housemult_w        armas_d_housemult_w
#define armas_x_hhouse             armas_d_hhouse
#define armas_x_hhouse_apply       armas_d_hhouse_apply
// marker for householder functions
#define __householder              1

// LQ functions
#define armas_x_lqbuild         armas_d_lqbuild
#define armas_x_lqbuild_w       armas_d_lqbuild_w
#define armas_x_lqbuild_work    armas_d_lqbuild_work
#define armas_x_lqfactor        armas_d_lqfactor
#define armas_x_lqfactor_w      armas_d_lqfactor_w
#define armas_x_lqfactor_work   armas_d_lqfactor_work
#define armas_x_lqmult          armas_d_lqmult
#define armas_x_lqmult_w        armas_d_lqmult_w
#define armas_x_lqmult_work     armas_d_lqmult_work
#define armas_x_lqreflector     armas_d_lqreflector
#define armas_x_lqsolve         armas_d_lqsolve
#define armas_x_lqsolve_w       armas_d_lqsolve_w
#define armas_x_lqsolve_work    armas_d_lqsolve_work
// internal LQ related functions available for others
#define __update_lq_left        __d_update_lq_left
#define __update_lq_right       __d_update_lq_right
#define __unblk_lq_reflector    __d_unblk_lq_reflector

// LU functions
#define armas_x_lufactor        armas_d_lufactor
#define armas_x_lusolve         armas_d_lusolve

// QL functions
#define armas_x_qlbuild         armas_d_qlbuild
#define armas_x_qlbuild_w       armas_d_qlbuild_w
#define armas_x_qlbuild_work    armas_d_qlbuild_work
#define armas_x_qlfactor        armas_d_qlfactor
#define armas_x_qlfactor_w      armas_d_qlfactor_w
#define armas_x_qlfactor_work   armas_d_qlfactor_work
#define armas_x_qlmult          armas_d_qlmult
#define armas_x_qlmult_w        armas_d_qlmult_w
#define armas_x_qlmult_work     armas_d_qlmult_work
#define armas_x_qlreflector     armas_d_qlreflector
#define armas_x_qlsolve         armas_d_qlsolve
#define armas_x_qlsolve_w       armas_d_qlsolve_w
// internal QL related function available for others
#define __update_ql_left        __d_update_ql_left
#define __update_ql_right       __d_update_ql_right
#define __unblk_ql_reflector    __d_unblk_ql_reflector

// QR functions
#define armas_x_qrfactor        armas_d_qrfactor
#define armas_x_qrfactor_w      armas_d_qrfactor_w
#define armas_x_qrfactor_work   armas_d_qrfactor_work
#define armas_x_qrmult          armas_d_qrmult
#define armas_x_qrmult_w        armas_d_qrmult_w
#define armas_x_qrmult_work     armas_d_qrmult_work
#define armas_x_qrbuild         armas_d_qrbuild
#define armas_x_qrbuild_w       armas_d_qrbuild_w
#define armas_x_qrbuild_work    armas_d_qrbuild_work
#define armas_x_qrreflector     armas_d_qrreflector
#define armas_x_qrsolve         armas_d_qrsolve
#define armas_x_qrsolve_w       armas_d_qrsolve_w
#define armas_x_qrsolve_work    armas_d_qrsolve_work
// internal QR related function available for others
#define __update_qr_left        __d_update_qr_left
#define __update_qr_right       __d_update_qr_right
#define __unblk_qr_reflector    __d_unblk_qr_reflector
#define __unblk_qrfactor        __d_unblk_qrfactor

// QRT functions
#define armas_x_qrtfactor       armas_d_qrtfactor
#define armas_x_qrtfactor_w     armas_d_qrtfactor_w
#define armas_x_qrtfactor_work  armas_d_qrtfactor_work
#define armas_x_qrtmult         armas_d_qrtmult
#define armas_x_qrtmult_w       armas_d_qrtmult_w
#define armas_x_qrtmult_work    armas_d_qrtmult_work

// RQ functions
#define armas_x_rqbuild         armas_d_rqbuild
#define armas_x_rqbuild_w       armas_d_rqbuild_w
#define armas_x_rqbuild_work    armas_d_rqbuild_work
#define armas_x_rqfactor        armas_d_rqfactor
#define armas_x_rqfactor_w      armas_d_rqfactor_w
#define armas_x_rqfactor_work   armas_d_rqfactor_work
#define armas_x_rqmult          armas_d_rqmult
#define armas_x_rqmult_w        armas_d_rqmult_w
#define armas_x_rqmult_work     armas_d_rqmult_work
#define armas_x_rqreflector     armas_d_rqreflector
#define armas_x_rqsolve         armas_d_rqsolve
#define armas_x_rqsolve_w       armas_d_rqsolve_w
// internal RQ related function available for others
#define __update_rq_left        __d_update_rq_left
#define __update_rq_right       __d_update_rq_right
#define __unblk_rq_reflector    __d_unblk_rq_reflector

// Tridiagonalization functions
#define armas_x_trdreduce       armas_d_trdreduce
#define armas_x_trdreduce_w     armas_d_trdreduce_w
#define armas_x_trdreduce_work  armas_d_trdreduce_work
#define armas_x_trdbuild        armas_d_trdbuild
#define armas_x_trdbuild_w      armas_d_trdbuild_w
#define armas_x_trdbuild_work   armas_d_trdbuild_work
#define armas_x_trdmult         armas_d_trdmult
#define armas_x_trdmult_w       armas_d_trdmult_w
#define armas_x_trdmult_work    armas_d_trdmult_work
// Tridiagonal EVD
#define armas_x_trdeigen        armas_d_trdeigen
#define armas_x_trdeigen_w      armas_d_trdeigen_w
#define armas_x_trdbisect       armas_d_trdbisect
#define armas_x_trdsec_solve    armas_d_trdsec_solve
#define armas_x_trdsec_eigen    armas_d_trdsec_eigen
#define armas_x_trdsec_solve_vec armas_d_trdsec_solve_vec

// Eigenvalue
#define armas_x_eigen_sym           armas_d_eigen_sym
#define armas_x_eigen_sym_w         armas_d_eigen_sym_w
#define armas_x_eigen_sym_selected  armas_d_eigen_sym_selected
#define armas_x_eigen_sym_selected_w armas_d_eigen_sym_selected_w

// Givens
#define armas_x_gvcompute       armas_d_gvcompute
#define armas_x_gvrotate        armas_d_gvrotate
#define armas_x_gvrot_vec       armas_d_gvrot_vec
#define armas_x_gvleft          armas_d_gvleft
#define armas_x_gvright         armas_d_gvright
#define armas_x_gvupdate        armas_d_gvupdate

// Bidiagonal SVD
#define armas_x_bdsvd           armas_d_bdsvd
#define armas_x_bdsvd_w         armas_d_bdsvd_w
#define armas_x_bdsvd_work      armas_d_bdsvd_work
#define armas_x_dqds            armas_d_dqds
#define armas_x_dqds_w          armas_d_dqds_w

// internal 
#define __bdsvd2x2              __d_bdsvd2x2
#define __bdsvd2x2_vec          __d_bdsvd2x2_vec
#define __bdsvd_golub		__d_bdsvd_golub
#define __bdsvd_demmel		__d_bdsvd_demmel

// SVD
#define armas_x_svd		armas_d_svd
#define armas_x_svd_w           armas_d_svd_w
#define armas_x_svd_work	armas_d_svd_work

// Butterfly
#define armas_x_gen_rbt         armas_d_gen_rbt
#define armas_x_size_rbt        armas_d_size_rbt
#define armas_x_mult_rbt        armas_d_mult_rbt
#define armas_x_update2_rbt     armas_d_update2_rbt
#define armas_x_update2_sym_rbt armas_d_update2_sym_rbt
#define __update2_rbt_descend   __d_update2_rbt_descend

// inverse
#define armas_x_inverse_trm     armas_d_inverse_trm
#define armas_x_inverse         armas_d_inverse
#define armas_x_inverse_w       armas_d_inverse_w
#define armas_x_inverse_psd     armas_d_inverse_psd
#define armas_x_inverse_psd_w   armas_d_inverse_psd_w
#define armas_x_ldlinverse_sym  armas_d_ldlinverse_sym
#define armas_x_ldlinverse_sym_w armas_d_ldlinverse_sym_w

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
#define armas_x_qdroots         armas_d_qdroots
#define armas_x_discriminant    armas_d_discriminant
#define armas_x_mult_diag       armas_d_mult_diag
#define armas_x_solve_diag      armas_d_solve_diag
#define armas_x_scale_to        armas_d_scale_to

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
