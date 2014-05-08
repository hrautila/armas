
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

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


#else
/* ---------------------------------------------------------------------------
 * Definitions for double precision real numbers.
 */

#define __SAFEMIN     DBL_MIN
// machine accurancy as in LAPACK library 
#define __EPS         (DBL_EPSILON/2.0)

// internal helpers
#define __swap_rows               __d_swap_rows
#define __swap_cols               __d_swap_cols
#define __apply_row_pivots        __d_apply_row_pivots

// Bidiagonal reduction
#define __armas_bdreduce         armas_d_bdreduce
#define __armas_bdreduce_work    armas_d_bdreduce_work
#define __armas_bdmult           armas_d_bdmult
#define __armas_bdmult_work      armas_d_bdmult_work

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
#define __unblk_ql_reflector    __d_unblk_ql_reflectorm

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

// Tridiagonalization functions
#define __armas_trdreduce       armas_d_trdreduce
#define __armas_trdreduce_work  armas_d_trdreduce_work

#endif /* FLOAT64 */


#endif  /* __ARMAS_DLPACK_H */
