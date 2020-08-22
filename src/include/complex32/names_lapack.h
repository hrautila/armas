
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef DOXYGEN
#ifndef ARMAS_NAMES_LAPACK_H
#define ARMAS_NAMES_LAPACK_H 1

#include <float.h>

#if 0 /* untestested */
/* ---------------------------------------------------------------------------
 * Definitions for double precision real numbers. (FLOAT64)
 */

#define SAFEMIN     FLT_MIN
// machine accurancy as in LAPACK library 
#define EPS         (FLT_EPSILON/2.0)

// internal helpers
#define armas_x_pivot_index             armas_c_pivot_index
// public pivot functions
#define armas_x_pivot_rows              armas_c_pivot_rows
#define armas_x_pivot_cols              armas_c_pivot_cols
#define armas_x_pivot                   armas_c_pivot
// marker for above function
#define armas_x_lapack_pivots  1


// Bidiagonal reduction
#define armas_x_bdreduce                armas_c_bdreduce
#define armas_x_bdreduce_w              armas_c_bdreduce_w
#define armas_x_bdmult                  armas_c_bdmult
#define armas_x_bdmult_w                armas_c_bdmult_w
#define armas_x_bdbuild                 armas_c_bdbuild
#define armas_x_bdbuild_w               armas_c_bdbuild_w

// Symmetric LDL functions
#define armas_x_bkfactor                 armas_c_bkfactor
#define armas_x_bkfactor_w               armas_c_bkfactor_w
#define armas_x_bksolve                  armas_c_bksolve
#define armas_x_bksolve_w                armas_c_bksolve_w
#define armas_x_ldlfactor                armas_c_ldlfactor
#define armas_x_ldlfactor_w              armas_c_ldlfactor_w
#define armas_x_ldlupdate                armas_c_ldlupdate
#define armas_x_ldlsolve                 armas_c_ldlsolve

// visible internal functions
#define armas_x_unblk_bkfactor_lower     armas_c_unblk_bkfactor_lower
#define armas_x_unblk_bkfactor_upper     armas_c_unblk_bkfactor_upper
#define armas_x_unblk_bksolve_lower      armas_c_unblk_bksolve_lower
#define armas_x_unblk_bksolve_upper      armas_c_unblk_bksolve_upper
#define armas_x_blk_bkfactor_lower       armas_c_blk_bkfactor_lower
#define armas_x_blk_bkfactor_upper       armas_c_blk_bkfactor_upper
// marker
#define armas_x_ldlbk 1

// Cholesky 
#define armas_x_cholesky                 armas_c_cholesky
#define armas_x_cholfactor               armas_c_cholfactor
#define armas_x_cholfactor_w             armas_c_cholfactor_w
#define armas_x_cholupdate               armas_c_cholupdate
#define armas_x_cholsolve                armas_c_cholsolve
// internal pivoting versions
#define armas_x_cholfactor_pv            armas_c_cholfactor_pv
#define armas_x_cholsolve_pv             armas_c_cholsolve_pv

// Hessenberg functions
#define armas_x_hessreduce               armas_c_hessreduce
#define armas_x_hessreduce_w             armas_c_hessreduce_w
#define armas_x_hessmult                 armas_c_hessmult
#define armas_x_hessmult_w               armas_c_hessmult_w

// householder functions
#define armas_x_compute_householder      armas_c_compute_householder
#define armas_x_compute_householder_vec  armas_c_compute_householder_vec
#define armas_x_compute_householder_rev  armas_c_compute_householder_rev
#define armas_x_apply_householder2x1     armas_c_apply_householder2x1
#define armas_x_apply_householder1x1     armas_c_apply_householder1x1
#define armas_x_house                    armas_c_house
#define armas_x_houseapply               armas_c_houseapply
#define armas_x_houseapply2x1            armas_c_houseapply2x1
#define armas_x_housemult                armas_c_housemult
#define armas_x_housemult_w              armas_c_housemult_w
#define armas_x_hhouse                   armas_c_hhouse
#define armas_x_hhouse_apply             armas_c_hhouse_apply
// marker for householder functions
#define armas_x_householder              1

// LQ functions
#define armas_x_lqbuild                  armas_c_lqbuild
#define armas_x_lqbuild_w                armas_c_lqbuild_w
#define armas_x_lqfactor                 armas_c_lqfactor
#define armas_x_lqfactor_w               armas_c_lqfactor_w
#define armas_x_lqmult                   armas_c_lqmult
#define armas_x_lqmult_w                 armas_c_lqmult_w
#define armas_x_lqreflector              armas_c_lqreflector
#define armas_x_lqsolve                  armas_c_lqsolve
#define armas_x_lqsolve_w                armas_c_lqsolve_w
// internal LQ related functions available for others
#define armas_x_update_lq_left           armas_c_update_lq_left
#define armas_x_update_lq_right          armas_c_update_lq_right
#define armas_x_unblk_lq_reflector       armas_c_unblk_lq_reflector

// LU functions
#define armas_x_lufactor                 armas_c_lufactor
#define armas_x_lusolve                  armas_c_lusolve

// QL functions
#define armas_x_qlbuild                  armas_c_qlbuild
#define armas_x_qlbuild_w                armas_c_qlbuild_w
#define armas_x_qlfactor                 armas_c_qlfactor
#define armas_x_qlfactor_w               armas_c_qlfactor_w
#define armas_x_qlmult                   armas_c_qlmult
#define armas_x_qlmult_w                 armas_c_qlmult_w
#define armas_x_qlreflector              armas_c_qlreflector
#define armas_x_qlsolve                  armas_c_qlsolve
#define armas_x_qlsolve_w                armas_c_qlsolve_w
// internal QL related function available for others
#define armas_x_update_ql_left           armas_c_update_ql_left
#define armas_x_update_ql_right          armas_c_update_ql_right
#define armas_unblk_ql_reflector         armas_c_unblk_ql_reflector

// QR functions
#define armas_x_qrfactor                 armas_c_qrfactor
#define armas_x_qrfactor_w               armas_c_qrfactor_w
#define armas_x_qrmult                   armas_c_qrmult
#define armas_x_qrmult_w                 armas_c_qrmult_w
#define armas_x_qrbuild                  armas_c_qrbuild
#define armas_x_qrbuild_w                armas_c_qrbuild_w
#define armas_x_qrreflector              armas_c_qrreflector
#define armas_x_qrsolve                  armas_c_qrsolve
#define armas_x_qrsolve_w                armas_c_qrsolve_w
// internal QR related function available for others
#define armas_x_update_qr_left           armas_c_update_qr_left
#define armas_x_update_qr_right          armas_c_update_qr_right
#define armas_x_unblk_qr_reflector       armas_c_unblk_qr_reflector
#define armas_x_unblk_qrfactor           armas_c_unblk_qrfactor

// QRT functions
#define armas_x_qrtfactor                armas_c_qrtfactor
#define armas_x_qrtfactor_w              armas_c_qrtfactor_w
#define armas_x_qrtfactor_work           armas_c_qrtfactor_work
#define armas_x_qrtmult                  armas_c_qrtmult
#define armas_x_qrtmult_w                armas_c_qrtmult_w
#define armas_x_qrtmult_work             armas_c_qrtmult_work

// RQ functions
#define armas_x_rqbuild                  armas_c_rqbuild
#define armas_x_rqbuild_w                armas_c_rqbuild_w
#define armas_x_rqfactor                 armas_c_rqfactor
#define armas_x_rqfactor_w               armas_c_rqfactor_w
#define armas_x_rqmult                   armas_c_rqmult
#define armas_x_rqmult_w                 armas_c_rqmult_w
#define armas_x_rqreflector              armas_c_rqreflector
#define armas_x_rqsolve                  armas_c_rqsolve
#define armas_x_rqsolve_w                armas_c_rqsolve_w
// internal RQ related function available for others
#define armas_x_update_rq_left           armas_c_update_rq_left
#define armas_x_update_rq_right          armas_c_update_rq_right
#define armas_x_unblk_rq_reflector       armas_c_unblk_rq_reflector

// Tridiagonalization functions
#define armas_x_trdreduce                armas_c_trdreduce
#define armas_x_trdreduce_w              armas_c_trdreduce_w
#define armas_x_trdbuild                 armas_c_trdbuild
#define armas_x_trdbuild_w               armas_c_trdbuild_w
#define armas_x_trdmult                  armas_c_trdmult
#define armas_x_trdmult_w                armas_c_trdmult_w
// Tridiagonal EVD
#define armas_x_trdeigen                 armas_c_trdeigen
#define armas_x_trdeigen_w               armas_c_trdeigen_w
#define armas_x_trdbisect                armas_c_trdbisect
#define armas_x_trdsec_solve             armas_c_trdsec_solve
#define armas_x_trdsec_eigen             armas_c_trdsec_eigen
#define armas_x_trdsec_solve_vec         armas_c_trdsec_solve_vec

// Eigenvalue
#define armas_x_eigen_sym                armas_c_eigen_sym
#define armas_x_eigen_sym_w              armas_c_eigen_sym_w
#define armas_x_eigen_sym_selected       armas_c_eigen_sym_selected
#define armas_x_eigen_sym_selected_w     armas_c_eigen_sym_selected_w

// Givens
#define armas_x_gvcompute                armas_c_gvcompute
#define armas_x_gvrotate                 armas_c_gvrotate
#define armas_x_gvrot_vec                armas_c_gvrot_vec
#define armas_x_gvleft                   armas_c_gvleft
#define armas_x_gvright                  armas_c_gvright
#define armas_x_gvupdate                 armas_c_gvupdate

// Bidiagonal SVD
#define armas_x_bdsvd                    armas_c_bdsvd
#define armas_x_bdsvd_w                  armas_c_bdsvd_w
#define armas_x_dqds                     armas_c_dqds
#define armas_x_dqds_w                   armas_c_dqds_w

// internal
#define armas_x_bdsvd2x2                 armas_c_bdsvd2x2
#define armas_x_bdsvd2x2_vec             armas_c_bdsvd2x2_vec
#define armas_x_bdsvd_golub		         armas_c_bdsvd_golub
#define armas_x_bdsvd_demmel		     armas_c_bdsvd_demmel

// SVD
#define armas_x_svd		                 armas_c_svd
#define armas_x_svd_w                    armas_c_svd_w

// Butterfly
#define armas_x_gen_rbt                  armas_c_gen_rbt
#define armas_x_size_rbt                 armas_c_size_rbt
#define armas_x_mult_rbt                 armas_c_mult_rbt
#define armas_x_update2_rbt              armas_c_update2_rbt
#define armas_x_update2_sym_rbt          armas_c_update2_sym_rbt
#define armas_x_update2_rbt_descend      armas_c_update2_rbt_descend

// inverse
#define armas_x_inverse_trm              armas_c_inverse_trm
#define armas_x_luinverse                armas_c_luinverse
#define armas_x_luinverse_w              armas_c_luinverse_w
#define armas_x_cholinverse              armas_c_cholinverse
#define armas_x_cholinverse_w            armas_c_cholinverse_w
#define armas_x_ldlinverse               armas_c_ldlinverse
#define armas_x_ldlinverse_w             armas_c_ldlinverse_w

// internal
#define armas_x_sym_eigen2x2             armas_c_sym_eigen2x2
#define armas_x_sym_eigen2x2vec          armas_c_sym_eigen2x2vec
#define armas_x_trdevd_qr                armas_c_trdevd_qr

// Sorting vectors (internal)
#define armas_x_pivot_sort               armas_c_pivot_sort
#define armas_x_eigen_sort               armas_c_eigen_sort
#define armas_x_abs_sort_vec             armas_c_abs_sort_vec
#define armas_x_sort_vec                 armas_c_sort_vec
#define armas_x_sort_eigenvec            armas_c_sort_eigenvec

// Additional
#define armas_x_qdroots                  armas_c_qdroots
#define armas_x_discriminant             armas_c_discriminant
#define armas_x_mult_diag                armas_c_mult_diag
#define armas_x_solve_diag               armas_c_solve_diag
#define armas_x_scale_to                 armas_c_scale_to

// common
#if defined(armas_x_update_lq_left) && defined(armas_x_update_lq_right) && defined(armas_x_unblk_lq_reflector)
#define armas_x_update_lq 1
#endif

#if defined(armas_x_update_ql_left) && defined(armas_x_update_ql_right) && defined(armas_x_unblk_ql_reflector)
#define armas_x_update_ql 1
#endif

#if defined(armas_x_update_qr_left) && defined(armas_x_update_qr_right) && defined(armas_x_unblk_qr_reflector)
#define armas_x_update_qr 1
#endif

#if defined(armas_x_update_rq_left) && defined(armas_x_update_rq_right) && defined(armas_x_unblk_rq_reflector)
#define armas_x_update_rq 1
#endif

#endif  /* 0 */
#endif  /* ARMAS_NAMES_LAPACK_H */
#endif  /* DOXYGEN */
