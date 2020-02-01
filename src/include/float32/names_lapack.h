
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef DOXYGEN
#ifndef ARMAS_NAMES_LAPACK_H
#define ARMAS_NAMES_LAPACK_H 1

#include <float.h>

/* ---------------------------------------------------------------------------
 * Definitions for double precision real numbers. (FLOAT64)
 */

#define SAFEMIN     FLT_MIN
// machine accurancy as in LAPACK library 
#define EPS         (FLT_EPSILON/2.0)

// internal helpers
#define armas_x_pivot_index             armas_s_pivot_index
// public pivot functions
#define armas_x_pivot_rows              armas_s_pivot_rows
#define armas_x_pivot_cols              armas_s_pivot_cols
#define armas_x_pivot                   armas_s_pivot
// marker for above function
#define armas_x_lapack_pivots  1


// Bidiagonal reduction
#define armas_x_bdreduce                armas_s_bdreduce
#define armas_x_bdreduce_w              armas_s_bdreduce_w
#define armas_x_bdmult                  armas_s_bdmult
#define armas_x_bdmult_w                armas_s_bdmult_w
#define armas_x_bdbuild                 armas_s_bdbuild
#define armas_x_bdbuild_w               armas_s_bdbuild_w

// Symmetric LDL functions
#define armas_x_bkfactor                 armas_s_bkfactor
#define armas_x_bkfactor_w               armas_s_bkfactor_w
#define armas_x_bksolve                  armas_s_bksolve
#define armas_x_bksolve_w                armas_s_bksolve_w
#define armas_x_ldlfactor                armas_s_ldlfactor
#define armas_x_ldlfactor_w              armas_s_ldlfactor_w
#define armas_x_ldlupdate                armas_s_ldlupdate
#define armas_x_ldlsolve                 armas_s_ldlsolve

// visible internal functions
#define armas_x_unblk_bkfactor_lower     armas_s_unblk_bkfactor_lower
#define armas_x_unblk_bkfactor_upper     armas_s_unblk_bkfactor_upper
#define armas_x_unblk_bksolve_lower      armas_s_unblk_bksolve_lower
#define armas_x_unblk_bksolve_upper      armas_s_unblk_bksolve_upper
#define armas_x_blk_bkfactor_lower       armas_s_blk_bkfactor_lower
#define armas_x_blk_bkfactor_upper       armas_s_blk_bkfactor_upper
// marker
#define armas_x_ldlbk 1

// Cholesky 
#define armas_x_cholesky                 armas_s_cholesky
#define armas_x_cholfactor               armas_s_cholfactor
#define armas_x_cholfactor_w             armas_s_cholfactor_w
#define armas_x_cholupdate               armas_s_cholupdate
#define armas_x_cholsolve                armas_s_cholsolve
// internal pivoting versions
#define armas_x_cholfactor_pv            armas_s_cholfactor_pv
#define armas_x_cholsolve_pv             armas_s_cholsolve_pv

// Hessenberg functions
#define armas_x_hessreduce               armas_s_hessreduce
#define armas_x_hessreduce_w             armas_s_hessreduce_w
#define armas_x_hessmult                 armas_s_hessmult
#define armas_x_hessmult_w               armas_s_hessmult_w

// householder functions
#define armas_x_compute_householder      armas_s_compute_householder
#define armas_x_compute_householder_vec  armas_s_compute_householder_vec
#define armas_x_compute_householder_rev  armas_s_compute_householder_rev
#define armas_x_apply_householder2x1     armas_s_apply_householder2x1
#define armas_x_apply_householder1x1     armas_s_apply_householder1x1
#define armas_x_house                    armas_s_house
#define armas_x_houseapply               armas_s_houseapply
#define armas_x_houseapply2x1            armas_s_houseapply2x1
#define armas_x_housemult                armas_s_housemult
#define armas_x_housemult_w              armas_s_housemult_w
#define armas_x_hhouse                   armas_s_hhouse
#define armas_x_hhouse_apply             armas_s_hhouse_apply
// marker for householder functions
#define armas_x_householder              1

// LQ functions
#define armas_x_lqbuild                  armas_s_lqbuild
#define armas_x_lqbuild_w                armas_s_lqbuild_w
#define armas_x_lqfactor                 armas_s_lqfactor
#define armas_x_lqfactor_w               armas_s_lqfactor_w
#define armas_x_lqmult                   armas_s_lqmult
#define armas_x_lqmult_w                 armas_s_lqmult_w
#define armas_x_lqreflector              armas_s_lqreflector
#define armas_x_lqsolve                  armas_s_lqsolve
#define armas_x_lqsolve_w                armas_s_lqsolve_w
// internal LQ related functions available for others
#define armas_x_update_lq_left           armas_s_update_lq_left
#define armas_x_update_lq_right          armas_s_update_lq_right
#define armas_x_unblk_lq_reflector       armas_s_unblk_lq_reflector

// LU functions
#define armas_x_lufactor                 armas_s_lufactor
#define armas_x_lusolve                  armas_s_lusolve

// QL functions
#define armas_x_qlbuild                  armas_s_qlbuild
#define armas_x_qlbuild_w                armas_s_qlbuild_w
#define armas_x_qlfactor                 armas_s_qlfactor
#define armas_x_qlfactor_w               armas_s_qlfactor_w
#define armas_x_qlmult                   armas_s_qlmult
#define armas_x_qlmult_w                 armas_s_qlmult_w
#define armas_x_qlreflector              armas_s_qlreflector
#define armas_x_qlsolve                  armas_s_qlsolve
#define armas_x_qlsolve_w                armas_s_qlsolve_w
// internal QL related function available for others
#define armas_x_update_ql_left           armas_s_update_ql_left
#define armas_x_update_ql_right          armas_s_update_ql_right
#define armas_unblk_ql_reflector         armas_s_unblk_ql_reflector

// QR functions
#define armas_x_qrfactor                 armas_s_qrfactor
#define armas_x_qrfactor_w               armas_s_qrfactor_w
#define armas_x_qrmult                   armas_s_qrmult
#define armas_x_qrmult_w                 armas_s_qrmult_w
#define armas_x_qrbuild                  armas_s_qrbuild
#define armas_x_qrbuild_w                armas_s_qrbuild_w
#define armas_x_qrreflector              armas_s_qrreflector
#define armas_x_qrsolve                  armas_s_qrsolve
#define armas_x_qrsolve_w                armas_s_qrsolve_w
// internal QR related function available for others
#define armas_x_update_qr_left           armas_s_update_qr_left
#define armas_x_update_qr_right          armas_s_update_qr_right
#define armas_x_unblk_qr_reflector       armas_s_unblk_qr_reflector
#define armas_x_unblk_qrfactor           armas_s_unblk_qrfactor

// QRT functions
#define armas_x_qrtfactor                armas_s_qrtfactor
#define armas_x_qrtfactor_w              armas_s_qrtfactor_w
#define armas_x_qrtfactor_work           armas_s_qrtfactor_work
#define armas_x_qrtmult                  armas_s_qrtmult
#define armas_x_qrtmult_w                armas_s_qrtmult_w
#define armas_x_qrtmult_work             armas_s_qrtmult_work

// RQ functions
#define armas_x_rqbuild                  armas_s_rqbuild
#define armas_x_rqbuild_w                armas_s_rqbuild_w
#define armas_x_rqfactor                 armas_s_rqfactor
#define armas_x_rqfactor_w               armas_s_rqfactor_w
#define armas_x_rqmult                   armas_s_rqmult
#define armas_x_rqmult_w                 armas_s_rqmult_w
#define armas_x_rqreflector              armas_s_rqreflector
#define armas_x_rqsolve                  armas_s_rqsolve
#define armas_x_rqsolve_w                armas_s_rqsolve_w
// internal RQ related function available for others
#define armas_x_update_rq_left           armas_s_update_rq_left
#define armas_x_update_rq_right          armas_s_update_rq_right
#define armas_x_unblk_rq_reflector       armas_s_unblk_rq_reflector

// Tridiagonalization functions
#define armas_x_trdreduce                armas_s_trdreduce
#define armas_x_trdreduce_w              armas_s_trdreduce_w
#define armas_x_trdbuild                 armas_s_trdbuild
#define armas_x_trdbuild_w               armas_s_trdbuild_w
#define armas_x_trdmult                  armas_s_trdmult
#define armas_x_trdmult_w                armas_s_trdmult_w
// Tridiagonal EVD
#define armas_x_trdeigen                 armas_s_trdeigen
#define armas_x_trdeigen_w               armas_s_trdeigen_w
#define armas_x_trdbisect                armas_s_trdbisect
#define armas_x_trdsec_solve             armas_s_trdsec_solve
#define armas_x_trdsec_eigen             armas_s_trdsec_eigen
#define armas_x_trdsec_solve_vec         armas_s_trdsec_solve_vec

// Eigenvalue
#define armas_x_eigen_sym                armas_s_eigen_sym
#define armas_x_eigen_sym_w              armas_s_eigen_sym_w
#define armas_x_eigen_sym_selected       armas_s_eigen_sym_selected
#define armas_x_eigen_sym_selected_w     armas_s_eigen_sym_selected_w

// Givens
#define armas_x_gvcompute                armas_s_gvcompute
#define armas_x_gvrotate                 armas_s_gvrotate
#define armas_x_gvrot_vec                armas_s_gvrot_vec
#define armas_x_gvleft                   armas_s_gvleft
#define armas_x_gvright                  armas_s_gvright
#define armas_x_gvupdate                 armas_s_gvupdate

// Bidiagonal SVD
#define armas_x_bdsvd                    armas_s_bdsvd
#define armas_x_bdsvd_w                  armas_s_bdsvd_w
#define armas_x_dqds                     armas_s_dqds
#define armas_x_dqds_w                   armas_s_dqds_w

// internal
#define armas_x_bdsvd2x2                 armas_s_bdsvd2x2
#define armas_x_bdsvd2x2_vec             armas_s_bdsvd2x2_vec
#define armas_x_bdsvd_golub		         armas_s_bdsvd_golub
#define armas_x_bdsvd_demmel		     armas_s_bdsvd_demmel

// SVD
#define armas_x_svd		                 armas_s_svd
#define armas_x_svd_w                    armas_s_svd_w

// Butterfly
#define armas_x_gen_rbt                  armas_s_gen_rbt
#define armas_x_size_rbt                 armas_s_size_rbt
#define armas_x_mult_rbt                 armas_s_mult_rbt
#define armas_x_update2_rbt              armas_s_update2_rbt
#define armas_x_update2_sym_rbt          armas_s_update2_sym_rbt
#define armas_x_update2_rbt_descend      armas_s_update2_rbt_descend

// inverse
#define armas_x_inverse_trm              armas_s_inverse_trm
#define armas_x_luinverse                armas_s_luinverse
#define armas_x_luinverse_w              armas_s_luinverse_w
#define armas_x_cholinverse              armas_s_cholinverse
#define armas_x_cholinverse_w            armas_s_cholinverse_w
#define armas_x_ldlinverse               armas_s_ldlinverse
#define armas_x_ldlinverse_w             armas_s_ldlinverse_w

// internal
#define armas_x_sym_eigen2x2             armas_s_sym_eigen2x2
#define armas_x_sym_eigen2x2vec          armas_s_sym_eigen2x2vec
#define armas_x_trdevd_qr                armas_s_trdevd_qr

// Sorting vectors (internal)
#define armas_x_pivot_sort               armas_s_pivot_sort
#define armas_x_eigen_sort               armas_s_eigen_sort
#define armas_x_abs_sort_vec             armas_s_abs_sort_vec
#define armas_x_sort_vec                 armas_s_sort_vec
#define armas_x_sort_eigenvec            armas_s_sort_eigenvec

// Additional
#define armas_x_qdroots                  armas_s_qdroots
#define armas_x_discriminant             armas_s_discriminant
#define armas_x_mult_diag                armas_s_mult_diag
#define armas_x_solve_diag               armas_s_solve_diag
#define armas_x_scale_to                 armas_s_scale_to

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

#endif  /* ARMAS_NAMES_LAPACK_H */
#endif  /* DOXYGEN */
