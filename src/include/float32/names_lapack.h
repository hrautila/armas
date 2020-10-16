
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef DOXYGEN
#ifndef ARMAS_NAMES_LAPACK_H
#define ARMAS_NAMES_LAPACK_H 1

#ifndef CONFIG_NOTYPENAMES

#include <float.h>

/* ---------------------------------------------------------------------------
 * Definitions for double precision real numbers. (FLOAT64)
 */

// internal helpers
#define armas_pivot_index             armas_s_pivot_index
// public pivot functions
#define armas_pivot_rows              armas_s_pivot_rows
#define armas_pivot_cols              armas_s_pivot_cols
#define armas_pivot                   armas_s_pivot
// marker for above function
#define armas_lapack_pivots  1


// Bidiagonal reduction
#define armas_bdreduce                armas_s_bdreduce
#define armas_bdreduce_w              armas_s_bdreduce_w
#define armas_bdmult                  armas_s_bdmult
#define armas_bdmult_w                armas_s_bdmult_w
#define armas_bdbuild                 armas_s_bdbuild
#define armas_bdbuild_w               armas_s_bdbuild_w

// Symmetric LDL functions
#define armas_bkfactor                 armas_s_bkfactor
#define armas_bkfactor_w               armas_s_bkfactor_w
#define armas_bksolve                  armas_s_bksolve
#define armas_bksolve_w                armas_s_bksolve_w
#define armas_ldlfactor                armas_s_ldlfactor
#define armas_ldlfactor_w              armas_s_ldlfactor_w
#define armas_ldlupdate                armas_s_ldlupdate
#define armas_ldlsolve                 armas_s_ldlsolve

// visible internal functions
#define armas_unblk_bkfactor_lower     armas_s_unblk_bkfactor_lower
#define armas_unblk_bkfactor_upper     armas_s_unblk_bkfactor_upper
#define armas_unblk_bksolve_lower      armas_s_unblk_bksolve_lower
#define armas_unblk_bksolve_upper      armas_s_unblk_bksolve_upper
#define armas_blk_bkfactor_lower       armas_s_blk_bkfactor_lower
#define armas_blk_bkfactor_upper       armas_s_blk_bkfactor_upper
// marker
#define armas_ldlbk 1

// Cholesky 
#define armas_cholesky                 armas_s_cholesky
#define armas_cholfactor               armas_s_cholfactor
#define armas_cholfactor_w             armas_s_cholfactor_w
#define armas_cholupdate               armas_s_cholupdate
#define armas_cholsolve                armas_s_cholsolve
// internal pivoting versions
#define armas_cholfactor_pv            armas_s_cholfactor_pv
#define armas_cholsolve_pv             armas_s_cholsolve_pv

// Hessenberg functions
#define armas_hessreduce               armas_s_hessreduce
#define armas_hessreduce_w             armas_s_hessreduce_w
#define armas_hessmult                 armas_s_hessmult
#define armas_hessmult_w               armas_s_hessmult_w

// householder functions
#define armas_compute_householder      armas_s_compute_householder
#define armas_compute_householder_vec  armas_s_compute_householder_vec
#define armas_compute_householder_rev  armas_s_compute_householder_rev
#define armas_apply_householder2x1     armas_s_apply_householder2x1
#define armas_apply_householder1x1     armas_s_apply_householder1x1
#define armas_house                    armas_s_house
#define armas_houseapply               armas_s_houseapply
#define armas_houseapply2x1            armas_s_houseapply2x1
#define armas_housemult                armas_s_housemult
#define armas_housemult_w              armas_s_housemult_w
#define armas_hhouse                   armas_s_hhouse
#define armas_hhouse_apply             armas_s_hhouse_apply
// marker for householder functions
#define armas_householder              1

// LQ functions
#define armas_lqbuild                  armas_s_lqbuild
#define armas_lqbuild_w                armas_s_lqbuild_w
#define armas_lqfactor                 armas_s_lqfactor
#define armas_lqfactor_w               armas_s_lqfactor_w
#define armas_lqmult                   armas_s_lqmult
#define armas_lqmult_w                 armas_s_lqmult_w
#define armas_lqreflector              armas_s_lqreflector
#define armas_lqsolve                  armas_s_lqsolve
#define armas_lqsolve_w                armas_s_lqsolve_w
// internal LQ related functions available for others
#define armas_update_lq_left           armas_s_update_lq_left
#define armas_update_lq_right          armas_s_update_lq_right
#define armas_unblk_lq_reflector       armas_s_unblk_lq_reflector

// LU functions
#define armas_lufactor                 armas_s_lufactor
#define armas_lusolve                  armas_s_lusolve

// QL functions
#define armas_qlbuild                  armas_s_qlbuild
#define armas_qlbuild_w                armas_s_qlbuild_w
#define armas_qlfactor                 armas_s_qlfactor
#define armas_qlfactor_w               armas_s_qlfactor_w
#define armas_qlmult                   armas_s_qlmult
#define armas_qlmult_w                 armas_s_qlmult_w
#define armas_qlreflector              armas_s_qlreflector
#define armas_qlsolve                  armas_s_qlsolve
#define armas_qlsolve_w                armas_s_qlsolve_w
// internal QL related function available for others
#define armas_update_ql_left           armas_s_update_ql_left
#define armas_update_ql_right          armas_s_update_ql_right
#define armas_unblk_ql_reflector         armas_s_unblk_ql_reflector

// QR functions
#define armas_qrfactor                 armas_s_qrfactor
#define armas_qrfactor_w               armas_s_qrfactor_w
#define armas_qrmult                   armas_s_qrmult
#define armas_qrmult_w                 armas_s_qrmult_w
#define armas_qrbuild                  armas_s_qrbuild
#define armas_qrbuild_w                armas_s_qrbuild_w
#define armas_qrreflector              armas_s_qrreflector
#define armas_qrsolve                  armas_s_qrsolve
#define armas_qrsolve_w                armas_s_qrsolve_w
// internal QR related function available for others
#define armas_update_qr_left           armas_s_update_qr_left
#define armas_update_qr_right          armas_s_update_qr_right
#define armas_unblk_qr_reflector       armas_s_unblk_qr_reflector
#define armas_unblk_qrfactor           armas_s_unblk_qrfactor

// QRT functions
#define armas_qrtfactor                armas_s_qrtfactor
#define armas_qrtfactor_w              armas_s_qrtfactor_w
#define armas_qrtfactor_work           armas_s_qrtfactor_work
#define armas_qrtmult                  armas_s_qrtmult
#define armas_qrtmult_w                armas_s_qrtmult_w
#define armas_qrtmult_work             armas_s_qrtmult_work

// RQ functions
#define armas_rqbuild                  armas_s_rqbuild
#define armas_rqbuild_w                armas_s_rqbuild_w
#define armas_rqfactor                 armas_s_rqfactor
#define armas_rqfactor_w               armas_s_rqfactor_w
#define armas_rqmult                   armas_s_rqmult
#define armas_rqmult_w                 armas_s_rqmult_w
#define armas_rqreflector              armas_s_rqreflector
#define armas_rqsolve                  armas_s_rqsolve
#define armas_rqsolve_w                armas_s_rqsolve_w
// internal RQ related function available for others
#define armas_update_rq_left           armas_s_update_rq_left
#define armas_update_rq_right          armas_s_update_rq_right
#define armas_unblk_rq_reflector       armas_s_unblk_rq_reflector

// Tridiagonalization functions
#define armas_trdreduce                armas_s_trdreduce
#define armas_trdreduce_w              armas_s_trdreduce_w
#define armas_trdbuild                 armas_s_trdbuild
#define armas_trdbuild_w               armas_s_trdbuild_w
#define armas_trdmult                  armas_s_trdmult
#define armas_trdmult_w                armas_s_trdmult_w
// Tridiagonal EVD
#define armas_trdeigen                 armas_s_trdeigen
#define armas_trdeigen_w               armas_s_trdeigen_w
#define armas_trdbisect                armas_s_trdbisect
#define armas_trdsec_solve             armas_s_trdsec_solve
#define armas_trdsec_eigen             armas_s_trdsec_eigen
#define armas_trdsec_solve_vec         armas_s_trdsec_solve_vec

// Eigenvalue
#define armas_eigen_sym                armas_s_eigen_sym
#define armas_eigen_sym_w              armas_s_eigen_sym_w
#define armas_eigen_sym_selected       armas_s_eigen_sym_selected
#define armas_eigen_sym_selected_w     armas_s_eigen_sym_selected_w

// Givens
#define armas_gvcompute                armas_s_gvcompute
#define armas_gvrotate                 armas_s_gvrotate
#define armas_gvrot_vec                armas_s_gvrot_vec
#define armas_gvleft                   armas_s_gvleft
#define armas_gvright                  armas_s_gvright
#define armas_gvupdate                 armas_s_gvupdate

// Bidiagonal SVD
#define armas_bdsvd                    armas_s_bdsvd
#define armas_bdsvd_w                  armas_s_bdsvd_w
#define armas_dqds                     armas_s_dqds
#define armas_dqds_w                   armas_s_dqds_w

// internal
#define armas_bdsvd2x2                 armas_s_bdsvd2x2
#define armas_bdsvd2x2_vec             armas_s_bdsvd2x2_vec
#define armas_bdsvd_golub		         armas_s_bdsvd_golub
#define armas_bdsvd_demmel		     armas_s_bdsvd_demmel

// SVD
#define armas_svd		                 armas_s_svd
#define armas_svd_w                    armas_s_svd_w

// Butterfly
#define armas_gen_rbt                  armas_s_gen_rbt
#define armas_size_rbt                 armas_s_size_rbt
#define armas_mult_rbt                 armas_s_mult_rbt
#define armas_update2_rbt              armas_s_update2_rbt
#define armas_update2_sym_rbt          armas_s_update2_sym_rbt
#define armas_update2_rbt_descend      armas_s_update2_rbt_descend

// inverse
#define armas_inverse_trm              armas_s_inverse_trm
#define armas_luinverse                armas_s_luinverse
#define armas_luinverse_w              armas_s_luinverse_w
#define armas_cholinverse              armas_s_cholinverse
#define armas_cholinverse_w            armas_s_cholinverse_w
#define armas_ldlinverse               armas_s_ldlinverse
#define armas_ldlinverse_w             armas_s_ldlinverse_w

// internal
#define armas_sym_eigen2x2             armas_s_sym_eigen2x2
#define armas_sym_eigen2x2vec          armas_s_sym_eigen2x2vec
#define armas_trdevd_qr                armas_s_trdevd_qr

// Sorting vectors (internal)
#define armas_pivot_sort               armas_s_pivot_sort
#define armas_eigen_sort               armas_s_eigen_sort
#define armas_abs_sort_vec             armas_s_abs_sort_vec
#define armas_sort_vec                 armas_s_sort_vec
#define armas_sort_eigenvec            armas_s_sort_eigenvec

// Additional
#define armas_qdroots                  armas_s_qdroots
#define armas_discriminant             armas_s_discriminant
#define armas_mult_diag                armas_s_mult_diag
#define armas_solve_diag               armas_s_solve_diag
#define armas_scale_to                 armas_s_scale_to

// common
#if defined(armas_update_lq_left) && defined(armas_update_lq_right) && defined(armas_unblk_lq_reflector)
#define armas_update_lq 1
#endif

#if defined(armas_update_ql_left) && defined(armas_update_ql_right) && defined(armas_unblk_ql_reflector)
#define armas_update_ql 1
#endif

#if defined(armas_update_qr_left) && defined(armas_update_qr_right) && defined(armas_unblk_qr_reflector)
#define armas_update_qr 1
#endif

#if defined(armas_update_rq_left) && defined(armas_update_rq_right) && defined(armas_unblk_rq_reflector)
#define armas_update_rq 1
#endif

#endif  /* CONFIG_NOTYPENAMES */
#endif  /* ARMAS_NAMES_LAPACK_H */
#endif  /* DOXYGEN */
