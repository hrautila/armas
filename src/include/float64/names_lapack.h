
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
#define armas_pivot_index             armas_d_pivot_index
// public pivot functions
#define armas_pivot_rows              armas_d_pivot_rows
#define armas_pivot_cols              armas_d_pivot_cols
#define armas_pivot                   armas_d_pivot
// marker for above function
#define armas_lapack_pivots  1


// Bidiagonal reduction
#define armas_bdreduce                armas_d_bdreduce
#define armas_bdreduce_w              armas_d_bdreduce_w
#define armas_bdmult                  armas_d_bdmult
#define armas_bdmult_w                armas_d_bdmult_w
#define armas_bdbuild                 armas_d_bdbuild
#define armas_bdbuild_w               armas_d_bdbuild_w

// Symmetric LDL functions
#define armas_bkfactor                 armas_d_bkfactor
#define armas_bkfactor_w               armas_d_bkfactor_w
#define armas_bksolve                  armas_d_bksolve
#define armas_bksolve_w                armas_d_bksolve_w
#define armas_ldlfactor                armas_d_ldlfactor
#define armas_ldlfactor_np             armas_d_ldlfactor_np
#define armas_ldlfactor_w              armas_d_ldlfactor_w
#define armas_ldlupdate                armas_d_ldlupdate
#define armas_ldlsolve                 armas_d_ldlsolve
#define armas_ldlsolve_np              armas_d_ldlsolve_np

// visible internal functions
#define armas_unblk_bkfactor_lower     armas_d_unblk_bkfactor_lower
#define armas_unblk_bkfactor_upper     armas_d_unblk_bkfactor_upper
#define armas_unblk_bksolve_lower      armas_d_unblk_bksolve_lower
#define armas_unblk_bksolve_upper      armas_d_unblk_bksolve_upper
#define armas_blk_bkfactor_lower       armas_d_blk_bkfactor_lower
#define armas_blk_bkfactor_upper       armas_d_blk_bkfactor_upper
// marker
#define armas_ldlbk 1

// Cholesky 
#define armas_cholesky                 armas_d_cholesky
#define armas_cholfactor               armas_d_cholfactor
#define armas_cholfactor_w             armas_d_cholfactor_w
#define armas_cholupdate               armas_d_cholupdate
#define armas_cholsolve                armas_d_cholsolve
// internal pivoting versions
#define armas_cholfactor_pv            armas_d_cholfactor_pv
#define armas_cholsolve_pv             armas_d_cholsolve_pv

// Hessenberg functions
#define armas_hessreduce               armas_d_hessreduce
#define armas_hessreduce_w             armas_d_hessreduce_w
#define armas_hessmult                 armas_d_hessmult
#define armas_hessmult_w               armas_d_hessmult_w

// householder functions
#define armas_compute_householder      armas_d_compute_householder
#define armas_compute_householder_vec  armas_d_compute_householder_vec
#define armas_compute_householder_rev  armas_d_compute_householder_rev
#define armas_apply_householder2x1     armas_d_apply_householder2x1
#define armas_apply_householder1x1     armas_d_apply_householder1x1
#define armas_house                    armas_d_house
#define armas_house_vec                armas_d_house_vec
#define armas_houseapply               armas_d_houseapply
#define armas_houseapply2x1            armas_d_houseapply2x1
#define armas_housemult                armas_d_housemult
#define armas_housemult_w              armas_d_housemult_w
#define armas_hhouse                   armas_d_hhouse
#define armas_hhouse_apply             armas_d_hhouse_apply
// marker for householder functions
#define armas_householder              1

// LQ functions
#define armas_lqbuild                  armas_d_lqbuild
#define armas_lqbuild_w                armas_d_lqbuild_w
#define armas_lqfactor                 armas_d_lqfactor
#define armas_lqfactor_w               armas_d_lqfactor_w
#define armas_lqmult                   armas_d_lqmult
#define armas_lqmult_w                 armas_d_lqmult_w
#define armas_lqreflector              armas_d_lqreflector
#define armas_lqsolve                  armas_d_lqsolve
#define armas_lqsolve_w                armas_d_lqsolve_w
// internal LQ related functions available for others
#define armas_update_lq_left           armas_d_update_lq_left
#define armas_update_lq_right          armas_d_update_lq_right
#define armas_unblk_lq_reflector       armas_d_unblk_lq_reflector

// LU functions
#define armas_lufactor                 armas_d_lufactor
#define armas_lusolve                  armas_d_lusolve

// QL functions
#define armas_qlbuild                  armas_d_qlbuild
#define armas_qlbuild_w                armas_d_qlbuild_w
#define armas_qlfactor                 armas_d_qlfactor
#define armas_qlfactor_w               armas_d_qlfactor_w
#define armas_qlmult                   armas_d_qlmult
#define armas_qlmult_w                 armas_d_qlmult_w
#define armas_qlreflector              armas_d_qlreflector
#define armas_qlsolve                  armas_d_qlsolve
#define armas_qlsolve_w                armas_d_qlsolve_w
// internal QL related function available for others
#define armas_update_ql_left           armas_d_update_ql_left
#define armas_update_ql_right          armas_d_update_ql_right
#define armas_unblk_ql_reflector       armas_d_unblk_ql_reflector

// QR functions
#define armas_qrfactor                 armas_d_qrfactor
#define armas_qrfactor_w               armas_d_qrfactor_w
#define armas_qrmult                   armas_d_qrmult
#define armas_qrmult_w                 armas_d_qrmult_w
#define armas_qrbuild                  armas_d_qrbuild
#define armas_qrbuild_w                armas_d_qrbuild_w
#define armas_qrreflector              armas_d_qrreflector
#define armas_qrsolve                  armas_d_qrsolve
#define armas_qrsolve_w                armas_d_qrsolve_w
// internal QR related function available for others
#define armas_update_qr_left           armas_d_update_qr_left
#define armas_update_qr_right          armas_d_update_qr_right
#define armas_unblk_qr_reflector       armas_d_unblk_qr_reflector
#define armas_unblk_qrfactor           armas_d_unblk_qrfactor

// QRT functions
#define armas_qrtfactor                armas_d_qrtfactor
#define armas_qrtfactor_w              armas_d_qrtfactor_w
#define armas_qrtfactor_work           armas_d_qrtfactor_work
#define armas_qrtmult                  armas_d_qrtmult
#define armas_qrtmult_w                armas_d_qrtmult_w
#define armas_qrtmult_work             armas_d_qrtmult_work

// RQ functions
#define armas_rqbuild                  armas_d_rqbuild
#define armas_rqbuild_w                armas_d_rqbuild_w
#define armas_rqfactor                 armas_d_rqfactor
#define armas_rqfactor_w               armas_d_rqfactor_w
#define armas_rqmult                   armas_d_rqmult
#define armas_rqmult_w                 armas_d_rqmult_w
#define armas_rqreflector              armas_d_rqreflector
#define armas_rqsolve                  armas_d_rqsolve
#define armas_rqsolve_w                armas_d_rqsolve_w
// internal RQ related function available for others
#define armas_update_rq_left           armas_d_update_rq_left
#define armas_update_rq_right          armas_d_update_rq_right
#define armas_unblk_rq_reflector       armas_d_unblk_rq_reflector

// Tridiagonalization functions
#define armas_trdreduce                armas_d_trdreduce
#define armas_trdreduce_w              armas_d_trdreduce_w
#define armas_trdbuild                 armas_d_trdbuild
#define armas_trdbuild_w               armas_d_trdbuild_w
#define armas_trdmult                  armas_d_trdmult
#define armas_trdmult_w                armas_d_trdmult_w
// Tridiagonal EVD
#define armas_trdeigen                 armas_d_trdeigen
#define armas_trdeigen_w               armas_d_trdeigen_w
#define armas_trdbisect                armas_d_trdbisect
#define armas_trdsec_solve             armas_d_trdsec_solve
#define armas_trdsec_eigen             armas_d_trdsec_eigen
#define armas_trdsec_solve_vec         armas_d_trdsec_solve_vec

// Eigenvalue
#define armas_eigen_sym                armas_d_eigen_sym
#define armas_eigen_sym_w              armas_d_eigen_sym_w
#define armas_eigen_sym_selected       armas_d_eigen_sym_selected
#define armas_eigen_sym_selected_w     armas_d_eigen_sym_selected_w

// Givens
#define armas_gvcompute                armas_d_gvcompute
#define armas_gvrotate                 armas_d_gvrotate
#define armas_gvrot_vec                armas_d_gvrot_vec
#define armas_gvleft                   armas_d_gvleft
#define armas_gvright                  armas_d_gvright
#define armas_gvupdate                 armas_d_gvupdate

// Bidiagonal SVD
#define armas_bdsvd                    armas_d_bdsvd
#define armas_bdsvd_w                  armas_d_bdsvd_w
#define armas_dqds                     armas_d_dqds
#define armas_dqds_w                   armas_d_dqds_w

// internal
#define armas_bdsvd2x2                 armas_d_bdsvd2x2
#define armas_bdsvd2x2_vec             armas_d_bdsvd2x2_vec
#define armas_bdsvd_golub		         armas_d_bdsvd_golub
#define armas_bdsvd_demmel		     armas_d_bdsvd_demmel
#define armas_bd_qrsweep               armas_d_bd_qrsweep
#define armas_bd_qlsweep               armas_d_bd_qlsweep
#define armas_bd_qrzero                armas_d_bd_qrzero
#define armas_bd_qlzero                armas_d_bd_qlzero
#define armas_trd_qrsweep              armas_d_trd_qrsweep
#define armas_trd_qlsweep              armas_d_trd_qlsweep

// SVD
#define armas_svd		                 armas_d_svd
#define armas_svd_w                    armas_d_svd_w

// Butterfly
#define armas_gen_rbt                  armas_d_gen_rbt
#define armas_size_rbt                 armas_d_size_rbt
#define armas_mult_rbt                 armas_d_mult_rbt
#define armas_update2_rbt              armas_d_update2_rbt
#define armas_update2_sym_rbt          armas_d_update2_sym_rbt
#define armas_update2_rbt_descend      armas_d_update2_rbt_descend

// inverse
#define armas_inverse_trm              armas_d_inverse_trm
#define armas_luinverse                armas_d_luinverse
#define armas_luinverse_w              armas_d_luinverse_w
#define armas_cholinverse              armas_d_cholinverse
#define armas_cholinverse_w            armas_d_cholinverse_w
#define armas_ldlinverse               armas_d_ldlinverse
#define armas_ldlinverse_w             armas_d_ldlinverse_w

// internal
#define armas_sym_eigen2x2             armas_d_sym_eigen2x2
#define armas_sym_eigen2x2vec          armas_d_sym_eigen2x2vec
#define armas_trdevd_qr                armas_d_trdevd_qr

// Sorting vectors (internal)
#define armas_pivot_sort               armas_d_pivot_sort
#define armas_eigen_sort               armas_d_eigen_sort
#define armas_abs_sort_vec             armas_d_abs_sort_vec
#define armas_sort_vec                 armas_d_sort_vec
#define armas_sort_eigenvec            armas_d_sort_eigenvec

// Additional
#define armas_qdroots                  armas_d_qdroots
#define armas_discriminant             armas_d_discriminant
#define armas_mult_diag                armas_d_mult_diag
#define armas_solve_diag               armas_d_solve_diag
#define armas_scale_to                 armas_d_scale_to

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
