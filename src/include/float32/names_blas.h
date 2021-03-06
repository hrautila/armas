
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_BLAS_H
#define ARMAS_NAMES_BLAS_H 1

#ifndef CONFIG_NOTYPENAMES

// public functions: blas level 3, matrix-matrix
#define armas_mult        armas_s_mult
#define armas_mult_sym    armas_s_mult_sym
#define armas_mult_trm    armas_s_mult_trm
#define armas_solve_trm   armas_s_solve_trm
#define armas_update_sym  armas_s_update_sym
#define armas_update2_sym armas_s_update2_sym
#define armas_update_trm  armas_s_update_trm

// marker for blas level 3
#define armas_blas3   1

// public functions: blas level 2, matrix-vector
#define armas_mvmult        armas_s_mvmult
#define armas_mvmult_sym    armas_s_mvmult_sym
#define armas_mvmult_trm    armas_s_mvmult_trm
#define armas_mvsolve_trm   armas_s_mvsolve_trm
#define armas_mvupdate      armas_s_mvupdate
#define armas_mvupdate_trm  armas_s_mvupdate_trm
#define armas_mvupdate_sym  armas_s_mvupdate_sym
#define armas_mvupdate2_sym armas_s_mvupdate2_sym
// marker for blas level 2
#define armas_blas2   1

#define armas_mvmult_diag   armas_s_mvmult_diag
#define armas_mvsolve_diag  armas_s_mvsolve_diag

// public functions: blas level 1, vector-vector
#define armas_nrm2    armas_s_nrm2
#define armas_asum    armas_s_asum
#define armas_scale   armas_s_scale
#define armas_iamax   armas_s_iamax
#define armas_dot     armas_s_dot
#define armas_adot    armas_s_adot
#define armas_axpy    armas_s_axpy
#define armas_axpby   armas_s_axpby
#define armas_swap    armas_s_swap
#define armas_copy    armas_s_copy
// marker for blas level 1
#define armas_blas1   1
//
#define armas_invscale armas_s_invscale
#define armas_sum      armas_s_sum
#define armas_amax     armas_s_amax
#define armas_add      armas_s_add
#define armas_iamin    armas_s_iamin

#if defined(armas_blas1) && defined(armas_blas2) && defined(armas_blas3)
#define armas_blas 1
#endif

// extended precision functions
#define armas_ext_adot                 armas_s_ext_adot
#define armas_ext_axpby                armas_s_ext_axpby
#define armas_ext_sum                  armas_s_ext_sum
#define armas_ext_asum                 armas_s_ext_asum
#define armas_ext_mvmult               armas_s_ext_mvmult
#define armas_ext_mvupdate             armas_s_ext_mvupdate
#define armas_ext_mvupdate_sym         armas_s_ext_mvupdate_sym
#define armas_ext_mvupdate2_sym        armas_s_ext_mvupdate2_sym
#define armas_ext_mvmult_trm           armas_s_ext_mvmult_trm
#define armas_ext_mvmult_sym           armas_s_ext_mvmult_sym
#define armas_ext_mvsolve_trm_w        armas_s_ext_mvsolve_trm_w
#define armas_ext_mvsolve_trm          armas_s_ext_mvsolve_trm
#define armas_ext_mult                 armas_s_ext_mult
#define armas_ext_mult_sym             armas_s_ext_mult_sym
#define armas_ext_update_trm           armas_s_ext_update_trm
#define armas_ext_update_sym           armas_s_ext_update_sym
#define armas_ext_update2_sym          armas_s_ext_update2_sym
#define armas_ext_mult_trm             armas_s_ext_mult_trm
#define armas_ext_solve_trm_w          armas_s_ext_solve_trm_w
#define armas_ext_solve_trm            armas_s_ext_solve_trm

// unsafe functions and other internal functions
#define armas_dot_unsafe               armas_s_dot_unsafe
#define armas_adot_unsafe              armas_s_adot_unsafe
#define armas_axpby_unsafe             armas_s_axpby_unsafe
#define armas_scale_unsafe             armas_s_scale_unsafe
#define armas_mvmult_trm_unsafe        armas_s_mvmult_trm_unsafe
#define armas_mvmult_unsafe            armas_s_mvmult_unsafe
#define armas_mvupdate_unsafe          armas_s_mvupdate_unsafe
#define armas_mvupdate_unb             armas_s_mvupdate_unb
#define armas_mvupdate_rec             armas_s_mvupdate_rec
#define armas_mult_kernel              armas_s_mult_kernel
#define armas_mult_kernel_nc           armas_s_mult_kernel_nc
#define armas_mult_kernel_inner        armas_s_mult_kernel_inner
#define armas_mult_sym_unsafe          armas_s_mult_sym_unsafe
#define armas_trmm_unb                 armas_s_trmm_unb
#define armas_trmm_blk                 armas_s_trmm_blk
#define armas_trmm_recursive           armas_s_trmm_recursive
#define armas_mult_trm_unsafe          armas_s_mult_trm_unsafe
#define armas_solve_recursive          armas_s_solve_recursive
#define armas_solve_unb                armas_s_solve_unb
#define armas_solve_blocked            armas_s_solver_blocked
#define armas_solve_trm_unsafe         armas_s_solve_trm_unsafe
#define armas_mvupdate_trm_unb         armas_s_mvupdate_trm_unb
#define armas_mvupdate_trm_rec         armas_s_mvupdate_trm_rec
#define armas_mvsolve_trm_unsafe       armas_s_mvsolve_trm_unsafe

// unsafe extended precision
#define armas_merge_unsafe             armas_s_merge_unsafe
#define armas_merge2_unsafe            armas_s_merge2_unsafe
#define armas_ext_mult_inner           armas_s_ext_mult_inner
#define armas_ext_mult_kernel          armas_s_ext_mult_kernel
#define armas_ext_mult_kernel_nc       armas_s_ext_mult_kernel_nc
#define armas_ext_panel_dB_unsafe      armas_s_ext_panel_dB_unsafe
#define armas_ext_panel_dA_unsafe      armas_s_ext_panel_dA_unsafe
#define armas_ext_panel_unsafe         armas_s_ext_panel_unsafe
#define armas_ext_scale_unsafe         armas_s_ext_scale_unsafe
#define armas_ext_dot_unsafe           armas_s_ext_dot_unsafe
#define armas_ext_adot_unsafe          armas_s_ext_adot_unsafe
#define armas_ext_adot_dx_unsafe       armas_s_ext_adot_dx_unsafe
#define armas_ext_axpy_unsafe          armas_s_ext_axpy_unsafe
#define armas_ext_axpby_unsafe         armas_s_ext_axpby_unsafe
#define armas_ext_axpby_dx_unsafe      armas_s_ext_axpby_dx_unsafe
#define armas_ext_asum_unsafe          armas_s_ext_asum_unsafe
#define armas_ext_sum_unsafe           armas_s_ext_sum_unsafe
#define armas_ext_mvmult_unsafe        armas_s_ext_mvmult_unsafe
#define armas_ext_mvmult_dx_unsafe     armas_s_ext_mvmult_dx_unsafe
#define armas_ext_mvmult_sym_unsafe    armas_s_ext_mvmult_sym_unsafe
#define armas_ext_mvmult_trm_unsafe    armas_s_ext_mvmult_trm_unsafe
#define armas_ext_mvsolve_trm_unsafe   armas_s_ext_mvsolve_trm_unsafe
#define armas_ext_mvupdate_unsafe      armas_s_ext_mvupdate_unsafe
#define armas_ext_mvupdate_trm_unsafe  armas_s_ext_mvupdate_trm_unsafe
#define armas_ext_mvupdate2_sym_unsafe armas_s_ext_mvupdate2_sym_unsafe
#define armas_ext_solve_trm_unb_unsafe armas_s_ext_solve_trm_unb_unsafe
#define armas_ext_solve_trm_blk_unsafe armas_s_ext_solve_trm_blk_unsafe
#define armas_ext_solve_trm_unsafe     armas_s_ext_solve_trm_unsafe
#define armas_ext_mult_trm_unsafe      armas_s_ext_mult_trm_unsafe

#endif  /* CONFIG_NOTYPENAMES */
#endif  /* ARMAS_NAMES_BLAS_H */

#endif /* __DOXYGEN */
