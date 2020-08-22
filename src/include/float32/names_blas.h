
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __DOXYGEN

#ifndef ARMAS_NAMES_BLAS_H
#define ARMAS_NAMES_BLAS_H 1

// public functions: blas level 3, matrix-matrix
#define armas_x_mult        armas_s_mult
#define armas_x_mult_sym    armas_s_mult_sym
#define armas_x_mult_trm    armas_s_mult_trm
#define armas_x_solve_trm   armas_s_solve_trm
#define armas_x_update_sym  armas_s_update_sym
#define armas_x_update2_sym armas_s_update2_sym
#define armas_x_update_trm  armas_s_update_trm

// marker for blas level 3
#define armas_x_blas3   1

// public functions: blas level 2, matrix-vector
#define armas_x_mvmult        armas_s_mvmult
#define armas_x_mvmult_sym    armas_s_mvmult_sym
#define armas_x_mvmult_trm    armas_s_mvmult_trm
#define armas_x_mvsolve_trm   armas_s_mvsolve_trm
#define armas_x_mvupdate      armas_s_mvupdate
#define armas_x_mvupdate_trm  armas_s_mvupdate_trm
#define armas_x_mvupdate_sym  armas_s_mvupdate_sym
#define armas_x_mvupdate2_sym armas_s_mvupdate2_sym
// marker for blas level 2
#define armas_x_blas2   1

#define armas_x_mvmult_diag   armas_s_mvmult_diag
#define armas_x_mvsolve_diag  armas_s_mvsolve_diag

// public functions: blas level 1, vector-vector
#define armas_x_nrm2    armas_s_nrm2
#define armas_x_asum    armas_s_asum
#define armas_x_scale   armas_s_scale
#define armas_x_iamax   armas_s_iamax
#define armas_x_dot     armas_s_dot
#define armas_x_adot    armas_s_adot
#define armas_x_axpy    armas_s_axpy
#define armas_x_axpby   armas_s_axpby
#define armas_x_swap    armas_s_swap
#define armas_x_copy    armas_s_copy
// marker for blas level 1
#define armas_x_blas1   1
//
#define armas_x_invscale armas_s_invscale
#define armas_x_sum      armas_s_sum
#define armas_x_amax     armas_s_amax
#define armas_x_add      armas_s_add
#define armas_x_iamin    armas_s_iamin

#if defined(armas_x_blas1) && defined(armas_x_blas2) && defined(armas_x_blas3)
#define armas_x_blas 1
#endif

// extended precision functions
#define armas_x_ext_adot                 armas_s_ext_adot
#define armas_x_ext_axpby                armas_s_ext_axpby
#define armas_x_ext_sum                  armas_s_ext_sum
#define armas_x_ext_asum                 armas_s_ext_asum
#define armas_x_ext_mvmult               armas_s_ext_mvmult
#define armas_x_ext_mvupdate             armas_s_ext_mvupdate
#define armas_x_ext_mvupdate_sym         armas_s_ext_mvupdate_sym
#define armas_x_ext_mvupdate2_sym        armas_s_ext_mvupdate2_sym
#define armas_x_ext_mvmult_trm           armas_s_ext_mvmult_trm
#define armas_x_ext_mvmult_sym           armas_s_ext_mvmult_sym
#define armas_x_ext_mvsolve_trm_w        armas_s_ext_mvsolve_trm_w
#define armas_x_ext_mvsolve_trm          armas_s_ext_mvsolve_trm
#define armas_x_ext_mult                 armas_s_ext_mult
#define armas_x_ext_mult_sym             armas_s_ext_mult_sym
#define armas_x_ext_update_trm           armas_s_ext_update_trm
#define armas_x_ext_update_sym           armas_s_ext_update_sym
#define armas_x_ext_update2_sym          armas_s_ext_update2_sym
#define armas_x_ext_mult_trm             armas_s_ext_mult_trm
#define armas_x_ext_solve_trm_w          armas_s_ext_solve_trm_w
#define armas_x_ext_solve_trm            armas_s_ext_solve_trm

// unsafe functions and other internal functions
#define armas_x_dot_unsafe               armas_s_dot_unsafe
#define armas_x_adot_unsafe              armas_s_adot_unsafe
#define armas_x_axpby_unsafe             armas_s_axpby_unsafe
#define armas_x_scale_unsafe             armas_s_scale_unsafe
#define armas_x_mvmult_trm_unsafe        armas_s_mvmult_trm_unsafe
#define armas_x_mvmult_unsafe            armas_s_mvmult_unsafe
#define armas_x_mvupdate_unsafe          armas_s_mvupdate_unsafe
#define armas_x_mvupdate_unb             armas_s_mvupdate_unb
#define armas_x_mvupdate_rec             armas_s_mvupdate_rec
#define armas_x_mult_kernel              armas_s_mult_kernel
#define armas_x_mult_kernel_nc           armas_s_mult_kernel_nc
#define armas_x_mult_kernel_inner        armas_s_mult_kernel_inner
#define armas_x_mult_sym_unsafe          armas_s_mult_sym_unsafe
#define armas_x_trmm_unb                 armas_s_trmm_unb
#define armas_x_trmm_blk                 armas_s_trmm_blk
#define armas_x_trmm_recursive           armas_s_trmm_recursive
#define armas_x_mult_trm_unsafe          armas_s_mult_trm_unsafe
#define armas_x_solve_recursive          armas_s_solve_recursive
#define armas_x_solve_unb                armas_s_solve_unb
#define armas_x_solve_blocked            armas_s_solver_blocked
#define armas_x_solve_trm_unsafe         armas_s_solve_trm_unsafe
#define armas_x_mvupdate_trm_unb         armas_s_mvupdate_trm_unb
#define armas_x_mvupdate_trm_rec         armas_s_mvupdate_trm_rec
#define armas_x_mvsolve_trm_unsafe       armas_s_mvsolve_trm_unsafe

// unsafe extended precision
#define armas_x_merge_unsafe             armas_s_merge_unsafe
#define armas_x_merge2_unsafe            armas_s_merge2_unsafe
#define armas_x_ext_mult_inner           armas_s_ext_mult_inner
#define armas_x_ext_mult_kernel          armas_s_ext_mult_kernel
#define armas_x_ext_mult_kernel_nc       armas_s_ext_mult_kernel_nc
#define armas_x_ext_panel_dB_unsafe      armas_s_ext_panel_dB_unsafe
#define armas_x_ext_panel_dA_unsafe      armas_s_ext_panel_dA_unsafe
#define armas_x_ext_panel_unsafe         armas_s_ext_panel_unsafe
#define armas_x_ext_scale_unsafe         armas_s_ext_scale_unsafe
#define armas_x_ext_dot_unsafe           armas_s_ext_dot_unsafe
#define armas_x_ext_adot_unsafe          armas_s_ext_adot_unsafe
#define armas_x_ext_adot_dx_unsafe       armas_s_ext_adot_dx_unsafe
#define armas_x_ext_axpy_unsafe          armas_s_ext_axpy_unsafe
#define armas_x_ext_axpby_unsafe         armas_s_ext_axpby_unsafe
#define armas_x_ext_axpby_dx_unsafe      armas_s_ext_axpby_dx_unsafe
#define armas_x_ext_asum_unsafe          armas_s_ext_asum_unsafe
#define armas_x_ext_sum_unsafe           armas_s_ext_sum_unsafe
#define armas_x_ext_mvmult_unsafe        armas_s_ext_mvmult_unsafe
#define armas_x_ext_mvmult_dx_unsafe     armas_s_ext_mvmult_dx_unsafe
#define armas_x_ext_mvmult_sym_unsafe    armas_s_ext_mvmult_sym_unsafe
#define armas_x_ext_mvmult_trm_unsafe    armas_s_ext_mvmult_trm_unsafe
#define armas_x_ext_mvsolve_trm_unsafe   armas_s_ext_mvsolve_trm_unsafe
#define armas_x_ext_mvupdate_unsafe      armas_s_ext_mvupdate_unsafe
#define armas_x_ext_mvupdate_trm_unsafe  armas_s_ext_mvupdate_trm_unsafe
#define armas_x_ext_mvupdate2_sym_unsafe armas_s_ext_mvupdate2_sym_unsafe
#define armas_x_ext_solve_trm_unb_unsafe armas_s_ext_solve_trm_unb_unsafe
#define armas_x_ext_solve_trm_blk_unsafe armas_s_ext_solve_trm_blk_unsafe
#define armas_x_ext_solve_trm_unsafe     armas_s_ext_solve_trm_unsafe
#define armas_x_ext_mult_trm_unsafe      armas_s_ext_mult_trm_unsafe


#endif  /* __ARMAS_DTYPE_H */

#endif /* __DOXYGEN */
