
#ifndef _UNIT_TESTING_H
#define _UNIT_TESTING_H 1

#if defined(FLOAT32)
#include <armas/smatrix.h>
typedef armas_s_dense_t __Matrix ;
typedef float __Dtype;

#define matrix_init            armas_s_init
#define matrix_diag            armas_s_diag
#define matrix_madd            armas_s_madd
#define matrix_set_values      armas_s_set_values
#define matrix_copy            armas_s_copy
#define matrix_size            armas_s_size
#define matrix_set             armas_s_set
#define matrix_get             armas_s_get
#define matrix_set_at          armas_s_set_at
#define matrix_get_at          armas_s_get_at
#define matrix_submatrix       armas_s_submatrix
#define matrix_mult            armas_s_mult
#define matrix_mult_sym        armas_s_mult_sym
#define matrix_mult_trm        armas_s_mult_trm
#define matrix_mult_trm        armas_s_mult_trm
#define matrix_make_trm        armas_s_make_trm
#define matrix_solve_trm       armas_s_solve_trm
#define matrix_mult_diag       armas_s_mult_diag
#define matrix_solve_diag      armas_s_solve_diag
#define matrix_update_sym      armas_s_update_sym
#define matrix_transpose       armas_s_transpose
#define matrix_release         armas_s_release
#define matrix_mcopy           armas_s_mcopy
#define matrix_mnorm           armas_s_mnorm
#define matrix_col_as_row      armas_s_col_as_row
#define matrix_nrm2            armas_s_nrm2
#define matrix_asum            armas_s_asum
#define matrix_dot             armas_s_dot
#define matrix_axpy            armas_s_axpy
#define matrix_scale           armas_s_scale
#define matrix_invscale        armas_s_invscale
#define matrix_printf          armas_s_printf
#define matrix_qrfactor        armas_s_qrfactor
#define matrix_qrfactor_work   armas_s_qrfactor_work
#define matrix_qrmult          armas_s_qrmult
#define matrix_qrmult_work     armas_s_qrmult_work
#define matrix_qrbuild         armas_s_qrbuild
#define matrix_qrbuild_work    armas_s_qrbuild_work
#define matrix_qrsolve         armas_s_qrsolve
#define matrix_qrsolve_work    armas_s_qrsolve_work
#define matrix_qlfactor        armas_s_qlfactor
#define matrix_qlfactor_work   armas_s_qlfactor_work
#define matrix_qlmult          armas_s_qlmult
#define matrix_qlmult_work     armas_s_qlmult_work
#define matrix_qlbuild         armas_s_qlbuild
#define matrix_qlbuild_work    armas_s_qlbuild_work
#define matrix_lqbuild         armas_s_lqbuild
#define matrix_lqbuild_work    armas_s_lqbuild_work
#define matrix_lqfactor        armas_s_lqfactor
#define matrix_lqfactor_work   armas_s_lqfactor_work
#define matrix_lqmult          armas_s_lqmult
#define matrix_lqmult_work     armas_s_lqmult_work
#define matrix_lqreflector     armas_s_lqreflector
#define matrix_lqsolve         armas_s_lqsolve
#define matrix_lqsolve_work    armas_s_lqsolve_work
#define matrix_rqbuild         armas_s_rqbuild
#define matrix_rqbuild_work    armas_s_rqbuild_work
#define matrix_rqfactor        armas_s_rqfactor
#define matrix_rqfactor_work   armas_s_rqfactor_work
#define matrix_rqmult          armas_s_rqmult
#define matrix_rqmult_work     armas_s_rqmult_work
#define matrix_rqreflector     armas_s_rqreflector
#define matrix_bkfactor        armas_s_bkfactor
#define matrix_bkfactor_work   armas_s_bkfactor_work
#define matrix_bksolve         armas_s_bksolve
#define matrix_bksolve_work    armas_s_bksolve_work
#define matrix_lufactor        armas_s_lufactor
#define matrix_lusolve         armas_s_lusolve
#define matrix_cholfactor      armas_s_cholfactor
#define matrix_cholsolve       armas_s_cholsolve
#define matrix_qdroots         armas_s_qdroots
#define matrix_discriminant    armas_s_discriminant
#define matrix_mult_diag       armas_s_mult_diag
#define matrix_solve_diag      armas_s_solve_diag
#define matrix_scale_to        armas_s_scale_to
#define matrix_gvcompute       armas_s_gvcompute
#define matrix_gvrotate        armas_s_gvrotate
#define matrix_gvleft          armas_s_gvleft
#define matrix_gvright         armas_s_gvright
#define matrix_gvupdate        armas_s_gvupdate
#define matrix_bdreduce        armas_s_bdreduce
#define matrix_bdreduce_work   armas_s_bdreduce_work
#define matrix_bdmult          armas_s_bdmult
#define matrix_bdmult_work     armas_s_bdmult_work
#define matrix_bdbuild         armas_s_bdbuild
#define matrix_bdbuild_work    armas_s_bdbuild_work
#define matrix_trdreduce       armas_s_trdreduce
#define matrix_trdreduce_work  armas_s_trdreduce_work
#define matrix_trdbuild        armas_s_trdbuild
#define matrix_trdbuild_work   armas_s_trdbuild_work
#define matrix_trdmult         armas_s_trdmult
#define matrix_trdmult_work    armas_s_trdmult_work
#define matrix_trdeigen        armas_s_trdeigen
#define matrix_trdsec_solve    armas_s_trdsec_solve
#define matrix_trdsec_eigen    armas_s_trdsec_eigen
#define matrix_trdsec_solve_vec armas_s_trdsec_solve_vec
#define matrix_eigen_sym       armas_s_eigen_sym
#define matrix_hessreduce      armas_s_hessreduce
#define matrix_hessreduce_work armas_s_hessreduce_work
#define matrix_hessmult        armas_s_hessmult
#define matrix_hessmult_work   armas_s_hessmult_work
#define matrix_bdsvd           armas_s_bdsvd
#define matrix_bdsvd_work      armas_s_bdsvd_work
#define matrix_svd             armas_s_svd
#define matrix_svd_work        armas_s_svd_work

#else
#include <armas/dmatrix.h>
typedef armas_d_dense_t __Matrix ;
typedef double __Dtype;

#define matrix_init          armas_d_init
#define matrix_diag          armas_d_diag
#define matrix_add           armas_d_add
#define matrix_set_values    armas_d_set_values
#define matrix_submatrix     armas_d_submatrix
#define matrix_mult          armas_d_mult
#define matrix_symm          armas_d_symm
#define matrix_mult_trm      armas_d_mult_trm
#define matrix_make_trm      armas_d_make_trm
#define matrix_solve_trm     armas_d_solve_trm
#define matrix_update_sym    armas_d_update_sym
#define matrix_transpose     armas_d_transpose
#define matrix_release       armas_d_release
#define matrix_mcopy         armas_d_mcopy
#define matrix_mnorm         armas_d_mnorm
#define matrix_printf        armas_d_printf
#define matrix_qrfactor      armas_d_qrfactor
#define matrix_qrfactor_work armas_d_qrfactor_work
#define matrix_qrmult        armas_d_qrmult
#define matrix_qrmult_work   armas_d_qrmult_work
#define matrix_qrbuild       armas_d_qrbuild
#define matrix_qrbuild_work  armas_d_qrbuild_work
#define matrix_qrsolve       armas_d_qrsolve
#define matrix_qrsolve_work  armas_d_qrsolve_work
#define matrix_qlfactor      armas_d_qlfactor
#define matrix_qlfactor_work armas_d_qlfactor_work
#define matrix_qlmult        armas_d_qlmult
#define matrix_qlmult_work   armas_d_qlmult_work
#define matrix_qlbuild       armas_d_qlbuild
#define matrix_qlbuild_work  armas_d_qlbuild_work
#define matrix_lqbuild       armas_d_lqbuild
#define matrix_lqbuild_work  armas_d_lqbuild_work
#define matrix_lqfactor      armas_d_lqfactor
#define matrix_lqfactor_work armas_d_lqfactor_work
#define matrix_lqmult        armas_d_lqmult
#define matrix_lqmult_work   armas_d_lqmult_work
#define matrix_lqreflector   armas_d_lqreflector
#define matrix_lqsolve       armas_d_lqsolve
#define matrix_lqsolve_work  armas_d_lqsolve_work

#endif
#include "helper.h"

#endif //_UNIT_TESTING_H

// Local Variables:
// indent-tabs-mode: nil
// c-basic-offset: 4
// End: