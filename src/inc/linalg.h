
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_LINALG_H
#define __ARMAS_LINALG_H 1

#include <armas/armas.h>

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif


extern int armas_x_scale_plus(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B,
                              int flags, armas_conf_t *conf);
extern ABSTYPE armas_x_mnorm(const armas_x_dense_t *A, int norm, armas_conf_t *conf);

extern int     armas_x_scale_to(armas_x_dense_t *A, DTYPE from, DTYPE to, int flags, armas_conf_t *conf);

// Blas level 1 functions
extern int     armas_x_iamax(const armas_x_dense_t *X, armas_conf_t *conf);
extern ABSTYPE armas_x_amax(const armas_x_dense_t *X, armas_conf_t *conf);
extern ABSTYPE armas_x_asum(const armas_x_dense_t *X, armas_conf_t *conf);
extern ABSTYPE armas_x_nrm2(const armas_x_dense_t *X, armas_conf_t *conf);
extern DTYPE   armas_x_dot(const armas_x_dense_t *X, const armas_x_dense_t *Y, armas_conf_t *conf);
extern int     armas_x_axpy(armas_x_dense_t *Y, DTYPE alpha, const armas_x_dense_t *X, armas_conf_t *conf);
extern int     armas_x_axpby(DTYPE beta, armas_x_dense_t *Y, DTYPE alpha, const armas_x_dense_t *X, armas_conf_t *conf);
extern int     armas_x_copy(armas_x_dense_t *Y, const armas_x_dense_t *X, armas_conf_t *conf);
extern int     armas_x_swap(armas_x_dense_t *Y, armas_x_dense_t *X, armas_conf_t *conf);

extern DTYPE   armas_x_sum(const armas_x_dense_t *X, armas_conf_t *conf);
extern int     armas_x_scale(const armas_x_dense_t *X, const DTYPE alpha, armas_conf_t *conf);
extern int     armas_x_invscale(const armas_x_dense_t *X, const DTYPE alpha, armas_conf_t *conf);
extern int     armas_x_add(const armas_x_dense_t *X, const DTYPE alpha, armas_conf_t *conf);


// Blas level 2 functions
  extern int armas_x_mvmult(DTYPE beta, armas_x_dense_t *Y,
			    DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *X,
			    int flags, armas_conf_t *conf);
  extern int armas_x_mvmult_sym(DTYPE beta, armas_x_dense_t *Y,
				DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *X, 
				int flags, armas_conf_t *conf);
  extern int armas_x_mvupdate(armas_x_dense_t *A,
			      DTYPE alpha, const armas_x_dense_t *X,  const armas_x_dense_t *Y,  
			      armas_conf_t *conf);
  extern int armas_x_mvupdate2_sym(armas_x_dense_t *A,
				   DTYPE alpha, const armas_x_dense_t *X,  const armas_x_dense_t *Y,  
				   int flags, armas_conf_t *conf);
  extern int armas_x_mvupdate_sym(armas_x_dense_t *A,
				  DTYPE alpha, const armas_x_dense_t *X,
				  int flags, armas_conf_t *conf);
  extern int armas_x_mvupdate_trm(armas_x_dense_t *A,
				  DTYPE alpha, const armas_x_dense_t *X,  const armas_x_dense_t *Y,  
				  int flags, armas_conf_t *conf);
  extern int armas_x_mvmult_trm(armas_x_dense_t *X,  DTYPE alpha, const armas_x_dense_t *A, 
				int flags, armas_conf_t *conf);
  extern int armas_x_mvsolve_trm(armas_x_dense_t *X, DTYPE alpha,  const armas_x_dense_t *A, 
				 int flags, armas_conf_t *conf);


// Blas level 3 functions
  extern int armas_x_mult(DTYPE beta, armas_x_dense_t *C,
			  DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
			  int flags, armas_conf_t *conf);

  extern int armas_x_mult_sym(DTYPE beta, armas_x_dense_t *C,
			      DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
			      int flags, armas_conf_t *conf);
                         
  extern int armas_x_mult_trm(armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A,
			      int flags, armas_conf_t *conf);

  extern int armas_x_solve_trm(armas_x_dense_t *B, DTYPE alpha, const armas_x_dense_t *A,
			       int flags, armas_conf_t *conf);

  extern int armas_x_update_trm(DTYPE beta, armas_x_dense_t *C,
				DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B,
				int flags, armas_conf_t *conf);

  extern int armas_x_update_sym(DTYPE beta, armas_x_dense_t *C,
				DTYPE alpha, const armas_x_dense_t *A,
				int flags, armas_conf_t *conf);

  extern int armas_x_update2_sym(DTYPE beta, armas_x_dense_t *C,
				 DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *B, 
				 int flags, armas_conf_t *conf);

// Lapack

// Bidiagonal reduction
extern int armas_x_bdreduce(armas_x_dense_t *A, armas_x_dense_t *tauq, armas_x_dense_t *taup,
                            armas_conf_t *conf);
extern int armas_x_bdbuild(armas_x_dense_t *A, const armas_x_dense_t *tau,
                           int K, int flags, armas_conf_t *conf);
extern int armas_x_bdmult(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                          int flags, armas_conf_t *conf);

// Cholesky
extern int armas_x_cholesky(armas_x_dense_t *A, int flags, armas_conf_t *conf);
extern int armas_x_cholfactor(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_cholsolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_pivot_t *P,
			     int flags, armas_conf_t *conf);
extern int armas_x_cholfactor_w(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

// Hessenberg reduction
extern int armas_x_hessreduce(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                              armas_conf_t *conf);
extern int armas_x_hessmult(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *tau,
                            armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_hessreduce_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_hessmult_work(armas_x_dense_t *A, int flags, armas_conf_t *conf);

// LU
extern int armas_x_lufactor(armas_x_dense_t *A, armas_pivot_t *P, armas_conf_t *conf);
extern int armas_x_lusolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_pivot_t *P,
                           int flags, armas_conf_t *conf);

// Symmetric LDL; Bunch-Kauffman
extern int armas_x_bkfactor(armas_x_dense_t *A, armas_x_dense_t *W,
                            armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_bksolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *W,
                           armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_bkfactor_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_bksolve_work(armas_x_dense_t *A, armas_conf_t *conf);

// LQ functions
extern int armas_x_lqbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                           int K, armas_conf_t *conf);
extern int armas_x_lqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                            armas_conf_t *conf);
extern int armas_x_lqmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                          armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_lqreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_lqsolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *tau,
                           armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_lqbuild_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_lqfactor_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_lqmult_work(armas_x_dense_t *C, int flags, armas_conf_t *conf);
extern int armas_x_lqsolve_work(armas_x_dense_t *B, armas_conf_t *conf);

// QL functions
extern int armas_x_qlbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                           int K, armas_conf_t *conf);
extern int armas_x_qlfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                            armas_conf_t *conf);
extern int armas_x_qlmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                          armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_qlreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_qlbuild_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_qlfactor_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_qlmult_work(armas_x_dense_t *C, int flags, armas_conf_t *conf);

// QR functions
extern int armas_x_qrbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                           int K, armas_conf_t *conf);
extern int armas_x_qrfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                            armas_conf_t *conf);
extern int armas_x_qrmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                          armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_qrreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_qrsolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *tau,
                           armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_qrbuild_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_qrfactor_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_qrmult_work(armas_x_dense_t *C, int flags, armas_conf_t *conf);
extern int armas_x_qrsolve_work(armas_x_dense_t *B, armas_conf_t *conf);

extern int armas_x_qrtfactor(armas_x_dense_t *A, armas_x_dense_t *T, armas_x_dense_t *W, armas_conf_t *conf);
extern int armas_x_qrtmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *T,
			   armas_x_dense_t *W, int flags, armas_conf_t *conf);

// RQ functions
extern int armas_x_rqbuild(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                           int K, armas_conf_t *conf);
extern int armas_x_rqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                            armas_conf_t *conf);
extern int armas_x_rqmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *tau,
                          armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_rqreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_rqsolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *tau,
                           armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_rqbuild_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_rqfactor_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_rqmult_work(armas_x_dense_t *C, int flags, armas_conf_t *conf);
extern int armas_x_rqsolve_work(armas_x_dense_t *B, armas_conf_t *conf);

// Tridiagonal reduction
extern int armas_x_trdreduce(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *W,
                             int flags, armas_conf_t *conf);
extern int armas_x_trdbuild(armas_x_dense_t *A, armas_x_dense_t *tau,
                            armas_x_dense_t *W, int K, int flags, armas_conf_t *conf);
extern int armas_x_trdmult(armas_x_dense_t *B, armas_x_dense_t *A, armas_x_dense_t *tau,
                           armas_x_dense_t *W, int flags, armas_conf_t *conf);
extern int armas_x_trdreduce_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_trdmult_work(armas_x_dense_t *A, int flags, armas_conf_t *conf);
extern int armas_x_trdbuild_work(armas_x_dense_t *A, armas_conf_t *conf);
extern int armas_x_trdeigen(armas_x_dense_t *D, armas_x_dense_t *E, armas_x_dense_t *V,
                            armas_x_dense_t *W, int flags, armas_conf_t *conf);
// Secular functions solvers
extern int armas_x_trdsec_solve(armas_x_dense_t *y, armas_x_dense_t *d,
                                armas_x_dense_t *z, armas_x_dense_t *delta, DTYPE rho,
                                armas_conf_t *conf);
extern int armas_x_trdsec_solve_vec(armas_x_dense_t *y, armas_x_dense_t *v, armas_x_dense_t *Qd,
                                    armas_x_dense_t *d, armas_x_dense_t *z,
                                    DTYPE rho, armas_conf_t *conf);

extern int armas_x_trdsec_eigen(armas_x_dense_t *Q, armas_x_dense_t *v, armas_x_dense_t *Qd,
                                armas_conf_t *conf);
  
// Givens
extern void armas_x_gvcompute(DTYPE *c, DTYPE *s, DTYPE *r, DTYPE a, DTYPE b);
extern void armas_x_gvrotate(DTYPE *v0, DTYPE *v1, DTYPE c, DTYPE s, DTYPE y0, DTYPE y1);
extern void armas_x_gvleft(armas_x_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol);
extern void armas_x_gvright(armas_x_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol);
extern int armas_x_gvupdate(armas_x_dense_t *A, int start, 
                            armas_x_dense_t *C, armas_x_dense_t *S, int nrot, int flags);
// Bidiagonal SVD
extern int armas_x_bdsvd(armas_x_dense_t *D, armas_x_dense_t *E, armas_x_dense_t *U, armas_x_dense_t *V,
                         int flags, armas_conf_t *conf);
extern int armas_x_bdsvd_work(armas_x_dense_t *D, armas_conf_t *conf);

extern int armas_x_svd(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V, armas_x_dense_t *A,
                       int flags, armas_conf_t *conf);
extern int armas_x_svd_work(armas_x_dense_t *D, int flags, armas_conf_t *conf);

  // DQDS
extern int armas_x_dqds(armas_x_dense_t *D, armas_x_dense_t *E, armas_conf_t *conf);

// Recursive Butterfly
extern int armas_x_mult_rbt(armas_x_dense_t *A, armas_x_dense_t *U, int flags, armas_conf_t *conf);
extern int armas_x_update2_rbt(armas_x_dense_t *A, armas_x_dense_t *U, armas_x_dense_t *V, armas_conf_t *conf);
extern void armas_x_gen_rbt(armas_x_dense_t *U);

// additional
extern int armas_x_qdroots(DTYPE *x1, DTYPE *x2, DTYPE a, DTYPE b, DTYPE c);
extern void armas_x_discriminant(DTYPE *d, DTYPE a, DTYPE b, DTYPE c);
extern int armas_x_mult_diag(armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *D, int flags, armas_conf_t *conf);
extern int armas_x_solve_diag(armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *D, int flags, armas_conf_t *conf);

extern int armas_x_pivot_rows(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);

#ifdef __cplusplus
}
#endif

#endif /* __ARMAS_LINALG_H */
