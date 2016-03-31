
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


extern int __armas_scale_plus(__armas_dense_t *A, const __armas_dense_t *B,
                              DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf);
extern ABSTYPE __armas_mnorm(const __armas_dense_t *A, int norm, armas_conf_t *conf);

extern int     __armas_scale_to(__armas_dense_t *A, DTYPE from, DTYPE to, int flags, armas_conf_t *conf);

// Blas level 1 functions
extern int     __armas_iamax(const __armas_dense_t *X, armas_conf_t *conf);
extern ABSTYPE __armas_amax(const __armas_dense_t *X, armas_conf_t *conf);
extern ABSTYPE __armas_asum(const __armas_dense_t *X, armas_conf_t *conf);
extern ABSTYPE __armas_nrm2(const __armas_dense_t *X, armas_conf_t *conf);
extern DTYPE   __armas_dot(const __armas_dense_t *X, const __armas_dense_t *Y, armas_conf_t *conf);
extern int     __armas_axpy(__armas_dense_t *Y, const __armas_dense_t *X, DTYPE alpha, armas_conf_t *conf);
extern int     __armas_axpby(__armas_dense_t *Y, const __armas_dense_t *X, DTYPE alpha, DTYPE beta, armas_conf_t *conf);
extern int     __armas_copy(__armas_dense_t *Y, const __armas_dense_t *X, armas_conf_t *conf);
extern int     __armas_swap(__armas_dense_t *Y, __armas_dense_t *X, armas_conf_t *conf);

extern DTYPE   __armas_sum(const __armas_dense_t *X, armas_conf_t *conf);
extern int     __armas_scale(const __armas_dense_t *X, const DTYPE alpha, armas_conf_t *conf);
extern int     __armas_invscale(const __armas_dense_t *X, const DTYPE alpha, armas_conf_t *conf);
extern int     __armas_add(const __armas_dense_t *X, const DTYPE alpha, armas_conf_t *conf);


// Blas level 2 functions
extern int __armas_mvmult(__armas_dense_t *Y,
                          const __armas_dense_t *A, const __armas_dense_t *X,
                          DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf);
extern int __armas_mvupdate(__armas_dense_t *A,
                            const __armas_dense_t *X,  const __armas_dense_t *Y,  
                            DTYPE alpha, armas_conf_t *conf);
extern int __armas_mvupdate2_sym(__armas_dense_t *A,
                                 const __armas_dense_t *X,  const __armas_dense_t *Y,  
                                 DTYPE alpha, int flags, armas_conf_t *conf);
extern int __armas_mvupdate_sym(__armas_dense_t *A,
                                const __armas_dense_t *X,
                                DTYPE alpha, int flags, armas_conf_t *conf);
extern int __armas_mvupdate_trm(__armas_dense_t *A,
                                const __armas_dense_t *X,  const __armas_dense_t *Y,  
                                DTYPE alpha, int flags, armas_conf_t *conf);
extern int __armas_mvmult_trm(__armas_dense_t *X,  const __armas_dense_t *A, 
                              DTYPE alpha, int flags, armas_conf_t *conf);
extern int __armas_mvsolve_trm(__armas_dense_t *X,  const __armas_dense_t *A, 
                               DTYPE alpha, int flags, armas_conf_t *conf);


// Blas level 3 functions
extern int __armas_mult(__armas_dense_t *C,
                        const __armas_dense_t *A, const __armas_dense_t *B,
                        DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf);

extern int __armas_mult_sym(__armas_dense_t *C,
                            const __armas_dense_t *A, const __armas_dense_t *B,
                            DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf);
                         
extern int __armas_mult_trm(__armas_dense_t *B, const __armas_dense_t *A,
                            DTYPE alpha, int flags, armas_conf_t *conf);

extern int __armas_solve_trm(__armas_dense_t *B, const __armas_dense_t *A,
                             DTYPE alpha, int flags, armas_conf_t *conf);

extern int __armas_update_trm(__armas_dense_t *C,
                              const __armas_dense_t *A, const __armas_dense_t *B,
                              DTYPE alpha, DTYPE beta, int flags,
                              armas_conf_t *conf);

extern int __armas_update_sym(__armas_dense_t *C, const __armas_dense_t *A,
                              DTYPE alpha, DTYPE beta, int flags,
                              armas_conf_t *conf);

extern int __armas_update2_sym(__armas_dense_t *C,
                               const __armas_dense_t *A, const __armas_dense_t *B, 
                               DTYPE alpha, DTYPE beta, int flags,
                               armas_conf_t *conf);

// Lapack

// Bidiagonal reduction
extern int __armas_bdreduce(__armas_dense_t *A, __armas_dense_t *tauq, __armas_dense_t *taup,
                            __armas_dense_t *W, armas_conf_t *conf);
extern int __armas_bdbuild(__armas_dense_t *A, __armas_dense_t *tau,
                           __armas_dense_t *W, int K, int flags, armas_conf_t *conf);
extern int __armas_bdmult(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                          __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_bdreduce_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_bdmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf);
extern int __armas_bdbuild_work(__armas_dense_t *A, int flags, armas_conf_t *conf);

// Cholesky
extern int __armas_cholfactor(__armas_dense_t *A, int flags, armas_conf_t *conf);
extern int __armas_cholsolve(__armas_dense_t *B, __armas_dense_t *A, int flags,
                             armas_conf_t *conf);

// Hessenberg reduction
extern int __armas_hessreduce(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                              armas_conf_t *conf);
extern int __armas_hessmult(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                            __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_hessreduce_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_hessmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf);

// LU
extern int __armas_lufactor(__armas_dense_t *A, armas_pivot_t *P, armas_conf_t *conf);
extern int __armas_lusolve(__armas_dense_t *B, __armas_dense_t *A, armas_pivot_t *P,
                           int flags, armas_conf_t *conf);

// Symmetric LDL; Bunch-Kauffman
extern int __armas_bkfactor(__armas_dense_t *A, __armas_dense_t *W,
                            armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int __armas_bksolve(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *W,
                           armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int __armas_bkfactor_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_bksolve_work(__armas_dense_t *A, armas_conf_t *conf);

// LQ functions
extern int __armas_lqbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                           int K, armas_conf_t *conf);
extern int __armas_lqfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                            armas_conf_t *conf);
extern int __armas_lqmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                          __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_lqreflector(__armas_dense_t *T, __armas_dense_t *V, __armas_dense_t *tau,
                               armas_conf_t *conf);
extern int __armas_lqsolve(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                           __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_lqbuild_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_lqfactor_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_lqmult_work(__armas_dense_t *C, int flags, armas_conf_t *conf);
extern int __armas_lqsolve_work(__armas_dense_t *B, armas_conf_t *conf);

// QL functions
extern int __armas_qlbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                           int K, armas_conf_t *conf);
extern int __armas_qlfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                            armas_conf_t *conf);
extern int __armas_qlmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                          __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_qlreflector(__armas_dense_t *T, __armas_dense_t *V, __armas_dense_t *tau,
                               armas_conf_t *conf);
extern int __armas_qlbuild_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_qlfactor_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_qlmult_work(__armas_dense_t *C, int flags, armas_conf_t *conf);

// QR functions
extern int __armas_qrbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                           int K, armas_conf_t *conf);
extern int __armas_qrfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                            armas_conf_t *conf);
extern int __armas_qrmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                          __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_qrreflector(__armas_dense_t *T, __armas_dense_t *V, __armas_dense_t *tau,
                               armas_conf_t *conf);
extern int __armas_qrsolve(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                           __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_qrbuild_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_qrfactor_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_qrmult_work(__armas_dense_t *C, int flags, armas_conf_t *conf);
extern int __armas_qrsolve_work(__armas_dense_t *B, armas_conf_t *conf);

extern int __armas_qrtfactor(__armas_dense_t *A, __armas_dense_t *T, __armas_dense_t *W, armas_conf_t *conf);
extern int __armas_qrtmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *T,
			   __armas_dense_t *W, int flags, armas_conf_t *conf);

// RQ functions
extern int __armas_rqbuild(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                           int K, armas_conf_t *conf);
extern int __armas_rqfactor(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                            armas_conf_t *conf);
extern int __armas_rqmult(__armas_dense_t *C, __armas_dense_t *A, __armas_dense_t *tau,
                          __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_rqreflector(__armas_dense_t *T, __armas_dense_t *V, __armas_dense_t *tau,
                               armas_conf_t *conf);
extern int __armas_rqsolve(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                           __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_rqbuild_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_rqfactor_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_rqmult_work(__armas_dense_t *C, int flags, armas_conf_t *conf);
extern int __armas_rqsolve_work(__armas_dense_t *B, armas_conf_t *conf);

// Tridiagonal reduction
extern int __armas_trdreduce(__armas_dense_t *A, __armas_dense_t *tau, __armas_dense_t *W,
                             int flags, armas_conf_t *conf);
extern int __armas_trdbuild(__armas_dense_t *A, __armas_dense_t *tau,
                            __armas_dense_t *W, int K, int flags, armas_conf_t *conf);
extern int __armas_trdmult(__armas_dense_t *B, __armas_dense_t *A, __armas_dense_t *tau,
                           __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_trdreduce_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_trdmult_work(__armas_dense_t *A, int flags, armas_conf_t *conf);
extern int __armas_trdbuild_work(__armas_dense_t *A, armas_conf_t *conf);
extern int __armas_trdeigen(__armas_dense_t *D, __armas_dense_t *E, __armas_dense_t *V,
                            __armas_dense_t *W, int flags, armas_conf_t *conf);
// Secular functions solvers
extern int __armas_trdsec_solve(__armas_dense_t *y, __armas_dense_t *d,
                                __armas_dense_t *z, __armas_dense_t *delta, DTYPE rho,
                                armas_conf_t *conf);
extern int __armas_trdsec_solve_vec(__armas_dense_t *y, __armas_dense_t *v, __armas_dense_t *Qd,
                                    __armas_dense_t *d, __armas_dense_t *z,
                                    DTYPE rho, armas_conf_t *conf);

extern int __armas_trdsec_eigen(__armas_dense_t *Q, __armas_dense_t *v, __armas_dense_t *Qd,
                                armas_conf_t *conf);
  
// Givens
extern void __armas_gvcompute(DTYPE *c, DTYPE *s, DTYPE *r, DTYPE a, DTYPE b);
extern void __armas_gvrotate(DTYPE *v0, DTYPE *v1, DTYPE c, DTYPE s, DTYPE y0, DTYPE y1);
extern void __armas_gvleft(__armas_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol);
extern void __armas_gvright(__armas_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol);
extern int __armas_gvupdate(__armas_dense_t *A, int start, 
                            __armas_dense_t *C, __armas_dense_t *S, int nrot, int flags);
// Bidiagonal SVD
extern int __armas_bdsvd(__armas_dense_t *D, __armas_dense_t *E, __armas_dense_t *U, __armas_dense_t *V,
                         __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_bdsvd_work(__armas_dense_t *D, armas_conf_t *conf);

extern int __armas_svd(__armas_dense_t *S, __armas_dense_t *U, __armas_dense_t *V, __armas_dense_t *A,
                       __armas_dense_t *W, int flags, armas_conf_t *conf);
extern int __armas_svd_work(__armas_dense_t *D, int flags, armas_conf_t *conf);

  // DQDS
extern int __armas_dqds(__armas_dense_t *D, __armas_dense_t *E, __armas_dense_t *W, armas_conf_t *conf);

// Recursive Butterfly
extern int __armas_mult_rbt(__armas_dense_t *A, __armas_dense_t *U, int flags, armas_conf_t *conf);
extern int __armas_update2_rbt(__armas_dense_t *A, __armas_dense_t *U, __armas_dense_t *V, armas_conf_t *conf);
extern void __armas_gen_rbt(__armas_dense_t *U);

// additional
extern int __armas_qdroots(DTYPE *x1, DTYPE *x2, DTYPE a, DTYPE b, DTYPE c);
extern void __armas_discriminant(DTYPE *d, DTYPE a, DTYPE b, DTYPE c);
extern int __armas_mult_diag(__armas_dense_t *A, const __armas_dense_t *D, int flags, armas_conf_t *conf);
extern int __armas_solve_diag(__armas_dense_t *A, const __armas_dense_t *D, int flags, armas_conf_t *conf);

extern int __armas_pivot_rows(__armas_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);

#ifdef __cplusplus
}
#endif

#endif /* __ARMAS_LINALG_H */
