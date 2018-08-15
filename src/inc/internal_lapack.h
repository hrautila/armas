
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_INTERNAL_LAPACK_H
#define __ARMAS_INTERNAL_LAPACK_H 1

#include "partition.h"

// internal declarations for lapack functionality.

// function that returns workspace size 
typedef int (*WSFUNC)(armas_x_dense_t *, int);

// function that returns workspace size 
typedef int (*WSSIZE)(int, int, int);

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

// Bunch-Kauffman algorithm constant ALPHA = (1 + sqrt(17))/8
#define bkALPHA  0.64038820320220756872

enum pivot_dir {
  PIVOT_FORWARD = 0,
  PIVOT_BACKWARD = 1
};

#define ONERROR(exp) \
  do { if ( (exp) ) return -1; } while (0)

// "empty" matrix;
#define __EMPTY (armas_x_dense_t ){                   \
    .elems = (DTYPE *)0,                              \
    .step = 0,                                        \
    .rows = 0,                                        \
    .cols = 0,                                        \
    .__data = (void *)0,                              \
    .__nbytes = 0}

 

#if __GNUC__
// macro to initialize matrix to "empty" to avoid GCC "maybe-uninitialized" errors
#define EMPTY(A) A = __EMPTY
#else
#define EMPTY(A)
#endif

extern int compute_lb(int M, int N, int wsize, WSSIZE wsizer);

static inline
long __IMAX(long a, long b) {
  return a > b ? a : b;
}

static inline
long __IMIN(long a, long b) {
  return a < b ? a : b;
}

// helper functions

extern
void __scale_pivots(armas_pivot_t *P, int offset);

extern
int __pivot_index(armas_x_dense_t *A, armas_conf_t *conf);
extern
int __pivot_sort(armas_x_dense_t *D, armas_pivot_t *P, int direction);
extern
int __eigen_sort(armas_x_dense_t *D, armas_x_dense_t *U, armas_x_dense_t *V, armas_x_dense_t *C,
		 armas_conf_t *conf);
extern
void __abs_sort_vec(armas_x_dense_t *D, int updown);
extern
void __sort_vec(armas_x_dense_t *D, int updown);
extern
int __sort_eigenvec(armas_x_dense_t *D, armas_x_dense_t *U, armas_x_dense_t *V, armas_x_dense_t *C, int updown);
		  
// internal householder
extern
void __compute_householder(armas_x_dense_t *a11, armas_x_dense_t *x,
                           armas_x_dense_t *tau, armas_conf_t *conf);
extern
void __compute_householder_vec(armas_x_dense_t *x, armas_x_dense_t *tau, armas_conf_t *conf);

extern
void __compute_householder_rev(armas_x_dense_t *x, armas_x_dense_t *tau, armas_conf_t *conf);

extern
int __apply_householder2x1(armas_x_dense_t *tau, armas_x_dense_t *v,
                           armas_x_dense_t *a1,  armas_x_dense_t *A2,
                           armas_x_dense_t *w1,  int flags, armas_conf_t *conf);

extern
int __apply_householder1x1(armas_x_dense_t *tau, armas_x_dense_t *v,
                           armas_x_dense_t *a1,  armas_x_dense_t *A2,
                           armas_x_dense_t *w1,  int flags, armas_conf_t *conf);


// internal LDL functions
extern int
__unblk_bkfactor_lower(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, armas_conf_t *conf);
extern int
__unblk_bkfactor_upper(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, armas_conf_t *conf);
extern int
__unblk_bksolve_lower(armas_x_dense_t *B, armas_x_dense_t *A, armas_pivot_t *P, int phase, armas_conf_t *conf);
extern int
__unblk_bksolve_upper(armas_x_dense_t *B, armas_x_dense_t *A, armas_pivot_t *P, int phase, armas_conf_t *conf);
extern int
__blk_bkfactor_lower(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, int lb, armas_conf_t *conf);
extern int
__blk_bkfactor_upper(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, int lb, armas_conf_t *conf);

// internal LQ functions
extern int
__update_lq_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		  armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_lq_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		 armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_lq_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
		     armas_conf_t *conf);

// internal QL functions
extern int
__update_ql_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		  armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_ql_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		 armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_ql_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
		     armas_conf_t *conf);

// internal QR functions
extern int
__update_qr_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		  armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_qr_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		 armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_qr_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
		     armas_conf_t *conf);
// internal RQ functions
extern int
__update_rq_right(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		  armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		  int transpose, armas_conf_t *conf);
extern int
__update_rq_left(armas_x_dense_t *C1, armas_x_dense_t *C2, armas_x_dense_t *Y1,
		 armas_x_dense_t *Y2, armas_x_dense_t *T, armas_x_dense_t *Wrk,
		 int transpose, armas_conf_t *conf);
extern int
__unblk_rq_reflector(armas_x_dense_t *T, armas_x_dense_t *A, armas_x_dense_t *tau,
		     armas_conf_t *conf);

// internal Bidiagonal SVD
extern DTYPE __bdsvd2x2(DTYPE *smin, DTYPE *smax, DTYPE f, DTYPE g, DTYPE h);
extern void __bdsvd2x2_vec(DTYPE *ssmin, DTYPE *ssmax,
			   DTYPE *cosl, DTYPE *sinl, DTYPE *cosr, DTYPE *sinr,
			   DTYPE f, DTYPE g, DTYPE h);
extern int __bdsvd_golub(armas_x_dense_t *D, armas_x_dense_t *E,
			 armas_x_dense_t *U, armas_x_dense_t *V,
			 armas_x_dense_t *CS, DTYPE tol, armas_conf_t *conf);
extern int __bdsvd_demmel(armas_x_dense_t *D, armas_x_dense_t *E,
			  armas_x_dense_t *U, armas_x_dense_t *V,
			  armas_x_dense_t *CS, DTYPE tol, int flags, armas_conf_t *conf);

// tridiagonal EVD
extern void __sym_eigen2x2(DTYPE *z1, DTYPE *z2, DTYPE a, DTYPE b, DTYPE c);
extern void __sym_eigen2x2vec(DTYPE *z1, DTYPE *z2, DTYPE *cs, DTYPE *sn, DTYPE a, DTYPE b, DTYPE c);
extern int __trdevd_qr(armas_x_dense_t *D, armas_x_dense_t *E,
		       armas_x_dense_t *V, armas_x_dense_t *CS, ABSTYPE tol, armas_conf_t *conf);



#endif /* __ARMAS_INTERNAL_LAPACK_H */
