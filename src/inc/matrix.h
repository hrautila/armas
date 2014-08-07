
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef __ARMAS_MATRIX_H
#define __ARMAS_MATRIX_H 1

#include <stdio.h>
#include <stdlib.h>
/* COMPLEX_H */
#include <armas/armas.h>

#ifdef __cplusplus
extern "C" {
#endif

// Internal 'type' independent declaration of matrix type. Public exported
// types are copies of this with explicit element type.
/**
 * @brief Column major matrix type.
 */
typedef struct __armas_dense {
  DTYPE *elems;   ///< Matrix elements, 
  int step;       ///< Row stride 
  int rows;       ///< Number of rows
  int cols;       ///< Number of columns
  void *__data;   ///< Non-null if stucture owns elements
  int __nbytes;   ///< sizeof __data buffer
} __armas_dense_t;

static inline int _M(__armas_dense_t *A) {
  return A->rows;
}

static inline int _N(__armas_dense_t *A) {
  return A->cols;
}

// function that constants
typedef DTYPE (*CONSTFUNC)();

// function that returns value for element
typedef DTYPE (*VALUEFUNC)(int, int);

// function that returns workspace size 
typedef int (*WSFUNC)(__armas_dense_t *, int);

// function that returns workspace size 
typedef int (*WSSIZE)(int, int, int);

extern void __armas_print(const __armas_dense_t *m, FILE *out);
extern void __armas_printf(FILE *out, const char *efmt, const __armas_dense_t *m);
extern int __armas_set_consts(__armas_dense_t *m, CONSTFUNC func, int flags);
extern int __armas_set_values(__armas_dense_t *m, VALUEFUNC func, int flags);

extern int __armas_allclose(__armas_dense_t *A, __armas_dense_t *B);
extern int __armas_intolerance(__armas_dense_t *A, __armas_dense_t *B, ABSTYPE atol, ABSTYPE rtol);

extern __armas_dense_t *__armas_mcopy(__armas_dense_t *d, __armas_dense_t *s);
extern __armas_dense_t *__armas_newcopy(__armas_dense_t *s);
extern __armas_dense_t *__armas_transpose(__armas_dense_t *d, __armas_dense_t *s);

extern int __armas_mscale(__armas_dense_t *d, DTYPE alpha, int flags);
extern int __armas_madd(__armas_dense_t *d, DTYPE alpha, int flags);

extern int __armas_scale_plus(__armas_dense_t *A, const __armas_dense_t *B,
                              DTYPE alpha, DTYPE beta, int flags, armas_conf_t *conf);
extern ABSTYPE __armas_mnorm(const __armas_dense_t *A, int norm, armas_conf_t *conf);

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

// -------------------------------------------------------------------------------------------
// inline functions

#ifndef __INLINE 
#define __INLINE extern inline
#endif

__INLINE
int __armas_isvector(const __armas_dense_t *m)
{
  return m && (m->rows == 1 || m->cols == 1);
}

__INLINE
int64_t __armas_size(const __armas_dense_t *m)
{
  return m ? m->rows * m->cols : 0;
}

__INLINE
int64_t __armas_real_index(const __armas_dense_t *m, int64_t ix)
{
  if (m->cols == 1 || m->step == m->rows)
    return ix;
  return m->step*(ix / m->rows) + (ix % m->rows);
}

__INLINE
int __armas_index_valid(const __armas_dense_t *m, int64_t ix)
{
  return m && (ix >= -__armas_size(m) && ix < __armas_size(m));
}


__INLINE
__armas_dense_t *__armas_init(__armas_dense_t *m, int r, int c)
{
  int doff;

  if (r <= 0 || c <= 0) {
    m->rows = 0; m->cols = 0;
    m->step = 0;
    m->elems = (DTYPE *)0;
    m->__data = (void *)0;
    m->__nbytes = 0;
    return m;
  }
  m->__data = calloc(r*c+7, sizeof(DTYPE));
  if ( !m->__data ) {
    return (__armas_dense_t *)0;
  }
  m->__nbytes = (r*c+7)*sizeof(DTYPE);
  m->rows = r;
  m->cols = c;
  m->step = r;
  doff = 0;
  if ((unsigned long)m->__data & 63) {
    doff = 64 - (((unsigned long)m->__data) & 63);
  }
  m->elems = (DTYPE *)&((unsigned char *)m->__data)[doff];
  return m;
}

__INLINE
void __armas_release(__armas_dense_t *m)
{
  if (m && m->__data) {
    free(m->__data);
    m->rows = 0; m->cols = 0; m->step = 0;
    m->elems = (DTYPE *)0;
    m->__data = (void *)0;
    m->__nbytes = 0;
  }
}

__INLINE
__armas_dense_t *__armas_alloc(int r, int c)
{
  __armas_dense_t *m;
  if (r <= 0 || c <= 0) {
    return (__armas_dense_t *)0;
  }
  m = malloc(sizeof(__armas_dense_t));
  if ( !m ) return m;
  return __armas_init(m, r, c);
}

__INLINE
void __armas_free(__armas_dense_t *m)
{
  if ( !m )
    return;
  __armas_release(m);
  free(m);
}

__INLINE
__armas_dense_t *__armas_make(__armas_dense_t *m, int r, int c, int s, DTYPE *elems)
{
  m->step = s;
  m->rows = r;
  m->cols = c;
  m->elems = elems;
  m->__data = (void *)0;
  m->__nbytes = 0;
}


// A = B[:, c], column vector as submatrix;
__INLINE
__armas_dense_t *__armas_column(__armas_dense_t *A, const __armas_dense_t *B, int c)
{
  if (__armas_size(B) == 0)
	return (__armas_dense_t*)0;
  if (c < 0) {
    c += B->cols;
  }
  A->cols = 1;
  A->rows = B->rows;
  A->step = B->step;
  A->elems = &B->elems[c*B->step];
  A->__data = (void *)0;
  A->__nbytes = 0;
  return A;
}

// A = B[r, :], row vector as submatrix;
__INLINE
__armas_dense_t *__armas_row(__armas_dense_t *A, const __armas_dense_t *B, int r)
{
  if (__armas_size(B) == 0)
    return (__armas_dense_t*)0;
  if (r < 0) {
    r += B->rows;
  }
  A->rows = 1;
  A->cols = B->cols;
  A->step = B->step;
  A->elems = &B->elems[r];
  A->__data = (void *)0;
  A->__nbytes = 0;
  return A;
}

//! \brief Make A a submatrix view of B with spesified row stride.
__INLINE
__armas_dense_t *__armas_submatrix_ext(__armas_dense_t *A, const __armas_dense_t *B,
                                       int r, int c, int nr, int nc, int step)
{
  if (__armas_size(B) == 0)
    return (__armas_dense_t*)0;
  if (r < 0) {
    r += B->rows;
  }
  if (c < 0) {
    c += B->cols;
  }
  if (nr < 0) {
    nr = B->rows - r;
  }
  if (nc < 0) {
    nc = B->cols - c;
  }
  if (step < 0) {
    step = B->step;
  }
  return __armas_make(A, nr, nc, step, &B->elems[c*B->step+r]);
}

//! \brief Make A submatrix view of B.
__INLINE
__armas_dense_t *__armas_submatrix(__armas_dense_t *A, const __armas_dense_t *B,
                                   int r, int c, int nr, int nc)
{
  return __armas_submatrix_ext(A, B, r, c, nr, nc, B->step);
}

//! \brief Make A diagonal row vector of B.
__INLINE
__armas_dense_t *__armas_diag(__armas_dense_t *A, const __armas_dense_t *B, int k)
{
  int nk;
  if (k > 0) {
    // super diagonal
    nk = B->rows < B->cols-k ? B->rows : B->cols-k;
    return __armas_submatrix_ext(A, B, 0, k, 1, nk, B->step+1);
  }
  if (k < 0) {
    // subdiagonal
    nk = B->rows+k < B->cols ? B->rows+k : B->cols;
    return __armas_submatrix_ext(A, B, -k, 0, 1, nk, B->step+1);
  }
  // main diagonal
  nk = B->rows < B->cols ? B->rows : B->cols;
  return __armas_submatrix_ext(A, B, 0, 0, 1, nk, B->step+1);
}

__INLINE
DTYPE __armas_get(const __armas_dense_t *m, int row, int col)
{
  if (__armas_size(m) == 0)
    return 0.0;
  if (row < 0)
    row += m->rows;
  if (col < 0)
    col += m->cols;
  return m->elems[col*m->step+row];
}

__INLINE
void __armas_set(__armas_dense_t *m, int row, int col, DTYPE val)
{
  if (__armas_size(m) == 0)
    return;
  if (row < 0)
    row += m->rows;
  if (col < 0)
    col += m->cols;
  m->elems[col*m->step+row] = val;
}

__INLINE
void __armas_set_at(__armas_dense_t *m, int ix, DTYPE val)
{
  if (__armas_size(m) == 0)
    return;
  if (ix < 0)
    ix += __armas_size(m);
  m->elems[__armas_real_index(m, ix)] = val;
}

__INLINE
DTYPE __armas_get_at(const __armas_dense_t *m, int ix)
{
  if (__armas_size(m) == 0)
    return 0.0;
  if (ix < 0)
    ix += __armas_size(m);
  return m->elems[__armas_real_index(m, ix)];
}

__INLINE
int __armas_index(const __armas_dense_t *m, int row, int col)
{
  if (__armas_size(m) == 0)
    return 0;
  if (row < 0)
    row += m->rows;
  if (col < 0)
    col += m->cols;
  return col*m->step + row;
}

__INLINE
DTYPE *__armas_data(__armas_dense_t *m)
{
  return m ? m->elems : (DTYPE *)0;
}

__INLINE
__armas_dense_t *__armas_col_as_row(__armas_dense_t *row, __armas_dense_t *col)
{
  __armas_make(row, 1, __armas_size(col), 1, __armas_data(col));
  return row;
}

#ifdef __cplusplus
}
#endif



#endif
  

// Local Variables:
// indent-tabs-mode: nil
// End:
