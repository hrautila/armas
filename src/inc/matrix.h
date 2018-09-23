
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * \file 
 * Matrix structure definition and public functions
 */

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
typedef struct armas_x_dense_s {
  DTYPE *elems;   ///< Matrix elements, 
  int step;       ///< Row stride 
  int rows;       ///< Number of rows
  int cols;       ///< Number of columns
  void *__data;   ///< Non-null if stucture owns elements
  int __nbytes;   ///< sizeof __data buffer
} armas_x_dense_t;


/**
 * \brief Eigenvalue selection parameters
 */
typedef struct armas_x_eigen_parameter_s {
    int ileft;          ///< Start index of half-open interval [ileft, iright)
    int iright;         ///< End index of half-open interval [ileft, iright) 
    DTYPE left;         ///< Start eigenvalue of half-open interval [left, right)
    DTYPE right;        ///< Last eigenvalue of half-open interval [left, right)
    DTYPE tau;          ///< Requested accurancy; must be > max{|T_i|}*eps
} armas_x_eigen_parameter_t;
  

//! \brief Define eigenvalue index range [l, r) with default accuracy
#define ARMAS_EIGEN_INT(l, r)  &(armas_x_eigen_parameter_t){(l), (r), 0.0, 0.0, 0.0}

//! \brief Define eigenvalue value range [low, high) with default accuracy
#define ARMAS_EIGEN_VAL(low, high)  &(armas_x_eigen_parameter_t){0, 0, (low), (high), 0.0}

//! \brief Define eigenvalue index range [l, r) and accuracy parameter tau
#define ARMAS_EIGEN_INT_TAU(l, r, tau)  &(armas_x_eigen_parameter_t){(l), (r), 0.0, 0.0, tau}

//! \brief Define eigenvalue value range [low, high) and accuracy parameter tau
#define ARMAS_EIGEN_VAL_TAU(low, high, tau)  &(armas_x_eigen_parameter_t){0, 0, (low), (high), tau}

//! \brief All eigenvalues with default accuracy
#define ARMAS_EIGEN_ALL  &(armas_x_eigen_parameter_t){-1, -1, 0.0, 0.0, 0.0}

//! \brief All eigenvalues with accuracy parameter
#define ARMAS_EIGEN_ALL_TAU(tau)  &(armas_x_eigen_parameter_t){-1, -1, 0.0, 0.0, tau}
  
// Null matrix 
//#define ARMAS_NULL (armas_x_dense_t *)0
  
static inline int _M(armas_x_dense_t *A) {
  return A->rows;
}

static inline int _N(armas_x_dense_t *A) {
  return A->cols;
}

// function that constants
typedef DTYPE (*armas_x_constfunc_t)();

//! Return value for element at [i, j]
typedef DTYPE (*armas_x_valuefunc_t)(int, int);

//! element wise operator functions, v := oper(x)
typedef DTYPE (*armas_x_operator_t)(DTYPE x);
//! element wise operator functions, v := oper(x, y)
typedef DTYPE (*armas_x_operator2_t)(DTYPE x, DTYPE y);

extern armas_x_dense_t *armas_x_init(armas_x_dense_t *m, int r, int c);

extern void armas_x_print(const armas_x_dense_t *m, FILE *out);
extern void armas_x_printf(FILE *out, const char *efmt, const armas_x_dense_t *m);
extern int armas_x_set_consts(armas_x_dense_t *m, armas_x_constfunc_t func, int flags);
extern int armas_x_set_values(armas_x_dense_t *m, armas_x_valuefunc_t func, int flags);

extern int armas_x_allclose(const armas_x_dense_t *A, const armas_x_dense_t *B);
extern int armas_x_intolerance(const armas_x_dense_t *A, const armas_x_dense_t *B, ABSTYPE atol, ABSTYPE rtol);

extern armas_x_dense_t *armas_x_mcopy(armas_x_dense_t *d, const armas_x_dense_t *s);
extern armas_x_dense_t *armas_x_newcopy(const armas_x_dense_t *s);
extern armas_x_dense_t *armas_x_transpose(armas_x_dense_t *d, const armas_x_dense_t *s);

extern void armas_x_make_trm(armas_x_dense_t *m, int flags);

extern int armas_x_mscale(armas_x_dense_t *d, DTYPE alpha, int flags);
extern int armas_x_madd(armas_x_dense_t *d, DTYPE alpha, int flags);
  // element wise operators
extern int armas_x_mul_elems(armas_x_dense_t *A, const armas_x_dense_t *B, int flags);
extern int armas_x_add_elems(armas_x_dense_t *A, const armas_x_dense_t *B, int flags);
extern int armas_x_div_elems(armas_x_dense_t *A, const armas_x_dense_t *B, int flags);
extern int armas_x_sub_elems(armas_x_dense_t *A, const armas_x_dense_t *B, int flags);
extern int armas_x_apply(armas_x_dense_t *A, armas_x_operator_t func, int flags);
extern int armas_x_apply2(armas_x_dense_t *A, armas_x_operator2_t func, DTYPE val, int flags);
  

// -------------------------------------------------------------------------------------------
// inline functions

#ifndef __ARMAS_INLINE 
#define __ARMAS_INLINE extern inline
#endif

//! \brief Test if matrix is a vector.
//! \ingroup matrix
__ARMAS_INLINE
int armas_x_isvector(const armas_x_dense_t *m)
{
  return m && (m->rows == 1 || m->cols == 1);
}

//! \brief Get number of elements in matrix.
//! \ingroup matrix
__ARMAS_INLINE
int64_t armas_x_size(const armas_x_dense_t *m)
{
  return m ? m->rows * m->cols : 0;
}

__ARMAS_INLINE
int64_t armas_x_real_index(const armas_x_dense_t *m, int64_t ix)
{
  if (m->cols == 1 || m->step == m->rows)
    return ix;
  return m->step*(ix / m->rows) + (ix % m->rows);
}

__ARMAS_INLINE
int armas_x_index_valid(const armas_x_dense_t *m, int64_t ix)
{
  return m && (ix >= -armas_x_size(m) && ix < armas_x_size(m));
}



//! \brief Release matrix allocated space
//! \ingroup matrix
__ARMAS_INLINE
void armas_x_release(armas_x_dense_t *m)
{
  if (m && m->__data) {
    free(m->__data);
    m->rows = 0; m->cols = 0; m->step = 0;
    m->elems = (DTYPE *)0;
    m->__data = (void *)0;
    m->__nbytes = 0;
  }
}

//! \brief Allocate a new matrix of size [r, c]
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_alloc(int r, int c)
{
  armas_x_dense_t *m;
  if (r <= 0 || c <= 0) {
    return (armas_x_dense_t *)0;
  }
  m = malloc(sizeof(armas_x_dense_t));
  if ( !m ) return m;
  return armas_x_init(m, r, c);
}

//! \brief Release matrix and its allocated space
//! \ingroup matrix
__ARMAS_INLINE
void armas_x_free(armas_x_dense_t *m)
{
  if ( !m )
    return;
  armas_x_release(m);
  free(m);
}

//! \brief Make matrix with provided buffer.
//! Make **r-by-c** matrix with stride s from buffer elems. Assumes elems is at least
//! of size **r*s**.
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_make(armas_x_dense_t *m, int r, int c, int s, DTYPE *elems)
{
  m->step = s;
  m->rows = r;
  m->cols = c;
  m->elems = elems;
  m->__data = (void *)0;
  m->__nbytes = 0;
  return m;
}


// \brief Make A a column vector of B; A = B[:, c]
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_column(armas_x_dense_t *A, const armas_x_dense_t *B, int c)
{
  if (armas_x_size(B) == 0)
	return (armas_x_dense_t*)0;
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

// \brief Make A a row vector of B; A = B[r, :]
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_row(armas_x_dense_t *A, const armas_x_dense_t *B, int r)
{
  if (armas_x_size(B) == 0)
    return (armas_x_dense_t*)0;
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
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_submatrix_ext(armas_x_dense_t *A, const armas_x_dense_t *B,
                                       int r, int c, int nr, int nc, int step)
{
  if (armas_x_size(B) == 0)
    return (armas_x_dense_t*)0;
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
  return armas_x_make(A, nr, nc, step, &B->elems[c*B->step+r]);
}

//! \brief Make A submatrix view of B.
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_submatrix(armas_x_dense_t *A, const armas_x_dense_t *B,
                                   int r, int c, int nr, int nc)
{
  return armas_x_submatrix_ext(A, B, r, c, nr, nc, B->step);
}

//! \brief Make A submatrix view of B. (Unsafe version without any limit checks.)
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_submatrix_unsafe(armas_x_dense_t *A, const armas_x_dense_t *B,
                                          int r, int c, int nr, int nc)
{
  return armas_x_make(A, nr, nc, B->step, &B->elems[c*B->step+r]);
}

//! \brief Make X subvector of Y
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_subvector(armas_x_dense_t *X, const armas_x_dense_t *Y,
                                   int n, int len)
{
  if (!armas_x_isvector(Y)) {
    X->rows = X->cols = 0;
  } else {
    if (Y->rows == 1) {
      armas_x_submatrix(X, Y, 0, n, 1, len);
    } else {
      armas_x_submatrix(X, Y, n, 0, len, 1);
    }
  }
  return X;
}

//! \brief Make X subvector of Y (Unsafe version without any limit checks.)
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_subvector_unsafe(armas_x_dense_t *X, const armas_x_dense_t *Y,
                                          int n, int len)
{
  if (Y->rows == 1) {
    armas_x_submatrix_unsafe(X, Y, 0, n, 1, len);
  } else {
    armas_x_submatrix_unsafe(X, Y, n, 0, len, 1);
  }
  return X;
}

//! \brief Make A diagonal row vector of B.
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_diag(armas_x_dense_t *A, const armas_x_dense_t *B, int k)
{
  int nk;
  if (k > 0) {
    // super diagonal
    nk = B->rows < B->cols-k ? B->rows : B->cols-k;
    return armas_x_submatrix_ext(A, B, 0, k, 1, nk, B->step+1);
  }
  if (k < 0) {
    // subdiagonal
    nk = B->rows+k < B->cols ? B->rows+k : B->cols;
    return armas_x_submatrix_ext(A, B, -k, 0, 1, nk, B->step+1);
  }
  // main diagonal
  nk = B->rows < B->cols ? B->rows : B->cols;
  return armas_x_submatrix_ext(A, B, 0, 0, 1, nk, B->step+1);
}

//! \brief Make A diagonal row vector of B. (unsafe version).
//! \ingroup matrix
__ARMAS_INLINE
armas_x_dense_t *armas_x_diag_unsafe(armas_x_dense_t *A, const armas_x_dense_t *B, int k)
{
  int nk;
  if (k > 0) {
    // super diagonal (starts at k'th colomn of A)
    nk = B->rows < B->cols-k ? B->rows : B->cols-k;
    return armas_x_make(A, 1, nk, B->step+1, &B->elems[k*B->step]);
  }
  if (k < 0) {
    // subdiagonal (starts at k'th row of A)
    nk = B->rows+k < B->cols ? B->rows+k : B->cols;
    return armas_x_make(A, 1, nk, B->step+1, &B->elems[-k]);
  }
  // main diagonal
  nk = B->rows < B->cols ? B->rows : B->cols;
  return armas_x_make(A, 1, nk, B->step+1, &B->elems[0]);
}

//! \brief Get element at `[row, col]`
__ARMAS_INLINE
DTYPE armas_x_get(const armas_x_dense_t *m, int row, int col)
{
  if (armas_x_size(m) == 0)
    return 0.0;
  if (row < 0)
    row += m->rows;
  if (col < 0)
    col += m->cols;
  return m->elems[col*m->step+row];
}

//! \brief Get unsafely element at `[row, col]`
//! \ingroup matrix
__ARMAS_INLINE
DTYPE armas_x_get_unsafe(const armas_x_dense_t *m, int row, int col)
{
  return m->elems[col*m->step+row];
}

//! \brief Set element at `[row, col]` to `val`
//! \ingroup matrix
__ARMAS_INLINE
void armas_x_set(armas_x_dense_t *m, int row, int col, DTYPE val)
{
  if (armas_x_size(m) == 0)
    return;
  if (row < 0)
    row += m->rows;
  if (col < 0)
    col += m->cols;
  m->elems[col*m->step+row] = val;
}

//! \brief Set element unsafely at `[row, col]` to `val`
//! \ingroup matrix
__ARMAS_INLINE
void armas_x_set_unsafe(armas_x_dense_t *m, int row, int col, DTYPE val)
{
  m->elems[col*m->step+row] = val;
}

//! \brief Set element of vector at index `ix` to `val`.
//! \ingroup matrix
__ARMAS_INLINE
void armas_x_set_at(armas_x_dense_t *m, int ix, DTYPE val)
{
  if (armas_x_size(m) == 0)
    return;
  if (ix < 0)
    ix += armas_x_size(m);
  m->elems[armas_x_real_index(m, ix)] = val;
}

//! \brief Set unsafely element of vector at index `ix` to `val`.
//! \ingroup matrix
__ARMAS_INLINE
void armas_x_set_at_unsafe(armas_x_dense_t *m, int ix, DTYPE val)
{
  m->elems[(m->rows == 1 ? ix*m->step : ix)] = val;
}

//! \brief Get element of vector at index `ix`.
//! \ingroup matrix
__ARMAS_INLINE
DTYPE armas_x_get_at(const armas_x_dense_t *m, int ix)
{
  if (armas_x_size(m) == 0)
    return 0.0;
  if (ix < 0)
    ix += armas_x_size(m);
  return m->elems[armas_x_real_index(m, ix)];
}

//! \brief Get unsafely element of vector at index `ix`.
//! \ingroup matrix
__ARMAS_INLINE
DTYPE armas_x_get_at_unsafe(const armas_x_dense_t *m, int ix)
{
  return m->elems[(m->rows == 1 ? ix*m->step : ix)];
}

__ARMAS_INLINE
int armas_x_index(const armas_x_dense_t *m, int row, int col)
{
  if (armas_x_size(m) == 0)
    return 0;
  if (row < 0)
    row += m->rows;
  if (col < 0)
    col += m->cols;
  return col*m->step + row;
}

//! \brief Get data buffer
//! \ingroup matrix
__ARMAS_INLINE
DTYPE *armas_x_data(const armas_x_dense_t *m)
{
  return m ? m->elems : (DTYPE *)0;
}

__ARMAS_INLINE
armas_x_dense_t *armas_x_col_as_row(armas_x_dense_t *row, armas_x_dense_t *col)
{
  armas_x_make(row, 1, armas_x_size(col), 1, armas_x_data(col));
  return row;
}

// -------------------------------------------------------------------------------------------
// 

#ifndef __ARMAS_LINALG_H

extern int armas_x_scale_plus(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B,
                              int flags, armas_conf_t *conf);
extern ABSTYPE armas_x_mnorm(const armas_x_dense_t *A, int norm, armas_conf_t *conf);
extern ABSTYPE armas_x_norm(const armas_x_dense_t *A, int norm, int flags, armas_conf_t *conf);
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
extern int armas_x_mvupdate(armas_x_dense_t *A,
                            DTYPE alpha, const armas_x_dense_t *X,  const armas_x_dense_t *Y,  
                            armas_conf_t *conf);
extern int armas_x_mvmult_sym(DTYPE beta, armas_x_dense_t *Y,
                              DTYPE alpha, const armas_x_dense_t *A, const armas_x_dense_t *X, 
                              int flags, armas_conf_t *conf);
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
extern int armas_x_mvsolve_trm(armas_x_dense_t *X,  DTYPE alpha, const armas_x_dense_t *A, 
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
extern int armas_x_bdreduce_w(armas_x_dense_t *A, armas_x_dense_t *tauq, armas_x_dense_t *taup,
                              armas_wbuf_t *w, armas_conf_t *conf);
extern int armas_x_bdbuild(armas_x_dense_t *A, const armas_x_dense_t *tau,
                           int K, int flags, armas_conf_t *conf);
extern int armas_x_bdbuild_w(armas_x_dense_t *A, const armas_x_dense_t *tau,
                             int K, int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_bdmult(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                          int flags, armas_conf_t *conf);
extern int armas_x_bdmult_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int flags, armas_wbuf_t *w, armas_conf_t *conf);

// Cholesky
extern int armas_x_cholesky(armas_x_dense_t *A, int flags, armas_conf_t *conf);
extern int armas_x_cholfactor(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_cholfactor_w(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_cholsolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_pivot_t *P,
                             int flags, armas_conf_t *conf);
extern int armas_x_cholupdate(armas_x_dense_t *A, armas_x_dense_t *X, int flags, armas_conf_t *conf);

// Hessenberg reduction
extern int armas_x_hessreduce(armas_x_dense_t *A, armas_x_dense_t *tau, armas_conf_t *conf);
extern int armas_x_hessmult(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int flags, armas_conf_t *conf);
extern int armas_x_hessreduce_w(armas_x_dense_t *A, armas_x_dense_t *tau, armas_wbuf_t *wrk,
                                armas_conf_t *conf);
extern int armas_x_hessmult_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                              int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

// LU
extern int armas_x_lufactor(armas_x_dense_t *A, armas_pivot_t *P, armas_conf_t *conf);
extern int armas_x_lusolve(armas_x_dense_t *B, armas_x_dense_t *A, armas_pivot_t *P,
                           int flags, armas_conf_t *conf);

// Symmetric LDL; Bunch-Kauffman
extern int armas_x_bkfactor(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_bkfactor_w(armas_x_dense_t *A,  armas_pivot_t *P, int flags,
                              armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_bksolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_pivot_t *P,
                           int flags, armas_conf_t *conf);
extern int armas_x_bksolve_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_pivot_t *P,
                             int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
// LDL.T symmetric
extern int armas_x_ldlfactor(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_ldlfactor_w(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_ldlsolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_pivot_t *P, int flags, armas_conf_t *conf);

// LQ functions
extern int armas_x_lqbuild(armas_x_dense_t *A, const armas_x_dense_t *tau, int K, armas_conf_t *conf);
extern int armas_x_lqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_conf_t *conf);
extern int armas_x_lqmult(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                          int flags, armas_conf_t *conf);
extern int armas_x_lqreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_lqsolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                           int flags, armas_conf_t *conf);
extern int armas_x_lqbuild_w(armas_x_dense_t *A, const armas_x_dense_t *tau, int K,
                             armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_lqfactor_w(armas_x_dense_t *A, armas_x_dense_t *tau, armas_wbuf_t *wrk,
                            armas_conf_t *conf);
extern int armas_x_lqmult_w(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_lqsolve_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                             int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

// QL functions
extern int armas_x_qlbuild(armas_x_dense_t *A, const armas_x_dense_t *tau, int K, armas_conf_t *conf);
extern int armas_x_qlfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_conf_t *conf);
extern int armas_x_qlmult(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                          int flags, armas_conf_t *conf);
extern int armas_x_qlreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_qlbuild_w(armas_x_dense_t *A, const armas_x_dense_t *tau, int K,
                             armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_qlfactor_w(armas_x_dense_t *A, armas_x_dense_t *tau, armas_wbuf_t *wrk,
                            armas_conf_t *conf);
extern int armas_x_qlmult_w(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_qlsolve_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                             int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

// QR functions
extern int armas_x_qrbuild(armas_x_dense_t *A, const armas_x_dense_t *tau, int K, armas_conf_t *conf);
extern int armas_x_qrfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_conf_t *conf);
extern int armas_x_qrmult(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                          int flags, armas_conf_t *conf);
extern int armas_x_qrreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_qrsolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                           int flags, armas_conf_t *conf);

extern int armas_x_qrbuild_w(armas_x_dense_t *A, const armas_x_dense_t *tau, int K,
                             armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_qrfactor_w(armas_x_dense_t *A, armas_x_dense_t *tau, armas_wbuf_t *wrk,
                              armas_conf_t *conf);
extern int armas_x_qrmult_w(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_qrsolve_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                             int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

extern int armas_x_qrtfactor(armas_x_dense_t *A, armas_x_dense_t *T, armas_x_dense_t *W, armas_conf_t *conf);
extern int armas_x_qrtmult(armas_x_dense_t *C, armas_x_dense_t *A, armas_x_dense_t *T,
			   armas_x_dense_t *W, int flags, armas_conf_t *conf);

// RQ functions
extern int armas_x_rqbuild(armas_x_dense_t *A, const armas_x_dense_t *tau, int K, armas_conf_t *conf);
extern int armas_x_rqfactor(armas_x_dense_t *A, armas_x_dense_t *tau, armas_conf_t *conf);
extern int armas_x_rqmult(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                          int flags, armas_conf_t *conf);
extern int armas_x_rqreflector(armas_x_dense_t *T, armas_x_dense_t *V, armas_x_dense_t *tau,
                               armas_conf_t *conf);
extern int armas_x_rqsolve(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                           int flags, armas_conf_t *conf);
extern int armas_x_rqbuild_w(armas_x_dense_t *A, const armas_x_dense_t *tau, int K, 
                             armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_rqfactor_w(armas_x_dense_t *A, armas_x_dense_t *tau, armas_wbuf_t *wrk,
                            armas_conf_t *conf);
extern int armas_x_rqmult_w(armas_x_dense_t *C, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_rqsolve_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                             int flags, armas_wbuf_t *W, armas_conf_t *conf);

// Tridiagonal reduction
extern int armas_x_trdreduce(armas_x_dense_t *A, armas_x_dense_t *tau, int flags, armas_conf_t *conf);
extern int armas_x_trdbuild(armas_x_dense_t *A, const armas_x_dense_t *tau,
                            int K, int flags, armas_conf_t *conf);
extern int armas_x_trdmult(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                           int flags, armas_conf_t *conf);
extern int armas_x_trdreduce_w(armas_x_dense_t *A, armas_x_dense_t *tau, int flags,
                               armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_trdbuild_w(armas_x_dense_t *A, const armas_x_dense_t *tau,
                              int K, int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_trdmult_w(armas_x_dense_t *B, const armas_x_dense_t *A, const armas_x_dense_t *tau,
                             int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

extern int armas_x_trdeigen(armas_x_dense_t *D, armas_x_dense_t *E, armas_x_dense_t *V,
                            int flags, armas_conf_t *conf);
extern int armas_x_trdeigen_w(armas_x_dense_t *D, armas_x_dense_t *E, armas_x_dense_t *V,
                              int flags, armas_wbuf_t *wb, armas_conf_t *conf);
extern int armas_x_trdbisect(armas_x_dense_t *Y, armas_x_dense_t *D, armas_x_dense_t *E,
                             const armas_x_eigen_parameter_t *params,  armas_conf_t *conf);
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
extern int armas_x_gvrot_vec(armas_x_dense_t *X, armas_x_dense_t *Y, DTYPE c, DTYPE s);

// Bidiagonal SVD
extern int armas_x_bdsvd(armas_x_dense_t *D, armas_x_dense_t *E, armas_x_dense_t *U, armas_x_dense_t *V,
                         int flags, armas_conf_t *conf);
extern int armas_x_bdsvd_w(armas_x_dense_t *D, armas_x_dense_t *E, armas_x_dense_t *U, armas_x_dense_t *V,
                           int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_bdsvd_work(armas_x_dense_t *D, armas_conf_t *conf);

extern int armas_x_svd(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V, armas_x_dense_t *A,
                       int flags, armas_conf_t *conf);
extern int armas_x_svd_w(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V, armas_x_dense_t *A,
                         int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_svd_work(armas_x_dense_t *D, int flags, armas_conf_t *conf);

// Eigen
extern int armas_x_eigen_sym(armas_x_dense_t *D, armas_x_dense_t *A,
                             armas_x_dense_t *W, int flags, armas_conf_t *conf); 
extern int armas_x_eigen_sym_w(armas_x_dense_t *D, armas_x_dense_t *A,
                               int flags, armas_wbuf_t *wrk, armas_conf_t *conf); 
extern int armas_x_eigen_sym_selected(armas_x_dense_t *D, armas_x_dense_t *A, armas_x_dense_t *W, 
                                      armas_x_eigen_parameter_t *params, int flags, armas_conf_t *conf);
extern int armas_x_eigen_sym_selected_w(armas_x_dense_t *D, armas_x_dense_t *A,
                                        const armas_x_eigen_parameter_t *params,
                                        int flags, armas_wbuf_t *wrk, armas_conf_t *conf);

// DQDS
extern int armas_x_dqds(armas_x_dense_t *D, armas_x_dense_t *E, armas_conf_t *conf);
extern int armas_x_dqds_w(armas_x_dense_t *D, armas_x_dense_t *E, armas_wbuf_t *wrk, armas_conf_t *conf);

// Householder functions
extern int armas_x_house(armas_x_dense_t *a11, armas_x_dense_t *x,
                         armas_x_dense_t *tau, int flags, armas_conf_t *conf);
extern int armas_x_houseapply(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *v,
                              armas_x_dense_t *w,  int flags, armas_conf_t *conf);
extern int armas_x_houseapply2x1(armas_x_dense_t *a1, armas_x_dense_t *A2, armas_x_dense_t *tau, armas_x_dense_t *v,
                                 armas_x_dense_t *w,  int flags, armas_conf_t *conf);
extern int armas_x_housemult(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *Q, int flags, armas_conf_t *conf);
extern int armas_x_housemult_w(armas_x_dense_t *A, armas_x_dense_t *tau, armas_x_dense_t *Q, int flags, armas_wbuf_t *wb, armas_conf_t *conf);

// Hyperbolic Householder functions
extern int armas_x_hhouse(armas_x_dense_t *a11, armas_x_dense_t *x,
                         armas_x_dense_t *tau, int flags, armas_conf_t *conf);
extern int armas_x_hhouse_apply(armas_x_dense_t *tau, armas_x_dense_t *v, armas_x_dense_t *a1,  armas_x_dense_t *A2,
                               armas_x_dense_t *w,  int flags, armas_conf_t *conf);

// Recursive Butterfly
extern int armas_x_mult_rbt(armas_x_dense_t *A, armas_x_dense_t *U, int flags, armas_conf_t *conf);
extern int armas_x_update2_rbt(armas_x_dense_t *A, armas_x_dense_t *U, armas_x_dense_t *V, armas_conf_t *conf);
extern void armas_x_gen_rbt(armas_x_dense_t *U);

// Inverse
extern int armas_x_inverse_trm(armas_x_dense_t *A, int flags, armas_conf_t *conf);
extern int armas_x_inverse(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, armas_conf_t *conf);
extern int armas_x_inverse_psd(armas_x_dense_t *A, int flags, armas_conf_t *conf);
extern int armas_x_ldlinverse_sym(armas_x_dense_t *A, armas_x_dense_t *W, armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_inverse_w(armas_x_dense_t *A, armas_pivot_t *P, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_inverse_psd_w(armas_x_dense_t *A, int flags, armas_wbuf_t *wrk, armas_conf_t *conf);
extern int armas_x_ldlinverse_sym_w(armas_x_dense_t *A, armas_pivot_t *P, int flags,
                                    armas_wbuf_t *wrk, armas_conf_t *conf);

  
// additional
extern int armas_x_qdroots(DTYPE *x1, DTYPE *x2, DTYPE a, DTYPE b, DTYPE c);
extern void armas_x_discriminant(DTYPE *d, DTYPE a, DTYPE b, DTYPE c);
extern int armas_x_mult_diag(armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *D, int flags, armas_conf_t *conf);
extern int armas_x_solve_diag(armas_x_dense_t *A, DTYPE alpha, const armas_x_dense_t *D, int flags, armas_conf_t *conf);

extern int armas_x_pivot(armas_x_dense_t *A, armas_pivot_t *P, unsigned int flags, armas_conf_t *conf);
extern int armas_x_pivot_rows(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);
extern int armas_x_pivot_cols(armas_x_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *conf);

#endif

  
#ifdef __cplusplus
}
#endif



#endif
  

// Local Variables:
// indent-tabs-mode: nil
// End:
