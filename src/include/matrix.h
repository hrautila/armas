
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/**
 * @file 
 * Matrix structure definition and public functions
 */

#ifndef ARMAS_MATRIX_H
#define ARMAS_MATRIX_H 1

#include <stdio.h>
#include <stdlib.h>
/* COMPLEX_H */
#include "armas.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal 'type' independent declaration of matrix type. Public exported
// types are copies of this with explicit element type.
/**
 * @brief Column major matrix type.
 *
 * @ingroup matrix
 */
typedef struct armas_dense {
    DTYPE *elems;  ///< Matrix elements,
    int step;      ///< Row stride
    int rows;      ///< Number of rows
    int cols;      ///< Number of columns
    void *__data;  ///< Non-null if stucture owns elements
    int __nbytes;  ///< sizeof __data buffer
} armas_dense_t;

/**
 * @brief Eigenvalue selection parameters
 * @ingroup lapack
 */
typedef struct armas_eigen_parameter {
    int ileft;    ///< Start index of half-open interval [ileft, iright)
    int iright;   ///< End index of half-open interval [ileft, iright)
    DTYPE left;   ///< Start eigenvalue of half-open interval [left, right)
    DTYPE right;  ///< Last eigenvalue of half-open interval [left, right)
    DTYPE tau;    ///< Requested accurancy; must be > max{|T_i|}*eps
} armas_eigen_parameter_t;

//! @brief Define eigenvalue index range [l, r) with default accuracy
#define ARMAS_EIGEN_INT(l, r) \
    &(armas_eigen_parameter_t) { (l), (r), 0.0, 0.0, 0.0 }

//! @brief Define eigenvalue value range [low, high) with default accuracy
#define ARMAS_EIGEN_VAL(low, high) \
    &(armas_eigen_parameter_t) { 0, 0, (low), (high), 0.0 }

//! @brief Define eigenvalue index range [l, r) and accuracy parameter tau
#define ARMAS_EIGEN_INT_TAU(l, r, tau) \
    &(armas_eigen_parameter_t) { (l), (r), 0.0, 0.0, tau }

//! @brief Define eigenvalue value range [low, high) and accuracy parameter tau
#define ARMAS_EIGEN_VAL_TAU(low, high, tau) \
    &(armas_eigen_parameter_t) { 0, 0, (low), (high), tau }

//! @brief All eigenvalues with default accuracy
#define ARMAS_EIGEN_ALL \
    &(armas_eigen_parameter_t) { -1, -1, 0.0, 0.0, 0.0 }

//! @brief All eigenvalues with accuracy parameter
#define ARMAS_EIGEN_ALL_TAU(tau) \
    &(armas_eigen_parameter_t) { -1, -1, 0.0, 0.0, tau }

//! @brief Function that returns a constants.
typedef DTYPE (*armas_constfunc_t)();

//! @brief Fucntion that returns value for element at [i, j].
typedef DTYPE (*armas_valuefunc_t)(int, int);

//! @brief Element wise operator function oper(elem)
typedef DTYPE (*armas_operator_t)(DTYPE x);

//! @brief Element wise operator functions, v := oper(x, y)
typedef DTYPE (*armas_operator2_t)(DTYPE x, DTYPE y);

// @cond
extern armas_dense_t *armas_init(armas_dense_t *m, int r, int c);

extern void armas_print(const armas_dense_t *m, FILE *out);
extern void armas_printf(FILE *out, const char *efmt, const armas_dense_t *m);
extern int armas_set_consts(armas_dense_t *m, armas_constfunc_t func, int flags);
extern int armas_set_values(armas_dense_t *m, armas_valuefunc_t func, int flags);

extern int armas_allclose(const armas_dense_t *A, const armas_dense_t *B);
extern int armas_intolerance(const armas_dense_t *A, const armas_dense_t *B, ABSTYPE atol, ABSTYPE rtol);

extern int armas_mcopy(armas_dense_t *d, const armas_dense_t *s, int flags, armas_conf_t *cf);
extern armas_dense_t *armas_newcopy(const armas_dense_t *s);

extern void armas_make_trm(armas_dense_t *m, int flags);

extern int armas_madd(armas_dense_t *d, DTYPE alpha, int flags, armas_conf_t *cf);
extern int armas_mscale(armas_dense_t *d, DTYPE alpha, int flags, armas_conf_t *cf);
extern int armas_mplus(DTYPE alpha, armas_dense_t *A, DTYPE beta, const armas_dense_t *B, int flags, armas_conf_t *cf);
// element wise operators
extern int armas_mul_elems(armas_dense_t *A, DTYPE beta, const armas_dense_t *B, int flags);
extern int armas_apply(armas_dense_t *A, armas_operator_t func, int flags);
extern int armas_apply2(armas_dense_t *A, armas_operator2_t func, DTYPE val, int flags);

extern int armas_mmload(armas_dense_t *A, int *flags, FILE *f);
extern int armas_mmdump(FILE *f, const armas_dense_t *A, int flags);
extern int armas_json_read(armas_dense_t **A, armas_iostream_t *ios);
extern int armas_json_write(armas_iostream_t *ios, const armas_dense_t *A, int flags);
extern int armas_json_load(armas_dense_t **A, FILE *fp);
extern int armas_json_dump(FILE *fp, const armas_dense_t *A, int flags);
// @endcond

// -------------------------------------------------------------------------------------------
// inline functions

#ifndef __ARMAS_INLINE
#define __ARMAS_INLINE extern inline
#endif

/**
 * @addtogroup matrix
 * @{
 */
//! @brief Test if matrix is a vector.
__ARMAS_INLINE
int armas_isvector(const armas_dense_t *m)
{
    return m && (m->rows == 1 || m->cols == 1);
}

//! @brief Get number of elements in matrix.
__ARMAS_INLINE
int64_t armas_size(const armas_dense_t *m)
{
    return m ? m->rows * m->cols : 0;
}

__ARMAS_INLINE
int64_t armas_real_index(const armas_dense_t *m, int64_t ix)
{
    if (m->cols == 1 || m->step == m->rows)
        return ix;
    return m->step * (ix / m->rows) + (ix % m->rows);
}

__ARMAS_INLINE
int armas_index_valid(const armas_dense_t *m, int64_t ix)
{
    return m && (ix >= -armas_size(m) && ix < armas_size(m));
}

//! @brief Release matrix allocated space
__ARMAS_INLINE
void armas_release(armas_dense_t *m)
{
    if (m && m->__data) {
        free(m->__data);
        m->rows = 0;
        m->cols = 0;
        m->step = 0;
        m->elems = (DTYPE *)0;
        m->__data = (void *)0;
        m->__nbytes = 0;
    }
}

//! @brief Allocate a new matrix of size [r, c]
__ARMAS_INLINE
armas_dense_t *armas_alloc(int r, int c)
{
    armas_dense_t *m;
    require(r >= 0 && c >= 0);
    if (r <= 0 || c <= 0) {
        return (armas_dense_t *)0;
    }
    m = (armas_dense_t *)malloc(sizeof(armas_dense_t));
    if (!m) return m;
    return armas_init(m, r, c);
}

//! @brief Release matrix and its allocated space
__ARMAS_INLINE
void armas_free(armas_dense_t *m)
{
    if (!m)
        return;
    armas_release(m);
    free(m);
}

//! @brief Make matrix with provided buffer.
//! Make **r-by-c** matrix with stride s from buffer elems. Assumes elems is at least
//! of size **r*s**.
__ARMAS_INLINE
armas_dense_t *armas_make(armas_dense_t *m, int r, int c, int s, DTYPE *elems)
{
    m->step = s;
    m->rows = r;
    m->cols = c;
    m->elems = elems;
    m->__data = (void *)0;
    m->__nbytes = 0;
    return m;
}

//! @brief Make A a column vector of B; A = B[:, c]
__ARMAS_INLINE
armas_dense_t *armas_column(armas_dense_t *A, const armas_dense_t *B, int c)
{
    if (armas_size(B) == 0)
        return (armas_dense_t *)0;
    if (c < 0) {
        c += B->cols;
    }
    require(c >= 0 && c < B->cols);
    A->cols = 1;
    A->rows = B->rows;
    A->step = B->step;
    A->elems = &B->elems[c * B->step];
    A->__data = (void *)0;
    A->__nbytes = 0;
    return A;
}

__ARMAS_INLINE
armas_dense_t *armas_column_unsafe(armas_dense_t *A, const armas_dense_t *B, int c)
{
    require(c >= 0 && c < B->cols);
    A->cols = 1;
    A->rows = B->rows;
    A->step = B->step;
    A->elems = &B->elems[c * B->step];
    A->__data = (void *)0;
    A->__nbytes = 0;
    return A;
}

//! @brief Make A a row vector of B; A = B[r, :]
__ARMAS_INLINE
armas_dense_t *armas_row(armas_dense_t *A, const armas_dense_t *B, int r)
{
    if (armas_size(B) == 0)
        return (armas_dense_t *)0;
    if (r < 0) {
        r += B->rows;
    }
    require(r >= 0 && r < B->rows);
    A->rows = 1;
    A->cols = B->cols;
    A->step = B->step;
    A->elems = &B->elems[r];
    A->__data = (void *)0;
    A->__nbytes = 0;
    return A;
}

__ARMAS_INLINE
armas_dense_t *armas_row_unsafe(armas_dense_t *A, const armas_dense_t *B, int r)
{
    require(r >= 0 && r < B->rows);
    A->rows = 1;
    A->cols = B->cols;
    A->step = B->step;
    A->elems = &B->elems[r];
    A->__data = (void *)0;
    A->__nbytes = 0;
    return A;
}

//! @brief Make A a submatrix view of B with spesified row stride.
__ARMAS_INLINE
armas_dense_t *armas_submatrix_ext(armas_dense_t *A, const armas_dense_t *B,
                                       int r, int c, int nr, int nc, int step)
{
    if (armas_size(B) == 0)
        return (armas_dense_t *)0;
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
    require(r >= 0 && (r < B->rows || nr == 0));
    require(c >= 0 && (c < B->cols || nc == 0));
    require(nr <= B->rows - r);
    require(nc <= B->cols - c);
    return armas_make(A, nr, nc, step, &B->elems[c * B->step + r]);
}

//! @brief Make A submatrix view of B.
__ARMAS_INLINE
armas_dense_t *armas_submatrix(armas_dense_t *A, const armas_dense_t *B,
                                   int r, int c, int nr, int nc)
{
    return armas_submatrix_ext(A, B, r, c, nr, nc, B->step);
}

//! @brief Make A submatrix view of B. (Unsafe version without any limit checks.)
__ARMAS_INLINE
armas_dense_t *armas_submatrix_unsafe(armas_dense_t *A, const armas_dense_t *B,
                                          int r, int c, int nr, int nc)
{
    return armas_make(A, nr, nc, B->step, &B->elems[c * B->step + r]);
}

//! @brief Make X subvector of Y
__ARMAS_INLINE
armas_dense_t *armas_subvector(armas_dense_t *X, const armas_dense_t *Y,
                                   int n, int len)
{
    if (!armas_isvector(Y)) {
        X->rows = X->cols = 0;
    } else {
        if (Y->rows == 1) {
            armas_submatrix(X, Y, 0, n, 1, len);
        } else {
            armas_submatrix(X, Y, n, 0, len, 1);
        }
    }
    return X;
}

//! @brief Make X subvector of Y (Unsafe version without any limit checks.)
__ARMAS_INLINE
armas_dense_t *armas_subvector_unsafe(armas_dense_t *X, const armas_dense_t *Y,
                                          int n, int len)
{
    if (Y->rows == 1) {
        armas_submatrix_unsafe(X, Y, 0, n, 1, len);
    } else {
        armas_submatrix_unsafe(X, Y, n, 0, len, 1);
    }
    return X;
}

//! @brief Make A diagonal row vector of B.
__ARMAS_INLINE
armas_dense_t *armas_diag(armas_dense_t *A, const armas_dense_t *B, int k)
{
    int nk;
    if (k > 0) {
        // super diagonal
        require(k < B->cols);
        nk = B->rows < B->cols - k ? B->rows : B->cols - k;
        return armas_submatrix_ext(A, B, 0, k, 1, nk, B->step + 1);
    }
    if (k < 0) {
        // subdiagonal
        require(-k < B->rows);
        nk = B->rows + k < B->cols ? B->rows + k : B->cols;
        return armas_submatrix_ext(A, B, -k, 0, 1, nk, B->step + 1);
    }
    // main diagonal
    nk = B->rows < B->cols ? B->rows : B->cols;
    return armas_submatrix_ext(A, B, 0, 0, 1, nk, B->step + 1);
}

//! @brief Make A diagonal row vector of B. (unsafe version).
__ARMAS_INLINE
armas_dense_t *armas_diag_unsafe(armas_dense_t *A, const armas_dense_t *B, int k)
{
    int nk;
    if (k > 0) {
        // super diagonal (starts at k'th colomn of A)
        require(k < B->cols);
        nk = B->rows < B->cols - k ? B->rows : B->cols - k;
        return armas_make(A, 1, nk, B->step + 1, &B->elems[k * B->step]);
    }
    if (k < 0) {
        // subdiagonal (starts at k'th row of A)
        require(-k < B->rows);
        nk = B->rows + k < B->cols ? B->rows + k : B->cols;
        return armas_make(A, 1, nk, B->step + 1, &B->elems[-k]);
    }
    // main diagonal
    nk = B->rows < B->cols ? B->rows : B->cols;
    return armas_make(A, 1, nk, B->step + 1, &B->elems[0]);
}

//! @brief Get element at `[row, col]`
__ARMAS_INLINE
DTYPE armas_get(const armas_dense_t *m, int row, int col)
{
    if (armas_size(m) == 0)
        return 0.0;
    if (row < 0)
        row += m->rows;
    if (col < 0)
        col += m->cols;
    require(row < m->rows && col < m->cols);
    return m->elems[col * m->step + row];
}

//! @brief Get unsafely element at `[row, col]`
__ARMAS_INLINE
DTYPE armas_get_unsafe(const armas_dense_t *m, int row, int col)
{
    require(row < m->rows && col < m->cols);
    return m->elems[col * m->step + row];
}

//! @brief Set element at `[row, col]` to `val`
__ARMAS_INLINE
void armas_set(armas_dense_t *m, int row, int col, DTYPE val)
{
    if (armas_size(m) == 0)
        return;
    if (row < 0)
        row += m->rows;
    if (col < 0)
        col += m->cols;
    require(row < m->rows && col < m->cols);
    m->elems[col * m->step + row] = val;
}

//! @brief Set element unsafely at `[row, col]` to `val`
__ARMAS_INLINE
void armas_set_unsafe(armas_dense_t *m, int row, int col, DTYPE val)
{
    require(row < m->rows && col < m->cols);
    m->elems[col * m->step + row] = val;
}

//! @brief Set element of vector at index `ix` to `val`.
__ARMAS_INLINE
void armas_set_at(armas_dense_t *m, int ix, DTYPE val)
{
    if (armas_size(m) == 0)
        return;
    if (ix < 0)
        ix += armas_size(m);
    require(ix < m->rows*m->cols);
    m->elems[armas_real_index(m, ix)] = val;
}

//! @brief Set unsafely element of vector at index `ix` to `val`.
__ARMAS_INLINE
void armas_set_at_unsafe(armas_dense_t *m, int ix, DTYPE val)
{
    require(ix < m->rows*m->cols);
    m->elems[(m->rows == 1 ? ix * m->step : ix)] = val;
}

//! @brief Get element of vector at index `ix`.
__ARMAS_INLINE
DTYPE armas_get_at(const armas_dense_t *m, int ix)
{
    if (armas_size(m) == 0)
        return 0.0;
    if (ix < 0)
        ix += armas_size(m);
    require(ix < m->rows*m->cols);
    return m->elems[armas_real_index(m, ix)];
}

//! @brief Get unsafely element of vector at index `ix`.
__ARMAS_INLINE
DTYPE armas_get_at_unsafe(const armas_dense_t *m, int ix)
{
    require(ix < m->rows*m->cols);
    return m->elems[(m->rows == 1 ? ix * m->step : ix)];
}

//! @brief Get index to element at [row, col].
__ARMAS_INLINE
int armas_index(const armas_dense_t *m, int row, int col)
{
    if (armas_size(m) == 0)
        return 0;
    if (row < 0)
        row += m->rows;
    if (col < 0)
        col += m->cols;
    require(row < m->rows && col < m->cols);
    return col * m->step + row;
}

__ARMAS_INLINE
int armas_index_unsafe(const armas_dense_t *m, int row, int col)
{
    require(row < m->rows && col < m->cols);
    return m ? col * m->step + row : 0;
}

//! @brief Get data buffer
__ARMAS_INLINE
DTYPE *armas_data(const armas_dense_t *m)
{
    return m ? m->elems : (DTYPE *)0;
}

__ARMAS_INLINE
armas_dense_t *armas_col_as_row(armas_dense_t *row, armas_dense_t *col)
{
    armas_make(row, 1, armas_size(col), 1, armas_data(col));
    return row;
}
/*! @} */

// -------------------------------------------------------------------------------------------
//
/*! @cond */

extern int armas_scale_plus(DTYPE alpha, armas_dense_t *A, DTYPE beta, const armas_dense_t *B,
                              int flags, armas_conf_t *cf);
extern ABSTYPE armas_mnorm(const armas_dense_t *A, int norm, armas_conf_t *cf);
extern ABSTYPE armas_norm(const armas_dense_t *A, int norm, int flags, armas_conf_t *cf);
extern int armas_scale_to(armas_dense_t *A, DTYPE from, DTYPE to, int flags, armas_conf_t *cf);

// Blas level 1 functions
extern int armas_iamax(const armas_dense_t *X, armas_conf_t *cf);
extern ABSTYPE armas_amax(const armas_dense_t *X, armas_conf_t *cf);
extern ABSTYPE armas_asum(const armas_dense_t *X, armas_conf_t *cf);
extern ABSTYPE armas_nrm2(const armas_dense_t *X, armas_conf_t *cf);
extern DTYPE armas_dot(const armas_dense_t *X, const armas_dense_t *Y, armas_conf_t *cf);
extern int armas_adot(DTYPE *result, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y, armas_conf_t *cf);
extern int armas_axpy(armas_dense_t *Y, DTYPE alpha, const armas_dense_t *X, armas_conf_t *cf);
extern int armas_axpby(DTYPE beta, armas_dense_t *Y, DTYPE alpha, const armas_dense_t *X, armas_conf_t *cf);
extern int armas_copy(armas_dense_t *Y, const armas_dense_t *X, armas_conf_t *cf);
extern int armas_swap(armas_dense_t *Y, armas_dense_t *X, armas_conf_t *cf);

extern DTYPE armas_sum(const armas_dense_t *X, armas_conf_t *cf);
extern int armas_scale(armas_dense_t *X, const DTYPE alpha, armas_conf_t *cf);
extern int armas_add(armas_dense_t *X, const DTYPE alpha, armas_conf_t *cf);

// Blas level 2 functions
extern int armas_mvmult(
    DTYPE beta, armas_dense_t *Y, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *X,
    int flags, armas_conf_t *cf);
extern int armas_mvupdate(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y, armas_conf_t *cf);
extern int armas_mvmult_sym(
    DTYPE beta, armas_dense_t *Y, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *X,
    int flags, armas_conf_t *cf);
extern int armas_mvupdate2_sym(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y,
    int flags, armas_conf_t *cf);
extern int armas_mvupdate_sym(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, int flags, armas_conf_t *cf);
extern int armas_mvupdate_trm(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y,
    int flags, armas_conf_t *cf);
extern int armas_mvmult_trm(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);
extern int armas_mvsolve_trm(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

// Blas level 3 functions
extern int armas_mult(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);

extern int armas_mult_sym(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);

extern int armas_mult_trm(
    armas_dense_t *B, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_solve_trm(
    armas_dense_t *B, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_update_trm(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);

extern int armas_update_sym(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_update2_sym(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);

// Extendded precision BLAS
extern int armas_ext_adot(
    DTYPE *result, DTYPE alpha, const armas_dense_t *x, const armas_dense_t *y, armas_conf_t *cf);

extern int armas_ext_asum(
    DTYPE *result, DTYPE alpha, const armas_dense_t *x, int flags, armas_conf_t *cf);

extern int armas_ext_axpby(
    DTYPE beta, armas_dense_t *Y, DTYPE alpha, const armas_dense_t *X, armas_conf_t *cf);

extern int armas_ext_mvmult(
    DTYPE beta, armas_dense_t *y, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *x,
    int flags, armas_conf_t *cf);

extern int armas_ext_mvmult_trm(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_ext_mvsolve_trm_w(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_wbuf_t *w, armas_conf_t *cf);

extern int armas_ext_mvsolve_trm(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_ext_mvmult_sym(
    DTYPE beta, armas_dense_t *y, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *x,
    int flags, armas_conf_t *cf);

extern int armas_ext_mvupdate(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y, armas_conf_t *cf);

extern int armas_ext_mvupdate2_sym(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y,
    int flags, armas_conf_t *cf);

extern int armas_ext_mvupdate_sym(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, int flags, armas_conf_t *cf);

extern int armas_ext_mvupdate_trm(
    DTYPE beta, armas_dense_t *A, DTYPE alpha, const armas_dense_t *X, const armas_dense_t *Y,
    int flags, armas_conf_t *cf);

extern int armas_ext_mult(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);

extern int armas_ext_mult_trm(
    armas_dense_t *B, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_ext_mult_sym(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);

extern int armas_ext_solve_trm_w(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_wbuf_t *w, armas_conf_t *cf);

extern int armas_ext_solve_trm(
    armas_dense_t *X, DTYPE alpha, const armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_ext_update_trm(
    DTYPE beta, armas_dense_t *C, DTYPE alpha, const armas_dense_t *A, const armas_dense_t *B,
    int flags, armas_conf_t *cf);
// Lapack

// Bidiagonal reduction
extern int armas_bdreduce(
    armas_dense_t *A, armas_dense_t *tauq, armas_dense_t *taup, armas_conf_t *cf);

extern int armas_bdreduce_w(
    armas_dense_t *A, armas_dense_t *tauq, armas_dense_t *taup, armas_wbuf_t *w, armas_conf_t *cf);

extern int armas_bdbuild(
    armas_dense_t *A, const armas_dense_t *tau, int K, int flags, armas_conf_t *cf);

extern int armas_bdbuild_w(
    armas_dense_t *A, const armas_dense_t *tau, int K, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_bdmult(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_bdmult_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *w, armas_conf_t *cf);

// Cholesky
extern int armas_cholesky(armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_cholfactor(armas_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_cholfactor_w(armas_dense_t *A, armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_cholsolve(
    armas_dense_t *B, const armas_dense_t *A, const armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_cholupdate(armas_dense_t *A, armas_dense_t *X, int flags, armas_conf_t *cf);

// Hessenberg reduction
extern int armas_hessreduce(armas_dense_t *A, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_hessmult(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_hessreduce_w(
    armas_dense_t *A, armas_dense_t *tau, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_hessmult_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// LU
extern int armas_lufactor(armas_dense_t *A, armas_pivot_t *P, armas_conf_t *cf);

extern int armas_lusolve(armas_dense_t *B, armas_dense_t *A, armas_pivot_t *P,
                           int flags, armas_conf_t *cf);

// Symmetric LDL; Bunch-Kauffman
extern int armas_bkfactor(armas_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_bkfactor_w(
    armas_dense_t *A, armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_bksolve(
    armas_dense_t *B, const armas_dense_t *A, const armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_bksolve_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// LDL.T symmetric
extern int armas_ldlfactor(armas_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_ldlfactor_w(armas_dense_t *A, armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_ldlsolve(armas_dense_t *B, const armas_dense_t *A, const armas_pivot_t *P, int flags, armas_conf_t *cf);

// LQ functions
extern int armas_lqbuild(armas_dense_t *A, const armas_dense_t *tau, int K, armas_conf_t *cf);

extern int armas_lqfactor(armas_dense_t *A, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_lqmult(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_lqreflector(
    armas_dense_t *T, armas_dense_t *V, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_lqsolve(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_lqbuild_w(
    armas_dense_t *A, const armas_dense_t *tau, int K, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_lqfactor_w(
    armas_dense_t *A, armas_dense_t *tau, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_lqmult_w(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_lqsolve_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// QL functions
extern int armas_qlbuild(armas_dense_t *A, const armas_dense_t *tau, int K, armas_conf_t *cf);

extern int armas_qlfactor(armas_dense_t *A, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_qlmult(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_qlreflector(
    armas_dense_t *T, armas_dense_t *V, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_qlbuild_w(
    armas_dense_t *A, const armas_dense_t *tau, int K, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qlfactor_w(
    armas_dense_t *A, armas_dense_t *tau, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qlmult_w(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qlsolve_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// QR functions
extern int armas_qrbuild(armas_dense_t *A, const armas_dense_t *tau, int K, armas_conf_t *cf);

extern int armas_qrfactor(armas_dense_t *A, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_qrmult(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_qrreflector(
    armas_dense_t *T, armas_dense_t *V, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_qrsolve(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_qrbuild_w(
    armas_dense_t *A, const armas_dense_t *tau, int K, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qrfactor_w(
    armas_dense_t *A, armas_dense_t *tau, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qrmult_w(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qrsolve_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_qrtfactor(armas_dense_t *A, armas_dense_t *T, armas_conf_t *cf);

extern int armas_qrtfactor_w(armas_dense_t *A, armas_dense_t *T, armas_wbuf_t *wb, armas_conf_t *cf);

extern int armas_qrtmult(
    armas_dense_t *C, armas_dense_t *A, armas_dense_t *T, armas_dense_t *W, int flags, armas_conf_t *cf);

// RQ functions
extern int armas_rqbuild(armas_dense_t *A, const armas_dense_t *tau, int K, armas_conf_t *cf);

extern int armas_rqfactor(armas_dense_t *A, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_rqmult(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_rqreflector(
    armas_dense_t *T, armas_dense_t *V, armas_dense_t *tau, armas_conf_t *cf);

extern int armas_rqsolve(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_rqbuild_w(
    armas_dense_t *A, const armas_dense_t *tau, int K, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_rqfactor_w(
    armas_dense_t *A, armas_dense_t *tau, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_rqmult_w(
    armas_dense_t *C, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_rqsolve_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *W, armas_conf_t *cf);

// Tridiagonal reduction
extern int armas_trdreduce(armas_dense_t *A, armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_trdbuild(
    armas_dense_t *A, const armas_dense_t *tau, int K, int flags, armas_conf_t *cf);

extern int armas_trdmult(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_trdreduce_w(
    armas_dense_t *A, armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_trdbuild_w(
    armas_dense_t *A, const armas_dense_t *tau, int K, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_trdmult_w(
    armas_dense_t *B, const armas_dense_t *A, const armas_dense_t *tau, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_trdeigen(
    armas_dense_t *D, armas_dense_t *E, armas_dense_t *V, int flags, armas_conf_t *cf);

extern int armas_trdeigen_w(
    armas_dense_t *D, armas_dense_t *E, armas_dense_t *V, int flags, armas_wbuf_t *wb, armas_conf_t *cf);

extern int armas_trdbisect(
    armas_dense_t *Y, armas_dense_t *D, armas_dense_t *E, const armas_eigen_parameter_t *params, armas_conf_t *cf);

// Secular functions solvers
extern int armas_trdsec_solve(
    armas_dense_t *y, armas_dense_t *d, armas_dense_t *z, armas_dense_t *delta, DTYPE rho, armas_conf_t *cf);

extern int armas_trdsec_solve_vec(
    armas_dense_t *y, armas_dense_t *v, armas_dense_t *Qd, armas_dense_t *d, armas_dense_t *z, DTYPE rho, armas_conf_t *cf);

extern int armas_trdsec_eigen(
    armas_dense_t *Q, armas_dense_t *v, armas_dense_t *Qd, armas_conf_t *cf);

// Givens
extern void armas_gvcompute(DTYPE *c, DTYPE *s, DTYPE *r, DTYPE a, DTYPE b);

extern void armas_gvrotate(DTYPE *v0, DTYPE *v1, DTYPE c, DTYPE s, DTYPE y0, DTYPE y1);

extern void armas_gvleft(armas_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol);

extern void armas_gvright(armas_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol);

extern int armas_gvupdate(
    armas_dense_t *A, int start, armas_dense_t *C, armas_dense_t *S, int nrot, int flags);

extern int armas_gvrot_vec(armas_dense_t *X, armas_dense_t *Y, DTYPE c, DTYPE s);

// Bidiagonal SVD
extern int armas_bdsvd(
    armas_dense_t *D, armas_dense_t *E, armas_dense_t *U, armas_dense_t *V, int flags, armas_conf_t *cf);

extern int armas_bdsvd_w(
    armas_dense_t *D, armas_dense_t *E, armas_dense_t *U, armas_dense_t *V, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_svd(
    armas_dense_t *S, armas_dense_t *U, armas_dense_t *V, armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_svd_w(
    armas_dense_t *S, armas_dense_t *U, armas_dense_t *V, armas_dense_t *A, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// Eigen
extern int armas_eigen_sym(
    armas_dense_t *D, armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_eigen_sym_w(
    armas_dense_t *D, armas_dense_t *A, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_eigen_sym_selected(
    armas_dense_t *D, armas_dense_t *A, const armas_eigen_parameter_t *params, int flags, armas_conf_t *cf);

extern int armas_eigen_sym_selected_w(
    armas_dense_t *D, armas_dense_t *A, const armas_eigen_parameter_t *params, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// DQDS
extern int armas_dqds(armas_dense_t *D, armas_dense_t *E, armas_conf_t *cf);

extern int armas_dqds_w(armas_dense_t *D, armas_dense_t *E, armas_wbuf_t *wrk, armas_conf_t *cf);

// Householder functions
extern int armas_house(
    armas_dense_t *a11, armas_dense_t *x, armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_houseapply(
    armas_dense_t *A, armas_dense_t *tau, armas_dense_t *v, armas_dense_t *w, int flags, armas_conf_t *cf);

extern int armas_houseapply2x1(
    armas_dense_t *a1, armas_dense_t *A2, armas_dense_t *tau, armas_dense_t *v, armas_dense_t *w, int flags, armas_conf_t *cf);

extern int armas_housemult(armas_dense_t *A, armas_dense_t *tau, armas_dense_t *Q, int flags, armas_conf_t *cf);

extern int armas_housemult_w(armas_dense_t *A, armas_dense_t *tau, armas_dense_t *Q, int flags, armas_wbuf_t *wb, armas_conf_t *cf);

// Hyperbolic Householder functions
extern int armas_hhouse(
    armas_dense_t *a11, armas_dense_t *x, armas_dense_t *tau, int flags, armas_conf_t *cf);

extern int armas_hhouse_apply(
    armas_dense_t *tau, armas_dense_t *v, armas_dense_t *a1, armas_dense_t *A2, armas_dense_t *w, int flags, armas_conf_t *cf);

// Recursive Butterfly
extern int armas_mult_rbt(armas_dense_t *A, armas_dense_t *U, int flags, armas_conf_t *cf);

extern int armas_update2_rbt(armas_dense_t *A, armas_dense_t *U, armas_dense_t *V, armas_conf_t *cf);

extern void armas_gen_rbt(armas_dense_t *U);

// Inverse
extern int armas_inverse_trm(armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_luinverse(armas_dense_t *A, const armas_pivot_t *P, armas_conf_t *cf);

extern int armas_cholinverse(armas_dense_t *A, int flags, armas_conf_t *cf);

extern int armas_ldlinverse(armas_dense_t *A, const armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_luinverse_w(armas_dense_t *A, const armas_pivot_t *P, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_cholinverse_w(armas_dense_t *A, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

extern int armas_ldlinverse_w(
    armas_dense_t *A, const armas_pivot_t *P, int flags, armas_wbuf_t *wrk, armas_conf_t *cf);

// additional
extern int armas_qdroots(DTYPE *x1, DTYPE *x2, DTYPE a, DTYPE b, DTYPE c);

extern void armas_discriminant(DTYPE *d, DTYPE a, DTYPE b, DTYPE c);

extern int armas_mult_diag(armas_dense_t *A, DTYPE alpha, const armas_dense_t *D, int flags, armas_conf_t *cf);

extern int armas_solve_diag(armas_dense_t *A, DTYPE alpha, const armas_dense_t *D, int flags, armas_conf_t *cf);

extern int armas_pivot(armas_dense_t *A, armas_pivot_t *P, unsigned int flags, armas_conf_t *cf);

extern int armas_pivot_rows(armas_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *cf);

extern int armas_pivot_cols(armas_dense_t *A, armas_pivot_t *P, int flags, armas_conf_t *cf);

/*! @endcond */

#ifdef __cplusplus
}
#endif

#endif /* ARMAS_MATRIX_H */
