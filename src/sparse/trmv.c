
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_mvmult_trm) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <armas/armas.h>
#include "matrix.h"
#include "sparse.h"

// b = alpha*A*x; A upper triangular
static
int csc_mvmult_un(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xj, *Ae = A->elems.v;
    int k;
    // start from upper left corner; update with linear combination
    for (int j = 0; j < A->cols; j++) {
        xj = x[j];
        for (k = A->ptr[j]; k < A->ptr[j+1] && A->ix[k] <= j; k++) {
            x[A->ix[k]] += alpha * Ae[k] * xj;
        }
        if (unit) {
            x[j] = alpha * xj;
        } else {
            // k-1 in index to last row in this column; if on diagonal 
            x[j] = A->ix[k-1] == j ? alpha * Ae[k-1] * xj : 0.0;
        }
    }
    return 0;
}

// b = alpha*A*x; A upper triangular
static
int csr_mvmult_un(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xv, *Ae = A->elems.v;
    int k;
    // start from upper left corner
    for (int j = 0; j < A->rows; j++) {
        xv = unit != 0 ? x[j] : 0.0;
        // skip positions below diagonal; if unit diagonal then skip diagonal
        for (k = A->ptr[j]; A->ix[k] < j + unit; k++);
        for ( ; k < A->ptr[j+1]; k++) {
            xv += Ae[k] * x[A->ix[k]];
        }
        x[j] = alpha*xv;
    }
    return 0;
}

// b = alpha*A^T*x; A upper triangular
static
int csc_mvmult_ut(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    // start from lower right corner
    DTYPE xv, *Ae = A->elems.v;
    int j, k;
    for (j = A->cols-1; j >= 0; j--) {
        xv = unit != 0 ? x[j] : 0.0;
        // per column only rows that are less or equal to column
        for (k = A->ptr[j]; k < A->ptr[j+1] && A->ix[k] <= j - unit; k++) {
            xv += Ae[k] * x[A->ix[k]];
        }
        x[j] = alpha*xv;
    }
    return 0;
}

static
int csr_mvmult_ut(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xj, *Ae = A->elems.v;
    int k;
    // start from lower right corner
    for (int j = A->rows-1; j >= 0; j--) {
        xj = x[j];
        for (k = A->ptr[j+1]-1; k >= A->ptr[j] && A->ix[k] > j; k--) {
            x[A->ix[k]] += alpha * Ae[k] * xj;
        }
        // diagonal element
        if (unit != 0) {
            x[j] = alpha * xj;
        } else {
            x[j] = A->ix[k] == j ? alpha * Ae[k] * xj : 0.0;
        }
    }
    return 0;
}

// b = alpha*A*x; A lower triangular
static
int csc_mvmult_ln(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xj, *Ae = A->elems.v;
    int i, k;
    // start from lower right corner
    for (int j = A->cols-1; j >= 0; j--) {
        xj = x[j];
        // skip position above diagonal
        for (k = A->ptr[j]; A->ix[k] < j + unit; k++);
        // update x rows below diagonal
        for (i = k; i < A->ptr[j+1]; i++) {
            x[A->ix[i]] += alpha * Ae[i] * xj;
        }
        // compute diagonal entry;
        if (unit != 0) {
            x[j] = alpha * xj;
        } else {
            x[j] = A->ix[k] == j ? alpha * Ae[k] * xj : 0.0;
        }
    }
    return 0;
}

// b = alpha*A*x; A lower triangular
static
int csr_mvmult_ln(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xv, *Ae = A->elems.v;
    int k, j;
    for (j = A->rows-1; j >= 0; j--) {
        xv = unit ? x[j] : 0.0;
        for (k = A->ptr[j]; k < A->ptr[j+1] && A->ix[k] <= j - unit; k++) {
            xv += Ae[k] * x[A->ix[k]];
        }
        x[j] = alpha*xv;
    }
    return 0;
}

// b = alpha*A^T*x; A lower triangular
static
int csc_mvmult_lt(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xv;
    int k, j;
    for (j = 0; j < A->cols; j++) {
        xv = unit != 0 ? x[j] : 0.0;
        // skip rows above diagonal
        for (k = A->ptr[j]; A->ix[k] < j + unit; k++);
        for (; k < A->ptr[j+1]; k++) {
            xv += A->elems.v[k] * x[A->ix[k]];
        }
        x[j] = alpha*xv;
    }
    return 0;
}

static
int csr_mvmult_lt(DTYPE *x, DTYPE alpha, const armas_x_sparse_t *A, int unit)
{
    DTYPE xj, *Ae = A->elems.v;
    int p;
    for (int j = 0; j < A->rows; j++) {
        xj = x[j];
        // update earlier rows with current entry
        for (p = A->ptr[j]; p < A->ptr[j+1] && A->ix[p] < j; p++) {
            x[A->ix[p]] += alpha * Ae[p] * xj;
        }
        // compute diagonal entry;
        if (unit != 0) {
            x[j] = alpha * xj;
        } else {
            x[j] = A->ix[p] == j ? alpha * Ae[p] * xj : 0.0;
        }
    }
    return 0;
}

/**
 * \brief Compute x = alpha*op(A)*b
 *
 * \param[in,out] x
 *     On entry the vector b, on exit result vector x.
 * \param[in] alpha
 *     Coefficient
 * \param[in] A
 *     Upper or lower triangular matrix A in column or row compressed storage format,
 *     per column rows or per row columns are sorted in ascending order.
 * \param[in] flags
 *     For lower (upper) triangular matrix A set to ARMAS_LOWER (ARMAS_UPPER). 
 *     If ARMAS_TRANS is  set then use transpose of A.
 *
 */
int armassp_x_mvmult_trm(armas_x_dense_t *x, DTYPE alpha, const armas_x_sparse_t *A, int flags, armas_conf_t *cf)
{
    if (!cf)
        cf = armas_conf_default();
    
    if (A->kind != ARMASSP_CSC && A->kind != ARMASSP_CSR)
        return -1;
    
    int ok = (flags & ARMAS_TRANS) == 0
        ? armas_x_size(x) == A->cols
        : armas_x_size(x) == A->rows;
    if (!ok) {
        return -1;
    }
    
    int unit = (flags & ARMAS_UNIT) != 0 ? 1 : 0;

    DTYPE *y = armas_x_data(x);
    
    // TODO: we assume column vector here; how about a row vector??
    switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANS)) {
    case ARMAS_UPPER|ARMAS_TRANS:
        if (A->kind == ARMASSP_CSR)
            csr_mvmult_ut(y, alpha, A, unit);
        else 
            csc_mvmult_ut(y, alpha, A, unit);
        break;
    case ARMAS_UPPER:
        if (A->kind == ARMASSP_CSR)
            csr_mvmult_un(y, alpha, A, unit);
        else
            csc_mvmult_un(y, alpha, A, unit);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
    case ARMAS_TRANS:
        if (A->kind == ARMASSP_CSR)
            csr_mvmult_lt(y, alpha, A, unit);
        else
            csc_mvmult_lt(y, alpha, A, unit);
        break;
    case ARMAS_LOWER:
    default:
        if (A->kind == ARMASSP_CSR)           
            csr_mvmult_ln(y, alpha, A, unit);
        else
            csc_mvmult_ln(y, alpha, A, unit);
        break;
    }
    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
