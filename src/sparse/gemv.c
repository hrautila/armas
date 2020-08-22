
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_mvmult)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "armas.h"
#include "matrix.h"
#include "sparse.h"

// CSC storage
static
int csc_mvmult(DTYPE beta, DTYPE * y, DTYPE alpha, const armas_x_sparse_t * A,
               const DTYPE * x, int flags)
{
    DTYPE ax, *Ae = A->elems.v;
    if (flags == 0) {

        for (int i = 0; i < A->rows; i++)
            y[i] *= beta;

        // update with linear combinations of A's columns
        for (int j = 0; j < A->cols; j++) {
            ax = alpha * x[j];
            for (int k = A->ptr[j]; k < A->ptr[j + 1]; k++) {
                y[A->ix[k]] += ax * Ae[k];
            }
        }
    } else {
        // y + alpha*A^T*x
        for (int j = 0; j < A->cols; j++) {
            ax = 0;
            for (int k = A->ptr[j]; k < A->ptr[j + 1]; k++) {
                ax += Ae[k] * x[A->ix[k]];
            }
            y[j] = beta * y[j] + alpha * ax;
        }
    }
    return 0;
}

static
int csr_mvmult(DTYPE beta, DTYPE * y, DTYPE alpha, const armas_x_sparse_t * A,
               const DTYPE * x, int flags)
{
    DTYPE ax, *Ae = A->elems.v;
    if (flags == 0) {
        // y + alpha*A*x
        for (int j = 0; j < A->rows; j++) {
            ax = 0;
            for (int k = A->ptr[j]; k < A->ptr[j + 1]; k++) {
                ax += Ae[k] * x[A->ix[k]];
            }
            y[j] = beta * y[j] + alpha * ax;
        }
    } else {
        // y + alpha*A^T*x
        for (int i = 0; i < A->rows; i++)
            y[i] *= beta;

        // update with linear combinations of A's rows
        for (int j = 0; j < A->rows; j++) {
            ax = alpha * x[j];
            for (int k = A->ptr[j]; k < A->ptr[j + 1]; k++) {
                y[A->ix[k]] += ax * Ae[k];
            }
        }
    }
    return 0;
}

/**
 * \brief Compute y = beta*y + alpha*A*x or y = beta*y + alpha*A^T*x
 *
 */
int armassp_x_mvmult(DTYPE beta, armas_x_dense_t * y,
                     DTYPE alpha, const armas_x_sparse_t * A,
                     const armas_x_dense_t * x, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    int ok = (flags & ARMAS_TRANS) == 0
        ? armas_x_size(y) == A->rows : armas_x_size(y) == A->cols;
    ok = ok && (flags & ARMAS_TRANS) == 0
        ? armas_x_size(x) == A->cols : armas_x_size(x) == A->rows;

    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }
    DTYPE *yd = armas_x_data(y);
    DTYPE *xd = armas_x_data(x);

    switch (A->kind) {
    case ARMASSP_CSC:
        return csc_mvmult(beta, yd, alpha, A, xd, flags);
    case ARMASSP_CSR:
        return csr_mvmult(beta, yd, alpha, A, xd, flags);
    default:
        break;
    }
    cf->error = ARMAS_EIMP;
    return -1;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
