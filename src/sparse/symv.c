
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_mvmult_sym)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "armas.h"
#include "matrix.h"
#include "sparse.h"

/**
 *   y0   (a00            )  (x0)
 *   y1   (a10 a11        )  (x1)
 *   y2   (a20 a21 a22    )  (x2)
 *   y3   (a30 a31 a32 a33)  (x3)
 *
 *   y0 = a00*x0 + a10*x1 + a20*x2 + a30*x3 
 *   y1 = a10*x0 + a11*x1 + a21*x2 + a31*x3
 *   y2 = a20*x0 + a21*x2 + a22*x2 + a32*x3
 *   y3 = a30*x0 + a31*x2 + a32*x2 + a33*x3
 */

#if 0
// CSR lower triangular
static
int csr_mvmult_lsym(DTYPE beta, DTYPE * y, DTYPE alpha,
                    const armas_sparse_t * A, const DTYPE * x, int flags)
{
    int j, p;
    DTYPE ax, yk, *Ae = A->elems.v;
    if (beta != ONE) {
        for (int i = 0; i < A->rows; i++)
            y[i] *= beta;
    }
    // update with dot product with A's rows
    for (j = A->rows - 1; j >= 0; j--) {
        ax = alpha * x[j];
        yk = 0.0;
        for (p = A->ptr[j]; p < A->ptr[j + 1] && A->ix[p] <= j; p++) {
            yk += x[A->ix[p]] * Ae[p];
            if (A->ix[p] != j)
                y[A->ix[p]] += ax * Ae[p];
        }
        y[j] += alpha * yk;
    }
    return 0;
}

// CSR storage; upper triangular
static
int csr_mvmult_usym(DTYPE beta, DTYPE * y, DTYPE alpha,
                    const armas_sparse_t * A, const DTYPE * x, int flags)
{
    int j, k;
    DTYPE yk, ax, *Ae = A->elems.v;

    if (beta != ONE) {
        for (int i = 0; i < A->rows; i++)
            y[i] *= beta;
    }
    // update with linear combinations of A's columns
    for (j = 0; j < A->rows; j++) {
        ax = alpha * x[j];
        // skip rows below diagonal
        for (k = A->ptr[j]; A->ix[k] < j && k < A->ptr[j + 1]; k++);
        //
        yk = 0.0;
        if (A->ix[k] == j) {
            yk = x[A->ix[k]] * Ae[k];
            k++;
        }
        for (; k < A->ptr[j + 1]; k++) {
            yk += x[A->ix[k]] * Ae[k];
            y[A->ix[p]] += ax * Ae[p];
        }
        y[j] += alpha * yk;
    }
    return 0;
}
#endif

// CSC lower triangular == CSR upper triangular
static
int csc_mvmult_lsym(DTYPE beta, DTYPE * y, DTYPE alpha,
                    const armas_sparse_t * A, const DTYPE * x, int flags)
{
    int j, k;
    DTYPE yk, ax, *Ae = A->elems.v;

    if (beta != ONE) {
        for (int i = 0; i < A->rows; i++)
            y[i] *= beta;
    }
    // update with linear combinations of A's columns
    for (j = 0; j < A->cols; j++) {
        ax = alpha * x[j];
        // skip element above diagonal (CSC) and below diagonal (CSR) 
        for (k = A->ptr[j]; A->ix[k] < j && k < A->ptr[j + 1]; k++);
        //
        yk = 0.0;
        if (A->ix[k] == j) {
            yk = x[A->ix[k]] * Ae[k];
            k++;
        }
        for (; k < A->ptr[j + 1]; k++) {
            yk += x[A->ix[k]] * Ae[k];
            y[A->ix[k]] += ax * Ae[k];
        }
        y[j] += alpha * yk;
    }
    return 0;
}

// CSC upper triangular == CSR lower triangular
static
int csc_mvmult_usym(DTYPE beta, DTYPE * y, DTYPE alpha,
                    const armas_sparse_t * A, const DTYPE * x, int flags)
{
    int i, j, k;
    DTYPE yk, ax, *Ae = A->elems.v;

    if (beta != ONE) {
        for (int i = 0; i < A->rows; i++)
            y[i] *= beta;
    }
    // update with linear combinations of A's columns
    for (j = A->cols - 1; j >= 0; j--) {
        ax = alpha * x[j];
        // skip rows below diagonal
        for (k = A->ptr[j + 1] - 1; A->ix[k] > j && k >= A->ptr[j]; k--);
        //
        yk = 0.0;
        if (A->ix[k] == j) {
            yk = x[A->ix[k]] * Ae[k];
            k--;
        }
        for (i = A->ptr[j]; i <= k; i++) {
            yk += x[A->ix[i]] * Ae[i];
            y[A->ix[i]] += ax * Ae[i];
        }
        y[j] += alpha * yk;
    }
    return 0;
}

/**
 * @brief Compute y = beta*y + alpha*A*x or y = beta*y + alpha*A^T*x
 *
 * @param[in] beta
 *   Scalar.
 * @param[in,out] y
 *   Dense result vector.
 * @param[in] alpha
 *   Scalar
 * @param[in] A
 *   Sparse symmetric matrix in CSC or CSR storage scheme.
 * @param[in] x
 *   Dense vector.
 * @param[in] flags
 *   *ARMAS_LOWER* or *ARMAS_UPPER*
 * @param[in] cf
 *   Configuration block.
 *
 * @retval  0  Success
 * @retval <0  Failure
 */
int armassp_mvmult_sym(DTYPE beta, armas_dense_t * y,
                         DTYPE alpha, const armas_sparse_t * A,
                         const armas_dense_t * x, int flags,
                         armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    if (A->kind != ARMASSP_CSC && A->kind != ARMASSP_CSR) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    int ok = (flags & ARMAS_TRANS) == 0
        ? armas_size(y) == A->rows : armas_size(y) == A->cols;
    ok = ok && (flags & ARMAS_TRANS) == 0
        ? armas_size(x) == A->cols : armas_size(x) == A->rows;

    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    DTYPE *yd = armas_data(y);
    DTYPE *xd = armas_data(x);

    switch (flags & (ARMAS_LOWER | ARMAS_UPPER)) {
    case ARMAS_UPPER:
        if (A->kind == ARMASSP_CSR) {
            return csc_mvmult_lsym(beta, yd, alpha, A, xd, flags);
        } else {
            return csc_mvmult_usym(beta, yd, alpha, A, xd, flags);
        }
        break;
    case ARMAS_LOWER:
    default:
        if (A->kind == ARMASSP_CSR) {
            return csc_mvmult_usym(beta, yd, alpha, A, xd, flags);
        } else {
            return csc_mvmult_lsym(beta, yd, alpha, A, xd, flags);
        }
        break;
    }
    return -ARMAS_EINVAL;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
