
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_mvsolve_trm)
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

// solve A*x = alpha*b; A upper triangular
static
int csc_mvsolve_un(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    DTYPE *Ae = A->elems.v;
    int i, j, k;
    for (j = A->cols - 1; j >= 0; j--) {
        // skip rows below diagonal;
        for (k = A->ptr[j + 1] - 1; A->ix[k] > j; k--);
        x[j] = unit ? x[j] : x[j] / Ae[k];
        for (i = A->ptr[j]; i < k; i++) {
            x[A->ix[i]] -= Ae[i] * x[j];
        }
    }
    return 0;
}

static
int csr_mvsolve_un(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    DTYPE xk, *Ae = A->elems.v;
    int i, j;
    for (j = A->rows - 1; j >= 0; j--) {
        xk = 0;
        for (i = A->ptr[j + 1] - 1; i >= A->ptr[j] && A->ix[i] > j; i--) {
            xk += Ae[i] * x[A->ix[i]];
        }
        // here Ax[i] == j
        if (unit) {
            x[j] -= xk;
        } else {
            x[j] = (x[j] - xk) / Ae[i];
        }
    }
    return 0;
}

// solve A^T*x = alpha*b; A upper triangular
static
int csc_mvsolve_ut(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    DTYPE xk, *Ae = A->elems.v;
    int i, j;
    for (j = 0; j < A->cols; j++) {
        xk = 0;
        for (i = A->ptr[j]; i < A->ptr[j + 1] - 1 && A->ix[i] < j; i++) {
            xk += Ae[i] * x[A->ix[i]];
        }
        // here Ax[i] == j
        if (unit) {
            x[j] -= xk;
        } else {
            x[j] = (x[j] - xk) / Ae[i];
        }
    }
    return 0;
}

static
int csr_mvsolve_ut(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    DTYPE *Ae = A->elems.v;
    int j, p;
    for (j = 0; j < A->rows; j++) {
        // skip elements below diagonal; assume diagonal always exists
        for (p = A->ptr[j]; A->ix[p] < j; p++);
        x[j] = unit ? x[j] : (A->ix[p] == j ? x[j] / Ae[p] : x[j] / 0.0);
        for (p++; p < A->ptr[j + 1]; p++) {
            x[A->ix[p]] -= Ae[p] * x[j];
        }
    }
    return 0;
}

// solve A*x = alpha*b; A lower triangular
static
int csc_mvsolve_ln(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    int i, j;
    for (j = 0; j < A->cols; j++) {
        // skip elements above diagonal
        for (i = A->ptr[j]; A->ix[i] < j; i++);
        // compute diagonal; assumes non-zero ; A->ix[i] == j
        if (!unit)
            x[j] /= A->elems.v[i];
        for (i += 1; i < A->ptr[j + 1]; i++) {
            x[A->ix[i]] -= A->elems.v[i] * x[j];
        }
    }
    return 0;
}

static
int csr_mvsolve_ln(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    DTYPE *Ae = A->elems.v;
    int j, p;

    if (!unit)
        x[0] /= Ae[A->ptr[0]];

    for (j = 1; j < A->rows; j++) {

        for (p = A->ptr[j]; p < A->ptr[j + 1] && A->ix[p] < j; p++) {
            x[j] -= Ae[p] * x[A->ix[p]];
        }
        // compute diagonal; assumes non-zero ; A->ix[k] == j
        if (!unit)
            x[j] /= Ae[p];
    }
    return 0;
}

// solve A^T*x = alpha*b; A lower triangular
static
int csc_mvsolve_lt(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    DTYPE *Ae = A->elems.v;
    int i, j, k;

    // skip elements above diagonal
    for (i = A->ptr[A->cols - 1]; A->ix[i] < A->cols - 1; i++);
    // compute diagonal; assumes non-zero ; A->ix[A->ptr[A->cols-1]] == A->cols-1
    if (!unit)
        x[A->cols - 1] /= Ae[i];

    for (j = A->cols - 2; j >= 0; j--) {
        // skip elements above diagonal
        for (k = A->ptr[j]; A->ix[k] < j; k++);
        // A->ix[k] == first row index equals column index
        for (i = k + 1; i < A->ptr[j + 1]; i++) {
            x[j] -= Ae[i] * x[A->ix[i]];
        }
        // compute diagonal; assumes non-zero ; A->ix[k] == j
        if (!unit)
            x[j] /= Ae[k];
    }
    return 0;
}

static
int csr_mvsolve_lt(DTYPE * x, DTYPE alpha, const armas_sparse_t * A, int unit)
{
    int j, p;
    DTYPE *Ae = A->elems.v;
    for (j = A->rows - 1; j >= 0; j--) {
        // skip elements above diagonal
        for (p = A->ptr[j + 1] - 1; A->ix[p] > j; p--);
        if (!unit)
            x[j] /= Ae[p];
        for (p--; p >= A->ptr[j]; p--) {
            x[A->ix[p]] -= Ae[p] * x[j];
        }
    }
    return 0;
}

/**
 * @brief Solve \f$ A^{-1} x = \alpha b \f$ or \f$ A^{-T} x = \alpha b \f$
 *
 * @param[in,out] x
 *   On entry the vector b, on exit result vector x.
 * @param[in] alpha
 *   Coefficient
 * @param[in] A
 *   Upper or lower triangular matrix A in column or row compressed storage format,
 *   per column rows or per row columns are sorted in ascending order.
 * @param[in] flags
 *   For lower (upper) triangular matrix A set to ARMAS_LOWER (ARMAS_UPPER).
 *   If ARMAS_TRANS is  set then use transpose of A.
 * @param[in] cf
 *   Configuration block.
 *
 * @retval  0  Success
 * @retval <0  Failure
 * @ingroup sparse
 */
int armassp_mvsolve_trm(armas_dense_t * x, DTYPE alpha,
                          const armas_sparse_t * A, int flags,
                          armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    if (A->kind != ARMASSP_CSC && A->kind != ARMASSP_CSR) {
        return -ARMAS_EINVAL;
    }

    int ok = (flags & ARMAS_TRANS) == 0
        ? armas_size(x) == A->cols : armas_size(x) == A->rows;
    if (!ok) {
        return -ARMAS_ESIZE;
    }

    int unit = (flags & ARMAS_UNIT) != 0 ? 1 : 0;

    DTYPE *y = armas_data(x);

    // TODO: we assume column vector here; how about a row vector??
    switch (flags & (ARMAS_UPPER | ARMAS_LOWER | ARMAS_TRANS)) {
    case ARMAS_UPPER | ARMAS_TRANS:
        if (A->kind == ARMASSP_CSR)
            csr_mvsolve_ut(y, alpha, A, unit);
        else
            csc_mvsolve_ut(y, alpha, A, unit);
        break;
    case ARMAS_UPPER:
        if (A->kind == ARMASSP_CSR)
            csr_mvsolve_un(y, alpha, A, unit);
        else
            csc_mvsolve_un(y, alpha, A, unit);
        break;
    case ARMAS_LOWER | ARMAS_TRANS:
    case ARMAS_TRANS:
        if (A->kind == ARMASSP_CSR)
            csr_mvsolve_lt(y, alpha, A, unit);
        else
            csc_mvsolve_lt(y, alpha, A, unit);
        break;
    case ARMAS_LOWER:
    default:
        if (A->kind == ARMASSP_CSR)
            csr_mvsolve_ln(y, alpha, A, unit);
        else
            csc_mvsolve_ln(y, alpha, A, unit);
        break;
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
