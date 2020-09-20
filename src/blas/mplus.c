
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

//! \file
//! Matrix element wise addition

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_mplus)
#define ARMAS_PROVIDES 1
#endif
// this this requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

static
void madd_lower(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B, int nR, int nC)
{
    register int i, j, k, lda, ldb;
    DTYPE *a, *b;

    a = armas_x_data(A);
    b = armas_x_data(B);
    lda = A->step; ldb = B->step;

    for (j = 0; j < nC-3; j += 4) {
        // top triangle
        for (k = j; k < j+3; k++) {
            for (i = k; i < j+3 && i < nR; i++) {
                a[i+k*lda] = alpha*a[i+k*lda] + beta*b[i+k*ldb];
            }
        }
        // rest of the column block
        for (i = j+3; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[i+(j+0)*ldb];
            a[i+(j+1)*lda] = alpha*a[i+(j+1)*lda] + beta*b[i+(j+1)*ldb];
            a[i+(j+2)*lda] = alpha*a[i+(j+2)*lda] + beta*b[i+(j+2)*ldb];
            a[i+(j+3)*lda] = alpha*a[i+(j+3)*lda] + beta*b[i+(j+3)*ldb];
        }
    }
    if (j == nC)
        return;
    for (; j < nC; j++) {
        for (i = j; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[i+(j+0)*ldb];
        }
    }
}

static
void madd_lower_abs(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B, int nR, int nC)
{
    register int i, j, k, lda, ldb;
    DTYPE *a, *b;

    a = armas_x_data(A);
    b = armas_x_data(B);
    lda = A->step; ldb = B->step;

    for (j = 0; j < nC-3; j += 4) {
        // top triangle
        for (k = j; k < j+3; k++) {
            for (i = k; i < j+3 && i < nR; i++) {
                a[i+k*lda] = alpha*ABS(a[i+k*lda]) + beta*ABS(b[i+k*ldb]);
            }
        }
        // rest of the column block
        for (i = j+3; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[i+(j+0)*ldb]);
            a[i+(j+1)*lda] = alpha*ABS(a[i+(j+1)*lda]) + beta*ABS(b[i+(j+1)*ldb]);
            a[i+(j+2)*lda] = alpha*ABS(a[i+(j+2)*lda]) + beta*ABS(b[i+(j+2)*ldb]);
            a[i+(j+3)*lda] = alpha*ABS(a[i+(j+3)*lda]) + beta*ABS(b[i+(j+3)*ldb]);
        }
    }
    if (j == nC)
        return;
    for (; j < nC; j++) {
        for (i = j; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[i+(j+0)*ldb]);
        }
    }
}

static
void madd_upper(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B, int nR, int nC)
{
    register int i, j, k, lda, ldb;
    DTYPE *a, *b;

    a = armas_x_data(A);
    b = armas_x_data(B);
    lda = A->step; ldb = B->step;

    for (j = 0; j < nC-3; j += 4) {
        // top column block
        for (i = 0; i <= j && i < nR; i++) {
            a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[i+(j+0)*ldb];
            a[i+(j+1)*lda] = alpha*a[i+(j+1)*lda] + beta*b[i+(j+1)*ldb];
            a[i+(j+2)*lda] = alpha*a[i+(j+2)*lda] + beta*b[i+(j+2)*ldb];
            a[i+(j+3)*lda] = alpha*a[i+(j+3)*lda] + beta*b[i+(j+3)*ldb];
        }
        // bottom triangle
        for (i = j+1; i < j+4 && i < nR; i++) {
            for (k = i; k < j+4; k++) {
                a[i+k*lda] = alpha*a[i+k*lda] + beta*b[i+k*ldb];
            }
        }
    }
    if (j == nC)
        return;
    for (; j < nC; j++) {
        for (i = 0; i <= j && i < nR; i++) {
            a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[i+(j+0)*ldb];
        }
    }
}

static
void madd_upper_abs(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B, int nR, int nC)
{
    register int i, j, k, lda, ldb;
    DTYPE *a, *b;

    a = armas_x_data(A);
    b = armas_x_data(B);
    lda = A->step; ldb = B->step;

    for (j = 0; j < nC-3; j += 4) {
        // top column block
        for (i = 0; i <= j && i < nR; i++) {
            a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[i+(j+0)*ldb]);
            a[i+(j+1)*lda] = alpha*ABS(a[i+(j+1)*lda]) + beta*ABS(b[i+(j+1)*ldb]);
            a[i+(j+2)*lda] = alpha*ABS(a[i+(j+2)*lda]) + beta*ABS(b[i+(j+2)*ldb]);
            a[i+(j+3)*lda] = alpha*ABS(a[i+(j+3)*lda]) + beta*ABS(b[i+(j+3)*ldb]);
        }
        // bottom triangle
        for (i = j+1; i < j+4 && i < nR; i++) {
            for (k = i; k < j+4; k++) {
                a[i+k*lda] = alpha*ABS(a[i+k*lda]) + beta*ABS(b[i+k*ldb]);
            }
        }
    }
    if (j == nC)
        return;
    for (; j < nC; j++) {
        for (i = 0; i <= j && i < nR; i++) {
            a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[i+(j+0)*ldb]);
        }
    }
}


static
void madd(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B, int nR, int nC, int flags)
{
    register int i, j, lda, ldb;
    DTYPE *a, *b;

    a = armas_x_data(A);
    b = armas_x_data(B);
    lda = A->step; ldb = B->step;

    if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
        for (j = 0; j < nC-3; j += 4) {
            // top column block
            for (i = 0; i < nR; i++) {
                a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[(j+0)+i*ldb];
                a[i+(j+1)*lda] = alpha*a[i+(j+1)*lda] + beta*b[(j+1)+i*ldb];
                a[i+(j+2)*lda] = alpha*a[i+(j+2)*lda] + beta*b[(j+2)+i*ldb];
                a[i+(j+3)*lda] = alpha*a[i+(j+3)*lda] + beta*b[(j+3)+i*ldb];
            }
        }
        if (j == nC)
            return;
        for (; j < nC; j++) {
            for (i = 0; i < nR; i++) {
                a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[(j+0)+i*ldb];
            }
        }
        return;
    }

    for (j = 0; j < nC-3; j += 4) {
        // top column block
        for (i = 0; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[i+(j+0)*ldb];
            a[i+(j+1)*lda] = alpha*a[i+(j+1)*lda] + beta*b[i+(j+1)*ldb];
            a[i+(j+2)*lda] = alpha*a[i+(j+2)*lda] + beta*b[i+(j+2)*ldb];
            a[i+(j+3)*lda] = alpha*a[i+(j+3)*lda] + beta*b[i+(j+3)*ldb];
        }
    }
    if (j == nC)
        return;
    for (; j < nC; j++) {
        for (i = 0; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*a[i+(j+0)*lda] + beta*b[i+(j+0)*ldb];
        }
    }
}

static
void madd_abs(DTYPE alpha, armas_x_dense_t *A, DTYPE beta, const armas_x_dense_t *B, int nR, int nC, int flags)
{
    register int i, j, lda, ldb;
    DTYPE *a, *b;

    a = armas_x_data(A);
    b = armas_x_data(B);
    lda = A->step; ldb = B->step;

    if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
        for (j = 0; j < nC-3; j += 4) {
            // top column block
            for (i = 0; i < nR; i++) {
                a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[(j+0)+i*ldb]);
                a[i+(j+1)*lda] = alpha*ABS(a[i+(j+1)*lda]) + beta*ABS(b[(j+1)+i*ldb]);
                a[i+(j+2)*lda] = alpha*ABS(a[i+(j+2)*lda]) + beta*ABS(b[(j+2)+i*ldb]);
                a[i+(j+3)*lda] = alpha*ABS(a[i+(j+3)*lda]) + beta*ABS(b[(j+3)+i*ldb]);
            }
        }
        if (j == nC)
            return;
        for (; j < nC; j++) {
            for (i = 0; i < nR; i++) {
                a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[(j+0)+i*ldb]);
            }
        }
        return;
    }

    for (j = 0; j < nC-3; j += 4) {
        // top column block
        for (i = 0; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[i+(j+0)*ldb]);
            a[i+(j+1)*lda] = alpha*ABS(a[i+(j+1)*lda]) + beta*ABS(b[i+(j+1)*ldb]);
            a[i+(j+2)*lda] = alpha*ABS(a[i+(j+2)*lda]) + beta*ABS(b[i+(j+2)*ldb]);
            a[i+(j+3)*lda] = alpha*ABS(a[i+(j+3)*lda]) + beta*ABS(b[i+(j+3)*ldb]);
        }
    }
    if (j == nC)
        return;
    for (; j < nC; j++) {
        for (i = 0; i < nR; i++) {
            a[i+(j+0)*lda] = alpha*ABS(a[i+(j+0)*lda]) + beta*ABS(b[i+(j+0)*ldb]);
        }
    }
}


/**
 * @brief Matrix summation.
 *
 *  Compute \f$A := \alpha*A + \beta*B\f$ or \f$A := \alpha*A + \beta*B^{T}\f$ if
 *  flag bit ARMAS_TRANS or ARMAS_TRANSB is set.
 *
 * @param [in] alpha
 *      Scaling constant for first operand.
 * @param [in,out] A
 *      On entry first operand matrix. On exit the result matrix.
 * @param [in] beta
 *      Scaling constant for second operand.
 * @param [in] B
 *      Second operand matrix. If B is 1x1 then operation equals to adding constant
 *      to first matrix.
 * @param [in] flags
 *      Indicator flags for matrix shape, ARMAS_UPPER, ARMAS_LOWER or ARMAS_TRANS
 * @param [in] conf
 *      Configuration block
 * @retval 0 OK
 * @retval <0 Error
 * @ingroup matrix
 */

int armas_x_mplus(
    DTYPE alpha,
    armas_x_dense_t *A,
    DTYPE beta,
    const armas_x_dense_t *B,
    int flags,
    armas_conf_t *cf)
{
    if (!cf)
        cf = armas_conf_default();

    require(A->step >= A->rows && B->step >= B->rows);

    if (armas_x_size(A) == 0 || armas_x_size(B) == 0)
        return 0;
    if (armas_x_size(B) == 1)
        return armas_x_madd(A, beta*armas_x_get_unsafe(B, 0, 0), flags, cf);
    if (armas_x_isvector(A) && armas_x_isvector(B))
        return armas_x_axpby(alpha, A, beta, B, cf);

    if (flags & (ARMAS_TRANS|ARMAS_TRANSB)) {
        if (A->rows != B->cols || A->cols != B->rows) {
            cf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
    } else {
        if (A->rows != B->rows || A->cols != B->cols) {
            cf->error = ARMAS_ESIZE;
            return -ARMAS_ESIZE;
        }
    }

    switch (flags & (ARMAS_LOWER|ARMAS_UPPER)) {
    case ARMAS_LOWER:
        if (flags & ARMAS_ABS) {
            madd_lower_abs(alpha, A, beta, B, A->rows, A->cols);
        } else {
            madd_lower(alpha, A, beta, B, A->rows, A->cols);
        }
        break;
    case ARMAS_UPPER:
        if (flags & ARMAS_ABS) {
            madd_upper_abs(alpha, A, beta, B, A->rows, A->cols);
        } else {
            madd_upper(alpha, A, beta, B, A->rows, A->cols);
        }
        break;
    default:
        if (flags & ARMAS_ABS) {
            madd_abs(alpha, A, beta, B, A->rows, A->cols, flags);
        } else {
            madd(alpha, A, beta, B, A->rows, A->cols, flags);
        }
        break;
    }
    return 0;
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
