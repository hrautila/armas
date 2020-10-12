
// Copyright (c) Harri Rautila, 2012-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_ext_mvmult_trm_unsafe) && defined(armas_x_ext_mvmult_trm)
#define ARMAS_PROVIDES 1
#endif
// this module requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = a00*b'0
 *  b1 = a10*b'0 + a11*b'1
 *  b2 = a20*b'0 + a21*b'1 + a22*b'2
 */
static
void trmv_ex_unb_ll(
    armas_x_dense_t *X,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int unit,
    int xinc)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;
    int N = armas_x_size(X);

    //printf("..ex_unb_lut: LOWER-NOTRANSPOSE...\n");
    for (i = N-1; i >= 0; --i) {
        s0 = unit ? X->elems[i*xinc] : 0.0;
        u0 = 0.0;
        for (j = 0; j < i+(1-unit); ++j) {
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->elems[i+j*A->step], X->elems[j*xinc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->elems[i*xinc] = s0 + u0 + c0;
    }
}


/*
 * LEFT-LOWER-TRANSPOSE
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =           a11*b'1 + a21*b'2
 *  b2 =                     a22*b'2
 */
static
void trmv_ex_unb_llt(
    armas_x_dense_t *X,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int unit,
    int xinc)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;
    int N = armas_x_size(X);

    //printf("..ex_unb_llt: LOWER-TRANSPOSE... unit=%d\n", unit);
    for (i = 0; i < N; ++i) {
        s0 = unit ? X->elems[i*xinc] : 0.0;
        u0 = 0.0;
        for (j = i+unit; j < N; ++j) {
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->elems[j+i*A->step], X->elems[j*xinc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->elems[i*xinc] = s0 + u0 + c0;
    }
}

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b'0
 *     0 |a11|a12  b'1
 *     0 | 0 |a22  b'2
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2
 *    b1 =           a11*b'1 + a12*b'2
 *    b2 =                     a22*b'2
 */
static
void trmv_ex_unb_lu(
    armas_x_dense_t *X,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int unit,
    int xinc)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;
    int N = armas_x_size(X);

    //printf("..ex_unb_lu: UPPER-NOTRANSPOSE...\n");
    for (i = 0; i < N; ++i) {
        s0 = unit ? X->elems[i*xinc] : 0.0;
        u0 = 0.0;
        for (j = i+unit; j < N; ++j) {
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->elems[i+j*A->step], X->elems[j*xinc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->elems[i*xinc] = s0 + u0 + c0;
    }
}

/*
 * LEFT-UPPER-TRANSPOSE
 *
 *  b0    a00|a01|a02  b'0
 *  b1 =   0 |a11|a12  b'1
 *  b2     0 | 0 |a22  b'2
 *
 *  b0 = a00*b'0
 *  b1 = a01*b'0 + a11*b'1
 *  b2 = a02*b'0 + a12*b'1 + a22*b'2
 */
static
void trmv_ex_unb_lut(
    armas_x_dense_t *X,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int unit,
    int xinc)
{
    register int i, j;
    DTYPE s0, u0, p0, r0, c0;
    int N = armas_x_size(X);

    for (i = N-1; i >= 0; --i) {
        s0 = unit ? X->elems[i*xinc] : 0.0;
        u0 = 0.0;
        for (j = 0; j < i+(1-unit); ++j) {
            // p + r = A[i,j]*X[j]
            twoprod(&p0, &r0, A->elems[j+i*A->step], X->elems[j*xinc]);
            // s + c = s + A[i,j]*X[j]
            twosum(&s0, &c0, s0, p0);
            u0 += c0 + r0;
        }
        twoprod(&s0, &c0, s0, alpha);
        u0 *= alpha;
        X->elems[i*xinc] = s0 + u0 + c0;
    }
}


int armas_x_ext_mvmult_trm_unsafe(
    armas_x_dense_t *X,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int xinc = X->rows == 1 ? X->step : 1;

    switch (flags & (ARMAS_TRANS|ARMAS_UPPER|ARMAS_LOWER)){
    case ARMAS_UPPER|ARMAS_TRANS:
        trmv_ex_unb_lut(X, A, alpha, unit, xinc);
        break;
    case ARMAS_LOWER|ARMAS_TRANS:
        trmv_ex_unb_llt(X, A, alpha, unit, xinc);
        break;
    case ARMAS_UPPER:
        trmv_ex_unb_lu(X, A, alpha, unit, xinc);
        break;
    case ARMAS_LOWER:
    default:
        trmv_ex_unb_ll(X, A, alpha, unit, xinc);
        break;
    }
    return 0;
}

/**
 * @brief Triangular matrix-vector multiply in extended precision
 *
 * Computes
 *    - \f$ X = alpha \times A X \f$
 *    - \f$ X = alpha \times A^T X  \f$   if *ARMAS_TRANS* set
 *    - \f$ X = alpha \times |A| |X| \f$  if *ARMAS_ABS* set
 *    - \f$ X = alpha \times |A^T| |X| \f$ if *ARMAS_ABS* and *ARMAS_TRANS* set
 *
 * where A is upper (lower) triangular matrix defined with flag bits *ARMAS_UPPER*
 * (*ARMAS_LOWER*).
 *
 * @param[in,out] x target and source vector
 * @param[in]     alpha scalar multiplier
 * @param[in]     A matrix
 * @param[in]     flags operand flags
 * @param[in]     cf  configuration block
 *
 * @retval  0  Success
 * @retval <0  Failed
 *
 * @ingroup blasext
 */
int armas_x_ext_mvmult_trm(
    armas_x_dense_t *x,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_conf_t *cf)
{
    int nx = armas_x_size(x);

    if (armas_x_size(A) == 0 || nx == 0)
        return 0;

    if (!cf)
        cf = armas_conf_default();

    if (!armas_x_isvector(x)) {
        cf->error = ARMAS_ENEED_VECTOR;
        return -ARMAS_ENEED_VECTOR;
    }
    if (A->cols != nx || A->rows != A->cols) {
        cf->error = ARMAS_ESIZE;
        return -ARMAS_ESIZE;
    }
    armas_x_ext_mvmult_trm_unsafe(x, alpha, A, flags);
    return 0;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
