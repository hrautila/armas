
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_solve_unb)
#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "nosimd/mvec.h"

/*
 * Functions here solves the matrix equations
 *
 *   op(A)*X = alpha*B or X*op(A) = alpha*B
 */

/*
 *   LEFT-UPPER
 *
 *     b0     a00 | a01 : a02     b'0
 *     ==     ===============     ====
 *     b1  =   0  | a11 : a12  *  b'1
 *     --     ---------------     ----
 *     b2      0  |  0  : a22     b'2
 *
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a12*b'2)/a00
 *
 */
static
void solve_unblk_lu(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    register int j;
    DTYPE scal;
    armas_x_dense_t B2, a1, b1;

    armas_x_row_unsafe(&b1, B, A->cols-1);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, A->cols-1, A->cols-1);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = A->cols-2; j >= 0; j--) {
        armas_x_submatrix_unsafe(&B2, B, j+1, 0, B->rows-1-j, B->cols);
        armas_x_row_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, j, j+1, 1, A->cols-1-j);
        // b1 = (alpha*b1 - B2.T*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B2, &a1, ARMAS_TRANS);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}


/*
 *    LEFT-UPPER-TRANS
 *
 *     b0     a00 | a01 : a02     b'0
 *     ==     ===============     ====
 *     b1  =   0  | a11 : a12  *  b'1
 *     --     ---------------     ----
 *     b2      0  |  0  : a22     b'2
 *
 *   b0 = a00*b'0                     --> b'0 =  b0/a00
 *   b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *   b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 */
static
void solve_unblk_lut(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B0, a1, b1;

    armas_x_row_unsafe(&b1, B, 0);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, 0, 0);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = 1; j < A->cols; j++) {
        armas_x_submatrix_unsafe(&B0, B, 0, 0, j, B->cols);
        // j'th column
        armas_x_row_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, 0, j, j, 1);
        // b1 = (alpha*b1 - B0*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B0, &a1, ARMAS_TRANS);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}
/*
 *    LEFT-LOWER
 *
 *     b0     a00 |  0  :  0      b'0
 *     ==     ===============     ====
 *     b1  =  a10 | a11 :  0   *  b'1
 *     --     ---------------     ----
 *     b2     a20 | a12 : a22     b'2
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a10*b'0 + a11*b'1           --> b'1 = (b1 - a10*b'0)/a11
 *    b2 = a20*b'0 + a21*b'1 + a22*b'2 --> b'2 = (b2 - a20*b'0 - a21*b'1)/a22
 */
static
void solve_unblk_ll(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B0, a1, b1;

    armas_x_row_unsafe(&b1, B, 0);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, 0, 0);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = 1; j < A->cols; j++) {
        armas_x_submatrix_unsafe(&B0, B, 0, 0, j, B->cols);
        // j'th column
        armas_x_row_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, j, 0, 1, j);
        // b1 = (alpha*b1 - B0*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B0, &a1, ARMAS_TRANS);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}
/*
 *   LEFT-LOWER-TRANS
 *
 *     b0     a00 |  0  :  0      b'0
 *     ==     ===============     ====
 *     b1  =  a10 | a11 :  0   *  b'1
 *     --     ---------------     ----
 *     b2     a20 | a12 : a22     b'2
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void solve_unblk_llt(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B2, a1, b1;

    armas_x_row_unsafe(&b1, B, A->cols-1);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, A->cols-1, A->cols-1);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = A->cols-2; j >= 0; j--) {
        armas_x_submatrix_unsafe(&B2, B, j+1, 0, A->cols-1-j, B->cols);
        armas_x_row_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, j+1, j, A->cols-1-j, 1);
        // b1 = (alpha*b1 - B2.T*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B2, &a1, ARMAS_TRANS);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}
/*
 *    RIGHT-UPPER
 *
 *                               a00 | a01 : a02
 *                               ===============
 *    b0|b1|b2 =  b'0|b'1|b'2 *   0  | a11 : a12
 *                               ---------------
 *                                0  |  0  : a22
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *    b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 *
 */
static
void solve_unblk_ru(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B0, a1, b1;

    armas_x_column_unsafe(&b1, B, 0);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, 0, 0);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = 1; j < A->cols; j++) {
        armas_x_submatrix_unsafe(&B0, B, 0, 0, B->rows, j);
        // j'th column
        armas_x_column_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, 0, j, j, 1);
        // b1 = (alpha*b1 - B0*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B0, &a1, 0);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}


/*
 *    RIGHT-UPPER-TRANS
 *
 *                               a00 | a01 : a02
 *                               ===============
 *    b0|b1|b2 =  b'0|b'1|b'2 *   0  | a11 : a12
 *                               ---------------
 *                                0  |  0  : a22
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a02*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1           - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void solve_unblk_rut(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B2, a1, b1;

    armas_x_column_unsafe(&b1, B, A->cols-1);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, A->cols-1, A->cols-1);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = A->cols-2; j >= 0; j--) {
        // j'th column
        armas_x_submatrix_unsafe(&B2, B, 0, j+1, B->rows, A->cols-1-j);
        armas_x_column_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, j, j+1, 1, A->cols-1-j);
        // b1 = (alpha*b1 - B0*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B2, &a1, 0);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}
/*
 *    RIGHT-LOWER
 *                               a00 |  0  :  0
 *                               ===============
 *    b0|b1|b2 =  b'0|b'1|b'2 *  a10 | a11 :  0
 *                               ---------------
 *                               a20 | a21 : a22
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a21*b'2           --> b'1 = (b1           - a21*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void solve_unblk_rl(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B2, a1, b1;

    armas_x_column_unsafe(&b1, B, A->cols-1);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, A->cols-1, A->cols-1);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = A->cols-2; j >= 0; j--) {
        // j'th column
        armas_x_submatrix_unsafe(&B2, B, 0, j+1, B->rows, A->cols-1-j);
        armas_x_column_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, j+1, j, A->cols-1-j, 1);
        // b1 = (b1 - B2*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B2, &a1, 0);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}

/*
 *    RIGHT-LOWER-TRANS
 *                               a00 |  0  :  0
 *                               ===============
 *    b0|b1|b2 =  b'0|b'1|b'2 *  a10 | a11 :  0
 *                               ---------------
 *                               a20 | a12 : a22
 *
 *    b00 = a00*b'00                       --> b'00 = b00/a00
 *    b01 = a10*b'00 + a11*b'01            --> b'01 = (b01 - a10*b'00)/a11
 *    b02 = a20*b'00 + a21*b'01 + a22*b'02 --> b'02 = (b02 - a20*b'00 - a21*b'01)/a22
 */
static
void solve_unblk_rlt(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    DTYPE scal;
    register int j;
    armas_x_dense_t B0, a1, b1;

    armas_x_column_unsafe(&b1, B, 0);
    scal = unit ? alpha : alpha/armas_x_get_unsafe(A, 0, 0);
    if (scal != ONE)
        armas_x_scale_unsafe(&b1, scal);

    for (j = 1; j < A->cols; j++) {
        armas_x_submatrix_unsafe(&B0, B, 0, 0, B->rows, j);
        // j'th column
        armas_x_column_unsafe(&b1, B, j);
        armas_x_submatrix_unsafe(&a1, A, j, 0, 1, j);
        // b1 = (alpha*b1 - B0*a1) / a11
        armas_x_mvmult_unsafe(alpha, &b1, -ONE, &B0, &a1, 0);
        if (!unit)
            armas_x_scale_unsafe(&b1, ONE/armas_x_get_unsafe(A, j, j));
    }
}


void armas_x_solve_unb(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags)
{
    if (flags & ARMAS_RIGHT) {
        switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANSA)) {
        case ARMAS_UPPER|ARMAS_TRANSA:
            solve_unblk_rut(B, A, alpha, flags);
            break;

        case ARMAS_UPPER:
            solve_unblk_ru(B, A, alpha, flags);
            break;

        case ARMAS_LOWER|ARMAS_TRANSA:
            solve_unblk_rlt(B, A, alpha, flags);
            break;

        case ARMAS_LOWER:
        default:
            solve_unblk_rl(B, A, alpha, flags);
            break;
        }
    }
    else {
        switch (flags & (ARMAS_UPPER|ARMAS_LOWER|ARMAS_TRANSA)) {
        case ARMAS_UPPER|ARMAS_TRANSA:
            solve_unblk_lut(B, A, alpha, flags);
            break;

        case ARMAS_UPPER:
            solve_unblk_lu(B, A, alpha, flags);
            break;

        case ARMAS_LOWER|ARMAS_TRANSA:
            solve_unblk_llt(B, A, alpha, flags);
            break;

        case ARMAS_LOWER:
        default:
            solve_unblk_ll(B, A, alpha, flags);
            break;
        }
    }
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
