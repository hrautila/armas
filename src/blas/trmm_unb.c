
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_trmm_unb)
#define ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"


// Functions here implement various versions of TRMM operation.


/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b0
 *     0 |a11|a12  b1
 *     0 | 0 |a22  b2
 *
 *    b00 = a00*b0 + a01*b1 + a02*b2
 *    b10 =          a11*b1 + a12*b2
 *    b20 =                   a22*b2
 *
 */
static
void trmm_unb_lu(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = 0; i < A->cols-1; i++) {
        armas_submatrix_unsafe(&b0, B, i, 0,   1, B->cols);
        armas_submatrix_unsafe(&a0, A, i, i+1, 1, A->cols-i-1);
        armas_submatrix_unsafe(&B1, B, i+1, 0, B->rows-i-1, B->cols);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, ARMAS_TRANS);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, i, 0, 1, B->cols);
    armas_scale_unsafe(&b0, scale);
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0
 *  b1 =   0 |a11|a12  b'1
 *  b2     0 | 0 |a22  b'2
 *
 *  b0 = a00*b'0
 *  b1 = a01*b'0 + a11*b'1
 *  b2 = a02*b'0 + a12*b'1 + a22*b'2
 *
 */
static
void trmm_unb_lut(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = A->cols-1; i > 0; i--) {
        armas_submatrix_unsafe(&b0, B, i, 0, 1, B->cols);
        armas_submatrix_unsafe(&a0, A, 0, i, i, 1);
        armas_submatrix_unsafe(&B1, B, 0, 0, i, B->cols);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, ARMAS_TRANS);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, 0, 0, 1, B->cols);
    armas_scale_unsafe(&b0, scale);
}

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
 *
 */
static
void trmm_unb_ll(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = A->cols-1; i > 0; i--) {
        armas_submatrix_unsafe(&b0, B, i, 0, 1, B->cols);
        armas_submatrix_unsafe(&a0, A, i, 0, 1, i);
        armas_submatrix_unsafe(&B1, B, 0, 0, i, B->cols);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, ARMAS_TRANS);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, 0, 0, 1, B->cols);
    armas_scale_unsafe(&b0, scale);
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0
 *  b1 =  a10|a11| 0   b'1
 *  b2    a20|a21|a22  b'2
 *
 *  b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =           a11*b'1 + a21*b'2
 *  b2 =                     a22*b'2
 *
 */
static
void trmm_unb_llt(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = 0; i < A->cols-1; i++) {
        armas_submatrix_unsafe(&b0, B, i, 0,   1, B->cols);
        armas_submatrix_unsafe(&a0, A, i+1, i, A->cols-i-1, 1);
        armas_submatrix_unsafe(&B1, B, i+1, 0, B->rows-i-1, B->cols);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, ARMAS_TRANS);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, i, 0, 1, B->cols);
    armas_scale_unsafe(&b0, scale);
}

/*
 *  RIGHT-UPPER
 *
 *                          a00|a01|a02
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12
 *                           0 | 0 |a22
 *
 *    b0 = b'0*a00
 *    b1 = b'0*a01 + a11*b'1
 *    b2 = b'0*a02 + a12*b'1 + a22*b'2
 *
 */
static
void trmm_unb_ru(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = A->cols-1; i > 0; i--) {
        armas_submatrix_unsafe(&b0, B, 0, i, B->rows, 1);
        armas_submatrix_unsafe(&a0, A, 0, i, i, 1);
        armas_submatrix_unsafe(&B1, B, 0, 0, B->rows, i);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, 0);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, 0, i, B->rows, 1);
    armas_scale_unsafe(&b0, scale);
}

/*
 * LOWER, RIGHT,
 *
 *                          a00| 0 | 0
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0
 *                          a20|a21|a22
 *
 *    b0 = b'0*a00 + b'1*a10 + b'2*a20
 *    b1 = b'1*a11 + b'2*a21
 *    b2 = b'2*a22
 *
 */
static
void trmm_unb_rl(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;
    for (i = 0; i < A->cols-1; i++) {
        armas_submatrix_unsafe(&b0, B, 0, i,   B->rows, 1);
        armas_submatrix_unsafe(&a0, A, i+1, i, A->cols-i-1, 1);
        armas_submatrix_unsafe(&B1, B, 0, i+1, B->rows, A->cols-i-1);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, 0);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, 0, i, B->rows, 1);
    armas_scale_unsafe(&b0, scale);
}

/*
 *  RIGHT-UPPER-TRANS
 *
 *                          a00|a01|a02
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12
 *                           0 | 0 |a22
 *
 *    b0 = b'0*a00 + b'1*a01 + b'2*a02
 *    b1 =           b'1*a11 + b'2*a12
 *    b2 =                     b'2*a22
 */
static
void trmm_unb_rut(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = 0; i < A->cols-1; i++) {
        armas_submatrix_unsafe(&b0, B, 0, i,   B->rows, 1);
        armas_submatrix_unsafe(&a0, A, i, i+1, 1, A->cols-i-1);
        armas_submatrix_unsafe(&B1, B, 0, i+1, B->rows, A->cols-i-1);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, 0);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, 0, i, B->rows, 1);
    armas_scale_unsafe(&b0, scale);
}

/* LOWER, RIGHT, TRANSA
 *
 *                          a00| 0 | 0
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0
 *                          a20|a21|a22
 *
 *    b0 = b'0*a00
 *    b1 = b'0*a10 + b'1*a11
 *    b2 = b'0*a20 + b'1*a21 + b'2*a22
 *
 */
static
void trmm_unb_rlt(armas_dense_t *B, const armas_dense_t *A, DTYPE alpha, int unit)
{
    int i;
    armas_dense_t b0, B1, a0;
    DTYPE scale;

    for (i = A->cols-1; i > 0; i--) {
        armas_submatrix_unsafe(&b0, B, 0, i, B->rows, 1);
        armas_submatrix_unsafe(&B1, B, 0, 0, B->rows, i);
        armas_submatrix_unsafe(&a0, A, i, 0, 1, i);
        scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
        armas_mvmult_unsafe(scale, &b0, alpha, &B1, &a0, 0);
    }
    scale = unit ? alpha : alpha *armas_get_unsafe(A, i, i);
    if (scale == ONE)
        return;
    armas_submatrix_unsafe(&b0, B, 0, i, B->rows, 1);
    armas_scale_unsafe(&b0, scale);
}

// X = A*X; unblocked version
void armas_trmm_unb(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags)
{
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    if (flags & ARMAS_RIGHT) {
        if (flags & ARMAS_UPPER) {
            if (flags & ARMAS_TRANSA) {
                trmm_unb_rut(B, A, alpha, unit);
            } else {
                trmm_unb_ru(B, A, alpha, unit);
            }
        } else {
            if (flags & ARMAS_TRANSA) {
                trmm_unb_rlt(B, A, alpha, unit);
            } else {
                trmm_unb_rl(B, A, alpha, unit);
            }
        }
    } else {
        if (flags & ARMAS_UPPER) {
            if (flags & ARMAS_TRANSA) {
                trmm_unb_lut(B, A, alpha, unit);
            } else {
                trmm_unb_lu(B, A, alpha, unit);
            }
        } else {
            if (flags & ARMAS_TRANSA) {
                trmm_unb_llt(B, A, alpha, unit);
            } else {
                trmm_unb_ll(B, A, alpha, unit);
            }
        }
    }
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
