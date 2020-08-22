
// Copyright (c) Harri Rautila, 2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_ext_mult_trm_unsafe) && defined(armas_x_ext_mult_trm)
#define ARMAS_PROVIDES 1
#endif
#if defined(armas_x_ext_adot_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "matrix.h"
#include "internal.h"
#include "kernel_ext.h"
#include "eft.h"

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b0     b0 = a00*b0 + a01*b1 + a02*b2
 *     0 |a11|a12  b1     b1 =          a11*b1 + a12*b2
 *     0 | 0 |a22  b2     b2 =                   a22*b2
 */
static
void trmm_ext_unb_upper(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    DTYPE s0, u0, xk;
    armas_x_dense_t a0, b0;
    register int i, k;

    for (k = 0; k < B->cols; ++k) {
        for (i = 0; i < A->cols; ++i) {
            xk = armas_x_get_unsafe(B, i, k);
            armas_x_submatrix_unsafe(&a0, A, i, i+unit, 1, A->cols-i-unit);
            armas_x_submatrix_unsafe(&b0, B, i+unit, k, A->cols-i-unit, 1);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, k, s0);
            armas_x_set_unsafe(dB, i, k, u0);
        }
    }
}

/* LEFT-UPPER
 *
 *   B0    (A00 A01 A02) (B0)      B0 = A00*B0 + A01*B1 + A02*B2
 *   B1 =  ( 0  A11 A12) (B1)      B1 = A11*B1 + A12*B2
 *   B2    ( 0   0  A22) (B2)      B2 = A22*B2
 */
static
void trmm_ext_blk_upper(
    armas_x_dense_t *B,
    const DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int NB = cache->NB;

    for (i = 0; i < A->rows; i += NB) {
        nI = A->rows - i < NB ? A->rows - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i, i+nI, nI, A->rows-i-nI);
        armas_x_submatrix_unsafe(&A1, A, i, i, nI, nI);

        for (j = 0; j < B->cols; j += NB) {
            nJ = B->cols - j < NB ? B->cols - j : NB;
            armas_x_submatrix_unsafe(&B0, B, i+nI, j, B->rows-i-nI, nJ);
            armas_x_submatrix_unsafe(&B1, B, i, j, nI, nJ);
            armas_x_make(&dB, nI, nJ, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_upper(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &A0, &B0, flags, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0    b0 = a00*b'0
 *  b1 =   0 |a11|a12  b'1    b1 = a01*b'0 + a11*b'1
 *  b2     0 | 0 |a22  b'2    b2 = a02*b'0 + a12*b'1 + a22*b'2
 */
static void
trmm_ext_unb_u_trans(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    register int i, j;
    armas_x_dense_t a0, b0;
    DTYPE s0, u0, xk;

    for (j = 0; j < B->cols; ++j) {
        for (i = A->rows-1; i >= 0; --i) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, 0, i, i+1-unit, 1);
            armas_x_submatrix_unsafe(&b0, B, 0, j, i+1-unit, 1);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*  LEFT-UPPER-TRANS
 *
 *  B0    (A00 A01 A02) (B0)        B0 = A00*B0
 *  B1 =  ( 0  A11 A12) (B1)        B1 = A01*B0 + A11*B1
 *  B2    ( 0   0  A22) (B1)        B2 = A02*B0 + A12*B1 + A22*B2
 */
static
void trmm_ext_blk_u_trans(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int NB = cache->NB;

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, 0, i-nI, i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        for (j = 0; j < B->cols; j += NB) {
            nJ = B->cols - j < NB ? B->cols - j : NB;
            armas_x_submatrix_unsafe(&B0, B, 0,    j, i-nI, nJ);
            armas_x_submatrix_unsafe(&B1, B, i-nI, j, nI, nJ);
            armas_x_make(&dB, nI, nJ, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_u_trans(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &A0, &B0, ARMAS_TRANSA, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0    b0 = a00*b'0
 *  b1 =  a10|a11| 0   b'1    b1 = a10*b'0 + a11*b'1
 *  b2    a20|a21|a22  b'2    b2 = a20*b'0 + a21*b'1 + a22*b'2
 */
static
void trmm_ext_unb_lower(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    register int i, j;
    armas_x_dense_t a0, b0;
    DTYPE s0, u0, xk;

    // for all columns in B
    for (j = 0; j < B->cols; ++j) {
        for (i = A->cols-1; i >= 0; --i) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, i, 0, 1, i+1-unit);
            armas_x_submatrix_unsafe(&b0, B, 0, j, i+1-unit, 1);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*  LEFT-LOWER
 *
 *   B0     (A00  0   0 ) (B0)       B0 = A00*B0
 *   B1 =   (A10 A11  0 ) (B1)       B1 = A10*B0 + A11*B1
 *   B2     (A20 A21 A21) (B1)       B2 = A20*B0 + A21*B1 + A22*B2
 */
static
void trmm_ext_blk_lower(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int NB = cache->NB;

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, i-nI, 0, nI, i-nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        for (j = 0; j < B->cols; j += NB) {
            nJ = B->cols - j < NB ? B->cols - j : NB;
            armas_x_submatrix_unsafe(&B0, B, 0,    j, i-nI, nJ);
            armas_x_submatrix_unsafe(&B1, B, i-nI, j, nI, nJ);
            armas_x_make(&dB, nI, nJ, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_lower(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &A0, &B0, 0, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0    b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =  a10|a11| 0   b'1    b1 =           a11*b'1 + a21*b'2
 *  b2    a20|a21|a22  b'2    b2 =                     a22*b'2
 *
 */
static
void trmm_ext_unb_l_trans(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    DTYPE s0, u0, xk;
    armas_x_dense_t a0, b0;
    register int i, j;

    for (j = 0; j < B->cols; ++j) {
        for (i = 0; i < A->cols; ++i) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, i+unit, i, A->cols-i-unit, 1);
            armas_x_submatrix_unsafe(&b0, B, i+unit, j, A->cols-i-unit, 1);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 *  LEFT-LOWER-TRANSA
 *
 *   B0     (A00  0   0 ) (B0)     B0 = A00*B0 + A10*B1 + A20*B2
 *   B1  =  (A10 A11  0 ) (B1)     B1 = A11*B1 + A21*B2
 *   B2     (A20 A21 A22) (B2)     B2 = A22*B2
 */
static
void trmm_ext_blk_l_trans(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int NB = cache->NB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    for (i = 0; i < A->cols; i += NB) {
        nI = A->cols - i < NB ? A->cols - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i+nI, i, A->rows-i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i,    i, nI, nI);

        for (j = 0; j < B->cols; j += NB) {
            nJ = B->cols - j < NB ? B->cols - j : NB;
            armas_x_submatrix_unsafe(&B0, B, i+nI, j, A->rows-i-nI, nJ);
            armas_x_submatrix_unsafe(&B1, B, i, j, nI, nJ);
            armas_x_make(&dB, nI, nJ, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_l_trans(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &A0, &B0, ARMAS_TRANSA, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 *  RIGHT-UPPER
 *
 *                          a00|a01|a02    b0 = b'0*a00
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12    b1 = b'0*a01 + a11*b'1
 *                           0 | 0 |a22    b2 = b'0*a02 + a12*b'1 + a22*b'2
 *
 */
static
void trmm_ext_unb_r_upper(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    register int i, j;
    armas_x_dense_t a0, b0;
    DTYPE s0, u0, xk;

    for (i = 0; i < B->rows; ++i) {
        for (j = A->cols-1; j >= 0; --j) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, 0, j, j+1-unit, 1);
            armas_x_submatrix_unsafe(&b0, B, i, 0, 1, j+1-unit);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 *  RIGHT-UPPER
 *
 *                              (A00 A01 A02)    B0 = B0*A00
 *   (B0 B1 B2) =  (B0 B1 B2) * ( 0  A11 A12)    B1 = B0*A01 + B1*A11
 *                              ( 0   0  A22)    B2 = B0*A02 + B1*A12 + B2*A22
 */
static
void trmm_ext_blk_r_upper(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int NB = cache->NB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    for (i = A->cols; i > 0; i -= NB) {

        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, 0,    i-nI, i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        for (j = 0; j < B->rows; j += NB) {
            nJ = B->rows - j < NB ? B->rows - j : NB;
            armas_x_submatrix_unsafe(&B0, B, j, 0, nJ, i-nI);
            armas_x_submatrix_unsafe(&B1, B, j, i-nI, nJ, nI);
            armas_x_make(&dB, nJ, nI, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_r_upper(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &B0, &A0, 0, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 * LOWER, RIGHT,
 *
 *                          a00| 0 | 0     b0 = b'0*a00 + b'1*a10 + b'2*a20
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0     b1 =           b'1*a11 + b'2*a21
 *                          a20|a21|a22    b2 =                     b'2*a22
 */
static
void trmm_ext_unb_r_lower(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    register int i, j;
    armas_x_dense_t a0, b0;
    DTYPE s0, u0, xk;

    for (i = 0; i < B->rows; ++i) {
        for (j = 0; j < A->cols; ++j) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, j+unit, j, A->cols-j-unit, 1);
            armas_x_submatrix_unsafe(&b0, B, i, j+unit, 1, A->cols-j-unit);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 * RIGHT-LOWER
 *
 *                            (A00  0   0 )     B0 = B0*A00 + B1*A01 + B2*A02
 *  (B0 B1 B2) = (B0 B1 B2) * (A01 A11  0 )     B1 = B1*A11 + B2*A12
 *                            (A02 A12 A22)     B2 = B2*A22
 */
static
void trmm_ext_blk_r_lower(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int NB = cache->NB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    for (i = 0; i < A->cols; i += NB) {
        nI = A->cols - i < NB ? A->cols - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i+nI, i, A->cols-i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i,    i, nI, nI);

        for (j = 0; j < B->rows; j += NB) {
            nJ = B->rows - j < NB ? B->rows - j : NB;
            armas_x_submatrix_unsafe(&B0, B, j, i+nI, nJ, A->cols-i-nI);
            armas_x_submatrix_unsafe(&B1, B, j, i, nJ, nI);
            armas_x_make(&dB, nJ, nI, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_r_lower(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &B0, &A0, 0, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 *  RIGHT-UPPER-TRANS
 *
 *                          a00|a01|a02    b0 = b'0*a00 + b'1*a01 + b'2*a02
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12    b1 =           b'1*a11 + b'2*a12
 *                           0 | 0 |a22    b2 =                     b'2*a22
 */
static
void trmm_ext_unb_ru_trans(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    register int i, j;
    armas_x_dense_t a0, b0;
    DTYPE s0, u0, xk;

    for (i = 0; i < B->rows; ++i) {
        for (j = 0; j < A->cols; ++j) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, j, j+unit, 1, A->cols-j-unit);
            armas_x_submatrix_unsafe(&b0, B, i, j+unit, 1, A->cols-j-unit);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 *  RIGHT-UPPER-TRANS 
 *
 *                             (A00 A01 A02)     B0 = B0*A00 + B1*A01 + B2*A02
 *   (B0 B1 B2) = (B0 B1 B2) * ( 0  A11 A12)     B1 = B1*A11 + B2*A12
 *                             ( 0   0  A22)     B2 = B2*A22
 */
static
void trmm_ext_blk_ru_trans(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int NB = cache->NB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    for (i = 0; i < A->cols; i += NB) {
        nI = A->cols - i < NB ? A->cols - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i, i+nI, A->cols-i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i, i, nI, nI);

        for (j = 0; j < B->rows; j += NB) {
            nJ = B->rows - j < NB ? B->rows - j : NB;
            armas_x_submatrix_unsafe(&B0, B, j, i+nI, nJ, A->cols-i-nI);
            armas_x_submatrix_unsafe(&B1, B, j, i, nJ, nI);
            armas_x_make(&dB, nJ, nI, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            trmm_ext_unb_ru_trans(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &B0, &A0, ARMAS_TRANSB, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

/*
 * LOWER, RIGHT, TRANSA
 *
 *                          a00| 0 | 0      b0 = b'0*a00
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0      b1 = b'0*a10 + b'1*a11
 *                          a20|a21|a22     b2 = b'0*a20 + b'1*a21 + b'2*a22
 */
static
void trmm_ext_unb_rl_trans(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int unit)
{
    register int i, j;
    armas_x_dense_t a0, b0;
    DTYPE s0, u0, xk;

    for (i = 0; i < B->rows; ++i) {
        for (j = A->cols-1; j >= 0; --j) {
            xk = armas_x_get_unsafe(B, i, j);
            armas_x_submatrix_unsafe(&a0, A, j, 0, 1, j+1-unit);
            armas_x_submatrix_unsafe(&b0, B, i, 0, 1, j+1-unit);

            s0 = unit ? xk : ZERO; u0 = ZERO;
            twoprod(&s0, &u0, s0, alpha);
            armas_x_ext_adot_unsafe(&s0, &u0, alpha, &a0, &b0);

            armas_x_set_unsafe(B, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 *  RIGHT-LOWER-TRANSA
 *
 *                            (A00  0   0 )     B0 = B0*A00
 *  (B0 B1 B2) = (B0 B1 B2) * (A01 A11  0 )     B1 = B0*A01 + B1*A11
 *                            (A02 A12 A22)     B2 = B0*A02 + B1*A12 + B2*A22
 */
static
void trmm_ext_blk_rl_trans(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, j, nI, nJ;
    armas_x_dense_t A0, A1, B0, B1, dB;
    int NB = cache->NB;
    int unit = flags & ARMAS_UNIT ? 1 : 0;

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, i-nI, 0, nI, i-nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        for (j = 0; j < B->rows; j += NB) {
            nJ = B->rows - j < NB ? B->rows - j : NB;
            armas_x_submatrix_unsafe(&B0, B, j, 0, nJ, i-nI);
            armas_x_submatrix_unsafe(&B1, B, j, i-nI, nJ, nI);
            armas_x_make(&dB, nJ, nI, cache->ab_step, cache->dC);
            armas_x_scale_unsafe(&dB, ZERO);

            // update current part with diagonal
            trmm_ext_unb_rl_trans(&B1, &dB, alpha, &A1, unit);
            armas_x_ext_panel_unsafe(&B1, &dB, alpha, &B0, &A0, ARMAS_TRANSB, cache);
            armas_x_merge_unsafe(&B1, &dB);
        }
    }
}

static inline
void trmm_ext_unb_unsafe(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *mcache)
{
    int i, nJ;
    armas_x_dense_t dB, B0;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int NB = mcache->NB;

    if (flags & ARMAS_RIGHT) {
        for (i = 0; i < B->rows; i += NB) {
            nJ = B->rows - i < NB ? B->rows - i : NB;
            armas_x_submatrix_unsafe(&B0, B, i, 0, nJ, B->cols);
            armas_x_make(&dB, nJ, B->cols, mcache->ab_step, mcache->dC);
            armas_x_scale_unsafe(&dB, ZERO);
            switch (flags & (ARMAS_UPPER|ARMAS_TRANSA)) {
            case ARMAS_UPPER|ARMAS_TRANSA:
                trmm_ext_unb_ru_trans(&B0, &dB, alpha, A, unit);
                break;
            case ARMAS_UPPER:
                trmm_ext_unb_r_upper(&B0, &dB, alpha, A, unit);
                break;
            case ARMAS_TRANSA:
                trmm_ext_unb_rl_trans(&B0, &dB, alpha, A, unit);
                break;
            default:
                trmm_ext_unb_r_lower(&B0, &dB, alpha, A, unit);
                break;
            }
            armas_x_merge_unsafe(&B0, &dB);
        }
    } else {
        for (i = 0; i < B->cols; i += NB) {
            nJ = B->cols - i < NB ? B->cols - i : NB;
            armas_x_submatrix_unsafe(&B0, B, 0, i, B->rows, nJ);
            armas_x_make(&dB, B->rows, nJ, mcache->ab_step, mcache->dC);
            armas_x_scale_unsafe(&dB, ZERO);
            switch (flags & (ARMAS_UPPER|ARMAS_TRANSA)) {
            case ARMAS_UPPER|ARMAS_TRANSA:
                trmm_ext_unb_u_trans(&B0, &dB, alpha, A, unit);
                break;
            case ARMAS_UPPER:
                trmm_ext_unb_upper(&B0, &dB, alpha, A, unit);
                break;
            case ARMAS_TRANSA:
                trmm_ext_unb_l_trans(&B0, &dB, alpha, A, unit);
                break;
            default:
                trmm_ext_unb_lower(&B0, &dB, alpha, A, unit);
                break;
            }
            armas_x_merge_unsafe(&B0, &dB);
        }
    }
}

void armas_x_ext_mult_trm_unsafe(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *mcache)
{
    if (A->cols < mcache->NB) {
        trmm_ext_unb_unsafe(B, alpha, A, flags, mcache);
        return;
    }
    switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA)) {
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        trmm_ext_blk_ru_trans(B, alpha, A, flags, mcache);
        break;
    case ARMAS_RIGHT|ARMAS_UPPER:
        trmm_ext_blk_r_upper(B, alpha, A, flags, mcache);
        break;
    case ARMAS_RIGHT|ARMAS_TRANSA:
        trmm_ext_blk_rl_trans(B, alpha, A, flags, mcache);
        break;
    case ARMAS_RIGHT:
        trmm_ext_blk_r_lower(B, alpha, A, flags, mcache);
        break;
    case ARMAS_UPPER|ARMAS_TRANSA:
        trmm_ext_blk_u_trans(B, alpha, A, flags, mcache);
        break;
    case ARMAS_UPPER:
        trmm_ext_blk_upper(B, alpha, A, flags, mcache);
        break;
    case ARMAS_TRANSA:
        trmm_ext_blk_l_trans(B, alpha, A, flags, mcache);
        break;
    default:
        trmm_ext_blk_lower(B, alpha, A, flags, mcache);
        break;
    }
}

int armas_x_ext_mult_trm(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_conf_t *cf)
{
    int ok;

    if (armas_x_size(B) == 0 || armas_x_size(A) == 0)
        return 0;
    if (!cf)
        cf = armas_conf_default();

    // check consistency
    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        ok = B->cols == A->rows && A->cols == A->rows;
        break;
    case ARMAS_LEFT:
    default:
        ok = B->rows == A->cols && A->cols == A->rows;
        break;
    }
    if (! ok) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;
    if (armas_cbuf_select(&cbuf, cf) < 0) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }

    cache_t cache;
    armas_cache_setup(&cache, &cbuf, 3, sizeof(DTYPE));

    if (cf->optflags & ARMAS_ONAIVE) {
        trmm_ext_unb_unsafe(B, alpha, A, flags, &cache);
        return 0;
    }
    armas_x_ext_mult_trm_unsafe(B, alpha, A, flags, &cache);
    armas_cbuf_release(&cbuf);
    return 0;
}
#else
#warning "Missing defines! No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
