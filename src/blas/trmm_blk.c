
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trmm_blk)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_kernel_nc) && defined(armas_x_trmm_recursive)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
//#include "mvec_nosimd.h"


// Functions here implement various versions of TRMM operation.

/*
 * LEFT-UPPER
 *
 *   B0    A00 | A01 | A02   B0
 *   --   ----------------   --
 *   B1 =   0  | A11 | A12   B1
 *   --   ----------------   --
 *   B2     0  |  0  | A22   B2
 *
 *    B0 = A00*B0 + A01*B1 + A02*B2
 *    B1 = A11*B1 + A12*B2
 *    B2 = A22*B2
 */
static
void trmm_blk_upper(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    const DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    // A0 is always off-diagonal block; B0 corresponding submatrix in B
    // A1 is always diagonal block; B1 is corresponding submatrix in B

    for (i = 0; i < A->rows; i += NB) {
        nI = A->rows - i < NB ? A->rows - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i, i+nI, nI, A->rows-i-nI);
        armas_x_submatrix_unsafe(&A1, A, i, i, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, i+nI, 0, B->rows-i-nI, B->cols);
        armas_x_submatrix_unsafe(&B1, B, i, 0, nI, B->cols);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &A0, &B0, aflags, cache);
    }
}
/*  LEFT-UPPER-TRANS
 *
 *  B0    A00 | A01 | A02   B0
 *  --   ----------------   --
 *  B1 =   0  | A11 | A12   B1
 *  --   ----------------   --
 *  B2     0  |  0  | A22   B2
 *
 *    B0 = A00*B0
 *    B1 = A01*B0 + A11*B1
 *    B2 = A02*B0 + A12*B1 + A22*B2
 */
static
void trmm_blk_u_trans(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    // A0 is always off-diagonal block; B0 corresponing submatrix in B
    // A1 is always diagonal block; B1 is corresponding submatrix in B

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, 0, i-nI, i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, 0,    0, i-nI, B->cols);
        armas_x_submatrix_unsafe(&B1, B, i-nI, 0, nI, B->cols);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &A0, &B0, ARMAS_TRANSA|aflags, cache);
    }
}


/*  LEFT-LOWER
 *
 *   B0     A00 |  0  |  0    B0
 *   --    ----------------   --
 *   B1 =   A10 | A11 |  0    B1
 *   --    ----------------   --
 *   B2     A20 | A21 | A22   B2
 *
 *    B0 = A00*B0
 *    B1 = A10*B0 + A11*B1
 *    B2 = A20*B0 + A21*B1 + A22*B2
 */
static
void trmm_blk_lower(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, i-nI, 0, nI, i-nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, 0,    0, i-nI, B->cols);
        armas_x_submatrix_unsafe(&B1, B, i-nI, 0, nI, B->cols);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &A0, &B0, aflags, cache);
    }
}
/*
 *  LEFT-LOWER-TRANSA
 *
 *   B0     A00 |  0  |  0    B0
 *   --    ----------------   --
 *   B1  =  A10 | A11 |  0    B1
 *   --    ----------------   --
 *   B2     A20 | A21 | A22   B2
 *
 *    B0 = A00*B0 + A10*B1 + A20*B2
 *    B1 = A11*B1 + A21*B2
 *    B2 = A22*B2
 */
static
void trmm_blk_l_trans(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    for (i = 0; i < A->cols; i += NB) {
        nI = A->cols - i < NB ? A->cols - i : NB;

        armas_x_submatrix_unsafe(&A0, A, i+nI, i, A->rows-i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i,    i, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, i+nI, 0, A->rows-i-nI, B->cols);
        armas_x_submatrix_unsafe(&B1, B, i,    0, nI, B->cols);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &A0, &B0, ARMAS_TRANSA|aflags, cache);
    }
}

/*
 *  RIGHT-UPPER
 *
 *                            A00 | A01 | A02
 *                           ----------------
 *   B0|B1|B2  =  B0|B1|B2 *   0  | A11 | A12
 *                           ----------------
 *                             0  |  0  | A22
 *
 *    B0 = B0*A00                   = trmm_unb(B0, A00)
 *    B1 = B0*A01 + B1*A11          = trmm_unb(B1, A11) + B0*A01
 *    B2 = B0*A02 + B1*A12 + B2*A22 = trmm_unb(B2, A22) + [B0; B1]*[A02; A12].T
 */
static
void trmm_blk_r_upper(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        armas_x_submatrix_unsafe(&A0, A, 0,    i-nI, i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, 0, 0, B->rows, i-nI);
        armas_x_submatrix_unsafe(&B1, B, 0, i-nI, B->rows, nI);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &B0, &A0, aflags, cache);
    }
}

/*
 *  RIGHT-UPPER-TRANS
 *
 *                            A00 | A01 | A02
 *                           ----------------
 *   B0|B1|B2  =  B0|B1|B2 *   0  | A11 | A12
 *                           ----------------
 *                             0  |  0  | A22
 *
 *  B0 = B0*A00 + B1*A01 + B2*A02  --> B0 = trmm(B0,A00) + [B1;B2]*[A01;A02].T
 *  B1 = B0*A11 + B2*A12           --> B1 = trmm(B1,A11) + B2*A12.T
 *  B2 = B2*A22                    --> B2 = trmm(B2,A22)
 */
static
void trmm_blk_ru_trans(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    for (i = 0; i < A->cols; i += NB) {
        nI = A->cols - i < NB ? A->cols - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i, i+nI, nI, A->cols-i-nI);
        armas_x_submatrix_unsafe(&A1, A, i, i, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, 0, i+nI, B->rows, B->cols-i-nI);
        armas_x_submatrix_unsafe(&B1, B, 0, i, B->rows, nI);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &B0, &A0, ARMAS_TRANSB|aflags, cache);
    }
}

/*
 * RIGHT-LOWER
 *
 *                             A00 |  0  |  0
 *                            ----------------
 *    B0|B1|B2  =  B0|B1|B2 *  A01 | A11 |  0
 *                            ----------------
 *                             A02 | A12 | A22
 *
 *    B0 = B0*A00 + B1*A01 + B2*A02
 *    B1 = B1*A11 + B2*A12
 *    B2 = B2*A22
 */
static
void trmm_blk_r_lower(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    for (i = 0; i < A->cols; i += NB) {
        nI = A->cols - i < NB ? A->cols - i : NB;
        armas_x_submatrix_unsafe(&A0, A, i+nI, i, A->cols-i-nI, nI);
        armas_x_submatrix_unsafe(&A1, A, i,    i, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, 0, i+nI, B->rows, B->cols-i-nI);
        armas_x_submatrix_unsafe(&B1, B, 0, i, B->rows, nI);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &B0, &A0, aflags, cache);
    }
}

/*
 *  RIGHT-LOWER-TRANSA
 *
 *                             A00 |  0  |  0
 *                            ----------------
 *    B0|B1|B2  =  B0|B1|B2 *  A01 | A11 |  0
 *                            ----------------
 *                             A02 | A12 | A22
 *
 *    B0 = B0*A00
 *    B1 = B0*A01 + B1*A11
 *    B2 = B0*A02 + B1*A12 + B2*A22
 */
static
void trmm_blk_rl_trans(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, aflags = 0;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    if (flags & ARMAS_ABS)
        aflags = ARMAS_ABSA|ARMAS_ABSB;

    for (i = A->cols; i > 0; i -= NB) {
        nI = i < NB ? i : NB;

        armas_x_submatrix_unsafe(&A0, A, i-nI, 0, nI, i-nI);
        armas_x_submatrix_unsafe(&A1, A, i-nI, i-nI, nI, nI);

        armas_x_submatrix_unsafe(&B0, B, 0, 0, B->rows, i-nI);
        armas_x_submatrix_unsafe(&B1, B, 0, i-nI, B->rows, nI);

        armas_x_trmm_recursive(&B1, alpha, &A1, flags, cache);
        armas_x_mult_kernel_nc(&B1, alpha, &B0, &A0, ARMAS_TRANSB|aflags, cache);
    }
}

void armas_x_trmm_blk(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *mcache)
{
    if (flags & ARMAS_RIGHT) {
        // B = alpha*B*op(A)
        if (flags & ARMAS_UPPER) {
            if (flags & ARMAS_TRANSA) {
                trmm_blk_ru_trans(B, A, alpha, flags, mcache);
            } else {
                trmm_blk_r_upper(B, A, alpha, flags, mcache);
            }
        } else {
            if (flags & ARMAS_TRANSA) {
                trmm_blk_rl_trans(B, A, alpha, flags, mcache);
            } else {
                trmm_blk_r_lower(B, A, alpha, flags, mcache);
            }
        }

    } else {
        // B = alpha*op(A)*B
        if (flags & ARMAS_UPPER) {
            if (flags & ARMAS_TRANSA) {
                trmm_blk_u_trans(B, A, alpha, flags, mcache);
            } else {
                trmm_blk_upper(B, A, alpha, flags, mcache);
            }
        } else {
            if (flags & ARMAS_TRANSA) {
                trmm_blk_l_trans(B, A, alpha, flags, mcache);
            } else {
                trmm_blk_lower(B, A, alpha, flags, mcache);
            }
        }
    }
}

#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
