
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_trmm_recursive)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_mult_kernel_nc) && defined(armas_trmm_unb)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "partition.h"

// Recursive versions of TRMM

/*
 *   LEFT-UPPER              LEFT-LOWER-TRANS
 *
 *    A00 | A01   B0         A00 |  0     B0
 *   -----------  --        -----------   --
 *     0  | A11   B1         A10 | A11    B1
 *
 *
 *   B0 = A00*B0 + A01*B1    B0 = A00*B0 + A10*B1
 *   B1 = A11*B1             B1 = A11*B1
 */
static
void mult_left_forward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BT, BB;

    if (A->cols < min_mblock_size) {
        armas_trmm_unb(B, alpha, A, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &BT,
        &BB, /**/ B, A->cols/2, ARMAS_PTOP);

    mult_left_forward(&BT, alpha, &ATL, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BT, alpha, &ATR, &BB, 0, cache);
    } else {
        armas_mult_kernel_nc(&BT, alpha, &ABL, &BB, ARMAS_TRANSA, cache);
    }
    mult_left_forward(&BB, alpha, &ABR, flags, min_mblock_size, cache);
}

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *
 *    A00 | A01   B0         A00 |  0     B0
 *   -----------  --        -----------   --
 *     0  | A11   B1         A10 | A11    B1
 *
 *
 *   B0 = A00*B0            B0 = A00*B0
 *   B1 = A01*B0 + A11*B1   B1 = A10*B0 + A11*B1
 */
static
void mult_left_backward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BT, BB;

    if (A->cols < min_mblock_size) {
        armas_trmm_unb(B, alpha, A, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &BT,
        &BB, /**/ B, A->cols/2, ARMAS_PBOTTOM);

    mult_left_backward(&BB, alpha, &ABR, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BB, alpha, &ATR, &BT, ARMAS_TRANSA, cache);
    } else {
        armas_mult_kernel_nc(&BB, alpha, &ABL, &BT, 0, cache);
    }
    mult_left_backward(&BT, alpha, &ATL, flags, min_mblock_size, cache);
}

/*
 *   RIGHT-UPPER-TRANS         RIGHT-LOWER
 *
 *            A00 | A01                 A00 |  0
 *   B0|B1 * -----------       B0|B1 * -----------
 *             0  | A11                 A10 | A11
 *
 *   B0 = B0*A00 + B1*A01      B0 = B0*A00 + B1*A10
 *   B1 = B1*A11               B1 = B1*A11
 */
static
void mult_right_forward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BL, BR;
    //armas_dense_t b0, b1, a0, a1;
    //int ops;
    //int N = A->cols;

    if (A->cols < min_mblock_size) {
        armas_trmm_unb(B, alpha, A, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &BL,  &BR,  /**/ B, A->cols/2, ARMAS_PLEFT);

    mult_right_forward(&BL, alpha, &ATL, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BL, alpha, &BR, &ATR, ARMAS_TRANSB, cache);
    } else {
        armas_mult_kernel_nc(&BL, alpha, &BR, &ABL, 0, cache);
    }
    mult_right_forward(&BR, alpha, &ABR, flags, min_mblock_size, cache);
}


/*
 *   RIGHT-UPPER               RIGHT-LOWER-TRANSA
 *
 *            A00 | A01                 A00 |  0
 *   B0|B1 * -----------       B0|B1 * -----------
 *             0  | A11                 A10 | A11
 *
 *   B0 = B0*A00               B0 = B0*A00
 *   B1 = B0*A01 + B1*A11      B1 = B0*A10 + B1*A11
 */
static
void mult_right_backward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BL, BR;
    //armas_dense_t b0, b1, a0, a1;
    //int N = A->cols;

    if (A->cols < min_mblock_size) {
        armas_trmm_unb(B, alpha, A, flags);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PBOTTOMRIGHT);
    mat_partition_1x2(
        &BL,  &BR,  /**/ B, A->cols/2, ARMAS_PRIGHT);

    mult_right_backward(&BR, alpha, &ABR, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BR, alpha, &BL, &ATR, 0, cache);
    } else {
        armas_mult_kernel_nc(&BR, alpha, &BL, &ABL, ARMAS_TRANSB, cache);
    }
    mult_right_backward(&BL, alpha, &ATL, flags, min_mblock_size, cache);
}

void armas_trmm_recursive(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    cache_t *mcache)
{
    armas_env_t *env = armas_getenv();

    switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA)) {
    case ARMAS_RIGHT|ARMAS_UPPER:
    case ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA:
        mult_right_backward(B, alpha, A, flags, env->blas2min, mcache);
        break;

    case ARMAS_RIGHT|ARMAS_LOWER:
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        mult_right_forward(B, alpha, A, flags, env->blas2min, mcache);
        break;

    case ARMAS_UPPER:
    case ARMAS_LOWER|ARMAS_TRANSA:
        mult_left_forward(B, alpha, A, flags, env->blas2min, mcache);
        break;

    case ARMAS_LOWER:
    case ARMAS_UPPER|ARMAS_TRANSA:
    default:
        mult_left_backward(B, alpha, A, flags, env->blas2min, mcache);
        break;
    }
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
