
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
// this file provides following type independent functions
#if defined(armas_solve_recursive)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_solve_unb) && defined(armas_mult_kernel_nc)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "partition.h"

// Recursive versions of TRSM

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *
 *    A00 | A01    B0         A00 |  0     B0
 *   ----------- * --        ----------- * --
 *     0  | A11    B1         A10 | A11    B1
 *
 *  upper:
 *    B'0 = A00*B0           --> B0 = trsm(B'0, A00)
 *    B'1 = A01*B0 + A11*B1  --> B1 = trsm(B'1 - A01*B0)
 *  lower:
 *    B'0 = A00*B0           --> B0 = trsm(B'0, A00)
 *    B'1 = A10*B0 + A11*B1  --> B1 = trsm(B'1 - A10*B0, A11)
 *
 *   Forward substitution.
 */
static
void solve_left_forward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BT, BB;

    if (A->cols < min_mblock_size) {
        armas_solve_unb(B, alpha, A, flags|ARMAS_LEFT);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PTOPLEFT);
    mat_partition_2x1(
        &BT,
        &BB, /**/ B, A->cols/2, ARMAS_PTOP);

    solve_left_forward(&BT, alpha, &ATL, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BB, -ONE, &ATR, &BT, ARMAS_TRANSA, cache);
    } else {
        armas_mult_kernel_nc(&BB, -ONE, &ABL, &BT, 0, cache);
    }
    solve_left_forward(&BB, alpha, &ABR, flags, min_mblock_size, cache);
}


/*
 *   LEFT-UPPER               LEFT-LOWER-TRANS
 *
 *    A00 | A01    B0         A00 |  0     B0
 *   ----------- * --        ----------- * --
 *     0  | A11    B1         A10 | A11    B1
 *
 *  upper:
 *    B'0 = A00*B0 + A01*B1  --> B0 = A00.-1*(B'0 - A01*B1)
 *    B'1 = A11*B1           --> B1 = A11.-1*B'1
 *  lower:
 *    B'0 = A00*B0 + A10*B1  --> B0 = trsm(B'0 - A10*B1, A00)
 *    B'1 = A11*B1           --> B1 = trsm(B'1, A11)
 *
 *   Backward substitution.
 */
static
void solve_left_backward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BT, BB;

    if (A->cols < min_mblock_size) {
        armas_solve_unb(B, alpha, A, flags|ARMAS_LEFT);
        return;
    }
    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PBOTTOMRIGHT);
    mat_partition_2x1(
        &BT,
        &BB, /**/ B, A->cols/2, ARMAS_PBOTTOM);

    solve_left_backward(&BB, alpha, &ABR, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BT, -ONE, &ATR, &BB, 0, cache);
    } else {
        armas_mult_kernel_nc(&BT, -ONE, &ABL, &BB, ARMAS_TRANSA, cache);
    }
    solve_left_backward(&BT, alpha, &ATL, flags, min_mblock_size, cache);
}


/*
 * Forward substitution for RIGHT-UPPER, RIGHT-LOWER-TRANSA
 */
static
void solve_right_forward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BL, BR;

    if (A->cols < min_mblock_size) {
        armas_solve_unb(B, alpha, A, flags|ARMAS_RIGHT);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PTOPLEFT);
    mat_partition_1x2(
        &BL, &BR,   /**/ B, A->cols/2, ARMAS_PLEFT);

    solve_right_forward(&BL, alpha, &ATL, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BR, -ONE, &BL, &ATR, 0, cache);
    } else {
        armas_mult_kernel_nc(&BR, -ONE, &BL, &ABL, ARMAS_TRANSB, cache);
    }
    solve_right_forward(&BR, alpha, &ABR, flags, min_mblock_size, cache);
}

/*
 * Backward substitution for RIGHT-UPPER-TRANSA and RIGHT-LOWER
 */
static
void solve_right_backward(
    armas_dense_t *B,
    DTYPE alpha,
    const armas_dense_t *A,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_dense_t ATL, ATR, ABL, ABR, BL, BR;

    if (A->cols < min_mblock_size) {
        armas_solve_unb(B, alpha, A, flags|ARMAS_RIGHT);
        return;
    }

    mat_partition_2x2(
        &ATL, &ATR,
        &ABL, &ABR, /**/ A, A->rows/2, A->cols/2, ARMAS_PBOTTOMRIGHT);
    mat_partition_1x2(
        &BL, &BR,   /**/ B, A->cols/2, ARMAS_PRIGHT);

    solve_right_backward(&BR, alpha, &ABR, flags, min_mblock_size, cache);
    if (flags & ARMAS_UPPER) {
        armas_mult_kernel_nc(&BL, -ONE, &BR, &ATR, ARMAS_TRANSB, cache);
    } else {
        armas_mult_kernel_nc(&BL, -ONE, &BR, &ABL, 0, cache);
    }
    solve_right_backward(&BL, alpha, &ATL, flags, min_mblock_size, cache);
}

void armas_solve_recursive(
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
        solve_right_forward(B, alpha, A, flags, env->blas2min, mcache);
        break;

    case ARMAS_RIGHT|ARMAS_LOWER:
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        solve_right_backward(B, alpha, A, flags, env->blas2min, mcache);
        break;

    case ARMAS_UPPER:
    case ARMAS_LOWER|ARMAS_TRANSA:
        solve_left_backward(B, alpha, A, flags, env->blas2min, mcache);
        break;

    case ARMAS_LOWER:
    case ARMAS_UPPER|ARMAS_TRANSA:
    default:
        solve_left_forward(B, alpha, A, flags, env->blas2min, mcache);
        break;
    }
}
#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
