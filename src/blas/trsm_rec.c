
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
#if defined(armas_x_solve_recursive)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_solve_unb) && \
  defined(armas_x_mult_kernel_nc)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
//#include "nosimd/mvec.h"

// Recursive versions of TRSM

/*
 *   RIGHT-UPPER             RIGHT-LOWER
 *
 *            A00 | A01               A00 |  0  
 *   B0|B1 * -----------     B0|B1 * -----------
 *             0  | A11               A10 | A11 
 *
 */

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
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_x_dense_t b0, b1, a0, a1;
    int N = A->cols;

    if (N < min_mblock_size) {
        armas_x_solve_unb(B, alpha, A, flags|ARMAS_LEFT);
        return;
    }

    armas_x_submatrix_unsafe(&b0, B, 0, 0, N/2, B->cols);
    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    solve_left_forward(&b0, &a0, alpha, flags, min_mblock_size, cache);

    // update B[0:N/2, S:E] with B[N/2:N, S:E]
    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a0, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a0, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_submatrix_unsafe(&b1, B, N/2, 0, N-N/2, B->cols);
    armas_x_mult_kernel_nc(&b1, -ONE, &a0, &b0, flags, cache);

    armas_x_submatrix_unsafe(&a1, A, N/2, N/2, N-N/2, N-N/2);
    solve_left_forward(&b1, &a1, alpha, flags, min_mblock_size, cache);
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
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_x_dense_t b0, b1, a0, a1;
    int N = A->cols;

    if (A->cols < min_mblock_size) {
        armas_x_solve_unb(B, alpha, A, flags|ARMAS_LEFT);
        return;
    }

    armas_x_submatrix_unsafe(&b1, B, N/2, 0, N-N/2, B->cols);
    armas_x_submatrix_unsafe(&a1, A, N/2, N/2, N-N/2, N-N/2);
    solve_left_backward(&b1, &a1, alpha, flags, min_mblock_size, cache);

    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a1, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a1, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_submatrix_unsafe(&b0, B, 0, 0, N/2, B->cols);
    armas_x_mult_kernel_nc(&b0, -ONE, &a1, &b1, flags, cache);

    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    solve_left_backward(&b0, &a0, alpha, flags, min_mblock_size, cache);
}


/*
 * Forward substitution for RIGHT-UPPER, RIGHT-LOWER-TRANSA
 */
static
void solve_right_forward(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_x_dense_t b0, b1, a0, a1;
    int ops, N = A->cols;

    if (A->cols < min_mblock_size) {
        armas_x_solve_unb(B, alpha, A, flags|ARMAS_RIGHT);
        return;
    }

    armas_x_submatrix_unsafe(&b0, B, 0, 0, B->rows, N/2);
    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    solve_right_forward(&b0, &a0, alpha, flags, min_mblock_size, cache);

    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a0, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a0, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_submatrix_unsafe(&b1, B, 0, N/2, B->rows, N-N/2);

    ops = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;
    armas_x_mult_kernel_nc(&b1, -ONE, &b0, &a0, ops, cache);

    armas_x_submatrix_unsafe(&a1, A, N/2, N/2, N-N/2, N-N/2);
    solve_right_forward(&b1, &a1, alpha, flags, min_mblock_size, cache);
}


/*
 * Backward substitution for RIGHT-UPPER-TRANSA and RIGHT-LOWER
 */
static
void solve_right_backward(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_x_dense_t b0, b1, a0;
    int ops;
    int N = A->cols;

    if (N < min_mblock_size) {
        armas_x_solve_unb(B, alpha, A, flags|ARMAS_RIGHT);
        return;
    }

    armas_x_submatrix_unsafe(&b0, B, 0, N/2, B->rows, N-N/2);
    armas_x_submatrix_unsafe(&a0, A, N/2, N/2, N-N/2, N-N/2);
    solve_right_backward(&b0, &a0, alpha, flags, min_mblock_size, cache);

    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a0, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a0, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_submatrix_unsafe(&b1, B, 0, 0, B->rows, N/2);

    ops = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;
    armas_x_mult_kernel_nc(&b1, -ONE, &b0, &a0, ops, cache);

    armas_x_submatrix_unsafe(&b0, B, 0, 0, B->rows, N/2);
    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    solve_right_backward(&b0, &a0, alpha, flags, min_mblock_size, cache);
}

void armas_x_solve_recursive(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *mcache)
{
    armas_env_t *env = armas_getenv();
    switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA)) {
    case ARMAS_RIGHT|ARMAS_UPPER:
    case ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA:
        solve_right_forward(B, A, alpha, flags, env->blas2min, mcache);
        break;

    case ARMAS_RIGHT|ARMAS_LOWER:
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        solve_right_backward(B, A, alpha, flags, env->blas2min, mcache);
        break;

    case ARMAS_UPPER:
    case ARMAS_LOWER|ARMAS_TRANSA:
        solve_left_backward(B, A, alpha, flags, env->blas2min, mcache);
        break;

    case ARMAS_LOWER:
    case ARMAS_UPPER|ARMAS_TRANSA:
    default:
        solve_left_forward(B, A, alpha, flags, env->blas2min, mcache);
        break;
    }
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
