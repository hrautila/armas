
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
// this file provides following type independet functions
#if defined(armas_x_trmm_recursive)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_kernel_nc) && defined(armas_x_trmm_unb)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"

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
        armas_x_trmm_unb(B, alpha, A, flags);
        return;
    }

    armas_x_submatrix_unsafe(&b0, B, 0, 0, N/2, B->cols);
    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    if (N/2 < min_mblock_size) {
        armas_x_trmm_unb(&b0, alpha, &a0, flags);
    } else {
        mult_left_forward(&b0, &a0, alpha, flags, min_mblock_size, cache);
    }

    // update B0 with A01/A10*B1
    armas_x_submatrix_unsafe(&b1, B, N/2, 0, N-N/2, B->cols);
    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a1, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a1, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_mult_kernel_nc(&b0, alpha, &a1, &b1, flags, cache);

    armas_x_submatrix_unsafe(&a1, A, N/2, N/2, N-N/2, N-N/2);
    if (N/2 < min_mblock_size) {
        armas_x_trmm_unb(&b1, alpha, &a1, flags);
    } else {
        mult_left_forward(&b1, &a1, alpha, flags, min_mblock_size, cache);
    }
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
        armas_x_trmm_unb(B, alpha, A, flags);
        return;
    }

    armas_x_submatrix_unsafe(&b1, B, N/2, 0, N-N/2, B->cols);
    armas_x_submatrix_unsafe(&a1, A, N/2, N/2, N-N/2, N-N/2);
    if (N/2 < min_mblock_size) {
        armas_x_trmm_unb(&b1, alpha, &a1, flags);
    } else {
        mult_left_backward(&b1, &a1, alpha, flags, min_mblock_size, cache);
    }

    // update b1, with A10*B0/A01*b0
    armas_x_submatrix_unsafe(&b0, B, 0, 0, N/2, B->cols);
    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a0, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a0, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_mult_kernel_nc(&b1, alpha, &a0, &b0, flags, cache);

    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    if (N/2 < min_mblock_size) {
        armas_x_trmm_unb(&b0, alpha, &a0, flags);
    } else {
        mult_left_backward(&b0, &a0, alpha, flags, min_mblock_size, cache);
    }
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
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    int min_mblock_size,
    cache_t *cache)
{
    armas_x_dense_t b0, b1, a0, a1;
    int ops;
    int N = A->cols;

    if (N < min_mblock_size) {
        armas_x_trmm_unb(B, alpha, A, flags);
        return;
    }

    armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
    armas_x_submatrix_unsafe(&b0, B, 0, 0, B->rows, N/2);
    if (N/2 < min_mblock_size) {
        armas_x_trmm_unb(&b0, alpha, &a0, flags);
    } else {
        mult_right_forward(&b0, &a0, alpha, flags, min_mblock_size, cache);
    }

    if (flags & ARMAS_UPPER) {
        armas_x_submatrix_unsafe(&a1, A, 0, N/2, N/2, N-N/2);
    } else {
        armas_x_submatrix_unsafe(&a1, A, N/2, 0, N-N/2, N/2);
    }
    armas_x_submatrix_unsafe(&b1, B, 0, N/2, B->rows, N-N/2);

    ops = flags & ARMAS_TRANSA ? ARMAS_TRANSB : ARMAS_NULL;

    armas_x_mult_kernel_nc(&b0, alpha, &b1, &a1, ops, cache);

    armas_x_submatrix(&a1, A, N/2, N/2, N-N/2, N-N/2);
    if (N/2 < min_mblock_size) {
        armas_x_trmm_unb(&b1, alpha, &a1, flags);
    } else {
        mult_right_forward(&b1, &a1, alpha, flags, min_mblock_size, cache);
    }
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
    armas_x_trmm_unb(B, alpha, A, flags);
    return;
  }

  armas_x_submatrix_unsafe(&b1, B, 0, N/2, B->rows, N-N/2);
  armas_x_submatrix_unsafe(&a1, A, N/2, N/2, N-N/2, N-N/2);
  if (N/2 < min_mblock_size) {
    armas_x_trmm_unb(&b1, alpha, &a1, flags);
  } else {
    mult_right_backward(&b1, &a1, alpha, flags, min_mblock_size, cache);
  }

  if (flags & ARMAS_UPPER) {
    armas_x_submatrix_unsafe(&a0, A, 0, N/2, N/2, N-N/2);
  } else {
    armas_x_submatrix_unsafe(&a0, A, N/2, 0, N-N/2, N/2);
  }
  armas_x_submatrix_unsafe(&b0, B, 0, 0, B->rows, N/2);
  int flgs = flags & ARMAS_TRANSA ? ARMAS_TRANSB : 0;
  armas_x_mult_kernel_nc(&b1, alpha, &b0, &a0, flgs, cache);

  armas_x_submatrix_unsafe(&a0, A, 0, 0, N/2, N/2);
  if (N/2 < min_mblock_size) {
    armas_x_trmm_unb(&b0, alpha, &a0, flags);
  } else {
    mult_right_backward(&b0, &a0, alpha, flags, min_mblock_size, cache);
  }
}

void armas_x_trmm_recursive(
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
        mult_right_backward(B, A, alpha, flags, env->blas2min, mcache);
        break;

    case ARMAS_RIGHT|ARMAS_LOWER:
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        mult_right_forward(B, A, alpha, flags, env->blas2min, mcache);
        break;

    case ARMAS_UPPER:
    case ARMAS_LOWER|ARMAS_TRANSA:
        mult_left_forward(B, A, alpha, flags, env->blas2min, mcache);
        break;

    case ARMAS_LOWER:
    case ARMAS_UPPER|ARMAS_TRANSA:
    default:
        mult_left_backward(B, A, alpha, flags, env->blas2min, mcache);
        break;
    }
}

#else
#warning "Missing defines; no code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
