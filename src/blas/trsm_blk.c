
// Copyright (c) Harri Rautila, 2013-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_solve_blocked)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_mult_kernel) && defined(armas_x_solve_recursive)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"


/*
 * UPPER, LEFT                 LOWER, TRANSA, LEFT
 *
 *    A00 | A01 | A02   B0       A00 |  0  |  0    B0
 *   ----------------   --      ----------------   --
 *     0  | A11 | A12   B1       A10 | A11 |  0    B1
 *   ----------------   --      ----------------   --
 *     0  |  0  | A22   B2       A20 | A21 | A22   B2
 *
 * upper:
 *    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = A00.-1*(B0 - A01*B'1 - A02*B'2)
 *    B1 = A11*B'1 + A12*B'2           --> B'1 = A11.-1*(B1           - A12*B'2)
 *    B2 = A22*B'2                     --> B'2 = A22.-1*B2
 * lower:
 *    c0*B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = A00.-1*(c0*B0 - A10*B'1 - A20*B'2)
 *    c0*B1 = A11*B'1 + A21*B'2           --> B'1 = A11.-1*(c0*B1           - A21*B'2)
 *    c0*B2 = A22*B'2                     --> B'2 = A22.-1*c0*B2
 */
static
void solve_blk_lu_llt(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    const  DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, cI;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    nI = NB < A->cols ? NB : A->cols;
    cI = NB < A->cols ? A->cols-NB : 0;
    armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
    armas_x_submatrix_unsafe(&B1, B, cI, 0,  nI, B->cols);
    armas_x_solve_recursive(&B1, alpha, &A1, flags|ARMAS_LEFT, cache);

    for (i = A->cols-NB; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        cI = i < NB ? 0 : i-NB;

        armas_x_submatrix_unsafe(&B0, B, i,  0, A->cols-i, B->cols);
        armas_x_submatrix_unsafe(&B1, B, cI, 0, nI, B->cols);
        if (flags & ARMAS_UPPER) {
            armas_x_submatrix_unsafe(&A0, A, cI, i, nI, A->cols-i);
        } else {
            armas_x_submatrix_unsafe(&A0, A, i, cI, A->cols-i, nI);
        }
        armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);

        // update and solve
        armas_x_mult_kernel(alpha, &B1, -ONE, &A0, &B0, flags, cache);
        armas_x_solve_recursive(&B1, ONE, &A1, flags|ARMAS_LEFT, cache);
    }
}

/*
 * LEFT-UPPER-TRANS              LEFT-LOWER
 *
 *    A00 | A01 | A02   B0         A00 |  0  |  0    B0
 *   ----------------   --        ----------------   --
 *     0  | A11 | A12   B1         A10 | A11 |  0    B1
 *   ----------------   --        ----------------   --
 *     0  |  0  | A22   B2         A20 | A21 | A22   B2
 *
 * upper:
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A01*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A01*B'0)
 *    B2 = A02*B'0 + A12*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A02*B'0 - A12*B'1)
 * lower:
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A10*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A10*B'0)
 *    B2 = A20*B'0 + A21*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A20*B'0 - A21*B'1)
 */
static
void solve_blk_lut_ll(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;

    // solve first block
    nI = NB < A->cols ? NB : A->cols;
    armas_x_submatrix_unsafe(&A1, A, 0, 0, nI, nI);
    armas_x_submatrix_unsafe(&B1, B, 0, 0, nI, B->cols);
    armas_x_solve_recursive(&B1, alpha, &A1, flags|ARMAS_LEFT, cache);

    for (i = NB; i < A->cols; i += NB) {
        nI = i < A->cols - NB ? NB : A->cols - i;

        armas_x_submatrix_unsafe(&B0, B, 0, 0, i,  B->cols);
        armas_x_submatrix_unsafe(&B1, B, i, 0, nI, B->cols);
        if (flags & ARMAS_UPPER) {
            armas_x_submatrix_unsafe(&A0, A, 0, i, i, nI);
        } else {
            armas_x_submatrix_unsafe(&A0, A, i, 0, nI, i);
        }
        armas_x_submatrix_unsafe(&A1, A, i, i, nI, nI);

        // update and solve
        armas_x_mult_kernel(alpha, &B1, -ONE, &A0, &B0, flags, cache);
        armas_x_solve_recursive(&B1, ONE, &A1, flags|ARMAS_LEFT, cache);
    }
}

/*
 *  RIGHT-UPPER                        LOWER, RIGHT, TRANSA
 *
 *                 A00 | A01 | A02                    A00 |  0  |  0
 *                ----------------                   ----------------
 *    B0|B1|B2      0  | A11 | A12       B0|B1|B2     A10 | A11 |  0
 *                ----------------                   ----------------
 *                  0  |  0  | A22                    A20 | A21 | A22
 *
 * upper:
 *    B0 = B'0*A00                     --> B'0 = B'0*A00.-1
 *    B1 = B'0*A01 + B'1*A11           --> B'1 = (B1 - B'0*A01)*A11.-1
 *    B2 = B'0*A02 + B'1*A12 + B'2*A22 --> B'2 = (B2 - B'0*A02 - B'1*A12)*A22.-1
 * lower:
 *    B0 = B'0*A00                     --> B'0 = B'0*A00.-1
 *    B1 = B'0*A10 + B'1*A11           --> B'1 = (B1 - B'0*A10)*A11.-1
 *    B2 = B'0*A20 + B'1*A21 + B'2*A22 --> B'2 = (B2 - B'0*A20 - B'1*A21)*A22.-1
 */
static
void solve_blk_ru_rlt(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;
    int transB = flags & ARMAS_LOWER ? ARMAS_TRANSB : 0;

    // solve first block
    nI = NB < A->cols ? NB : A->cols;
    armas_x_submatrix_unsafe(&A1, A, 0, 0, nI, nI);
    armas_x_submatrix_unsafe(&B1, B, 0, 0, B->rows, nI);
    armas_x_solve_recursive(&B1, alpha, &A1, flags|ARMAS_RIGHT, cache);

    for (i = NB; i < A->cols; i += NB) {
        nI = i < A->cols - NB ? NB : A->cols - i;

        armas_x_submatrix_unsafe(&B0, B, 0, 0, B->rows, i);
        armas_x_submatrix_unsafe(&B1, B, 0, i, B->rows, nI);
        if (flags & ARMAS_UPPER) {
            armas_x_submatrix_unsafe(&A0, A, 0, i, i, nI);
        } else {
            armas_x_submatrix_unsafe(&A0, A, i, 0, nI, i);
        }
        armas_x_submatrix_unsafe(&A1, A, i, i, nI, nI);

        // update block and solve;
        armas_x_mult_kernel(alpha, &B1, -ONE, &B0, &A0, transB, cache);
        armas_x_solve_recursive(&B1, ONE, &A1, flags|ARMAS_RIGHT, cache);
    }
}

/*
 *   RIGHT-UPPER-TRANSA                RIGHT-LOWER
 *
 *                 A00 | A01 | A02                    A00 |  0  |  0
 *                ----------------                   ----------------
 *    B0|B1|B2      0  | A11 | A12       B0|B1|B2     A10 | A11 |  0
 *                ----------------                   ----------------
 *                  0  |  0  | A22                    A20 | A21 | A22
 *
 *  upper:
 *    B0 = B'0*A00 + B'1*A01 + B'2*A02 --> B'0 = (B0 - B'1*A01 - B'2*A02)*A00.-1
 *    B1 = B'1*A11 + B'2*A12           --> B'1 = (B1           - B'2*A12)*A11.-1
 *    B2 = B'2*A22                     --> B'2 = B2*A22.-1
 *  lower:
 *    B0 = B'0*A00 + B'1*A10 + B'2*A20 --> B'0 = (B0 - B'1*A10 - B'2*A20)*A00.-1
 *    B1 = B'1*A11 + B'2*A21           --> B'1 = (B1 - B'2*A21)*A11.-1
 *    B2 = B'2*A22                     --> B'2 = B2*A22.-1
 */
static
void solve_blk_rut_rl(
    armas_x_dense_t *B,
    const armas_x_dense_t *A,
    DTYPE alpha,
    int flags,
    cache_t *cache)
{
    register int i, nI, cI;
    armas_x_dense_t A0, A1, B0, B1;
    int NB = cache->NB;
    int transB = flags & ARMAS_UPPER ? ARMAS_TRANSB : 0;

    // solve first block
    nI = NB < A->cols ? NB : A->cols;
    cI = NB < A->cols ? A->cols-NB : 0;
    armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
    armas_x_submatrix_unsafe(&B1, B, 0,  cI, B->rows, nI);
    armas_x_solve_recursive(&B1, alpha, &A1, flags|ARMAS_RIGHT, cache);

    for (i = A->cols-NB; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        cI = i < NB ? 0 : i-NB;

        armas_x_submatrix_unsafe(&B0, B, 0, i,  B->rows, B->cols-i );
        armas_x_submatrix_unsafe(&B1, B, 0, cI, B->rows, nI);
        if (flags & ARMAS_UPPER) {
            armas_x_submatrix_unsafe(&A0, A, cI, i, A->rows-i, nI);
        } else {
            armas_x_submatrix_unsafe(&A0, A, i, cI, nI, A->cols-i);
        }
        armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);

        // update and solve
        armas_x_mult_kernel(alpha, &B1, -1.0, &B0, &A0, transB, cache);
        armas_x_solve_recursive(&B1, ONE, &A1, flags|ARMAS_RIGHT, cache);
    }
}

void armas_x_solve_blocked(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *mcache)
{
    switch (flags&(ARMAS_UPPER|ARMAS_LOWER|ARMAS_RIGHT|ARMAS_TRANSA)) {
    case ARMAS_RIGHT|ARMAS_UPPER:
    case ARMAS_RIGHT|ARMAS_LOWER|ARMAS_TRANSA:
        solve_blk_ru_rlt(B, A, alpha, flags, mcache);
        break;

    case ARMAS_RIGHT|ARMAS_LOWER:
    case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        solve_blk_rut_rl(B, A, alpha, flags, mcache);
        break;

    case ARMAS_UPPER:
    case ARMAS_LOWER|ARMAS_TRANSA:
        solve_blk_lu_llt(B, A, alpha, flags, mcache);
        break;

    case ARMAS_LOWER:
    case ARMAS_UPPER|ARMAS_TRANSA:
        solve_blk_lut_ll(B, A, alpha, flags, mcache);
        break;
    }
}
#else
#warning "Missing defines; no code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
