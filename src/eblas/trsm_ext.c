
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independent functions
#if defined(armas_x_ext_solve_trm_blk_unsafe) && \
  defined(armas_x_ext_solve_trm_unb_unsafe) &&  \
  defined(armas_x_ext_solve_trm_unsafe)
#define ARMAS_PROVIDES 1
#endif
#if defined(armas_x_ext_adot_unsafe)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "eft.h"
#include "kernel_ext.h"

// alignment to 256 bit boundary (256/8)
#define __DTYPE_ALIGN_SIZE (32/sizeof(DTYPE))
#define __DTYPE_ALIGN_MASK (32/sizeof(DTYPE) - 1)

#define __ALIGNED(x)  ((x) & ~__DTYPE_ALIGN_MASK)

#define __ALIGNED_PTR(_T, _x)                                               \
  ((_T*)((unsigned long)( ((char *)(_x)) + __DTYPE_ALIGN_SIZE-1) & ~__DTYPE_ALIGN_MASK))


// (1) Ph. Langlois, N. Louvet
//     Solving Triangular Systems More Accurately and Efficiently
//

/*
 * Functions here solves the matrix equations
 *
 *   A*X   = alpha*B  --> X = alpha*A.-1*B
 *   A.T*X = alpha*B  --> X = alpha*A.-T*B
 *   X*A   = alpha*B  --> X = alpha*B*A.-1
 *   X*A.T = alpha*B  --> X = alpha*B*A.-T
 */

/*
 *  LEFT-UPPER
 *     (b0)     (a00 a01 a02)   (b'0)
 *     (b1)  =  ( 0  a11 a12) * (b'1)
 *     (b2)     ( 0   0  a22)   (b'2)
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a02*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 *
 *  LEFT-LOWER-TRANS
 *     (b0)     (a00  0   0 )   (b'0)
 *     (b1)  =  (a10 a11  0 ) * (b'1)
 *     (b2)     (a20 a21 a22)   (b'2)
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a21*b'2           --> b'1 = (b1 - a21*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void solve_ext_lu_llt(
    armas_x_dense_t *Bc,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *Ac,
    int unit,
    int upper)
{
    register int i, j;
    armas_x_dense_t a0, b0, d0;
    DTYPE p0, s0, r0, u0, bk, ak, dk;

    // assume dB holds valid initial cumulative values
    for (j = 0; j < Bc->cols; ++j) {
        bk = armas_x_get_unsafe(Bc, Ac->cols-1, j);
        twoprod(&bk, &dk, alpha, bk);
        dk = dk + alpha * armas_x_get_unsafe(dB, Ac->cols-1, j);
        if (!unit) {
            ak = armas_x_get_unsafe(Ac, Ac->cols-1, Ac->cols-1);
            approx_twodiv(&s0, &r0, bk, ak);
            fastsum(&bk, &dk, s0, dk/ak + r0);
        }
        armas_x_set_unsafe(Bc, Ac->cols-1, j, bk);
        armas_x_set_unsafe(dB, Ac->cols-1, j, dk);

        for (i = Ac->cols-2; i >= 0; --i) {
            if (upper) {
                armas_x_submatrix_unsafe(&a0, Ac, i, i+1, 1, Ac->cols-i-1);
            } else {
                armas_x_submatrix_unsafe(&a0, Ac, i+1, i, Ac->cols-i-1, 1);
            }
            armas_x_submatrix_unsafe(&b0, Bc, i+1, j, Ac->cols-i-1, 1);
            armas_x_submatrix_unsafe(&d0, dB, i+1, j, Ac->cols-i-1, 1);

            bk = armas_x_get_unsafe(Bc, i, j);
            twoprod(&s0, &u0, alpha, bk);
            u0 += alpha * armas_x_get_unsafe(dB, i, j);
            armas_x_ext_adot_dx_unsafe(&s0, &u0, -ONE, &b0, &d0, &a0);

            if (!unit) {
                ak = armas_x_get_unsafe(Ac, i, i);
                approx_twodiv(&p0, &r0, s0, ak);
                fastsum(&s0, &u0, p0, u0/ak + r0);
            }
            // here: (s0, u0) is the final value
            armas_x_set_unsafe(Bc, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 * LEFT-UPPER
 *   (B0)   (A00 A01 A02)   (B'0)
 *   (B1) = ( 0  A11 A12) * (B'1)
 *   (B2)   ( 0   0  A22)   (B'2)
 *
 *    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = A00.-1*(B0 - A01*B'1 - A02*B'2)
 *    B1 = A11*B'1 + A12*B'2           --> B'1 = A11.-1*(B1 - A12*B'2)
 *    B2 = A22*B'2                     --> B'2 = A22.-1*B2
 *
 * LEFT-LOWER-TRANSA
 *   (B0)    (A00  0   0 )   (B'0)
 *   (B1) =  (A10 A11  0 ) * (B'1)
 *   (B2)    (A20 A21 A22)   (B'2)
 *
 *    B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = A00.-1*(B0 - A10*B'1 - A20*B'2)
 *    B1 = A11*B'1 + A21*B'2           --> B'1 = A11.-1*(B1 - A21*B'2)
 *    B2 = A22*B'0                     --> B'2 = A22.-1*B2
 */
static
void solve_ext_blk_lu_llt(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, nI, cI;
    armas_x_dense_t A0, A1, B0, B1, dB0, dB1;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int upper = (flags & ARMAS_UPPER) != 0 ? 1 : 0;
    int mflags = (flags & ARMAS_TRANS) != 0 ? ARMAS_TRANSA : 0;
    int NB = dB->cols;

    nI = A->cols < NB ? A->cols : NB;
    cI = A->cols < NB ? 0 : A->cols-NB;
    armas_x_submatrix_unsafe(&A0, A, cI, cI, nI, nI);
    armas_x_submatrix_unsafe(&B0, B, cI, 0, nI, B->cols);
    armas_x_submatrix_unsafe(&dB0,dB, cI, 0, nI, dB->cols);
    solve_ext_lu_llt(&B0, &dB0, alpha, &A0, unit, upper);

    for (i = A->cols-NB; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        cI = i < NB ? 0 : i-NB;
        if (upper) {
            armas_x_submatrix_unsafe(&A0, A, cI, i, nI, A->cols-i);
        } else {
            armas_x_submatrix_unsafe(&A0, A, i, cI, A->cols-i, nI);
        }
        armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
        armas_x_submatrix_unsafe(&B0,  B,  i, 0, A->cols-i, B->cols);
        armas_x_submatrix_unsafe(&dB0, dB, i, 0, A->cols-i, dB->cols);
        armas_x_submatrix_unsafe(&B1,  B,  cI, 0, nI, B->cols);
        armas_x_submatrix_unsafe(&dB1, dB, cI, 0, nI, dB->cols);

        if (alpha != ONE) {
            armas_x_ext_scale_unsafe(&B1, &dB1, alpha, &B1);
        } else {
            armas_x_scale_unsafe(&dB1, ZERO);
        }
        armas_x_ext_panel_dB_unsafe(&B1, &dB1, -ONE, &A0, &B0, &dB0, mflags, cache);
        solve_ext_lu_llt(&B1, &dB1, ONE, &A1, unit, upper);
    }
    armas_x_merge_unsafe(B, dB);
}


/*
 *   LEFT-UPPER-TRANS
 *
 *     (b0)     (a00 a01 a02)   (b'0)
 *     (b1)  =  ( 0  a11 a12) * (b'1)
 *     (b2)     ( 0   0  a22)   (b'2)
 *
 *   b0 = a00*b'0                     --> b'0 =  b0/a00
 *   b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *   b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 *
 *   LEFT-LOWER
 *
 *     (b0)     (a00  0   0 )   (b'0)
 *     (b1)  =  (a10 a11  0 ) * (b'1)
 *     (b2)     (a20 a21 a22)   (b'2)
 *
 *   b0 = a00*b'0                     --> b'0 =  b0/a00
 *   b1 = a10*b'0 + a11*b'1           --> b'1 = (b1 - a10*b'0)/a11
 *   b2 = a20*b'0 + a21*b'1 + a22*b'2 --> b'2 = (b2 - a20*b'0 - a21*b'1)/a22
 */
static
void solve_ext_lut_ll(
    armas_x_dense_t *Bc,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *Ac,
    int unit,
    int upper)
{
    register int i, j;
    armas_x_dense_t a0, b0, d0;
    DTYPE p0, s0, r0, u0, bk, ak, dk;

    // assume dB holds valid initial cumulative values
    for (j = 0; j < Bc->cols; ++j) {
        bk = armas_x_get_unsafe(Bc, 0, j);
        twoprod(&bk, &dk, alpha, bk);
        dk = dk + alpha * armas_x_get_unsafe(dB, 0, j);
        if (!unit) {
            ak = armas_x_get_unsafe(Ac, 0, 0);
            approx_twodiv(&s0, &r0, bk, ak);
            fastsum(&bk, &dk, s0, dk/ak + r0);
        }
        armas_x_set_unsafe(Bc, 0, j, bk);
        armas_x_set_unsafe(dB, 0, j, dk);

        for (i = 1; i < Ac->cols; ++i) {
            if (upper) {
                armas_x_submatrix_unsafe(&a0, Ac, 0, i, i, 1);
            } else {
                armas_x_submatrix_unsafe(&a0, Ac, i, 0, 1, i);
            }
            armas_x_submatrix_unsafe(&b0, Bc, 0, j, i, 1);
            armas_x_submatrix_unsafe(&d0, dB, 0, j, i, 1);

            bk = armas_x_get_unsafe(Bc, i, j);
            twoprod(&s0, &u0, alpha, bk);
            u0 += alpha * armas_x_get_unsafe(dB, i, j);
            armas_x_ext_adot_dx_unsafe(&s0, &u0, -ONE, &b0, &d0, &a0);

            if (!unit) {
                ak = armas_x_get_unsafe(Ac, i, i);
                approx_twodiv(&p0, &r0, s0, ak);
                fastsum(&s0, &u0, p0, u0/ak + r0);
            }
            armas_x_set_unsafe(Bc, i, j, s0);
            armas_x_set_unsafe(dB, i, j, u0);
        }
    }
}

/*
 * LEFT-UPPER-TRANS
 *    B0   (A00 A01 A02)   (B'0)
 *    B1 = ( 0  A11 A12) * (B'1)
 *    B2   ( 0   0  A22)   (B'2)
 *
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A01*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A01*B'0)
 *    B2 = A02*B'0 + A12*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A02*B'0 - A12*B'1)
 *
 * LEFT-LOWER
 *    B0   (A00  0   0 )   (B'0)
 *    B1 = (A10 A11  0 ) * (B'1)
 *    B2   (A20 A21 A22)   (B'2)
 *
 *    B0 = A00*B'0                     --> B'0 = A00.-1*B0
 *    B1 = A10*B'0 + A11*B'1           --> B'1 = A11.-1*(B1 - A10*B'0)
 *    B2 = A20*B'0 + A21*B'1 + A22*B'2 --> B'2 = A22.-1*(B2 - A20*B'0 - A21*B'1)
 */
static
void solve_ext_blk_lut_ll(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, nI, cI;
    armas_x_dense_t A0, A1, B0, B1, dB0, dB1;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int upper = flags & ARMAS_UPPER ? 1 : 0;
    int mflags = flags & ARMAS_UPPER ? ARMAS_TRANS : 0;
    int NB = cache->NB;

    nI = A->cols < NB ? A->cols : NB;
    armas_x_submatrix_unsafe(&A1, A, 0, 0, nI, nI);
    armas_x_submatrix_unsafe(&B1,  B, 0, 0, nI, B->cols);
    armas_x_submatrix_unsafe(&dB1, dB, 0, 0, nI, dB->cols);
    solve_ext_lut_ll(&B1, &dB1, alpha, &A1, unit, upper);

    for (i = NB; i < A->cols; i += NB) {
        nI = i < A->cols - NB ? NB : A->cols - i;
        cI = nI < NB ? A->cols-nI : i;

        if (upper) {
            armas_x_submatrix_unsafe(&A0, A, 0, cI, cI, nI);
        } else {
            armas_x_submatrix_unsafe(&A0, A, cI, 0, nI, cI);
        }
        armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
        armas_x_submatrix_unsafe(&B0,  B,  0, 0, cI, B->cols);
        armas_x_submatrix_unsafe(&dB0, dB, 0, 0, cI, B->cols);
        armas_x_submatrix_unsafe(&B1,  B,  cI, 0, nI, B->cols);
        armas_x_submatrix_unsafe(&dB1, dB, cI, 0, nI, B->cols);

        if (alpha != ONE) {
            armas_x_ext_scale_unsafe(&B1, &dB1, alpha, &B1);
        } else {
            armas_x_scale_unsafe(&dB1, ZERO);
        }
        armas_x_ext_panel_dB_unsafe(&B1, &dB1, -ONE, &A0, &B0, &dB0, mflags, cache);
        solve_ext_lut_ll(&B1, &dB1, ONE, &A1, unit, upper);
    }
    armas_x_merge_unsafe(B, dB);
}


/*
 * RIGHT-UPPER
 *                                (a00 a01 a02)
 *    (b0 b1 b2) =  (b'0 b'1 b'2) ( 0  a11 a12) 
 *                                ( 0   0  a22) 
 *
 *    b0 = a00*b'0                     --> b'0 =  b0/a00
 *    b1 = a01*b'0 + a11*b'1           --> b'1 = (b1 - a01*b'0)/a11
 *    b2 = a02*b'0 + a12*b'1 + a22*b'2 --> b'2 = (b2 - a02*b'0 - a12*b'1)/a22
 *
 * RIGHT-LOWER-TRANS
 *                                (a00  0   0 )
 *    (b0 b1 b2) =  (b'0 b'1 b'2) (a10 a11  0 )
 *                                (a20 a21 a22)
 *
 *    b0 = a00*b'0                     --> b'0 = b0/a00
 *    b1 = a10*b'0 + a11*b'1           --> b'1 = (b1 - a10*b'0)/a11
 *    b2 = a20*b'0 + a21*b'1 + a22*b'2 --> b'2 = (b2 - a20*b'0 - a21*b'1)/a22
 */
static
void solve_ext_ru_rlt(
    armas_x_dense_t *Bc,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *Ac,
    int unit,
    int upper)
{
    register int i, j;
    armas_x_dense_t a0, b0, d0;
    DTYPE p0, s0, r0, u0, bk, ak, dk;

    for (j = 0; j < Bc->rows; ++j) {
        bk = armas_x_get_unsafe(Bc, j, 0);
        twoprod(&bk, &dk, alpha, bk);
        dk = dk + alpha * armas_x_get_unsafe(dB, j, 0);
        if (!unit) {
            ak = armas_x_get_unsafe(Ac, 0, 0);
            approx_twodiv(&s0, &r0, bk, ak);
            fastsum(&bk, &dk, s0, dk/ak + r0);
        }
        armas_x_set_unsafe(Bc, j, 0, bk);
        armas_x_set_unsafe(dB, j, 0, dk);

        for (i = 1; i < Ac->cols; ++i) {
            if (upper) {
                armas_x_submatrix_unsafe(&a0, Ac, 0, i, i, 1);
            } else {
                armas_x_submatrix_unsafe(&a0, Ac, i, 0, 1, i);
            }
            armas_x_submatrix_unsafe(&b0, Bc, j, 0, 1, i);
            armas_x_submatrix_unsafe(&d0, dB, j, 0, 1, i);

            bk = armas_x_get_unsafe(Bc, j, i);
            twoprod(&s0, &u0, alpha, bk);
            u0 += alpha * armas_x_get_unsafe(dB, i, j);
            armas_x_ext_adot_dx_unsafe(&s0, &u0, -ONE, &b0, &d0, &a0);

            if (!unit) {
                ak = armas_x_get_unsafe(Ac, i, i);
                approx_twodiv(&p0, &r0, s0, ak);
                fastsum(&s0, &u0, p0, u0/ak + r0);
            }
            armas_x_set_unsafe(Bc, j, i, s0);
            armas_x_set_unsafe(dB, j, i, u0);
        }
    }
}

/*
 * RIGHT-UPPER
 *                                (A00 A01 A02)
 *    (B0 B1 B2) =  (B'0 B'1 B'2) ( 0  A11 A12)
 *                                ( 0   0  A22)
 *
 *    B0 = B'0*A00                     --> B'0 =  B0*A00.-1
 *    B1 = B'0*A01 + B'1*A11           --> B'1 = (B1 - B'0*A01)*A11.-1
 *    B2 = B'0*A02 + B'1*A12 + B'2*A22 --> B'2 = (B2 - B'0*A02 - B'1*A12)*A22.-1
 *
 * RIGHT-LOWER-TRANS
 *                                (A00  0   0 )
 *    (B0 B1 B2) =  (B'0 B'1 B'2) (A10 A11  0 )
 *                                (A20 A21 A22)
 *
 *    B0 = B'0*A00                     --> B'0 =  B0*A00.-1
 *    B1 = B'0*A10 + B'1*A11           --> B'1 = (B1 - B'0*A10)*A11.-1
 *    B2 = B'0*A20 + B'1*A21 + B'2*A22 --> B'2 = (B2 - B'0*A20 - B'1*A21)*A22.-1
 */
static
void solve_ext_blk_ru_rlt(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, nI, cI;
    armas_x_dense_t A0, A1, B0, B1, dB0, dB1;
    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int upper = flags & ARMAS_UPPER ? 1 : 0;
    int mflags = flags & ARMAS_UPPER ? 0 : ARMAS_TRANSB;
    int NB = cache->NB;

    nI = A->cols < NB ? A->cols : NB;
    armas_x_submatrix_unsafe(&A1, A, 0, 0, nI, nI);
    armas_x_submatrix_unsafe(&B1,  B, 0, 0, B->rows, nI);
    armas_x_submatrix_unsafe(&dB1, dB, 0, 0, dB->rows, nI);
    solve_ext_ru_rlt(&B1, &dB1, alpha, &A1, unit, upper);

    for (i = NB; i < A->cols; i += NB) {
        nI = i < A->cols - NB ? NB : A->cols - i;
        cI = nI < NB ? A->cols-nI : i;

        if (upper) {
            armas_x_submatrix_unsafe(&A0, A, 0, cI, cI, nI);
        } else {
            armas_x_submatrix_unsafe(&A0, A, cI, 0, nI, cI);
        }
        armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
        armas_x_submatrix_unsafe(&B0,  B,  0, 0, B->rows, cI);
        armas_x_submatrix_unsafe(&dB0, dB, 0, 0, B->rows, cI);
        armas_x_submatrix_unsafe(&B1,  B,  0, cI, B->rows, nI);
        armas_x_submatrix_unsafe(&dB1, dB, 0, cI, B->rows, nI);

        if (alpha != ONE) {
            armas_x_ext_scale_unsafe(&B1, &dB1, alpha, &B1);
        } else {
            armas_x_scale_unsafe(&dB1, ZERO);
        }
        armas_x_ext_panel_dA_unsafe(&B1, &dB1, -ONE, &B0, &dB0, &A0, mflags, cache);
        solve_ext_ru_rlt(&B1, &dB1, ONE, &A1, unit, upper);
    }
    armas_x_merge_unsafe(&B1, &dB1);
}

/*
 * RIGHT-UPPER-TRANS
 *
 *                                (a00 a01 a02)
 *    (b0 b1 b2) =  (b'0 b'1 b'2) ( 0  a11 a12)
 *                                ( 0   0  a22)
 *
 *    b0 = a00*b'0 + a01*b'1 + a02*b'2 --> b'0 = (b0 - a01*b'1 - a02*b'2)/a00
 *    b1 = a11*b'1 + a12*b'2           --> b'1 = (b1 - a12*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 *
 * RIGHT-LOWER
 *                                (a00  0  :  0 )
 *    (b0 b1 b2) =  (b'0 b'1 b'2) (a10 a11 :  0 )
 *                                (a20 a21 : a22)
 *
 *    b0 = a00*b'0 + a10*b'1 + a20*b'2 --> b'0 = (b0 - a10*b'1 - a20*b'2)/a00
 *    b1 = a11*b'1 + a21*b'2           --> b'1 = (b1 - a21*b'2)/a11
 *    b2 = a22*b'2                     --> b'2 =  b2/a22
 */
static
void solve_ext_rut_rl(
    armas_x_dense_t *Bc,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *Ac,
    int unit,
    int upper)
{
    register int i, j;
    armas_x_dense_t a0, b0, d0;
    DTYPE p0, s0, r0, u0, bk, ak, dk;

    // assume dB holds valid initial cumulative values
    for (j = 0; j < Bc->rows; ++j) {
        bk = armas_x_get_unsafe(Bc, j, Ac->cols-1);
        twoprod(&bk, &dk, alpha, bk);
        dk = dk + alpha * armas_x_get_unsafe(dB, j, Ac->cols-1);
        if (!unit) {
            ak = armas_x_get_unsafe(Ac, Ac->cols-1, Ac->cols-1);
            approx_twodiv(&s0, &r0, bk, ak);
            fastsum(&bk, &dk, s0, dk/ak + r0);
        }
        armas_x_set_unsafe(Bc, j, Ac->cols-1, bk);
        armas_x_set_unsafe(dB, j, Ac->cols-1, dk);

        for (i = Ac->cols-2; i >= 0; --i) {
            if (upper) {
                armas_x_submatrix_unsafe(&a0, Ac, i, i+1, 1, Ac->cols-i-1);
            } else {
                armas_x_submatrix_unsafe(&a0, Ac, i+1, i, Ac->cols-i-1, 1);
            }
            armas_x_submatrix_unsafe(&b0, Bc, j, i+1, 1, Ac->cols-i-1);
            armas_x_submatrix_unsafe(&d0, dB, j, i+1, 1, Ac->cols-i-1);

            bk = armas_x_get_unsafe(Bc, j, i);
            twoprod(&s0, &u0, alpha, bk);
            u0 += alpha * armas_x_get_unsafe(dB, j, i);
            armas_x_ext_adot_dx_unsafe(&s0, &u0, -ONE, &b0, &d0, &a0);

            if (!unit) {
                ak = armas_x_get_unsafe(Ac, i, i);
                approx_twodiv(&p0, &r0, s0, ak);
                fastsum(&s0, &u0, p0, u0/ak + r0);
            }
            armas_x_set_unsafe(Bc, j, i, s0);
            armas_x_set_unsafe(dB, j, i, u0);
        }
    }
}

/*
 * RIGHT-UPPER-TRANS
 *                                (A00 A01 A02)
 *    (B0 B1 B2) =  (B'0 B'1 B'2) ( 0  A11 A12)
 *                                ( 0   0  A22)
 *
 *    B0 = A00*B'0 + A01*B'1 + A02*B'2 --> B'0 = (B0 - A01*B'1 - A02*B'2)*A00.-1
 *    B1 = A11*B'1 + A12*B'2           --> B'1 = (B1 - A12*B'2)*A11.-1
 *    B2 = A22*B'2                     --> B'2 =  B2*A22.-1
 *
 * RIGHT-LOWER
 *                                (A00  0   0 )
 *    (B0 B1 B2) =  (B'0 B'1 B'2) (A10 A11  0 )
 *                                (A20 A21 A22)
 *
 *    B0 = A00*B'0 + A10*B'1 + A20*B'2 --> B'0 = (B0 - A10*B'1 - A20*B'2)*A00.-1
 *    B1 = A11*B'1 + A21*B'2           --> B'1 = (B1 - A21*B'2)*A11.-1
 *    B2 = A22*B'2                     --> B'2 =  B2*A22.-1
 *
 */
static
void solve_ext_blk_rut_rl(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    register int i, nI, cI;
    armas_x_dense_t A0, A1, B0, B1, dB0, dB1;

    int unit = flags & ARMAS_UNIT ? 1 : 0;
    int upper = flags & ARMAS_UPPER ? 1 : 0;
    int mflags = flags & ARMAS_TRANS ? 0 : ARMAS_TRANSB;
    int NB = cache->NB;

    nI = A->cols < NB ? A->cols : NB;
    cI = nI < NB ? 0 : A->cols - NB;
    armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
    armas_x_submatrix_unsafe(&B1, B, 0, cI, B->rows, nI);
    armas_x_submatrix_unsafe(&dB1, dB, 0, cI, dB->rows, nI);
    solve_ext_rut_rl(&B1, &dB1, alpha, &A1, unit, upper);

    for (i = A->cols-NB; i > 0; i -= NB) {
        nI = i < NB ? i : NB;
        cI = i < NB ? 0 : i-NB;

        if (upper) {
            armas_x_submatrix_unsafe(&A0, A, cI, cI+nI, nI, A->cols-nI-cI);
        } else {
            armas_x_submatrix_unsafe(&A0, A, cI+nI, cI, A->cols-nI-cI, nI);
        }
        armas_x_submatrix_unsafe(&B0,  B,  0, cI+nI, B->rows, A->cols-nI-cI);
        armas_x_submatrix_unsafe(&dB0, dB, 0, cI+nI, B->rows, A->cols-nI-cI);
        armas_x_submatrix_unsafe(&A1, A, cI, cI, nI, nI);
        armas_x_submatrix_unsafe(&B1,  B,  0, cI, B->rows, nI);
        armas_x_submatrix_unsafe(&dB1, dB, 0, cI, B->rows, nI);

        if (alpha != ONE) {
            armas_x_ext_scale_unsafe(&B1, &dB1, alpha, &B1);
        } else {
            armas_x_scale_unsafe(&dB1, ZERO);
        }
        armas_x_ext_panel_dA_unsafe(&B1, &dB1, -ONE, &B0, &dB0, &A0, mflags, cache);
        solve_ext_rut_rl(&B1, &dB1, ONE, &A1, unit, upper);
    }
    armas_x_merge_unsafe(B, dB);
}


/*
 * @param B
 *     M,N (right) or N,M (left) result matrix
 * @param dB
 *     K,N (right) or N,K (left) workspace matrix; K defines the block size
 * @param A
 *     N,N upper or left triangular matrix
 */
void armas_x_ext_solve_trm_unb_unsafe(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags)
{
    armas_x_dense_t B0, dB0;
    int nJ, cJ, i;
    int right = (flags & ARMAS_RIGHT) != 0 ? 1 : 0;
    int unit = (flags & ARMAS_UNIT) != 0 ? 1 : 0;
    int NB = right ? dB->rows : dB->cols;
    int M = right ? B->rows : B->cols;
    int upper = (flags & ARMAS_UPPER) != 0 ? 1 : 0;

    for (i = 0; i < M; i += NB) {
        nJ = i < M - NB ? NB : M - i;
        cJ = nJ < NB ? M - nJ : i;

        if (right) {
            armas_x_submatrix_unsafe(&B0, B, cJ, 0, nJ, B->cols);
            armas_x_submatrix_unsafe(&dB0, dB, cJ, 0, nJ, dB->cols);
        } else {
            armas_x_submatrix_unsafe(&B0, B, 0, cJ, B->rows, nJ);
            armas_x_submatrix_unsafe(&dB0, dB, 0, cJ, B->rows, nJ);
        }
        armas_x_scale_unsafe(&dB0, ZERO);

        // solve column or row panel;
        switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA)) {
        case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        case ARMAS_RIGHT:
            solve_ext_rut_rl(&B0, &dB0, alpha, A, unit, upper);
            break;

        case ARMAS_RIGHT|ARMAS_UPPER:
        case ARMAS_RIGHT|ARMAS_TRANSA:
            solve_ext_ru_rlt(&B0, &dB0, alpha, A, unit, upper);
            break;

            // LEFT from here
        case ARMAS_UPPER:
        case ARMAS_TRANSA:
            solve_ext_lu_llt(&B0, &dB0, alpha, A, unit, upper);
            break;

        case ARMAS_UPPER|ARMAS_TRANSA:
        default:
            solve_ext_lut_ll(&B0, &dB0, alpha, A, unit, upper);
            break;
        }
        armas_x_merge_unsafe(&B0, &dB0);
    }
}

/*
 * @param B
 *     M,N (right) or N,M (left) result matrix
 * @param dB
 *     K,N (right) or N,K (left) workspace matrix; K defines the block size
 * @param A
 *     N,N upper or left triangular matrix
 */
void armas_x_ext_solve_trm_blk_unsafe(
    armas_x_dense_t *B,
    armas_x_dense_t *dB,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    cache_t *cache)
{
    int i;
    armas_x_dense_t B0, dB0;
    int right = (flags & ARMAS_RIGHT) != 0 ? 1 : 0;
    int NB = right ? dB->rows : dB->cols;
    int M = right ? B->rows : B->cols;

    for (i = 0; i < M; i += NB) {
        int nJ = i < M - NB ? NB : M - i;
        int cJ = nJ < NB ? M - nJ : i;

        if (right) {
            armas_x_submatrix_unsafe(&B0, B, cJ, 0, nJ, B->cols);
            armas_x_submatrix_unsafe(&dB0, dB, cJ, 0, nJ, B->cols);
        } else {
            armas_x_submatrix_unsafe(&B0, B, 0,  cJ, B->rows, nJ);
            armas_x_submatrix_unsafe(&dB0, dB, 0,  cJ, B->rows, nJ);
        }
        armas_x_scale_unsafe(&dB0, ZERO);

        // solve column or row panel;
        switch (flags & (ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA)) {
        case ARMAS_RIGHT|ARMAS_UPPER|ARMAS_TRANSA:
        case ARMAS_RIGHT:
            solve_ext_blk_rut_rl(&B0, &dB0, alpha, A, flags, cache);
            break;

        case ARMAS_RIGHT|ARMAS_UPPER:
        case ARMAS_RIGHT|ARMAS_TRANSA:
            solve_ext_blk_ru_rlt(&B0, &dB0, alpha, A, flags, cache);
            break;

            // LEFT from here
        case ARMAS_UPPER:
        case ARMAS_TRANSA:
            solve_ext_blk_lu_llt(&B0, &dB0, alpha, A, flags, cache);
            break;

        case ARMAS_UPPER|ARMAS_TRANSA:
        default:
            solve_ext_blk_lut_ll(&B0, &dB0, alpha, A, flags, cache);
            break;
        }
        armas_x_merge_unsafe(&B0, &dB0);
    }
}

#if 1
/**
 * @brief Solve \$f X = A^-1*B \$f in extended precision.
 */
int armas_x_ext_solve_trm_w(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_wbuf_t *wb,
    armas_conf_t *cf)
{
    armas_x_dense_t dX;
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
    if (!wb) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }

    armas_cbuf_t cbuf = ARMAS_CBUF_EMPTY;
    if (armas_cbuf_select(&cbuf, cf) < 0) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }

    cache_t cache;
    armas_cache_setup(&cache, &cbuf, 3, sizeof(DTYPE));

    if (wb->bytes == 0) {
        wb->bytes = A->rows*min(A->cols, cache.NB)*sizeof(DTYPE);
        return 0;
    }
    int ncol = wb->bytes/(A->rows*sizeof(DTYPE));
    if (ncol < 4) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }
    if (ncol > cache.NB)
        ncol = cache.NB;
    armas_x_make(&dX, A->cols, ncol, A->cols, (DTYPE *)armas_wptr(wb));
    armas_x_ext_solve_trm_blk_unsafe(B, &dX, alpha, A, flags, &cache);
    return 0;
}

/**
 * @brief Solve \$f X = A^-1*B \$f in extended precision.
 */
int armas_x_ext_solve_trm(
    armas_x_dense_t *B,
    DTYPE alpha,
    const armas_x_dense_t *A,
    int flags,
    armas_conf_t *cf)
{
    int err = 0;
    armas_wbuf_t wb = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();
    if (armas_x_ext_solve_trm_w(B, alpha, A, flags,&wb, cf) < 0) {
        return -1;
    }
    if (!armas_walloc(&wb, wb.bytes)) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }
    err = armas_x_ext_solve_trm_w(B, alpha, A, flags, &wb, cf);
    armas_wrelease(&wb);
    return err;
}
#endif

#else
#warning "Missing defines! No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
