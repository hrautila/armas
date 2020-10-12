
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Givens rotations

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_gvleft) && defined(armas_x_gvright)
#define ARMAS_PROVIDES 1
#endif
#if defined(armas_x_gvrotate)
#define ARMAS_REQUIRES 1
#endif
// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
//! \endcond

/*
 * Notes about applying Givens rotation.
 *
 * See Golub, 3rd edition, Section 5.1.8
 *
 *   Premultiplication by G(i,k,t).T amounts to a counterclockwise rotation of
 *   t radians in the (i, k) coordinate plane.
 *
 * Then 5.1.9 gives algorithm for prepostmultiplication of a matrix.
 *
 *   A([i,k],:) = ( c s ).T * A([i,k],:) 
 *                (-s c )
 *
 *   for i = 1:n
 *      A[i,j] = c*A[i,j] - s*A[k,j]
 *      A[k,j] = s*A[i,j] + c*A[k,j]
 *
 *  If we compare this to BLAS DROT subroutine that plane rotation to vectors DX and DY
 *  it is given as 
 *
 *   DROT(N, dx, dy, c, s):            |  which is equal to clockwise rotation.
 *   for i = 1:N                       |
 *     dx[i] = c*dx[i] + s*dy[i]       |  ( dx ) = ( c s ) ( dx )
 *     dy[i] = c*dy[i] - s*dx[i]       |  ( dy ) = (-s c ) ( dy )
 *
 *  The relative position of the vectors is defined but one would assume that DY is in
 *  higher index than DX. And looking code in function DBDSQR we see calls like
 *  DROT(N, VT[M-1,1], VT[M,1], C, S) which updates eigenvectors on rows M-1 and M.
 *
 *  It propably does not matter whether the rotation is clockwise or counterclockwise
 *  if it is applied consistently. (19.8.2014)
 */

/*
 * \brief Apply plane rotations from left.
 *
 * See armas_gvleft() for details.
 *
 * \ingroup lapackaux internal
 */
static inline
void gvleft(armas_x_dense_t * A, DTYPE c, DTYPE s, int r1, int r2,
            int col, int ncol)
{
    DTYPE t0, *y0, *y1;
    int k, n;

    y0 = &A->elems[col * A->step + r1];
    y1 = &A->elems[col * A->step + r2];
    for (k = 0, n = 0; k < ncol; k++, n += A->step) {
        t0 = c * y0[n] + s * y1[n];
        y1[n] = c * y1[n] - s * y0[n];
        y0[n] = t0;
    }
}


/*
 * \brief Apply plane rotations from right.
 *
 * See armas_gvright() for details.
 *
 * \ingroup lapackaux internal
 */
static inline
void gvright(armas_x_dense_t * A, DTYPE c, DTYPE s, int c1, int c2,
             int row, int nrow)
{
    DTYPE t0, *y0, *y1;
    int k;

    y0 = &A->elems[c1 * A->step + row];
    y1 = &A->elems[c2 * A->step + row];
    for (k = 0; k < nrow; k++) {
        t0 = c * y0[k] + s * y1[k];
        y1[k] = c * y1[k] - s * y0[k];
        y0[k] = t0;
    }
}


/**
 * @brief Apply Givens rotation (c, s) to rows of A.
 *
 * @param A
 *      Target matrix
 * @param c, s
 *      Givens rotation paramers
 * @param r1
 *      Index to first row
 * @param r2
 *      Index to second row
 * @param col
 *      Start column
 * @param ncol
 *      Number of columns 
 *
 * @ingroup lapack
 */
void armas_x_gvleft(armas_x_dense_t * A, DTYPE c, DTYPE s, int r1, int r2,
                    int col, int ncol)
{
    DTYPE v0;
    int k, lastc;

    if (col >= A->cols)
        return;

    lastc = min(col + ncol, A->cols);
    if (r1 == r2) {
        // one row
        for (k = col; k < lastc; k++) {
            v0 = armas_x_get(A, r1, k);
            armas_x_set(A, r1, k, c * v0);
        }
        return;
    }
    gvleft(A, c, s, r1, r2, col, ncol);
}


/**
 * @brief Apply Givens rotation (c, s) to columns of A.
 *
 * @param[out] A
 *      Target matrix
 * @param[in] c, s
 *      Givens rotation paramers
 * @param[in] c1
 *      Index to first column
 * @param[in] c2
 *      Index to second column
 * @param[in] row
 *      Start row
 * @param[in] nrow
 *      Number of rows
 *
 * @ingroup lapack
 */
void armas_x_gvright(armas_x_dense_t * A, DTYPE c, DTYPE s, int c1, int c2,
                     int row, int nrow)
{
    DTYPE v0;
    int k, lastr;

    if (row >= A->rows)
        return;

    lastr = min(A->rows, row + nrow);
    if (c1 == c2) {
        // one column
        for (k = row; k < lastr; k++) {
            v0 = armas_x_get(A, k, c1);
            armas_x_set(A, k, c1, c * v0);
        }
        return;
    }
    gvright(A, c, s, c1, c2, row, nrow);
}

/**
 * @brief Apply multiple Givens rotations to matrix A from left or right.
 *
 * Applies sequence of plane rotations to matrix A from either left or right.
 * The transformation has the form A = P*A (left) or A = A*P.T (right) where
 * where P is orthogonal matrix of sequence of k plane rotations.
 *
 * P is either forward (ARMAS_FORWARD) or backward (ARMAS_BACKWARD) sequence
 *
 *    \f$ P = P_{k-1}...P_1 P_0,     P = P_0 *P_1...P_{k-1} \f$
 *
 * where \f$ P_n \f$ is plane rotation defined by 2x2 matrix
 *```txt
 *     P(n) = ( c(n), s(n) )
 *            (-s(n), c(n) )
 *```
 * Left/right is indicated with flags parameter.
 *
 * @param [in,out] A
 *      Target matrix
 * @param [in] start
 *      Start row (left update) or column (right update)
 * @param[in] S, C
 *      Rotation parameters
 * @param[in] nrot
 *      Number of rotation to apply.
 * @param[in] flags
 *      Select from left (ARMAS_LEFT) or right (ARMAS_RIGHT). Or with direction
 *      flag backward (ARMAS_BACKWARD)  or forward (ARMAS_FORWARD). Forward is
 *      default direction for updates.
 * @return
 *      Number of rotation applied, min(nrot, A->rows-start) for left and
 *      min(nrot, A->cols-start) for right.
 *
 * @ingroup lapack
 */
int armas_x_gvupdate(armas_x_dense_t * A, int start,
                     armas_x_dense_t * C, armas_x_dense_t * S, int nrot,
                     int flags)
{
    int k, n, end = start + nrot;
    DTYPE c, s;

    if (flags & ARMAS_BACKWARD) {
        if (flags & ARMAS_LEFT) {
            end = min(A->rows, end);
            for (k = end, n = nrot; n > 0 && k > start; n--, k--) {
                c = armas_x_get_at_unsafe(C, n - 1);
                s = armas_x_get_at_unsafe(S, n - 1);
                if (c != 1.0 || s != 0.0)
                    gvleft(A, c, s, k - 1, k, 0, A->cols);
            }
        } else {
            end = min(A->cols, end);
            for (k = end, n = nrot; n > 0 && k > start; n--, k--) {
                c = armas_x_get_at_unsafe(C, n - 1);
                s = armas_x_get_at_unsafe(S, n - 1);
                if (c != 1.0 || s != 0.0)
                    gvright(A, c, s, k - 1, k, 0, A->rows);
            }
        }
    } else {
        // here we apply forward direction
        if (flags & ARMAS_LEFT) {
            end = min(A->rows, end);
            for (k = start, n = 0; n < nrot && k < end; n++, k++) {
                c = armas_x_get_at_unsafe(C, n);
                s = armas_x_get_at_unsafe(S, n);
                if (c != 1.0 || s != 0.0)
                    gvleft(A, c, s, k, k + 1, 0, A->cols);
            }
        } else {
            end = min(A->cols, end);
            for (k = start, n = 0; n < nrot && k < end; n++, k++) {
                c = armas_x_get_at_unsafe(C, n);
                s = armas_x_get_at_unsafe(S, n);
                if (c != 1.0 || s != 0.0)
                    gvright(A, c, s, k, k + 1, 0, A->rows);
            }
        }
    }
    return n;
}


/**
 * @brief Rotate vectors
 *
 *  Computes
 *```txt
 *    (X^T) = G(c,s)*(X^T)   or (X Y) = (X Y)*G(c,s)
 *    (Y^T)          (Y^T)
 *```
 *  Assumes len(X) == len(Y).
 * @ingroup lapackaux
 */
int armas_x_gvrot_vec(armas_x_dense_t * X, armas_x_dense_t * Y, DTYPE c,
                      DTYPE s)
{
    DTYPE x, y;
    int k;
    for (k = 0; k < armas_x_size(X); k++) {
        x = armas_x_get_at_unsafe(X, k);
        y = armas_x_get_at_unsafe(Y, k);
        armas_x_gvrotate(&x, &y, c, s, x, y);
        armas_x_set_at_unsafe(X, k, x);
        armas_x_set_at_unsafe(Y, k, y);
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
