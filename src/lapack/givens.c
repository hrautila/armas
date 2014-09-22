
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_gvcompute) && defined(__armas_gvrotate)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
#include "auxiliary.h"

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
 * \brief Compute Givens rotation.
 *
 * Compatible to blas.DROTG and lapack.DLARTG.
 *
 * \ingroup lapack givens
 */
void __armas_gvcompute(DTYPE *c, DTYPE *s, DTYPE *r, DTYPE a, DTYPE b)
{
    __gvrotg(c, s, r, a, b);
}

/*
 * \brief Apply Givens rotation.
 *
 * Computes
 *
 *    ( v0 )  = G(c, s) * ( y0 )  or ( v0 v1 ) = ( y0 y1 ) * G(c, s)
 *    ( v1 )              ( y1 )
 *
 *    G(c, s) = ( c  s )  => ( v0 ) = ( c*y0 + s*y1 )
 *              (-s  c )     ( v1 )   ( c*y1 - s*y0 )
 *
 * \ingroup lapack givens
 */
void __armas_gvrotate(DTYPE *v0, DTYPE *v1, DTYPE c, DTYPE s, DTYPE y0, DTYPE y1)
{
    __gvrot(v0, v1, c, s, y0, y1);
}

/*
 * \brief Apply Givens rotation (c, s) to rows of A.
 *
 * \param A
 *      Target matrix
 * \param c, s
 *      Givens rotation paramers
 * \param r1
 *      Index to first row
 * \param r2
 *      Index to second row
 * \param col
 *      Start column
 * \param ncol
 *      Number of columns 
 *
 * \ingroup lapack givens
 */
void __armas_gvleft(__armas_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol)
{
    DTYPE v0, v1, t0, t1;
    int k, lastc;

    if (col >= A->cols)
        return;

    lastc = min(col+ncol, A->cols);
    if (r1 == r2) {
        // one row
        for (k = col; k < lastc; k++) {
            v0 = __armas_get(A, r1, k);
            __armas_set(A, r1, k, c*v0);
        }
        return;
    }
    __gvleft(A, c, s, r1, r2, col, ncol);
}


/*
 * \brief Apply Givens rotation (c, s) to columns of A.
 *
 * \param A
 *      Target matrix
 * \param c, s
 *      Givens rotation paramers
 * \param c1
 *      Index to first column
 * \param c2
 *      Index to second column
 * \param row
 *      Start row
 * \param nrow
 *      Number of rows
 *
 * \ingroup lapack givens
 */
void __armas_gvright(__armas_dense_t *A, DTYPE c, DTYPE s, int c1, int c2, int row, int nrow)
{
    DTYPE v0, v1, t0, t1;
    int k, lastr;

    if (row >= A->rows)
        return;

    lastr = min(A->rows, row+nrow);
    if (c1 == c2) {
        // one column
        for (k = row; k < lastr; k++) {
            v0 = __armas_get(A, k, c1);
            __armas_set(A, k, c1, c*v0);
        }
        return;
    }
    __gvright(A, c, s, c1, c2, row, nrow);
}

/*
 * \brief Apply multiple Givens rotations to matrix A from left or right.
 *
 * Applies sequence of plane rotations to matrix A from either left or right.
 * The transformation has the form A = P*A (left) or A = A*P.T (right) where
 * where P is orthogonal matrix of sequence of k plane rotations.
 *
 * P is either forward (ARMAS_FORWARD) or backward (ARMAS_BACKWARD) sequence
 *
 *    P = P(k-1)*...P(1)*P(0),     P = P(0)*P(1)...*P(k-1)
 *
 * where P(n) is plane rotation defined by 2x2 matrix
 *
 *    R(n) = ( c(n), s(n) )
 *           (-s(n), c(n) )
 *
 * Left/right is indicated with flags parameter.
 *
 * \param A [in,out]
 *      Target matrix
 * \param start [in]
 *      Start row (left update) or column (right update)
 * \param S, C [in]
 *      Rotation parameters
 * \param nrot [in]
 *      Number of rotation to apply.
 * \param flags [in]
 *      Select from left (ARMAS_LEFT) or right (ARMAS_RIGHT). Or with direction
 *      flag backward (ARMAS_BACKWARD)  or forward (ARMAS_FORWARD). Forward is
 *      default direction for updates.
 * \return
 *      Number of rotation applied, min(nrot, A->rows-start) for left and
 *      min(nrot, A->cols-start) for right.
 *
 * \ingroup lapack givens
 */
int __armas_gvupdate(__armas_dense_t *A, int start, 
                     __armas_dense_t *C, __armas_dense_t *S, int nrot, int flags)
{
    int k, n, end = start+nrot;
    DTYPE c, s;

    if (flags & ARMAS_BACKWARD) {
        if (flags & ARMAS_LEFT) {
            end = min(A->rows, end);
            for (k = end, n = nrot; n > 0 && k > start; n--, k--) {
                c = __armas_get_at_unsafe(C, n-1);
                s = __armas_get_at_unsafe(S, n-1);
                if (c != 1.0 || s != 0.0)
                    __gvleft(A, c, s, k-1, k, 0, A->cols);
            }
        } else {
            end = min(A->cols, end);
            for (k = end, n = nrot; n > 0 && k > start; n--, k--) {
                c = __armas_get_at_unsafe(C, n-1);
                s = __armas_get_at_unsafe(S, n-1);
                if (c != 1.0 || s != 0.0)
                    __gvright(A, c, s, k-1, k, 0, A->rows);
            }
        }
    } else {
        // here we apply forward direction
        if (flags & ARMAS_LEFT) {
            end = min(A->rows, end);
            for (k = start, n = 0; n < nrot && k < end; n++, k++) {
                c = __armas_get_at_unsafe(C, n);
                s = __armas_get_at_unsafe(S, n);
                if (c != 1.0 || s != 0.0)
                    __gvleft(A, c, s, k, k+1, 0, A->cols);
            }
        } else {
            end = min(A->cols, end);
            for (k = start, n = 0; n < nrot && k < end; n++, k++) {
                c = __armas_get_at_unsafe(C, n);
                s = __armas_get_at_unsafe(S, n);
                if (c != 1.0 || s != 0.0)
                    __gvright(A, c, s, k, k+1, 0, A->rows);
            }
        }
    }
    return n;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

