
// Copyright (c) Harri Rautila, 2012-2015

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__update_ger_ext_unb) || defined(__update_trmv_ext_unb) || defined(__update2_symv_ext_unb)
#define __ARMAS_PROVIDES 1
#endif
// this file requires no external public functions
#if EXT_PRECISION
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "eft.h"

#if defined(__update_ger_ext_unb)
/*
 * Unblocked update of general M-by-N matrix. A[i,j] = A[i,j] + alpha*x[i]*y[j]
 */
int __update_ger_ext_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                         DTYPE alpha, int flags, int N, int M)
{
    DTYPE p, r, s, c, q;
    register int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            // p + r = X[i]*Y[j]
            twoprod(&p, &r, X->md[i*X->inc], Y->md[j*Y->inc]);
            // p + c = alpha*p
            twoprod(&p, &c, alpha, p);
            // s + q = A[i,j] + p;
            twosum(&s, &q, A->md[i+j*A->step], p);
            A->md[i+j*A->step] = s + (alpha*r + c + q);
        }
    }
    return 0;
}
#endif

#if defined(__update_trmv_ext_unb)
/*
 * Unblocked update of triangular (M == N) and trapezoidial (M != N) matrix.
 * (M is rows, N is columns.)
 */
int __update_trmv_ext_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                          DTYPE alpha, int flags, int N, int M)
{
    DTYPE p, r, s, c, q;
    register int i, j;
    switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        for (i = 0; i < M; i++) {
            for (j = i; j < N; j++) {
                // p + r = X[i]*Y[j]
                twoprod(&p, &r, X->md[i*X->inc], Y->md[j*Y->inc]);
                // p + c = alpha*p
                twoprod(&p, &c, alpha, p);
                // s + q = A[i,j] + p;
                twosum(&s, &q, A->md[i+j*A->step], p);
                A->md[i+j*A->step] = s + (alpha*r + c + q);
            }
        }
        break;
    case ARMAS_LOWER:
    default:
        for (j = 0; j < N; j++) {
            for (i = 0; i < j+1; i++) {
                // p + r = X[i]*Y[j]
                twoprod(&p, &r, X->md[i*X->inc], Y->md[j*Y->inc]);
                // p + c = alpha*p
                twoprod(&p, &c, alpha, p);
                // s + q = A[i,j] + p;
                twosum(&s, &q, A->md[i+j*A->step], p);
                A->md[i+j*A->step] = s + (alpha*r + c + q);
            }
        }
        break;
    }
    return 0;
}
#endif

#if defined(__update2_symv_ext_unb)
// A = A + alpha*x.T*y + alpha*y.T*x
int __update2_symv_ext_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                           DTYPE alpha, int flags, int N)
{
    DTYPE p0, p1, r, s, c0, c1, q0, q1;
    register int i, j;
    switch (flags & (ARMAS_UPPER|ARMAS_LOWER)) {
    case ARMAS_UPPER:
        for (i = 0; i < N; i++) {
            for (j = i; j < N; j++) {
                // p + r = X[i]*Y[j]
                twoprod(&p0, &r, X->md[i*X->inc], Y->md[j*Y->inc]);
                // p + c = alpha*p
                twoprod(&p0, &c0, alpha, p0);
                c0 += alpha*r;

                // p + r = X[j]*Y[i]
                twoprod(&p1, &r, X->md[j*X->inc], Y->md[i*Y->inc]);
                // p + c = alpha*p
                twoprod(&p1, &c1, alpha, p1);
                c1 += alpha*r;
                c0 += c1;
                // s + q = A[i,j] + p;
                twosum(&s, &q0, A->md[i+j*A->step], p0);
                twosum(&s, &q1, s, p1);
                A->md[i+j*A->step] = s + (c0 + q0 + q1);
            }
        }
        break;
    case ARMAS_LOWER:
    default:
        for (j = 0; j < N; j++) {
            for (i = 0; i < j+1; i++) {
                // p + r = X[i]*Y[j]
                twoprod(&p0, &r, X->md[i*X->inc], Y->md[j*Y->inc]);
                // p + c = alpha*p
                twoprod(&p0, &c0, alpha, p0);
                c0 += alpha*r;

                // p + r = X[j]*Y[i]
                twoprod(&p1, &r, X->md[j*X->inc], Y->md[i*Y->inc]);
                // p + c = alpha*p
                twoprod(&p1, &c1, alpha, p1);
                c1 += alpha*r;
                c0 += c1;
                // s + q = A[i,j] + p;
                twosum(&s, &q0, A->md[i+j*A->step], p0);
                twosum(&s, &q1, s, p1);
                A->md[i+j*A->step] = s + (c0 + q0 + q1);
            }
        }
        break;
    }
    return 0;
}
#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
