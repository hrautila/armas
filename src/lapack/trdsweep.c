
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trd_qrsweep) && defined(armas_x_trd_qlsweep)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"


int armas_x_trd_qlsweep(armas_x_dense_t * D, armas_x_dense_t * E,
                        armas_x_dense_t * Cr, armas_x_dense_t * Sr, DTYPE f0,
                        DTYPE g0, int saves)
{
    DTYPE cosr, sinr, r, d0, d1, e0, e1, e0r, e0c, w0, f, g;
    int k;
    int N = armas_x_size(D);

    e0c = w0 = e1 = ZERO;

    f = f0;
    g = g0;
    d0 = armas_x_get_at_unsafe(D, N - 1);
    e0 = armas_x_get_at_unsafe(E, N - 2);

    for (k = N - 1; k > 0; k--) {
        d1 = armas_x_get_at_unsafe(D, k - 1);
        armas_x_gvcompute(&cosr, &sinr, &r, f, g);
        if (k < N - 1)
            armas_x_set_at_unsafe(E, k, r);
        // rows
        armas_x_gvrotate(&d0, &e0c, cosr, sinr, d0, e0);
        armas_x_gvrotate(&e0r, &d1, cosr, sinr, e0, d1);
        armas_x_gvrotate(&d0, &e0r, cosr, sinr, d0, e0r);
        armas_x_gvrotate(&e0c, &d1, cosr, sinr, e0c, d1);
        if (k > 1) {
            e1 = armas_x_get_at_unsafe(E, k - 2);
            armas_x_gvrotate(&w0, &e1, cosr, sinr, 0.0, e1);
        }
        armas_x_set_at_unsafe(D, k, d0);
        d0 = d1;
        e0 = e1;
        f = e0r;
        g = w0;
        if (saves) {
            armas_x_set_at_unsafe(Cr, k - 1, cosr);
            armas_x_set_at_unsafe(Sr, k - 1, -sinr);
        }
    }
    armas_x_set_at_unsafe(D, 0, d0);
    armas_x_set_at_unsafe(E, 0, e0c);
    return N - 1;
}

/*
 * \brief Implicit tridiagonal QR sweep from top to bottom.
 *
 *   - zx is the bulge element
 *
 *   d0 | e0 | z0  .
 *  -------------------> 1st(b)
 *   e0 | d1 | e1  z1
 *  -------------------> 2nd(b)
 *   z0 | e1 | d1  e2
 *   .  | z1 | e2  d2
 *      V    V
 *     1st  2nd
 */

int armas_x_trd_qrsweep(armas_x_dense_t * D, armas_x_dense_t * E,
                        armas_x_dense_t * Cr, armas_x_dense_t * Sr, DTYPE f0,
                        DTYPE g0, int saves)
{
    DTYPE cosr, sinr, r, d0, d1, e0, e1, e0r, e0c, w0, f, g;
    int k;
    int N = armas_x_size(D);

    e0r = e1 = w0 = ZERO;
    f = f0;
    g = g0;
    d0 = armas_x_get_at_unsafe(D, 0);
    e0 = armas_x_get_at_unsafe(E, 0);

    for (k = 0; k < N - 1; k++) {
        d1 = armas_x_get_at_unsafe(D, k + 1);
        armas_x_gvcompute(&cosr, &sinr, &r, f, g);
        if (k > 0)
            armas_x_set_at_unsafe(E, k - 1, r);
        armas_x_gvrotate(&d0, &e0c, cosr, sinr, d0, e0);
        armas_x_gvrotate(&e0r, &d1, cosr, sinr, e0, d1);
        armas_x_gvrotate(&d0, &e0r, cosr, sinr, d0, e0r);
        armas_x_gvrotate(&e0c, &d1, cosr, sinr, e0c, d1);
        if (k < N - 2) {
            e1 = armas_x_get_at_unsafe(E, k + 1);
            armas_x_gvrotate(&w0, &e1, cosr, sinr, 0.0, e1);
        }
        armas_x_set_at_unsafe(D, k, d0);
        d0 = d1;
        e0 = e1;
        f = e0r;
        g = w0;
        if (saves) {
            armas_x_set_at_unsafe(Cr, k, cosr);
            armas_x_set_at_unsafe(Sr, k, sinr);
        }
    }
    armas_x_set_at_unsafe(D, N - 1, d0);
    armas_x_set_at_unsafe(E, N - 2, e0r);
    //return k > 0 ? N : 0;
    return N - 1;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
