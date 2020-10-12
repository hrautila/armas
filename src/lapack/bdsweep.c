
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bd_qrsweep) && defined(armas_x_bd_qlsweep)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"

#include "auxiliary.h"

/*
 *  Implicit QR step with initial G(f0, g0)
 *
 *    d0 | e0   g  .
 *  -----+----------->  2. left inside loop
 *     f | d1  e1  *
 *     . | x1  d2  e2
 *     . | .   x2  d3
 *       |
 *       V 1. right inside the loop
 *
 */


/**
 * @brief Bidiagonal top to bottom QR sweep.
 *
 * @param D [in,out]
 *      Diagonal elements.
 * @param E [in,out]
 *      Off-diagonal elements.
 * @param Cr, Sr, Cl, Sl [out]
 *      Saved plane rotations.  Accessed only if rotations are saved.
 * @param f0, g0 [in]
 *      Initial plane rotation parameters.
 * @param saves [in]
 *      If set then rotation are saved.
 * @return
 *      Number of rotations performed (N-2)
 *
 * Assume: len(D)-1 == len(E)
 *
 * @ingroup lapackaux
 */
int armas_x_bd_qrsweep(armas_x_dense_t * D, armas_x_dense_t * E,
                       armas_x_dense_t * Cr, armas_x_dense_t * Sr,
                       armas_x_dense_t * Cl, armas_x_dense_t * Sl,
                       DTYPE f0, DTYPE g0, int saves)
{
    DTYPE d1, e1, d2, e2, f, g, cosr, cosl, sinr, sinl, r;
    int k, N = armas_x_size(D);

    d2 = e2 = ZERO;
    d1 = armas_x_get_at_unsafe(D, 0);
    e1 = armas_x_get_at_unsafe(E, 0);
    f = f0;
    g = g0;
    for (k = 0; k < N - 1; k++) {
        d2 = armas_x_get_at_unsafe(D, k + 1);

        armas_x_gvcompute(&cosr, &sinr, &r, f, g);
        if (k > 0) {
            // e[i-1] = r
            armas_x_set_at_unsafe(E, k - 1, r);
        }
        // f = cosr*d[i] + sinr*e[i]; e[i] = cosr*e[i] - sinr*d[i]
        armas_x_gvrotate(&f, &e1, cosr, sinr, d1, e1);
        // g = sinr*d[i+1]; d[i+1] = cosr*d[i+];
        armas_x_gvrotate(&g, &d2, cosr, sinr, 0.0, d2);
        armas_x_gvcompute(&cosl, &sinl, &r, f, g);
        // d[i] = r
        d1 = r;
        // f = cosl*e[i] + sinl*d[i+1]; d[i+1] = cosl*d[i+1] - sinl*e[i];
        armas_x_gvrotate(&f, &d2, cosl, sinl, e1, d2);
        if (k < N - 2) {
            e2 = armas_x_get_at_unsafe(E, k + 1);
            // g = sinl*e[i+1]; e[i+1] = cosl*e[i+1];
            armas_x_gvrotate(&g, &e2, cosl, sinl, 0.0, e2);
            armas_x_set_at_unsafe(E, k + 1, e2);
        }

        armas_x_set_at_unsafe(D, k, d1);
        armas_x_set_at_unsafe(D, k + 1, d2);
        d1 = d2;
        e1 = e2;

        // save rotations
        if (saves) {
            armas_x_set_at_unsafe(Cr, k, cosr);
            armas_x_set_at_unsafe(Sr, k, sinr);
            armas_x_set_at_unsafe(Cl, k, cosl);
            armas_x_set_at_unsafe(Sl, k, sinl);
        }
    }
    // e[-1] = f
    armas_x_set_at_unsafe(E, N - 2, f);
    return N - 1;
}

/**
 * @brief Implicit zero shift QR sweep.
 *
 * @param D [in,out]
 *      Diagonal elements
 * @param E [in,out]
 *      Off-diagonal elements
 * @param Cr, Sr, Cl, Sl [out]
 *      Saved plane rotations. Accessed only if rotations are saved.
 * @param saves [in]
 *      If set then rotation are saved.
 * @return
 *      Number of rotations performed (N-2)
 *
 * Assume: len(D)-1 == len(E).
 * As described in Demmel-Kahan, 1990.
 *
 * @ingroup lapackaux
 */
int armas_x_bd_qrzero(armas_x_dense_t * D, armas_x_dense_t * E,
                      armas_x_dense_t * Cr, armas_x_dense_t * Sr,
                      armas_x_dense_t * Cl, armas_x_dense_t * Sl, int saves)
{
    DTYPE d1, e1, d2, cosr, cosl, sinr, sinl, r;
    int k, N = armas_x_size(D);

    d2 = ZERO;
    d1 = armas_x_get_at_unsafe(D, 0);
    sinl = ONE;
    cosr = ONE;
    cosl = ONE;
    for (k = 0; k < N - 1; k++) {
        e1 = armas_x_get_at_unsafe(E, k);
        d2 = armas_x_get_at_unsafe(D, k + 1);
        armas_x_gvcompute(&cosr, &sinr, &r, d1 * cosr, e1);
        if (k > 0)
            armas_x_set_at_unsafe(E, k - 1, sinl * r);
        armas_x_gvcompute(&cosl, &sinl, &r, cosl * r, sinr * d2);
        armas_x_set_at_unsafe(D, k, r);
        d1 = d2;
        if (saves) {
            armas_x_set_at_unsafe(Cr, k, cosr);
            armas_x_set_at_unsafe(Sr, k, sinr);
            armas_x_set_at_unsafe(Cl, k, cosl);
            armas_x_set_at_unsafe(Sl, k, sinl);
        }
    }
    d2 = cosr * d2;
    armas_x_set_at_unsafe(D, N - 1, d2 * cosl);
    armas_x_set_at_unsafe(E, N - 2, d2 * sinl);
    return N - 1;
}

/**
 * @brief Bidiagonal bottom to top QL sweep.
 *
 * @param D [in,out]
 *      Diagonal elements
 * @param E [in,out]
 *      Off-diagonal elements
 * @param Cr, Sr, Cl, Sl [out]
 *      Saved plane rotations. Accessed only if rotations are saved.
 * @param f0, g0 [in]
 *      Initial plane rotation parameters
 * @param saves [in]
 *      If set then rotation are saved.
 * @return
 *      Number of rotations performed (N-2)
 *
 * Assume: len(D)-1 == len(E)
 *
 * @ingroup lapackaux
 */
int armas_x_bd_qlsweep(armas_x_dense_t * D, armas_x_dense_t * E,
                       armas_x_dense_t * Cr, armas_x_dense_t * Sr,
                       armas_x_dense_t * Cl, armas_x_dense_t * Sl, DTYPE f0,
                       DTYPE g0, int saves)
{
    DTYPE d1, e1, d2, e2, f, g, cosr, cosl, sinr, sinl, r;
    int k, n, N = armas_x_size(D);

    d1 = e2 = ZERO;
    d1 = armas_x_get_at_unsafe(D, N - 1);
    e1 = armas_x_get_at_unsafe(E, N - 2);
    f = f0;
    g = g0;
    for (n = 0, k = N - 1; k > 0; k--, n++) {
        d2 = armas_x_get_at_unsafe(D, k - 1);

        armas_x_gvcompute(&cosr, &sinr, &r, f, g);
        if (k < N - 1) {
            armas_x_set_at_unsafe(E, k, r);
        }
        //f  = cosr*d1 + sinr*e1;   e1 = cosr*e1 - sinr*d1;
        armas_x_gvrotate(&f, &e1, cosr, sinr, d1, e1);
        //g  = sinr*d2;  d2 = cosr*d2;
        armas_x_gvrotate(&g, &d2, cosr, sinr, 0.0, d2);

        armas_x_gvcompute(&cosl, &sinl, &r, f, g);
        d1 = r;
        //f  = cosl*e1 + sinl*d2;  d2 = cosl*d2 - sinl*e1;
        armas_x_gvrotate(&f, &d2, cosl, sinl, e1, d2);
        if (k > 1) {
            e2 = armas_x_get_at_unsafe(E, k - 2);
            //g  = sinl*e2;  e2 = cosl*e2;
            armas_x_gvrotate(&g, &e2, cosl, sinl, 0.0, e2);
            armas_x_set_at_unsafe(E, k - 2, e2);
        }

        armas_x_set_at_unsafe(D, k, d1);
        armas_x_set_at_unsafe(D, k - 1, d2);
        // save values;
        d1 = d2;
        e1 = e2;
        // save rotations
        if (saves) {
            armas_x_set_at_unsafe(Cr, k - 1, cosr);
            armas_x_set_at_unsafe(Sr, k - 1, -sinr);
            armas_x_set_at_unsafe(Cl, k - 1, cosl);
            armas_x_set_at_unsafe(Sl, k - 1, -sinl);
        }
    }
    armas_x_set_at_unsafe(E, 0, f);
    return N - 1;
}


/**
 * @brief Implicit zero shift QL sweep.
 *
 * @param D [in,out]
 *      Diagonal elements
 * @param E [in,out]
 *      Off-diagonal elements
 * @param Cr, Sr, Cl, Sl [out]
 *      Saved plane rotations. Accessed only if rotations are saved.
 * @param saves [in]
 *      If set then rotation are saved.
 * @return
 *      Number of rotations performed (N-2)
 *
 * As described in Demmel-Kahan, 1990.
 *
 * @ingroup lapackaux
 */
int armas_x_bd_qlzero(armas_x_dense_t * D, armas_x_dense_t * E,
                      armas_x_dense_t * Cr, armas_x_dense_t * Sr,
                      armas_x_dense_t * Cl, armas_x_dense_t * Sl, int saves)
{
    DTYPE d1, e1, d2, cosr, cosl, sinr, sinl, r;
    int k, n, N = armas_x_size(D);

    sinl = ONE;
    d1 = armas_x_get_at_unsafe(D, N - 1);
    cosr = ONE;
    cosl = ONE;
    for (n = 0, k = N - 1; k > 0; k--, n++) {
        e1 = armas_x_get_at_unsafe(E, k - 1);
        d2 = armas_x_get_at_unsafe(D, k - 1);
        armas_x_gvcompute(&cosr, &sinr, &r, d1 * cosr, e1);
        if (k < N - 1)
            armas_x_set_at_unsafe(E, k, sinl * r);
        armas_x_gvcompute(&cosl, &sinl, &r, cosl * r, sinr * d2);
        armas_x_set_at_unsafe(D, k, r);
        d1 = d2;
        if (saves) {
            armas_x_set_at_unsafe(Cr, k - 1, cosr);
            armas_x_set_at_unsafe(Sr, k - 1, -sinr);
            armas_x_set_at_unsafe(Cl, k - 1, cosl);
            armas_x_set_at_unsafe(Sl, k - 1, -sinl);
        }
    }
    d2 = cosr * armas_x_get_at_unsafe(D, 0);
    armas_x_set_at_unsafe(D, 0, d2 * cosl);
    armas_x_set_at_unsafe(E, 0, d2 * sinl);
    return N - 1;
}

#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
