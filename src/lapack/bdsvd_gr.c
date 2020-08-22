
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bdsvd_golub)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_gvcompute) && defined(armas_x_gvupdate) \
    && defined(armas_x_bdsvd2x2_vec)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "auxiliary.h"

/*
 *  @brief Deterimine blocking for bidiagonal/tridiagonal matrix B
 *
 *   ( B00  0   0  )
 *   (  0  B11  0  )
 *   (  0   0  B22 )
 *
 *  such that
 *     B22 is diagonal (all off-diagonal elements are zero) (start index iq)
 *     B11 is bidiagonal without no zero off-diagonal entries (start index ip)
 *     B00 is bidiagonal 
 *  
 * @param ip [in,out]
 *      On entry, current start index. On exit, new start index.
 * @param iq [in,out]
 *      On entry, current end index. On exit, new end index.
 * @param D, E [in]
 *      Diagonal (D) and off-diagonal (E) elements
 * @return
 *      non-zero value, if either index has changed, zero otherwise.
 */
static
int find_diagonal_blocks(int *ip, int *iq, armas_x_dense_t * D,
                         armas_x_dense_t * E)
{
    int p, q, k, firstzero = 0, nonzero = 0;
    int ipold = *ip, iqold = *iq;
    DTYPE v;

    p = 0;
    q = armas_x_size(D);
    for (k = armas_x_size(E); k > 0; k--) {
        v = armas_x_get_at(E, k - 1);
        // B22 has all zeros on off-diagonal
        if (!nonzero && v == ZERO) {
            q = k;
            continue;
        }
        // we have seen first non-zero off-diagonal
        nonzero = 1;
        // A11 block has non-zeros on off-diagonal
        if (!firstzero && v != ZERO) {
            p = k - 1;
            continue;
        }
        // here we have seen zero again
        firstzero = 1;
        break;
    }
    if (!nonzero)
        q = 0;
    if (!firstzero)
        p = 0;
    *ip = p;
    *iq = q;
    return p != ipold || q != iqold ? 1 : 0;
}


/*
 * \brief Compute shift for 2x2 bidiagonal matrix
 *
 * Computes Golub-Kahan BSVD shift for 2x2 matrix T = B.T*B
 *
 *      ( e  0 )
 *  B = ( f  g )  T = B.T*B = ( f*f+e*e    f*g   )
 *      ( 0  h )              (   f*g    g*g+h*h )
 */
static
DTYPE compute_bsvd_shift(DTYPE e, DTYPE f, DTYPE g, DTYPE h)
{
    DTYPE e1, e2, a, b, c, T, D;
    a = f * f + e * e;
    b = g * f;
    c = h * h + g * g;
    T = (a + c) / 2.0;
    D = a * c - b * b;
    armas_x_qdroots(&e1, &e2, 1.0, T, D);
    if (ABS(e1) - c < ABS(e2) - c)
        return e1;
    return e2;
}



/*
 * Golub-Kahan-Reisch SVD algorithm as described in
 *  - Hogben, Handbook of Linear Algebra 2007, 45.2 algorithm 1b,1c
 *  - Golub, Matrix Computation, 3rd ed. section 8.6.2
 */
int armas_x_bdsvd_golub(armas_x_dense_t * D, armas_x_dense_t * E,
                  armas_x_dense_t * U, armas_x_dense_t * V,
                  armas_x_dense_t * CS, DTYPE tol, armas_conf_t * conf)
{
    int N, work, lc, i, n, nrot, ip, iq, zeros, saves;
    DTYPE e0, e1, d0, d1, dp, c, s, r, ushift;
    armas_x_dense_t sD, sE, Cr, Sr, Cl, Sl;

    EMPTY(sD);
    EMPTY(sE);

    N = armas_x_size(D);
    lc = 6 * N * N;
    saves = 0;

    if (U || V) {
        armas_x_subvector(&Cr, CS, 0, N);
        armas_x_subvector(&Sr, CS, N, N);
        armas_x_subvector(&Cl, CS, 2 * N, N);
        armas_x_subvector(&Sl, CS, 3 * N, N);
        saves = 1;
    }

    ip = 0;
    iq = N;
    for (work = 1, n = 0; work && lc > 0; lc--, n++) {
        // convergence: |E(i)| < epsilon*(|D(i)| + D(i+1)|) => E(i) = 0.0
        for (i = 0; i < iq - 1; i++) {
            e1 = armas_x_get_at_unsafe(E, i);
            if (e1 == 0.0)
                continue;
            d0 = armas_x_get_at_unsafe(D, i);
            d1 = armas_x_get_at_unsafe(D, i + 1);
            dp = ABS(d0) + ABS(d1);
            if (ABS(e1) < tol * dp) {
                armas_x_set_at(E, i, 0.0);
            }
        }
        // step: divide to blocks A00, A11, A22 where A22 is diagonal
        // and A11 has no zero off-diagonals (This should be intergrated with
        // the deflation loop above)
        find_diagonal_blocks(&ip, &iq, D, E);
        if (iq == 0) {
            work = 0;
            break;
        }
        // from column iq already diagonal; work ip:iq
        zeros = 0;
        for (i = ip; i < iq; i++) {
            d0 = armas_x_get_at(D, i);
            if (d0 == 0.0) {
                e1 = armas_x_get_at(E, i);
                armas_x_gvcompute(&c, &s, &r, d0, e1);
                armas_x_set_at_unsafe(D, i, r);
                armas_x_set_at_unsafe(E, i, 0.0);
                zeros = 1;
            }
        }
        if (zeros)
            continue;

        if ((iq - ip) == 2) {
            // 2x2 block
            DTYPE smin, smax, cosl, sinl, cosr, sinr;
            d0 = armas_x_get_at_unsafe(D, ip);
            d1 = armas_x_get_at_unsafe(D, ip + 1);
            e1 = armas_x_get_at_unsafe(E, ip);
            armas_x_bdsvd2x2_vec(&smin, &smax,
                                 &cosl, &sinl, &cosr, &sinr, d0, e1,
                           d1);
            armas_x_set_at_unsafe(D, ip, smax);
            armas_x_set_at_unsafe(D, ip + 1, smin);
            armas_x_set_at_unsafe(E, ip, ZERO);
            if (U) {
                armas_x_gvright(U, cosl, sinl, ip, ip + 1, 0, U->rows);
            }
            if (V) {
                armas_x_gvleft(V, cosr, sinr, ip, ip + 1, 0, V->cols);
            }
            continue;
        }

        armas_x_subvector(&sD, D, ip, iq - ip);
        armas_x_subvector(&sE, E, ip, iq - ip - 1);

        // get elements to compute shift from trailing B.T*T 2x2 matrix
        d0 = armas_x_get_at_unsafe(D, iq - 2);
        d1 = armas_x_get_at_unsafe(D, iq - 1);
        e0 = iq < 3 ? 0.0 : armas_x_get_at_unsafe(E, iq - 3);
        e1 = armas_x_get_at_unsafe(E, iq - 2);
        ushift = compute_bsvd_shift(e0, d0, e1, d1);

        d0 = armas_x_get_at_unsafe(D, ip);
        e0 = armas_x_get_at_unsafe(E, ip);
        nrot =
            armas_x_bd_qrsweep(&sD, &sE, &Cr, &Sr,
                               &Cl, &Sl, (d0 * d0 - ushift), (d0 * e0), saves);
        // update
        if (U) {
            armas_x_gvupdate(U, ip, &Cl, &Sl, nrot, ARMAS_RIGHT);
        }
        if (V) {
            armas_x_gvupdate(V, ip, &Cr, &Sr, nrot, ARMAS_LEFT);
        }
    }

    if (lc > 0) {
        // finished properly
        for (i = 0; i < N; i++) {
            d0 = armas_x_get_at_unsafe(D, i);
            if (d0 < 0) {
                armas_x_set_at_unsafe(D, i, -d0);
                if (V) {
                    armas_x_row(&sD, V, i);
                    armas_x_scale(&sD, -1.0, conf);
                }
            }
        }
    }
    return lc > 0 ? 0 : -1;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
