
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_bdsvd) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_gvcompute) && defined(__armas_gvupdate) && defined(__bdsvd2x2_vec)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
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
int find_diagonal_blocks(int *ip, int *iq, __armas_dense_t *D, __armas_dense_t *E)
{
    int p, q, k, firstzero = 0, nonzero = 0;
    int ipold = *ip, iqold = *iq;
    DTYPE v;

    q = __armas_size(D);
    for (k = __armas_size(E); k > 0; k--) {
        v = __armas_get_at(E, k-1);
        // B22 has all zeros on off-diagonal
        if ( !nonzero && v == __ZERO) {
            q = k;
            continue;
        }
        // we have seen first non-zero off-diagonal
        nonzero = 1;
        // A11 block has non-zeros on off-diagonal
        if ( !firstzero && v != __ZERO ) {
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
    DTYPE e1, e2, ha, a, b, c, T, D;
    a = f*f+e*e;
    b = g*f;
    c = h*h+g*g;
    T = (a + c)/2.0;
    D = a*c - b*b;
    __armas_qdroots(&e1, &e2, 1.0, T, D);
    if (__ABS(e1)-c < __ABS(e2)-c)
        return e1;
    return e2;
}



/*
 * Golub-Kahan-Reisch SVD algorithm as described in
 *  - Hogben, Handbook of Linear Algebra 2007, 45.2 algorithm 1b,1c
 *  - Golub, Matrix Computation, 3rd ed. section 8.6.2
 */
static
int __bsvd_golub_reisch(__armas_dense_t *D, __armas_dense_t *E,
                        __armas_dense_t *U, __armas_dense_t *V,
                        __armas_dense_t *CS, DTYPE tol, armas_conf_t *conf)
{
    int N, work, lc, i, n, nrot, ip, iq, zeros, saves;
    DTYPE e0, e1, d0, d1, dp, c, s, r, ushift;
    __armas_dense_t sD, sE, Cr, Sr, Cl, Sl;

    N = __armas_size(D);
    lc = 6*N*N;
    saves = 0;
    
    if (U || V) {
        __armas_subvector(&Cr, CS, 0, N);
        __armas_subvector(&Sr, CS, N, N);
        __armas_subvector(&Cl, CS, 2*N, N);
        __armas_subvector(&Sl, CS, 3*N, N);
        saves = 1;
    }

    ip = 0; iq = N; 
    for (work = 1, n = 0; work && lc > 0; lc--, n++) {
        // convergence: |E(i)| < epsilon*(|D(i)| + D(i+1)|) => E(i) = 0.0
        for (i = 0; i < iq-1; i++) {
            e1 = __armas_get_at_unsafe(E, i);
            if (e1 == 0.0)
                continue;
            d0 = __armas_get_at_unsafe(D, i);
            d1 = __armas_get_at_unsafe(D, i+1);
            dp = __ABS(d0) + __ABS(d1);
            if (__ABS(e1) < tol*dp) {
                __armas_set_at(E, i, 0.0);
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
            d0 = __armas_get_at(D, i);
            if (d0 == 0.0) {
                e1 = __armas_get_at(E, i);
                __armas_gvcompute(&c, &s, &r, d0, e1);
                __armas_set_at_unsafe(D, i, r);
                __armas_set_at_unsafe(E, i, 0.0);
                zeros = 1;
            }
        }
        if (zeros)
            continue;

        if ((iq - ip) == 2) {
            // 2x2 block
            DTYPE smin, smax, cosl, sinl, cosr, sinr;
            d0 = __armas_get_at_unsafe(D, ip);
            d1 = __armas_get_at_unsafe(D, ip+1);
            e1 = __armas_get_at_unsafe(E, ip);
            __bdsvd2x2_vec(&smin, &smax, &cosl, &sinl, &cosr, &sinr, d0, e1, d1);
            __armas_set_at_unsafe(D, ip, smax);
            __armas_set_at_unsafe(D, ip+1, smin);
            __armas_set_at_unsafe(E, ip, __ZERO);
            if (U) {
                __armas_gvright(U, cosl, sinl, ip, ip+1, 0, U->rows);
            }
            if (V) {
                __armas_gvleft(V, cosr, sinr, ip, ip+1, 0, V->cols);
            }
            continue;
        }

        __armas_subvector(&sD, D, ip, iq-ip);
        __armas_subvector(&sE, E, ip, iq-ip-1);

        // get elements to compute shift from trailing B.T*T 2x2 matrix
        d0 = __armas_get_at_unsafe(D, iq-2);
        d1 = __armas_get_at_unsafe(D, iq-1);
        e0 = iq < 3 ? 0.0 :__armas_get_at_unsafe(E, iq-3);
        e1 = __armas_get_at_unsafe(E, iq-2);
        ushift = compute_bsvd_shift(e0, d0, e1, d1);

        d0 = __armas_get_at_unsafe(D, ip);
        e0 = __armas_get_at_unsafe(E, ip);
        nrot = __bd_qrsweep(&sD, &sE, &Cr, &Sr, &Cl, &Sl, d0*d0-ushift, d0*e0, saves);
        // update 
        if (U) {
            __armas_gvupdate(U, ip, &Cl, &Sl, nrot, ARMAS_RIGHT);
        }
        if (V) {
            __armas_gvupdate(V, ip, &Cr, &Sr, nrot, ARMAS_LEFT);
        }
    }

    if (lc > 0) {
        // finished properly 
        for (i = 0; i < N; i++) {
            d0 = __armas_get_at_unsafe(D, i);
            if (d0 < 0) {
                __armas_set_at_unsafe(D, i, -d0);
                if (V) {
                    __armas_row(&sD, V, i);
                    __armas_scale(&sD, -1.0, conf);
                }
            }
        }
    }
    return lc > 0 ? 0 : -1;
}

/*
 * \brief Rotate lower bidiagonal matrix to upper bidiagonal matrix.
 */
static inline
void bdmake_upper(__armas_dense_t *D, __armas_dense_t *E,
                  __armas_dense_t *U, __armas_dense_t *C,
                  __armas_dense_t *CS)
{
    __armas_dense_t Cl, Sl;
    DTYPE cosl, sinl, r, d0, e0, d1;
    int nrot, saves, k, N = __armas_size(D);

    saves = 0;
    if (U || C) {
        __armas_subvector(&Cl, CS, 0, N);
        __armas_subvector(&Sl, CS, N, N);
        saves = 1;
    }
    d0 = __armas_get_at_unsafe(D, 0);
    for (k = 0; k < N-1; k++) {
        e0 = __armas_get_at_unsafe(E, k);
        d1 = __armas_get_at_unsafe(D, k+1);
        __armas_gvcompute(&cosl, &sinl, &r,  d0, e0);
        __armas_set_at_unsafe(D, k, r);
        __armas_set_at_unsafe(E, k, sinl*d1);
        d0 = cosl*d1;
        __armas_set_at_unsafe(D, k+1, d0);
        if (saves) {
            __armas_set_at_unsafe(&Cl, k, cosl);
            __armas_set_at_unsafe(&Sl, k, sinl);
        }
    }

    if (U && k > 0) {
        __armas_gvupdate(U, 0, &Cl, &Sl, k+1, ARMAS_RIGHT);
    }
    if (C && k > 0) {
        __armas_gvupdate(C, 0, &Cl, &Sl, k+1, ARMAS_LEFT);
    }
}

/*
 * \brief Compute SVD for bidiagonal matrix
 */
int __armas_bdsvd(__armas_dense_t *D, __armas_dense_t *E,
                  __armas_dense_t *U, __armas_dense_t *V,
                  __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    __armas_dense_t CS, *uu, *vv;
    int err, N = __armas_size(D);

    uu = (__armas_dense_t *)0;
    vv = (__armas_dense_t *)0;
    // check for sizes
    if (! (__armas_isvector(D) && __armas_isvector(E))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (flags & ARMAS_WANTU) {
        if (! U) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        if (U->cols != N) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        uu = U;
    }
    if (flags & ARMAS_WANTV) {
        if (! V) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        if (V->rows != N) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        if (flags & ARMAS_WANTU) {
            if (V->rows != U->cols) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
        }
        vv = V;
    }
    if (__armas_size(E) != N-1) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if ((uu || vv) && __armas_size(W) < 4*N) {
        // if eigenvectors needed then must have workspace
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (uu || vv) {
        __armas_make(&CS, 4*N, 1, 4*N, __armas_data(W));
    } else {
        __armas_make(&CS, 0, 0, 1, (DTYPE *)0);
    }
    if (flags & ARMAS_LOWER) {
        // rotate to UPPER bidiagonal
        bdmake_upper(D, E, U, __nil, &CS);
    }
    err =__bsvd_golub_reisch(D, E, uu, vv, &CS, __EPS,  conf);
    if (err == 0) {
        __eigen_sort(D, uu, vv, __nil, conf);
    } else {
        conf->error = ARMAS_ECONVERGE;
    }
    return err;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

