
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Eigenvalues of symmetric tridiagonal matrix

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdeigen) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_gvupdate) && defined(armas_x_gvright) && defined(__sym_eigen2x2vec)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
#include "auxiliary.h"
//! \endcond



/*
 * \brief Implicit symmetric QR iteration  (Golub algorithm 8.3.3)
 *
 * Lapack: DSTEQR
 */
int __trdevd_qr(armas_x_dense_t *D, armas_x_dense_t *E,
                armas_x_dense_t *V, armas_x_dense_t *CS, ABSTYPE tol, armas_conf_t *conf)
{
    int ip, iq, iqold, ipold, k, N, n, maxiter, nrot;
    int forwards = 1, stop = 0, saves = 0;
    ABSTYPE e0, e1, d0, d1, f0, g0, ushift;
    armas_x_dense_t Cr, Sr, sD, sE;

    EMPTY(sD); EMPTY(sE);

    N = armas_x_size(D);
    if (V) {
        armas_x_subvector(&Cr, CS, 0, N);
        armas_x_subvector(&Sr, CS, N, N);
        saves = 1;
    }

    maxiter = 6*N;
    iq = iqold = N; ip = ipold = 0;
    for (stop = 0, n = 0; !stop && maxiter > 0 && iq > 0; maxiter--, n++) {
        // 1. deflate off-diagonal entries if they are small
        d0 = __ABS(armas_x_get_at_unsafe(D, iq-1));
        for (k = iq-1; k > 0; k--) {
            e1 = __ABS(armas_x_get_at_unsafe(E, k-1));
            d1 = __ABS(armas_x_get_at_unsafe(D, k-1));
            if (e1 < tol*(d0+d1)) {
                armas_x_set_at_unsafe(E, k-1, __ZERO);
                if (k == (iq-1)) {
                    // convergence of bottom value;
                    iq = iq - 1;
                    stop = k == 0;
                    goto Next; 
                } else if (k-1 == ip) {
                    // convergence of top value;
                    ip = k;
                    //stop = ip == iq;
                    goto Next; 
                }
                ip = k;
                break;
            }
            ip = k - 1;
            d0 = d1;
        }

        if (iq <= ip) {
            stop = 1;
            continue;
        }

        if ((iq-ip) == 2) {
            // 2x2 block
            DTYPE a, b, c, e0, e1, cs, sn;
            a = armas_x_get_at_unsafe(D, ip);
            b = armas_x_get_at_unsafe(E, ip);
            c = armas_x_get_at_unsafe(D, ip+1);
            __sym_eigen2x2vec(&e0, &e1, &cs, &sn, a, b, c);
            //printf(".. 2x2 block [%d,%d] e0 = %e, e1 = %e\n", ip, iq, e0, e1);
            armas_x_set_at_unsafe(D, ip, e0);
            armas_x_set_at_unsafe(D, ip+1, e1);
            armas_x_set_at_unsafe(E, ip, __ZERO);
            if (V) {
                armas_x_gvright(V, cs, sn, ip, ip+1, 0, V->rows);
            }
            iq -= 2;
            goto Next;
        }

        if (n == 0 || iq != iqold || ip != ipold) {
            // disjoint block, select direction
            ipold = ip; iqold = iq;
            d0 = __ABS(armas_x_get_at_unsafe(D, ip));
            d1 = __ABS(armas_x_get_at_unsafe(D, iq-1));
            forwards = d1 >= d0;
        }

        armas_x_subvector(&sD, D, ip, iq-ip);
        armas_x_subvector(&sE, E, ip, iq-ip-1);

        if (forwards) {
            // implicit QR sweep on subvector
            // d1 last in D, d0 before d1, e0 last in E
            d0 = armas_x_get_at_unsafe(D, iq-2);
            e0 = armas_x_get_at_unsafe(E, iq-2);
            d1 = armas_x_get_at_unsafe(D, iq-1);
            // Wilkinson shift from trailing 2x2 matrix
            ushift = __wilkinson(d0, e0, d1);
            f0 = armas_x_get_at_unsafe(&sD, 0) - ushift;
            g0 = armas_x_get_at_unsafe(&sE, 0);
            nrot = __trd_qrsweep(&sD, &sE, &Cr, &Sr, f0, g0, saves);
            // update eigenvectors
            if (V) {
                armas_x_gvupdate(V, ip, &Cr, &Sr, nrot, ARMAS_RIGHT);
            }
        } else {
            // implicit QL sweep on subvector
            d0 = armas_x_get_at_unsafe(D, ip);
            d1 = armas_x_get_at_unsafe(D, ip+1);
            e0 = armas_x_get_at_unsafe(E, ip);
            // Wilkinson shift from leading 2x2 matrix
            ushift = __wilkinson(d1, e0, d0);
            f0 = armas_x_get_at_unsafe(D, iq-1) - ushift;
            g0 = armas_x_get_at_unsafe(E, iq-2);
            nrot = __trd_qlsweep(&sD, &sE, &Cr, &Sr, f0, g0, saves);
            // update eigenvectors
            if (V) {
                armas_x_gvupdate(V, ip, &Cr, &Sr, nrot, ARMAS_RIGHT|ARMAS_BACKWARD);
            }
        }

    Next:
        ;
    }
    return maxiter > 0 ? 0 : -1;
}

/**
 * \brief Compute eigenvalues of a symmetric tridiagonal matrix T.
 *
 * Computes all eigenvalues and, optionally, eigenvectors of a symmetric
 * tridiagonal matrix T.
 *
 * \param[in,out] D
 *      On entry, the diagonal elements of B. On exit, the eigenvalues
 *      of T in inreasing order.
 * \param[in] E
 *      On entry, the offdiagonal elements of T. On exit, E is destroyed.
 * \param[in,out] V
 *      On entry, initial orthogonal matrix of eigenvectors. On exit,
 *      updated eigenvectors.
 * \param[out] W
 *      Workspace of size 2*N.
 * \param[in] flags
 *      Indicators *ARMAS_WANTV*
 * \param[in,out] conf
 *      Configuration block.
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 * \ingroup lapack
 */
int armas_x_trdeigen(armas_x_dense_t *D, armas_x_dense_t *E,
                     armas_x_dense_t *V, armas_x_dense_t *W, int flags, armas_conf_t *conf)
{
    armas_x_dense_t CS, *vv;
    int err, N = armas_x_size(D);
    ABSTYPE tol = 5.0;

    vv = (armas_x_dense_t *)0;
    // check for sizes
    if (! (armas_x_isvector(D) && armas_x_isvector(E))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
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
        vv = V;
    }
    if (armas_x_size(E) != N-1) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (vv && armas_x_size(W) < 2*N) {
        // if eigenvectors needed then must have workspace
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (vv) {
        armas_x_make(&CS, 2*N, 1, 2*N, armas_x_data(W));
    } else {
        armas_x_make(&CS, 0, 0, 1, (DTYPE *)0);
    }

    tol = tol*__EPS;
    if (conf->tolmult > 0) {
        tol = ((ABSTYPE)conf->tolmult) * __EPS;
    }
    err =__trdevd_qr(D, E, vv, &CS, tol, conf);

    if (err == 0) {
        __sort_eigenvec(D, vv, __nil, __nil, ARMAS_ASC);
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

