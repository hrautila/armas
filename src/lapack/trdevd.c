
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Eigenvalues of symmetric tridiagonal matrix

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdeigen) && defined(armas_x_trdeigen_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_gvupdate) && defined(armas_x_gvright) \
    && defined(armas_x_sym_eigen2x2vec) && defined(armas_x_trd_qrsweep) \
    && defined(armas_x_trd_qlsweep)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "auxiliary.h"
//! \endcond



/*
 * \brief Implicit symmetric QR iteration  (Golub algorithm 8.3.3)
 *
 * Lapack: DSTEQR
 */
static
int armas_x_trdevd_qr(armas_x_dense_t * D, armas_x_dense_t * E,
                      armas_x_dense_t * V, armas_x_dense_t * CS, ABSTYPE tol,
                      int flags, armas_conf_t * conf)
{
    int ip, iq, iqold, ipold, k, N, n, maxiter, nrot;
    int forwards = 1, stop = 0, saves = 0;
    ABSTYPE e0, e1, d0, d1, f0, g0, ushift;
    armas_x_dense_t Cr, Sr, sD, sE;

    EMPTY(sD);
    EMPTY(sE);

    N = armas_x_size(D);
    if (V) {
        armas_x_subvector(&Cr, CS, 0, N);
        armas_x_subvector(&Sr, CS, N, N);
        saves = 1;
    }

    maxiter = 6 * N;
    iq = iqold = N;
    ip = ipold = 0;
    for (stop = 0, n = 0; !stop && maxiter > 0 && iq > 0; maxiter--, n++) {
        // 1. deflate off-diagonal entries if they are small
        d0 = ABS(armas_x_get_at_unsafe(D, iq - 1));
        for (k = iq - 1; k > 0; k--) {
            e1 = ABS(armas_x_get_at_unsafe(E, k - 1));
            d1 = ABS(armas_x_get_at_unsafe(D, k - 1));
            if (e1 < tol * (d0 + d1)) {
                armas_x_set_at_unsafe(E, k - 1, ZERO);
                if (k == (iq - 1)) {
                    // convergence of bottom value;
                    iq = iq - 1;
                    stop = k == 0;
                    goto Next;
                } else if (k - 1 == ip) {
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

        if ((iq - ip) == 1) {
            iq -= 1;
            continue;
        }

        if ((iq - ip) == 2) {
            // 2x2 block
            DTYPE a, b, c, e0, e1, cs, sn;
            a = armas_x_get_at_unsafe(D, ip);
            b = armas_x_get_at_unsafe(E, ip);
            c = armas_x_get_at_unsafe(D, ip + 1);
            armas_x_sym_eigen2x2vec(&e0, &e1, &cs, &sn, a, b, c);
            //printf(".. 2x2 block [%d,%d] e0 = %e, e1 = %e\n", ip, iq, e0, e1);
            armas_x_set_at_unsafe(D, ip, e0);
            armas_x_set_at_unsafe(D, ip + 1, e1);
            armas_x_set_at_unsafe(E, ip, ZERO);
            if (V) {
                armas_x_gvright(V, cs, sn, ip, ip + 1, 0, V->rows);
            }
            iq -= 2;
            goto Next;
        }

        if (n == 0 || iq != iqold || ip != ipold) {
            // disjoint block, select direction
            ipold = ip;
            iqold = iq;
            d0 = ABS(armas_x_get_at_unsafe(D, ip));
            d1 = ABS(armas_x_get_at_unsafe(D, iq - 1));
            forwards = d1 >= d0 || (flags & ARMAS_FORWARD) != 0;
        }

        armas_x_subvector(&sD, D, ip, iq - ip);
        armas_x_subvector(&sE, E, ip, iq - ip - 1);

        if (forwards) {
            // implicit QR sweep on subvector
            // d1 last in D, d0 before d1, e0 last in E
            d0 = armas_x_get_at_unsafe(D, iq - 2);
            e0 = armas_x_get_at_unsafe(E, iq - 2);
            d1 = armas_x_get_at_unsafe(D, iq - 1);
            // Wilkinson shift from trailing 2x2 matrix
            ushift = wilkinson_shift(d0, e0, d1);
            f0 = armas_x_get_at_unsafe(&sD, 0) - ushift;
            g0 = armas_x_get_at_unsafe(&sE, 0);
            nrot = armas_x_trd_qrsweep(&sD, &sE, &Cr, &Sr, f0, g0, saves);
            // update eigenvectors
            if (V) {
                armas_x_gvupdate(V, ip, &Cr, &Sr, nrot, ARMAS_RIGHT);
            }
        } else {
            // implicit QL sweep on subvector
            d0 = armas_x_get_at_unsafe(D, ip);
            d1 = armas_x_get_at_unsafe(D, ip + 1);
            e0 = armas_x_get_at_unsafe(E, ip);
            // Wilkinson shift from leading 2x2 matrix
            ushift = wilkinson_shift(d1, e0, d0);
            f0 = armas_x_get_at_unsafe(D, iq - 1) - ushift;
            g0 = armas_x_get_at_unsafe(E, iq - 2);
            nrot = armas_x_trd_qlsweep(&sD, &sE, &Cr, &Sr, f0, g0, saves);
            // update eigenvectors
            if (V) {
                armas_x_gvupdate(V, ip, &Cr, &Sr, nrot,
                                 ARMAS_RIGHT | ARMAS_BACKWARD);
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
 * \param[in] flags
 *      Indicators *ARMAS_WANTV*
 * \param[in,out] conf
 *      Configuration block.
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 * \ingroup lapack
 */
int armas_x_trdeigen(armas_x_dense_t * D,
                     armas_x_dense_t * E,
                     armas_x_dense_t * V, int flags, armas_conf_t * conf)
{
    if (!conf)
        conf = armas_conf_default();

    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;
    if (armas_x_trdeigen_w(D, E, V, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    int stat = armas_x_trdeigen_w(D, E, V, flags, wbs, conf);
    armas_wrelease(&wb);
    return stat;
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
 * \param[in] flags
 *      Indicators *ARMAS_WANTV*
 * \param[out] wb
 *      Workspace of size 2*N if eigenvector wanted. If eigenvectors are not 
 *      wanted call with constant *ARMAS_NOWORK*. If eigenvector wanted and called
 *      with wb.bytes set to zero size of workspace is returned in wb.bytes and function
 *      returns with success.
 * \param[in,out] conf
 *      Configuration block.
 *
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 *
 * Last error codes:
 *   - ARMAS_EINVAL if D or E null or V null when flags ARMAS_WANTV set.
 *   - ARMAS_ESIZE  if len(E) != len(D)-1 or m(V) != n(V) != len(D)
 *   - ARMAS_ENEED_VECTOR if D and E are not vectors.
 *   - ARMAS_EWORK  if eigenvectors wanted and workspace less than 2*len(D) elements.
 *   - ARMAS_ECONVERGE if algorigthm does not converge
 *
 * \ingroup lapack
 */
int armas_x_trdeigen_w(armas_x_dense_t * D,
                       armas_x_dense_t * E,
                       armas_x_dense_t * V,
                       int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t CS, *vv;
    int err, N = armas_x_size(D);
    ABSTYPE tol = 5.0;

    if (!conf)
        conf = armas_conf_default();

    if (!D) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    if (wb && wb->bytes == 0) {
        // workspace only if eigenvector wanted
        if ((flags & ARMAS_WANTV) != 0)
            wb->bytes = 2 * N * sizeof(DTYPE);
        return 0;
    }

    vv = (armas_x_dense_t *) 0;
    // check for sizes
    if (!(armas_x_isvector(D) && armas_x_isvector(E))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (flags & ARMAS_WANTV) {
        if (!V) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        if (V->rows != N || V->rows != V->cols) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        vv = V;
    }
    if (armas_x_size(E) != N - 1) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (vv && (!wb || armas_wbytes(wb) < 2 * N * sizeof(DTYPE))) {
        // if eigenvectors needed then must have workspace
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (vv) {
        armas_x_make(&CS, 2 * N, 1, 2 * N, (DTYPE *) armas_wptr(wb));
    } else {
        armas_x_make(&CS, 0, 0, 1, (DTYPE *) 0);
    }

    tol = tol * EPS;
    if (conf->tolmult > 0) {
        tol = ((ABSTYPE) conf->tolmult) * EPS;
    }
    err = armas_x_trdevd_qr(D, E, vv, &CS, tol, flags, conf);

    if (err == 0) {
        armas_x_sort_eigenvec(D, vv, __nil, __nil, ARMAS_ASC);
    } else {
        conf->error = ARMAS_ECONVERGE;
    }
    return err;
}
#else
#warning "Missing defines. No code"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
