
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__bdsvd_demmel) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_gvcompute) && defined(armas_x_gvupdate) && defined(__bdsvd2x2_vec)
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
 * Demmel-Kahan SVD algorithm as described in
 * (1)  Demmel, Kahan 1990, Accurate Singular Values of Bidiagonal Matrices
 *      (Lapack Working Notes #3)
 * (2)  Hogben 2007, 45.2 algorithm 2a,2b
 */


/*
 * Estimate min, max singular values for bidiagonal matrix B;
 * See (1) 2.4
 */
static
void estimate_sval(DTYPE *smin, DTYPE *smax, armas_x_dense_t *D, armas_x_dense_t *E)
{
    DTYPE ssmin, ssmax, mu, e0, e1, d1;
    int k, N = armas_x_size(D);

    d1 = __ABS(armas_x_get_at_unsafe(D, 0));
    e1 = __ABS(armas_x_get_at_unsafe(E, 0));
    e0 = e1;
    ssmax = d1 > e1 ? d1 : e1;
    mu = ssmin = d1;
    for (k = 1; k < N; k++) {
        d1 = __ABS(armas_x_get_at_unsafe(D, k));
        if (k < N-1)
            e1 = __ABS(armas_x_get_at_unsafe(E, k));
        if (d1 > ssmax)
            ssmax = d1;
        if (e1 > ssmax)
            ssmax = e1;
        if (ssmin != __ZERO) {
            mu = d1 * ( mu / ( mu + e0 ) );
            if (mu < ssmin)
                ssmin = mu;
        }
        e0 = e1;
    }
    *smax = ssmax;
    *smin = ssmin / __SQRT((DTYPE)N);
}

/*
 * from (1)
 *
 * Convergence criteria:
 * 1a) mu(0) = |s(0)|
 *     for j = 0 to N-2: mu(j+1) = |s(j+1)|*(mu(j)/(mu(j) + |e(j)|)
 *     if |e(j)/mu(j)| <= tol then e(j) = 0.0

 * 1b) mu(N-1) = |s(N-1)|
 *     for j = N-2 to 0: mu(j) = |s(j)|*(mu(j+1)/(mu(j+1) + |e(j)|)
 *     if |e(j)/mu(j+1)| <= tol then e(j) = 0.0
 *
 * Applying the convergence criteria. In section 4 we presented four convergence criteria.
 * Since applying the convergence criteria costs approximately as many floating point
 * operations (O(n)) as performing a QR sweep, it is important to test criteria only
 * when they are likely to be satisfied. Our decision is based on the following empirical
 * observation: When chasing the bulge down (up) , the bottom most (topmost) entry s(n-1) (s(0))
 * often tends to converge to the smallest singular value, with e(n − 2) (e(0)) tending
 * to zero fastest of all offdiagonal entries. Therefore, when chasing the bulge down,
 * we expect convergence criteria 1a an d 2a to be successful, and possibly 1b but only for
 * the bottom most entry e(n−2) . Criteria 2b an d 1b for the other off diagonal entries
 * are not as likely to succeed. Conversely, when chasing the bulge up, we only apply
 * convergence criteria 1b, 2b and 1a for e(0) . One advantage of this scheme is that
 * testing 2a (for e(n−2) , and if the test succeeds, for e(n-3) too) costs only a few more
 * operations after testing 1a, since they share the same recurrence from (4.3).
 * Similarly, 2b (for e(0) , and if the test succeeds, for e(1) too) is very cheap after
 * applying 1b.
 */


/*
 * Demmel-Kahan SVD algorithm as described in (1)
 *
 * from  (1)
 *   EPS    = machine precision
 *   UNFL   = underflow threshold (smallest positive normalized number = DBL_MIN)
 *   N      = dimension of the matrix
 *   tol    = relative error tolerance (currently 100*EPS)
 *   maxit  = maximum number of QR inner loops (6*N^2)
 *
 *   slow   = estimate of minimum singular value 
 *   shigh  = estimate of maximum singular value max(|D(i)|, |E(i)|)
 */
int __bdsvd_demmel(armas_x_dense_t *D, armas_x_dense_t *E,
                   armas_x_dense_t *U, armas_x_dense_t *V,
                   armas_x_dense_t *CS, DTYPE tol, int flags, armas_conf_t *conf)
{
    int N, work, maxit, i, n, k, nrot, ip, iq, zero, saves, abstol, forwards;
    int ipold, iqold;
    DTYPE e0, e1, d0, d1, dp, f0, g0, r, ushift, slow, shigh, threshold, mu;
    armas_x_dense_t sD, sE, Cr, Sr, Cl, Sl;

    EMPTY(sD); EMPTY(sE);
    forwards = 1; zero = 0;

    N = armas_x_size(D);
    maxit = 6*N*N;
    saves = 0;
    abstol = conf->optflags & ARMAS_OABSTOL;
    
    if (U || V) {
        armas_x_subvector(&Cr, CS, 0, N);
        armas_x_subvector(&Sr, CS, N, N);
        armas_x_subvector(&Cl, CS, 2*N, N);
        armas_x_subvector(&Sl, CS, 3*N, N);
        saves = 1;
    }

    estimate_sval(&slow, &shigh, D, E);
    if (conf->optflags & ARMAS_OABSTOL) {
        threshold = maxit*__SAFEMIN;
        if (tol*shigh < threshold)
            threshold = tol*shigh;
    } else {
        threshold = tol*slow;
        if (maxit*__SAFEMIN < threshold)
            threshold = maxit*__SAFEMIN;
    }

    // iq points past the last entry i.e. last entry on index iq-1
    // First divide B to blocks B0, B1, B2 where B2 has all off diagonal
    // entries less than treshold value. B1 has all off-diagonal entries
    // greater than threshold value. 
    ip = 0; iq = N; ipold = ip; iqold = iq;
    for (work = 1, n = 0; work && maxit > 0 && iq > 0; maxit--, n++) {
        // from (1);
        // a. find largest iq such that |E(iq)|...|E(N-2)| < threshold
        //    and if none then iq = N
        // b. find smallest ip < iq such that |E(iq)| < threshold or if none
        //    then ip = 0
        // In effect iq is index to start of B2, ip is index to start of B1.
        // We search from bottom of the matrix to top and zero values below threshold
        // on the way upwards. And update estimates of singular values.
        d0 = __ABS(armas_x_get_at_unsafe(D, iq-1));
        if (abstol && d0 < threshold)
            armas_x_set_at_unsafe(D, iq-1, __ZERO);
        shigh = __ABS(armas_x_get_at_unsafe(D, iq-1));
        slow  = shigh; 
        // proceed from bottom to top; 
        for (k = iq-1; k > 0; k--) {
            d0 = __ABS(armas_x_get_at_unsafe(D, k-1));
            e0 = __ABS(armas_x_get_at_unsafe(E, k-1));
            if (e0 < threshold) {
                armas_x_set_at_unsafe(E, k-1, __ZERO);
                if (k == (iq - 1)) {
                    // convergence of bottom singular value
                    iq = iq - 1;
                    // work until we hit bottom at index 0
                    work = k > 0; 
                    goto Next;
                }
                // E[k] == 0.0; E[k+1] ... E[iq-2] != 0
                ip = k;
                break;
            }
            ip = k - 1;
            // update estimates
            slow = __MIN(slow, d0);
            shigh = __MAX(__MAX(shigh, d0), e0);
        }

        if (iq <= ip) {
            work = 0;
            continue;
        }

        if ((iq - ip) == 2) {
            // 2x2 block, do separately
            DTYPE smin, smax, cosl, sinl, cosr, sinr;
            d0 = armas_x_get_at_unsafe(D, ip);
            d1 = armas_x_get_at_unsafe(D, ip+1);
            e1 = armas_x_get_at_unsafe(E, ip);
            __bdsvd2x2_vec(&smin, &smax, &cosl, &sinl, &cosr, &sinr, d0, e1, d1);
            armas_x_set_at_unsafe(D, ip, smax);
            armas_x_set_at_unsafe(D, ip+1, smin);
            armas_x_set_at_unsafe(E, ip, __ZERO);
            if (U) {
                armas_x_gvright(U, cosl, sinl, ip, ip+1, 0, U->rows);
            }
            if (V) {
                armas_x_gvleft(V, cosr, sinr, ip, ip+1, 0, V->cols);
            }
            iq -= 2;
            goto Next;
        }
        
        if (n == 0 || iq != iqold || ip != ipold) {
            // this first time or when new disjoint block selected
            ipold = ip; iqold = iq;
            d0 = __ABS(armas_x_get_at_unsafe(D, ip));
            d1 = __ABS(armas_x_get_at_unsafe(D, iq-1));
            // select direction
            forwards = d1 >= d0 || (flags & ARMAS_FORWARD) != 0;
        }

        // convergence
        if (forwards) {
            // criterion 1b, 1a, 2a
            e0 = __ABS(armas_x_get_at_unsafe(E, iq-2));
            d0 = __ABS(armas_x_get_at_unsafe(D, iq-1));
            // this is the standard convergence test 
            if ((abstol && e0 < threshold) || e0 < tol*d0) {
                armas_x_set_at_unsafe(E, iq-2, __ZERO);
                iq = iq - 1;
                goto Next;
            }
            // if relative tolerance then criteria 1a.
            if (!abstol) {
                mu = __ABS(armas_x_get_at_unsafe(D, ip));
                for (k = ip; k < iq-1; k++) {
                    e0 = armas_x_get_at_unsafe(E, k);
                    if (__ABS(e0) <= tol*mu) {
                        // test recurrence |e(j)/mu(j)| <= tol
                        armas_x_set_at_unsafe(E, k, __ZERO);
                        goto Next;
                    }
                    d0 = __ABS(armas_x_get_at_unsafe(D, k+1));
                    mu = d0 * (mu / (mu + __ABS(e0)));
                }
            }
        } else {
            // criterion 1a, 1b, 2b
            e0 = __ABS(armas_x_get_at_unsafe(E, ip));
            d0 = __ABS(armas_x_get_at_unsafe(D, ip));
            if ((abstol && e0 < threshold) || e0 < tol*d0) {
                armas_x_set_at_unsafe(E, ip, __ZERO);
                ip = ip + 1;
                goto Next;
            }
            // if relative tolerance then criteria 1b.
            if (!abstol) {
                mu = __ABS(armas_x_get_at_unsafe(D, iq-1));
                for (k = iq-1; k > ip; k--) {
                    e0 = armas_x_get_at_unsafe(E, k-1);
                    if (__ABS(e0) <= tol*mu) {
                        // test recurrence |e(j)/mu(j)| <= tol
                        armas_x_set_at_unsafe(E, k-1, __ZERO);
                        goto Next;
                    }
                    d0 = __ABS(armas_x_get_at_unsafe(D, k-1));
                    mu = d0 * (mu / (mu + __ABS(e0)));
                }
            }
        }
        // compute shift
        if (!abstol && (N*tol*(slow/shigh)) <= max(__EPS, 0.01*tol)) {
            zero = 1;
        } else {
            if (forwards) {
                d0 = __ABS(armas_x_get_at_unsafe(D, ip));
                d1 = armas_x_get_at_unsafe(D, iq-2);
                e1 = armas_x_get_at_unsafe(E, iq-2);
                dp = armas_x_get_at_unsafe(D, iq-1);
                __bdsvd2x2(&ushift, &r, d1, e1, dp);
            } else {
                d0 = __ABS(armas_x_get_at_unsafe(D, iq-1));
                d1 = armas_x_get_at_unsafe(D, ip);
                e1 = armas_x_get_at_unsafe(E, ip);
                dp = armas_x_get_at_unsafe(D, ip+1);
                __bdsvd2x2(&ushift, &r, d1, e1, dp);
            }
            if (d0 > __ZERO)
                zero = (ushift/d0)*(ushift/d0) <= __EPS;
        }

        armas_x_subvector(&sD, D, ip, iq-ip);
        armas_x_subvector(&sE, E, ip, iq-ip-1);
        // QR iteration
        if (forwards) {
            if (zero) {
                // implicit zero shift QR
                nrot = __bd_qrzero(&sD, &sE, &Cr, &Sr, &Cl, &Sl, saves);
            } else {
                // standard shifted QR
                f0 = (__ABS(d0) - ushift) * (copysign(__ONE, d0) + ushift/d0);
                g0 = armas_x_get_at_unsafe(E, ip);
                nrot = __bd_qrsweep(&sD, &sE, &Cr, &Sr, &Cl, &Sl, f0, g0, saves);
            }
            e0 = __ABS(armas_x_get_at_unsafe(E, iq-2));
            if (e0 <= threshold) {
                armas_x_set_at_unsafe(E, iq-2, __ZERO);
            }
            if (U) {
                armas_x_gvupdate(U, ip, &Cl, &Sl, nrot, ARMAS_RIGHT);
            }
            if (V) {
                armas_x_gvupdate(V, ip, &Cr, &Sr, nrot, ARMAS_LEFT);
            }
        } else {
            if (zero) {
                // implicit zero shift QL
                nrot = __bd_qlzero(&sD, &sE, &Cr, &Sr, &Cl, &Sl, saves);
            } else {
                // standard shifted QL
                f0 = (__ABS(d0) - ushift) * (copysign(__ONE, d0) + ushift/d0);
                g0 = armas_x_get_at_unsafe(E, iq-2);
                nrot = __bd_qlsweep(&sD, &sE, &Cr, &Sr, &Cl, &Sl, f0, g0, saves);
            }
            e0 = __ABS(armas_x_get_at_unsafe(E, ip));
            if (e0 <= threshold) {
                armas_x_set_at_unsafe(E, ip, __ZERO);
            }
            if (U) {
                armas_x_gvupdate(U, ip, &Cr, &Sr, nrot, ARMAS_RIGHT|ARMAS_BACKWARD);
            }
            if (V) {
                armas_x_gvupdate(V, ip, &Cl, &Sl, nrot, ARMAS_LEFT|ARMAS_BACKWARD);
            }
        }
    Next:
        // next round of the iteration
        ;
    }

    // ready here....
    if (maxit > 0) {
        // finished properly 
        for (i = 0; i < N; i++) {
            d0 = armas_x_get_at(D, i);
            if (d0 < 0) {
                armas_x_set_at(D, i, -d0);
                if (V) {
                    armas_x_row(&sD, V, i);
                    armas_x_scale(&sD, -1.0, conf);
                }
            }
        }
        return 0;
    }
    // error return  here, no convergence
    return -1;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

