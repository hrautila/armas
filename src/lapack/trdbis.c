
// Copyright (c) Harri Rautila, 2016-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Eigenvalues of symmetric diagonal matrix by bisection

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdbisect)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_blas)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include <math.h>
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
//! \endcond

/*
 * Reference:
 * (1) Demmel, Dhillon, Ben
 *     On the Correctnes of Parallel Bisection in Floating Point
 *     UCB//CSD-94-805, 1994
 */

/*
 * From (1) section 3.
 *
 * subroutine   Ser_Bisec( n, T, left, right, n_left, n_right, tau) 
 *   --computes the eigenvalues of T in the interval [ left; right ) 
 *   --to the desired accuracy
 *
 *   if ( n_left >= n_right or left > right ) return;
 *   if (FloatingCount( left ) > n_left or FloatingCount( right ) < n_right ) return;
 *
 *    enqueue (left, right, n_left, n_right, I_{n_left}^{n_right}) to Worklist
 *    while ( Worklist is not empty)
 *       dequeue (alpha, beta n_alpha, n_beta ; I{n_beta}{n_alpha}) from Worklist ;
 *       mid = inside (alpha, beta);
 *       if ( beta - alpha < MIN i=n_alpha+1, n_beta {tau_i} ) then
 *           -- Eigenvalue MIN(MAX(( alpha+ beta)/2, alpha), beta) 
 *           -- has multiplicity n_beta-n_alpha
 *       else
 *          n_mid = MIN(MAX(FloatingCount(mid) ; n_alpha) ; n_beta);
 *          if ( n_mid > n_alpha ) then
 *              enqueue (alpha mid; n_alpha, n_mid,  I_{n_alpha}^{n_mid}) to Worklist ;
 *          end if
 *          if ( n_mid < n_beta) then
 *              enqueue (mid, beta, n_mid, n_beta, I_{n_mid}^{n_beta}) to Worklist 
 *          end if
 *       end if
 *     end while
 *  end subroutine
 */


/*
 * Return the number of eigenvalues of a real symmetric tridiagonal matrix (D,E)
 * that are less than x.
 * See (1) 5.3 algorithm 4.
 */
static
int float_count_ieee(armas_x_dense_t * D, armas_x_dense_t * E, DTYPE x)
{
    int k, count = 0;
    DTYPE d = ONE, bim1 = ZERO;

    for (k = 0; k < armas_x_size(D); k++) {
        d = (armas_x_get_at_unsafe(D, k) - x) - bim1 * bim1 / d;
        bim1 = armas_x_get_at_unsafe(E, k);
        count += signbit(d) ? 1 : 0;
    }
    return count;
}

/*
 * Compute Gerschgorin interval (glow,gup), See (1) section 4.3

 *  gl = gl - fudge*tnorm*ulp*n - fudge*two*pivmin
 *  gu = gu + fudge*tnorm*ulp*n + fudge*pivmin
 *  fudge = 2.1; pivmin = min e(i)^2  ; pivmin = pivmin*safemn
 */
static
void compute_gerschgorin(DTYPE * glow, DTYPE * gup, armas_x_dense_t * D,
                           armas_x_dense_t * E)
{
    int k, n = armas_x_size(D);
    DTYPE gl, gu, di, ei, eim1, t, bnorm, pivmin;
    ei = ZERO;
    pivmin = ONE;
    gl = gu = armas_x_get_at_unsafe(D, 0);

    eim1 = ZERO;
    for (k = 0; k < n - 1; k++) {
        di = armas_x_get_at_unsafe(D, k);
        ei = ABS(armas_x_get_at_unsafe(E, k));
        t = ei + eim1;
        if (gl > di - t)
            gl = di - t;
        if (gu < di + t)
            gu = di + t;
        eim1 = ei;
        if (ei * ei < pivmin)
            pivmin = ei * ei;
    }
    pivmin = SAFEMIN * pivmin;
    di = armas_x_get_at_unsafe(D, n - 1);
    if (gl > di - ei)
        gl = di - ei;
    if (gu < di + ei)
        gu = di + ei;

    // widen the interval to ensure that all eigenvalues are included.
    bnorm = ABS(gl) > ABS(gu) ? ABS(gl) : ABS(gu);
    *glow = gl - bnorm * (n * EPS) * 2.0 - 4.2 * pivmin;
    *gup = gu + bnorm * (n * EPS) * 2.0 + 2.1 * pivmin;

}


/*
 * \brief Compute eigenvalues by serial bisection as described in (1)
 *
 * \param D
 *      Diagonal of tridiagonal matrix T
 * \param E
 *      Subdiagonal of tridiagonal matrix T 
 * \param left
 *      Left value of eigenvalue interval
 * \param right
 *      Right value of eigenvalue interval
 * \param nleft
 *      Value of count(left)
 * \param nright
 *      value of count(right)
 * \param tau
 *      requested precision; must be < max|T_i|*eps
 *
 * \retval eigenvalue or NaN
 */
static
DTYPE trd_bisect_one(armas_x_dense_t * D,
                       armas_x_dense_t * E,
                       DTYPE left,
                       DTYPE right, int nleft, int nright, DTYPE tau)
{
    int nl, nr, nmid, working = 1;
    DTYPE mid, eigen;

    if (nleft >= nright || left > right) {
        return NAN;
    }

    nl = float_count_ieee(D, E, left);
    nr = float_count_ieee(D, E, right);
    if (nl > nleft || nr < nright) {
        return NAN;
    }

    for (int count = 0; working; count++) {
        // inside(l, r) (see Assumption 3 in (1))
        mid = (right + left) / 2.0;
        if (right - left < tau) {
            eigen = MIN(MAX(mid, left), right);
            printf("  eigen: %e [%d iterations]\n", eigen, count);
            working = 0;
        } else {
            nmid = float_count_ieee(D, E, mid);
            nmid = IMIN(IMAX(nmid, nleft), nright);
            if (nmid > nleft) {
                right = mid;
                nright = nmid;
            }
            if (nmid < nright) {
                left = mid;
                nleft = nmid;
            }
        }
    }
    return eigen;
}


/**
 * \brief Compute selected eigenvalues of symmetric tridiagonal matrix T
 *
 * Computes all or selected eigenvalues of symmetric tridiagonal matrix to desired accuracy
 * by bisection algorightm. Eigenvalue selection by half-open index or value range. Use 
 * macros ARMAS_EIGEN_INT to define index range [left, right) and macro ARMAS_EIGEN_VAL
 * to define value range [low, high). 
 *
 * \param[out] Y
 *      Computed eigenvalues sorted to increasing order.
 * \param[in] D
 *      Diagonal elements of tridiagonal matrix T
 * \param[in] E
 *      Off-diagonal elements of matrix T
 * \param[in] params
 *      Eigenvalue selection parameters. If null pointer then all eigenvalues are computed.
 * \param[in] conf
 *      Configuration block
 *
 * \retval  0  OK
 * \retval <0  Failed to store all requested eigenvalues to result vector
 */
int armas_x_trdbisect(armas_x_dense_t * Y,
                      armas_x_dense_t * D,
                      armas_x_dense_t * E,
                      const armas_x_eigen_parameter_t * params,
                      armas_conf_t * conf)
{
    //armas_x_dense_t Xrow;
    DTYPE gleft, gright, bnorm, tau, eigen;
    int nleft, nright;
    int first, last;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(D) == 0 || armas_x_size(E) == 0)
        return 0;

    compute_gerschgorin(&gleft, &gright, D, E);
    bnorm = MAX(ABS(gleft), ABS(gright));
    tau = params ? params->tau : 0.0;
    if (tau < bnorm * EPS)
        tau = bnorm * EPS * 2.1;

    if (params && params->ileft != -1 && params->iright != -1) {
        if (params->ileft == 0 && params->iright == 0) {
            // index parameters are zero; eigenvalue range defined.
            gleft = MAX(gleft, params->left);
            gright = MIN(gright, params->right);
            nleft = float_count_ieee(D, E, gleft);
            nright = IMAX(0, float_count_ieee(D, E, gright));
        } else {
            // index range
            nleft = IMAX(0, params->ileft);
            nright = IMIN(params->iright, armas_x_size(D));
        }
        first = nleft;
        last = nright;
    } else {
        // all eigenvalues
        first = 0;
        last = armas_x_size(D);
    }

    for (int k = first; k < last; k++) {
        eigen = trd_bisect_one(D, E, gleft, gright, k, k + 1, tau);
        if (isnan(eigen)) {
            return -1;
        }
        armas_x_set_unsafe(Y, k, 0, eigen);
        gleft = (eigen > ZERO ? (ONE - EPS) : (ONE + EPS)) * gleft;
    }
    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
