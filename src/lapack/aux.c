
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

// Function here are pretty much copies of corresponding function in NETLIB
// LAPACK library. They are here to acknowledge this fact.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__bdsvd2x2) && defined(__bdsvd2x2_vec)
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

/*
 * NOTES:
 *
 * Singular Value Decomposition is similiarity transformation 
 *
 *    Q.T*A*P = S
 *
 *  where diag(S)^2 = E are the non-zero latent roots (eigenvalues) of A.T*A.
 *
 *  SVD of bidiagonal 2x2 matrix B is then computed by finding eigenvalues of
 *  matrix B.T*B and taking square roots of them.
 *
 *  The characteristic function is det(B.T*B - Ix) = x^2 - T*x + D
 *  where
 *      T = trace(B.T*B) = f^2 + h^2 + g^2 >= 0
 *      D = det(B.T*B)   = f^2*h^2 >= 0
 *
 *  The eigenvalues of B.T*B are then
 *      e1 = C/2; C = T + sqrt(T^2 - 4*D)
 *      e2 = 2*D/C
 *
 *  and the singular values of B are s1 = sqrt(e1), s2 = sqrt(e2)
 *
 *      C = (f^2 + h^2 + g^2) + sqrt((f^2 + h^2 + g^2)^2 - 4*f^2*h^2)
 *
 *  firstly we can rewrite:
 *      sqrt((f^2 + h^2 + g^2)^2 - 4*f^2*h^2) ==
 *      sqrt((g^2 + (|f| + |h|)^2)*(g^2 + (|f| - |h|)^2))
 *
 *  secondly:
 *      (f^2 + h^2 + g^2) ==
 *      (1/2)*[(g^2 + (|f| + |h|)^2) + (g^2 + (|f| - |h|)^2)]
 *
 *  therefore:
 *      2*C = [ sqrt(g^2 + (|f|+|h|)^2) + sqrt(g^2 + (|f|-|h|)^2) ]^2 
 *  and
 *      s1 = sqrt(C/2) =   sqrt(g^2 + (|f|+|h|)^2) + sqrt(g^2 + (|f|-|h|)^2)
 *      s2 = sqrt(2*D/C) = |h||f|/sqrt(g^2 + (|f|+|h|)^2) + sqrt(g^2 + (|f|-|h|)^2)
 *
 *  In order to avoid overflow/underflow calculations need to be done
 *  with scaled variables.
 *
 *  1. fhmax = MAX(|f|, |h|); fhmin = MIN(|f|, |h|) and |g| < fhmax and set
 *     gs = |g|/fhmax, t = 1 + fhmin/fhmax and d = (fhmax - fhmin)/fhmax
 *
 *     C  = sqrt(gs^2 + t^2) + sqrt(gs^2 + d^2)
 *     s1 = 2*(fhmin/C), s2 = fhmax*(C/2)
 *
 *  2. fhmax = MAX(|f|, |h|); fhmin = MIN(|f|, |h|) and |g| >= fhmax and set
 *     gs = fhmax/|g|, t = 1 + fhmin/fhmax and d = (fhmax - fhmin)/fhmax
 *
 *     C  = sqrt(1.0 + (gs*t)^2) + sqrt(1.0 + (gs*d)^2)
 *     s1 = 2*(fhmin/C)*gs,  s2 = fhmax*C/(2*gs),
 *
 *  Above is how singular values of 2x2 bidiagonal matrix are computed in
 *  LAPACK function DLAS2.  Bsvd2x2() implements above computations and is therefore
 *  mostly copy of DLAS2.
 */

/*
 * \brief Compute SVD of 2x2 bidiagonal matrix
 *
 * \param smin [out]
 *      Smaller of the computed singular values
 * \param smax [out]
 *      Larger of the computed singular values
 * \param f, g, h [in]
 *      Bidiagonal matrix entries
 * \return
 *      Value of the smaller singular value.
 */
DTYPE __bdsvd2x2(DTYPE *smin, DTYPE *smax, DTYPE f, DTYPE g, DTYPE h)
{
    DTYPE C, fa, ga, ha, fhmax, fhmin, gs, d, t;

    fa = __ABS(f);
    ha = __ABS(h);
    ga = __ABS(g);
    if (fa > ha) {
        fhmax = fa;
        fhmin = ha;
    } else {
        fhmax = ha;
        fhmin = fa;
    }

    if (fhmin == __ZERO) {
        *smin = __ZERO;
        if (fhmax == __ZERO) {
            *smax = ga;
        } else {
            if (fhmax > ga) {
                fhmin = ga;
            } else {
                fhmin = fhmax;
                fhmax = ga;
            }
            *smax = fhmax * __SQRT(1.0 + (fhmin/fhmax)*(fhmin/fhmax));
        }
        return *smin;
    }

    t  = 1.0 + fhmin/fhmax;
    d  = (fhmax - fhmin)/fhmax;
    if (ga < fhmax) {
        gs = ga/fhmax;
        C = (__SQRT(gs*gs + t*t) + __SQRT(gs*gs + d*d))/2.0;
    } else {
        gs = fhmax / ga;
        if (gs == __ZERO) {
            *smin = (fhmin * fhmax) / ga;
            *smax = ga;
            return *smin;
        } 
        C = __SQRT(1.0 + (gs*t)*(gs*t)) + __SQRT(1.0 + (gs*d)*(gs*d));
        C /= 2.0*gs;
    }
    *smin = fhmin / C;
    *smax = fhmax * C;
    return *smin;
}

static inline DTYPE sign(DTYPE a, DTYPE b)
{
    return signbit(b) ? -a : a;
}

#ifndef __TWO
#define __TWO 2.0
#endif

/*
 * \brief Compute singular values and vectors of bidiagonal 2x2 matrix.
 *
 * Computes singular values and vectors such that following holds.
 *
 *    (  cosl sinl ) ( f  g ) ( cosr -sinr ) = ( ssmax   0   )
 *    ( -sinl cosl ) ( 0  h ) ( sinr  cosr )   (   0   ssmin )
 *
 * \param[out] ssmin
 *      On return abs(ssmin) is the smaller singular value
 * \param[out] ssmax
 *      On return abs(ssmax) is the larger singular value
 * \param[out] cosl, sinl
 *      The vector (cosl, sinl) is the unit left singular vector for abs(ssmax).
 * \param[out] cosr, sinr
 *      The vector (cosr, sinr) is the unit right singular vector for abs(ssmin).
 * \param[in] f, g, h
 *      Elements of the upper bidiagonal matrix.
 *
 * Compatible with LAPACK xLASV2.
 */
void __bdsvd2x2_vec(DTYPE *ssmin, DTYPE *ssmax,
                    DTYPE *cosl, DTYPE *sinl, DTYPE *cosr, DTYPE *sinr,
                    DTYPE f, DTYPE g, DTYPE h)
{
    DTYPE smin, smax, clt, slt, crt, srt;
    DTYPE amax, amin, fhmax, fhmin;
    DTYPE gt, ga, fa, ha, d, t, l, m, t2, m2, s, r, a, tsign;
    int swap = 0, gmax = 0;

    fa = __ABS(f); ha = __ABS(h); ga = __ABS(g);
    if (fa > ha) {
        fhmax = f; amax = fa;
        fhmin = h; amin = ha;
    } else {
        fhmax = h; amax = ha;
        fhmin = f; amin = fa;
        swap = 1;
    }
    gt = g;

    if (ga == __ZERO) {
        smin = amin;
        smax = amax;
        clt = __ONE;
        crt = __ONE;
        slt = __ZERO;
        srt = __ZERO;
        goto signs;
    }

    if (ga > amax) {
        gmax = 1;
        if (amax/ga < __EPS) {
            // very large ga
            smax = ga;
            if (amin > 1.0) {
                smin  =  amax / ( ga / amin );
            } else {
                smin = (amax / ga)*amin;
            }
            clt = __ONE;
            slt = fhmin / gt;  
            crt = fhmax / gt;
            srt = __ONE;
            goto signs;
        }
    }
    // normal case here
    d = amax  - amin;
    if (d == amax) {
        // infinite F or H
        l = __ONE;
    } else {
        l = d / amax;   // l = (amax - amin)/amax;
    }
    m = gt/fhmax;
    t = __TWO - l;  // t = 1 + (amax/amin)

    m2 = m*m;
    t2 = t*t;
    s  = __SQRT(t2 + m2);
    r  = l == __ZERO ? fabs(m) : __SQRT(l*l + m2);

    a  = 0.5*(s + r);
    smin = amin / a;
    smax = amax * a;
    // upto here this same as function bsvd2x2(). Code below
    // is not understood :(

    if (m2 == __ZERO) {
        if (l == __ZERO) {
            t = sign(2.0, fhmax)*sign(1.0, gt); // 2.0 or -2.0
        } else {
            t = gt / sign(d, fhmax) + m/t;
        }
    } else {
        t = (m / (s+t) + m /(r+l)) * (1.0 + a);
    }
    l = sqrt(t*t + 4);
    crt = 2.0/l;
    srt = t / l;
    clt = (crt + srt*m) / a;
    slt = (fhmin / fhmax) * srt / a;
    
 signs:
    if (swap) {
        *cosr = slt;
        *sinr = clt;
        *cosl = srt;
        *sinl = crt;
        tsign = sign(__ONE, *cosr)*sign(__ONE, *cosl)*sign(__ONE, f);
    } else {
        *cosr = crt;
        *sinr = srt;
        *cosl = clt;
        *sinl = slt;
        tsign = sign(__ONE, *sinr)*sign(__ONE, *sinl)*sign(__ONE, h);
    }
    if (gmax)
        tsign = sign(__ONE, *sinr)*sign(__ONE, *cosl)*sign(__ONE, g);
        
    // correct signs of smin & smax
    *ssmax = sign(smax, tsign);
    *ssmin = sign(smin, tsign*sign(__ONE,f)*sign(__ONE,h));
}



/*
 * Eigenvalues of 2x2 symmetric matrix
 *
 *  ( a  b ) are roots of characteristic polynomial x^2 - T*x + D = 0
 *  ( b  c ) where T = a + c, D = a*c - b*b
 *
 *  z1 = T + sign(T)*sqrt(T^2 - 4D)/2, z2 = 2*D/(T + sign(T)*sqrt(T^2 - 4D))
 *
 *  sqrt(T^2 - 4*D)  == sqrt((a + c)^2 - 4*(a*c - b*b)) == sqrt(a^2 + 2ac + c^2 - 4ac + 4b^2)
 *  == sqrt(a^2 - 2ac + c^2 + 4b^2) == sqrt((a - c)^2 + (2b)^2) == HYPOT(|a-c|,|2b|)
 *
 * computing HYPOT(|a-c|, |2b|):
 *  |a - c| > |2b|:
 *    |a - c|*sqrt(1.0 + (|2b|/|a - c|)^2)
 *  |a - c| < |2b|
 *    |2b|*sqrt((|a - c|/|2b|)^2 + 1.0)
 *  |a - c| = |2b|
 *    |2b|*sqrt(2.0)
 *
 * Zt = T + copysign(hypot(|a-c|, |2b|), T)
 * z1 = Zt/2
 * z2 = 2*(a*c - b*b)/Zt
 */
void __sym_eigen2x2(DTYPE *z1, DTYPE *z2, DTYPE a, DTYPE b, DTYPE c)
{
    DTYPE T, b2a, amca, Zt, acmax, acmin;

    acmax = a; acmin = c;
    if (__ABS(c) > __ABS(a)) {
        acmax = c; acmin = a;
    }
    b2a = __ABS(b + b);
    T = a + c;
    amca = __ABS(a - c);
    Zt = T + __COPYSIGN(__HYPOT(amca, b2a), T);
    *z1 = __HALF*Zt;
    *z2 = __TWO*((acmax/Zt) * acmin - (b/Zt)*b);
}


/*
 * Eigenvector corresponding eigenvalue z1 for symmetric 2x2 matrix.
 *
 *  ( a - z1    b    ) ( x0 ) = ( 0 )
 *  (   b     c - z1 ) ( x1 )   ( 0 )
 *
 * gives
 *  (a - z1)*x0 + b*x1 = 0  => x1 = (z1 - a)/b * x0
 *  b*x0 + (c - z1)*x1 = 0  => x1 = b/(z1 - c) * x0
 *
 *     ht =  hypot(1, (z1-a)/b)
 *  => x0 = 1/ht
 *     x1 = (z1 - a)/(ht*b)
 */
void __sym_eigen2x2vec(DTYPE *z1, DTYPE *z2, DTYPE *cs, DTYPE *sn, DTYPE a, DTYPE b, DTYPE c)
{
    DTYPE T, b2a, amca, Zt, acmax, acmin, ht;

    acmax = a; acmin = c;
    if (__ABS(c) > __ABS(a)) {
        acmax = c; acmin = a;
    }
    b2a = __ABS(b + b);
    T = a + c;
    amca = __ABS(a - c);
    Zt = T + __COPYSIGN(__HYPOT(amca, b2a), T);
    *z1 = __HALF*Zt;
    *z2 = __TWO*((acmax/Zt) * acmin - (b/Zt)*b);
        
    // these should be thought out for cancellation effects....
    ht = __HYPOT(1.0, (*z1 - a)/b);
    *cs = 1.0/ht;
    *sn = (*z1 - a)/(b*ht);
}



#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

