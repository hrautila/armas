
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_dqds) && defined(__armas_scale_to)
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

// References
// (1) Parlett, Marquies
//     An Implementation of the dqds Algorighm (Positive Case),
//     1999 (LAWN #155)
// (2) Ogita, Rump, Oishi
//     Accurate Sum and Dot Product,
//     2005, SIAM Journal on Scientific Computing
// (3) Li,
//     A Modified DQDS Algorithm

#define __THIRD  0.333
#define __QUART  0.25
#define PING 0
#define PONG 1
#define CFLIP 1.5
#define EPS2 (__EPS*__EPS*1e4)
#define __SAFEMAX  (1/__SAFEMIN)

// structure to hold internal state variables
typedef struct dmin_data {
    DTYPE dmin;
    DTYPE dmin1;
    DTYPE dmin2;
    DTYPE dn;
    DTYPE dn1;
    DTYPE dn2;
    DTYPE emin;         // last emin
    DTYPE qmax;
    DTYPE g;            // saved coefficient state
    DTYPE tau;          // last tau
    DTYPE cterm;        // Summation correction term
    int   imin;         // index of dmin
    int   ttype;        // shift calculation type
    int   niter;
    int   nfail;
} dmin_data_t;

/*
 * \brief Error-free transformation x + y = a + b and x = fl(a+b)
 *
 * From (2) algorithm 3.1, originally by Knuth in TAOCP vol #2.
 */
static inline
void twosum(DTYPE *x, DTYPE *y, DTYPE a, DTYPE b)
{
    // volatile locals to deny optimizing out
    volatile DTYPE z, y0;  
    *x = a + b;
    z  = *x - a;
    y0 = (a - (*x - z));
    *y = y0 + (b - z);
}


static
int __dqds_sweep(__armas_dense_t *dq, __armas_dense_t *de, 
                 __armas_dense_t *sq, __armas_dense_t *se, 
                 int N, DTYPE tau, dmin_data_t *dm)
{
    int k;
    register DTYPE t, d, qk, qk1, ek, thresh, emin;

    if (N <= 1) {
        return 0;
    }

    d    = __armas_get_at_unsafe(sq, 0) - tau;
    emin = __armas_get_at_unsafe(se, 0);
    dm->dmin = d; dm->dmin1 = d; dm->dmin2 = d;
    dm->dn2  = d; dm->dn1   = d; dm->dn    = d;
    for (k = 0; k < N-3; k++) {
        qk = d + __armas_get_at_unsafe(se, k);
        t  = __armas_get_at_unsafe(sq, k+1)/qk;
        ek = __armas_get_at_unsafe(se, k) * t;
        d  = d*t - tau;

        if (ek < emin) 
            emin = ek;
        if (d < dm->dmin) {
            dm->dmin  = d;
            dm->imin  = k + 1;
        }
        __armas_set_at_unsafe(dq, k, qk);
        __armas_set_at_unsafe(de, k, ek);
    }
    // unroll last steps and compute value of d as algorithm dqds(2) in (1) section 2
    // Within the loop above d = d*(qk1/qk) - tau, in unrolled steps d = qk1*(d/qk) - tau.
    // Here k = N-3 || zero
    switch (N-k) {
    case 3:
        dm->dn2 = d;
        dm->dmin2 = dm->dmin;
        qk  = d + __armas_get_at_unsafe(se, k);
        qk1 = __armas_get_at_unsafe(sq, k+1);
        ek  = __armas_get_at_unsafe(se, k)*(qk1/qk);
#if 0
        // LAPACK dlasq5 does not check emin values for unrolled tail elements
        // this or next
        if (ek < emin) 
            emin = ek;
#endif
        dm->dn1 = d = qk1*(dm->dn2/qk) - tau;
        if (dm->dn1 < dm->dmin) {
            dm->dmin = dm->dn1;
            dm->imin = k+1;
        }
        __armas_set_at_unsafe(dq, k, qk);
        __armas_set_at_unsafe(de, k, ek);
        k++;
        // fall through
    case 2:
        // k = N-2;
        dm->dmin1 = dm->dmin;
        qk = d + __armas_get_at_unsafe(se, k);
        qk1 = __armas_get_at_unsafe(sq, k+1);
        ek  = __armas_get_at_unsafe(se, k)*(qk1/qk);
#if 0
        // enabling this affects segment split checks, last entries are
        // small and may cause tail splitting for a segment. 
        if (ek < emin) 
            emin = ek;
#endif
        dm->dn = d = qk1*(dm->dn1/qk) - tau;
        if (dm->dn < dm->dmin) {
            dm->dmin = dm->dn;
            dm->imin = k+1;
        }
        __armas_set_at_unsafe(dq, k, qk);
        __armas_set_at_unsafe(de, k, ek);
        // fall through
    case 1:
        // k = N-1
        __armas_set_at_unsafe(dq, N-1, dm->dn);
        break;
    default:
        break;
    }
    dm->emin = emin;
    return dm->imin;
}


/*
 * \brief Compute sum of products as described in (1) section 7.
 */
static inline
DTYPE sum_of_prod(__armas_dense_t *q, __armas_dense_t *e, int N, DTYPE start)
{
    register DTYPE sum, prod, oldprod, qk, ek;
    int k;
    sum  = start;
    oldprod = prod = __ONE;
    for (k = N-2; k >= 0 && 100.0*__MAX(oldprod, prod) >= sum; k--) {
        oldprod = prod;
        qk = __armas_get_at_unsafe(q, k);
        ek = __armas_get_at_unsafe(e, k);
        prod *= ek/qk;
        sum += prod;
    }
    // increase by 5% to compensate for truncated loop
    return 1.05*sum;
}

/*
 * \brief Compute sum of products with upper bound as described in (1) section 7.
 */
static inline
int sum_of_prod_bounded(DTYPE *res, __armas_dense_t *q, __armas_dense_t *e, int N, DTYPE start, DTYPE bound)
{
    register DTYPE sum, prod, oldprod, qk, ek;
    int k, err = 0;
    sum  = start;
    oldprod = prod = __ONE;
    for (k = N-2; k >= 0 && 100.0*__MAX(oldprod, prod) > sum && bound > sum; k--) {
        oldprod = prod;
        qk = __armas_get_at_unsafe(q, k);
        ek = __armas_get_at_unsafe(e, k);
        // LAPACK code in dlasq4 breaks out from the corresponding loop and returns
        // directly without changing last tau value if ek > qk. Intentional?
        // We return error indication.
        if (ek > qk && err == 0)
            err = -k;
        prod *= ek/qk;
        sum += prod;
    }
    // increase by 5% to compensate for truncated loop
    *res = 1.05*sum;
    return err;
}


/*
 * \brief Compute new shift;
 */
static
void __dqds_shift(DTYPE *tau, __armas_dense_t *q, __armas_dense_t *e,
                  __armas_dense_t *q0, __armas_dense_t *e0, int N, int neigen, dmin_data_t *dm)
{
    DTYPE qn, qn1, qn2, en1, en2, en3, an, an1, bn1, bn2;
    DTYPE gap1, gap2, x1, x2, stau, gamma, phi, rho;
    int ie;

    // restore old shift; case 4a, 4b may return without changing tau 
    *tau = dm->tau;
    stau = __ZERO;

    if (dm->dmin <= __ZERO) {
        //printf("..[shift] case 0: dmin [%e] < 0\n", dm->dmin);
        dm->ttype = -1;
        *tau = - dm->dmin;
        return; //-dm->dmin;
    }

    //N -= neigen;
    // see (1) end of sec 6.3.2; N must be at least 3 here
    qn = an = dm->dn;
    qn1 = __armas_get_at_unsafe(q, N-2);
    en1 = __armas_get_at_unsafe(e, N-2);
    en2 = __armas_get_at_unsafe(e, N-3);
    an1 = qn1 + en1;
    bn1 = __SQRT(qn)*__SQRT(en1);

    switch (neigen) {
    case 0:
        // see (1) sec 6.3.3
        stau = 0.25*dm->dmin;
        bn2 = __SQRT(qn1)*__SQRT(en2);

        if (dm->dmin == dm->dn && dm->dmin1 == dm->dn1) {
            gap2 = dm->dmin2 - an1 - 0.25*dm->dmin2; // this trick is from dlasq4
            if (gap2 > __ZERO && gap2 > bn2) {
                gap1 = an1 - bn2*(bn2/gap2) - an;
            } else {
                gap1 = an1 - an - (bn1 + bn2);
            }
            if (gap1 > __ZERO && gap1 > bn1) {
                //printf("..[shift] case 2: dmin == dn && dmin1 == dn1\n");
                // case 2
                x1 = qn - bn1*(bn1/gap1);
                stau = __MAX(x1, 0.5*qn);
                dm->ttype = -2;
            } else {
                //printf("..[shift] case 3: dmin == dn && dmin1 == dn1\n");
                // case 3
                //printf("..[shift] 3: qn=%e, bn1=%e, qn1=%e, en1=%e, bn2=%e\n", qn, bn1, qn1, en1, bn2);
                x1 = __MAX(__ZERO, dm->dn - bn1);
                x2 = __MAX(__ZERO, an1 - (bn1 + bn2));
                stau = __MAX(dm->dmin/3.0, __MIN(x1, x2));
                dm->ttype = -3;
            }
            //printf("..[shift] tau=%e\n", stau);
            break;
        } else if (dm->dmin == dm->dn && dm->dmin1 != dm->dn1) {
            // case 4(a)
            //printf("..[shift] case 4(a):\n");
            dm->ttype = -40;
            gamma = dm->dmin;
            if (en1 > qn) {
                //printf("..[shift] case 4a: en1 > qn ...\n");
            }
            // negative value returned if any e(k) > q(k)
            ie = sum_of_prod_bounded(&phi, q, e, N, __ZERO, 9.0/16.0);
            if (ie < 0) {
                //printf("..[shift] case 4a, e > q at %d\n", -ie);
            }
            if (phi < 9.0/16.0) {
                stau = gamma*(( 1.0 - __SQRT(phi))/(1.0+phi));
            } 
            //printf("..[shift] qe & se:\n");
            break;
        } else if (dm->dmin != dm->dn && dm->dmin1 == dm->dn1) {
            // case 4(b); twisted factorization; twist at N-2
            // printf("..[shift] case 4(b): imin = %d\n", dm->imin);
            dm->ttype = -41;
            en1 = __armas_get_at_unsafe(e0, N-2);  // old e(n-1)
            qn  = __armas_get_at_unsafe(q0, N-1);  // old q(n)
            gamma = dm->dn1;
            if (en1 > qn) {
                //printf("..[shift] case 4(b): en1 > qn  [%e > %e]\n", en1, qn);
            }
            // negative value returned if any e(k) > q(k)
            ie = sum_of_prod_bounded(&phi, q, e, N-1, en1/qn, 9.0/16.0);
            if (ie < 0) {
                //printf("..[shift] case 4b, e > q at %d\n", -ie);
            }
            if (phi < 9.0/16.0) {
                stau = gamma*(( 1.0 - __SQRT(phi))/(1.0+phi));
            } 
            //printf("..[shift] qe & se:\n");
            break;
        }

        if (dm->dmin == dm->dn2) {
            // case 5; twisted factorization; twist at N-3
            //printf("..[shift] case 5: imin = %d\n", dm->imin);
            stau = 0.25*dm->dmin;
            qn  = __armas_get_at_unsafe(q0, N-1);  // old q(n)
            qn1 = __armas_get_at_unsafe(q0, N-2);  // old q(n-1)
            en1 = __armas_get_at_unsafe(e0, N-2);  // old e(n-1)
            en2 = __armas_get_at_unsafe(e0, N-3);  // old e(n-2)
            gamma = dm->dn2;
            x1 = (en2/qn1)*(1.0 + en1/qn);
            ie = sum_of_prod_bounded(&phi, q, e, N-2, x1, 9.0/16.0);
            if (ie < 0) {
                //printf("..[shift] case 5, e > q at %d\n", -ie);
            }
            if (phi < 9.0/16.0) {
                stau = gamma*(( 1.0 - __SQRT(phi))/(1.0+phi));
            }
            dm->ttype = -5;
            break;
        }
        // case 6: dmin != dn && dmin != dn1 && dmin != dn2
        // (twisted factorization at k < N-3)
        //printf("..[shift] case 6\n");
        if (dm->ttype == -6) {
            dm->g = dm->g + __THIRD*(1.0 - dm->g);
        } else if (dm->ttype == -18) {
            // tau was too big
            dm->g = __THIRD*0.25;
        } else {
            dm->g = 0.25;
        }
        dm->ttype = -6;
        stau = dm->g*dm->dmin;
        break;

    case 1:
        // 1 eigenvalue deflated; see (1) section 6.3.4
        if (dm->dmin1 == dm->dn1 && dm->dmin2 == dm->dn2) {
            // cases 7, 8
            stau  = dm->dmin1/3.0;
            // from (1) section 7:
            //     ||r||   = rho*sqrt(||v||^2 - 1)  = rho*||x||
            //     ||v||^2 = 1 + ||x||^2
            //     rho     = q(n)/(1 + ||x||^2)
            // ||x||^2 is calculated correct to 1% therefore coefficient 1.01
            x1   = __SQRT(sum_of_prod(q, e, N/*-1*/, __ZERO));
            rho  = dm->dmin1/(1.0 + x1*x1);
            // from section 6.3.4
            gap2 = 0.5*dm->dmin2 - rho;
            // and again from section 7
            if (gap2 > __ZERO && gap2 > rho*x1) {
                // tau = rho - ||r||^2/gap == rho - rho^2*||x||^2/gap
                //printf("..[shift] case 7\n");
                x2 = rho*(1.0 - 1.01*rho*(x1/gap2)*x1);
                stau = __MAX(stau, x2);
                dm->ttype = -7;
            } else {
                //printf("..[shift] case 8\n");
                stau = __MAX(stau, rho*(1.0 - 1.01*x1));
                dm->ttype = -8;
            }
            break;
        }
        // case 9
        //printf("..[shift] case 9\n ");
        if (dm->dmin1 == dm->dn1) {
            stau = 0.5*dm->dmin1;
        } else {
            stau = 0.25*dm->dmin1;
        }
        dm->ttype = -9;
        break;
        
    case 2:
        // 2 eigenvalues deflated; see (1) section 6.3.5
        bn2 = __SQRT(qn1)*__SQRT(en2);
        if (dm->dmin2 == dm->dn2 && 2.0*en1 < qn1) {
            // case 10;
            //printf("..[shift] case 10\n");
            stau  = __THIRD*dm->dmin2;
            x1   = __SQRT(sum_of_prod(q, e, N-1, __ZERO));
            rho  = dm->dmin1/(1.0 + x1*x1);
            gap2 = an1 - bn2 - rho;
            if (gap2 > __ZERO && gap2 > rho*x1) {
                x2 = rho*(1.0 - 1.01*rho*(x1/gap2)*x1);
                stau = __MAX(stau, x2);
            } else {
                stau = __MAX(stau, rho*(1.0 - 1.01*x1));
            }
            dm->ttype = -10;
        } else {
            //printf("..[shift] case 11\n");
            stau = 0.25*dm->dmin2;
            dm->ttype = -11;
        }
        break;

    default:
        // more than 2 deflations; see (1) sectoin 6.3.6
        stau = __ZERO;
        dm->ttype = -12;
        break;
    }

 finish:
    *tau = stau;
    dm->tau = stau;
    return;
}


/*
 * 2x2 case from (1), Section 8
 *
 * \param[out] r1, r2
 *      Roots, r1 is the larger and r2 is the smaller root
 */
static
void __dqds2x2_plain(DTYPE *r1, DTYPE *r2, DTYPE q1, DTYPE e1, DTYPE q2)
{
    DTYPE t, s;
    if (q1 < q2) {
        //printf("..[dqds2x2] q1 < q2, swap!\n");
        t  = q1;
        q1 = q2;
        q2 = t;
    }
    t = 0.5*((q1 - q2) + e1);
    if (e1 > EPS2*q2 && t != __ZERO) {
        s = q2*(e1/t);
        //printf("..[dqds2x2] q1=%13e, q2=%13e, e1=%13e, t=%13e, s=%13e, e1/t=%13e\n", q1, q2, e1, t, s, e1/t);
        if (s <= t) {
            s = q2*e1/(t * (1.0 + __SQRT(1.0 + s/t)));
        } else {
            s = q2*e1/(t + __SQRT(t)*__SQRT(t + s));
        }
        t  = q1 + (s + e1);
        //printf("..[dqds2x2] q1=%13e, q2=%13e, e1=%13e, t=%13e, s=%13e, q1/t=%13e\n", q1, q2, e1, t, s, q1/t);
        q2 = q2*(q1/t);
        q1 = t;
    }
    *r1 = q1;
    *r2 = q2;
}

static
void __dqds2x2(DTYPE *r1, DTYPE *r2, DTYPE q1, DTYPE e1, DTYPE q2, DTYPE sigma)
{
    __dqds2x2_plain(r1, r2, q1, e1, q2);
    *r1 += sigma;
    *r2 += sigma;
}


/*
 * \brief Test if e(n-1)/e(n-2) are neglible
 *
 *  Z[4,N] = q0,  q1,  ..., qn1,  qn     [qrow]  
 *           qq0, qq1, ..., qqn1, qqn    [qqrow]
 *           e0,  e1,  ..., en1,  0      [erow]
 *           ee0, ee1, ..., een1, 0      [eerow]
 */
static
int __dqds_neglible(__armas_dense_t *Z, int N, int ping, DTYPE sigma)
{
    int nnew, k, neglible, qrow, erow, qqrow, eerow;
    DTYPE q0, q1, qn, qn1, qn2, en1, en2, en1p, en2p, r1, r2;

    qrow  = ping;
    qqrow = 1 - ping;
    erow  = 2 + ping;
    eerow = 2 + (1 - ping);

    if (N <= 0)
        return 0;

    // from (1)
    //  e(n-1) neglible if
    //    old.e(n-1) <= eps^2*q(n-1) || e(n-1) <= eps^2*(sigma + q(n))
    //  e(n-2) neglible if
    //    old.e(n-2) <= eps^2*q(n-2) || e(n-2) <= eps^2*(sigma + q(n-1)*q(n)/(q(n)+e(n-1)))

    en1  = __armas_get_unsafe(Z, erow,  N-2); 
    en1p = __armas_get_unsafe(Z, eerow, N-2); 
    qn   = __armas_get_unsafe(Z, qrow,  N-1);
    qn1  = __armas_get_unsafe(Z, qrow,  N-2);   

    // deflate if N == 1 or if last E is neglibe --> 1 eigenvalue 
    if (N == 1 || en1 <= EPS2*(sigma + qn) || en1p <= EPS2*qn1) {
        r1 = __armas_get_unsafe(Z, qqrow,  N-1);
        //printf("..[neglible] deflate, 1 eigenvalue, k=%d, eig=%9.7f (%9.7f)\n", N-1, __SQRT(qn+sigma), __SQRT(r1+sigma));
        __armas_set_unsafe(Z, 0, N-1, qn+sigma);
        return 1;
    }

    if (N > 2) {
        // check if second to last E is neglible; 2 eigenvalues
        en2  = __armas_get_unsafe(Z, erow,  N-3); 
        en2p = __armas_get_unsafe(Z, eerow, N-3); 
        qn2  = __armas_get_unsafe(Z, qrow,  N-3);   
        if (en2 > EPS2*sigma && en2p > EPS2*qn2) {
            return 0;
        }
    }

    // 2 eigenvalues; from (1) sec 6.1
    __dqds2x2_plain(&r1, &r2, qn1, en1, qn);
    __armas_set_unsafe(Z, 0, N-1, qn*(qn1/r1)+sigma);
    __armas_set_unsafe(Z, 0, N-2, r1+sigma);
    //printf("..[neglible] deflate, 2 eigenvalues, k=%d, 1.eig=%9.6f, 2.eig=%9.6f\n",
    //     N-1, __SQRT(qn*(qn1/r1)+sigma), __SQRT(r1+sigma));
    return 2;
}

/*
 * \brief Flip N first entries of q and N-1 of e.
 */
static 
int __dqds_flip(__armas_dense_t *Z, /*__armas_dense_t *e,*/ int N)
{
    DTYPE qt, et;
    int k;
    for (k = 0; k < N/2; k++) {
        qt = __armas_get_unsafe(Z, 0, k);
        __armas_set_unsafe(Z, 0, k, __armas_get_unsafe(Z, 0, N-1-k));
        __armas_set_unsafe(Z, 0, N-1-k, qt);
        qt = __armas_get_unsafe(Z, 1, k);
        __armas_set_unsafe(Z, 1, k, __armas_get_unsafe(Z, 1, N-1-k));
        __armas_set_unsafe(Z, 1, N-1-k, qt);

        if (k != N-2-k) {
            et = __armas_get_unsafe(Z, 2, k);
            __armas_set_unsafe(Z, 2, k, __armas_get_unsafe(Z, 2, N-2-k));
            __armas_set_unsafe(Z, 2, N-2-k, et);
            et = __armas_get_unsafe(Z, 3, k);
            __armas_set_unsafe(Z, 3, k, __armas_get_unsafe(Z, 3, N-2-k));
            __armas_set_unsafe(Z, 3, N-2-k, et);
        }
    }
    // we have flipped q and e, normal dqds_sweep does not include last 2
    // e-values to emin, now these values have been flipped to the top of
    // e-vector, need to do something about emin.
    return N/2;
}


/*
 *  Z = q0,  q1,  ..., qn1,  qn    (N columns, 4 rows)
 *      qq0, qq1, ..., qqn1, qqn
 *      e0,  e1,  ..., en1,  0
 *      ee0, ee1, ..., een1, 0
 *
 * \return number of deflations
 */
static
int __dqds_goodstep(DTYPE *ssum, __armas_dense_t *Z, int N, int pp, dmin_data_t *dmind)
{
    __armas_dense_t sq, dq, se, de;
    int n, k, t, ncnt, idk, nfail, newseg;
    DTYPE x, y, q0, q1, qn, qn1, qn2, en1, en2, r1, r2, sigma, tau, tau0;

    sigma = *ssum;
    newseg = __SIGN(dmind->dmin);
    //printf("..[goodstep] entering ping=%d, N=%d, newseg=%d, sigma=%e, dmin=%e\n",
    //     pp, N, newseg, sigma, dmind->dmin);

    armas_d_row(&sq, Z, pp);
    armas_d_row(&dq, Z, 1 - pp);
    armas_d_row(&se, Z, 2 + pp);
    armas_d_row(&de, Z, 3 - pp);

    dmind->niter++;
    if (N == 1) {
        DTYPE t0;
        t0 = __armas_get_at_unsafe(&sq, 0);
        __armas_set_unsafe(Z, 0, 0, __armas_get_at_unsafe(&sq, 0)+(sigma+dmind->cterm));
        //printf("..[goodstep] deflated single valued vector eig=%9.6f\n", __SQRT(__armas_get_unsafe(Z, 0, 0)));
        return 1;
    }

    // 1. Look for neglible E-values
    ncnt = 0;
    if (! newseg) {
        do {
            n = __dqds_neglible(Z, N-ncnt, pp, sigma+dmind->cterm);
            ncnt += n;
        } while (n != 0 && N-ncnt > 0);
    }
    if (N-ncnt == 0) {
        //printf("..[goodstep] deflated (%d) to zero length\n", ncnt);
        return ncnt;
    }
    // 2 test flipping 1.5*q(0) < q(N-1) if new segment or deflated values
    if (newseg || ncnt > 0) {
        q0 = __armas_get_at_unsafe(&sq, 0);
        q1 = __armas_get_at_unsafe(&sq, N-ncnt-1);
        if (CFLIP*q0 < q1) {
            //printf("..[goodstep] need flipping.. [0, %d]\n", N-ncnt-1);
            __dqds_flip(Z, N-ncnt);
        }
    }
    
    // 3a. if no overflow or no new segment, choose shift
    __dqds_shift(&tau, &sq, &se, &dq, &de, N-ncnt, ncnt, dmind);
    //printf("..[goodstep]: tau=%e [type=%d, dmin=%e,%e, dmin1=%e]\n", tau, t, dmind->ttype, dmind->dmin, dmind->dn, dmind->dmin1);
    
    // 4.  run dqds until dmin > 0
    nfail = 0;
    do {
        idk = __dqds_sweep(&dq, &de, &sq, &se, N-ncnt, tau, dmind);
        if (dmind->dmin < __ZERO) {
            // failure here
            DTYPE en1 = __armas_get_at_unsafe(&de, N-ncnt-2);
            if (dmind->dmin1 > __ZERO && en1 < __EPS*(sigma+dmind->dn1) &&
                __ABS(dmind->dn) < __EPS*sigma) {
                // convergence masked by negative d (section 6.4)
                __armas_set_at_unsafe(&dq, N-ncnt-1, __ZERO);
                dmind->dmin = __ZERO;
                //printf("..[masked] dmin1 > %e, setting qn to zero.!\n", dmind->dmin1);
                // break out from loop
                //break;
            }
    
            nfail++;
            if (nfail > 1) {
                tau = __ZERO;
            } else if (dmind->dmin1 > __ZERO) {
                // late failure
                tau = (tau + dmind->dmin)*(1.0 - 2.0*__EPS);
                dmind->ttype -= 11;
            } else {
                tau *= 0.25;
                dmind->ttype -= 12;
            }
            //printf("..failure[%d]: tau=%e\n", nfail, tau);
            dmind->niter++;
            dmind->nfail++;
        }
    } while (dmind->dmin < __ZERO || dmind->dmin1 < __ZERO);

    // 5. update sigma; sequence of tau values.
    //    use cascaded summation from (2), algorithm 4.1
    //    this here is one step of the algorithm, error term
    //    is summated to dmind->cterm
 update:
    twosum(&x, &y, *ssum, tau);
    //printf("..[goodstep] 2sum x=%13e, y=%13e, a=%13e, b=%13e\n", x, y, *ssum, tau);
    *ssum = x;
    dmind->cterm += y;
    //printf("..[goodstep] 2sum c=%13e, eig=%13e\n", dmind->cterm, __SQRT(x+dmind->cterm));
    
    return ncnt;
}


/*
 * \brief Run DQDS on one segment
 */
static
int __dqds_segment(__armas_dense_t *Z, int N, DTYPE sigma, DTYPE dmin, dmin_data_t *dmind, int level)
{
    __armas_dense_t sZ, Ztail;
    DTYPE emins[2], emax, qmin, qmax, emin, q0, e0, e1, old_dmin, cterm;
    int ip, iq, ping, neigen, maxiter, niter, k, j, taillen;

    //printf("\n** segment %d start [N=%d, sigma=%e, dmin=%e, qmax=%e, emin=%e] **\n",
    //     level, N, sigma, dmind->dmin, dmind->qmax, dmind->emin);

    ping = PING;
    ip = 0; iq = N;

    if (__SIGN(dmin)) {
        dmind->dmin = dmin;
    } else {
        dmind->dmin = -dmin;
    }

    maxiter = 30*(iq - ip);
    qmax = dmind->qmax;
    for (niter = 0; niter < maxiter && iq > ip; niter++) {

        //printf("\n-- segment loop: niter=%d/%d, N=%d %d:%d dmin=%e --\n", niter, maxiter, iq-ip, ip, iq, dmind->dmin);

        __armas_submatrix(&sZ, Z, 0, ip, 4, iq-ip);
        neigen = __dqds_goodstep(&sigma, &sZ, iq-ip, ping, dmind);
        ping = PONG - ping;
        emins[ping] = dmind->emin;
        if (neigen > 0) {
            //printf("..[segment]: goodstep deflated %d values\n", neigen);
            iq -= neigen;
        }

#if 0
        /*
         * This is part of (1) that checks for splitting after one PING-PONG transfomations.
         * Could not make work correctly and it is here just to remind that something could
         * be done.
         */
        if (ping == PING && (iq-ip) > 2) {
            // emins[0] current emin from ZZ->Z (PONG) phase,
            // emins[1] old emin from Z -> ZZ (PING) phase
            if (emins[0] < EPS2*sigma || emins[1] < EPS2*qmax) {
                // check for splits
                qmin = __armas_get_unsafe(&sZ, 0, iq-1);
                emin = __armas_get_unsafe(&sZ, 2, iq-1);
                qmax = qmin;
                emax = emin;
                for (k = iq-1; k > ip; k--) {
                    q0 = __armas_get_unsafe(&sZ, 0, k);
                    e0 = __armas_get_unsafe(&sZ, 2, k);
                    e1 = __armas_get_unsafe(&sZ, 3, k);
                    if (e1 <= EPS2*q0 || e0 <= EPS2*sigma) {
                        // split here
                        __armas_dense_t zZ;
                        __armas_submatrix(&zZ, &sZ, 0, k, sZ.rows, iq-k);
                        dmind->qmax = __MAX(qmax, q0);
                        qmin = __MIN(qmin, q0);
                        emax = __MAX(emax, e0);
                        dmin = - __MAX(__ZERO, qmin - 2.0*__SQRT(qmin)*__SQRT(emax));
                        dmind->emin = emin;
                        cterm = dmind->cterm;
                        neigen = __dqds_segment(&zZ, iq-k, sigma, dmin, dmind, level+1);
                        if (neigen < 0) {
                            return -N;
                        }
                        iq = k;
                        dmind->cterm = cterm;
                        dmind->dmin = - __ZERO;
                        // reset limits
                        qmin = __armas_get_unsafe(&sZ, 0, iq-1);
                        emin = __armas_get_unsafe(&sZ, 2, iq-1);
                        qmax = qmin;
                        emax = emin;
                    } else {
                        qmax = __MAX(qmax, q0);
                        emin = __MIN(emin, e0);
                        qmin = __MIN(qmin, q0);
                        emax = __MAX(emax, e0);
                    }
                }
                q0 = __armas_get_unsafe(&sZ, 0, k);
                qmax = __MAX(qmax, q0);
                //printf("..[segment] continue with N=%d ip=%d, iq=%d, qmax=%e\n", iq-ip, ip, iq, qmax);
            }
        }
#endif
    }
    //printf("\n** segment %d end [N=%d, niter=%d] **\n", level, N, niter);
    return niter == maxiter ? -N : N;
}


/*
 * \brief Run Li's test from (1), section 3.3
 */ 
static
int __dqds_li_test(int ping, __armas_dense_t *Z, int N, DTYPE *emin, DTYPE *qmax)
{
    __armas_dense_t sq, se, dq, de;
    DTYPE qk, qk1, ek, d, t, emn, qmx;
    int k;

    armas_d_row(&sq, Z, ping);
    armas_d_row(&dq, Z, 1 - ping);
    armas_d_row(&se, Z, 2 + ping);
    armas_d_row(&de, Z, 3 - ping);

    d = __armas_get_at_unsafe(&sq, N-1);
    // run Li's reverse test
    for (k = N-1; k > 0; k--) {
        qk = __armas_get_at_unsafe(&sq, k-1);
        ek = __armas_get_at_unsafe(&se, k-1);
        if (ek <= EPS2*d) {
            __armas_set_at_unsafe(&se, k-1, - __ZERO);
            d = qk;
        } else {
            d = qk * (d / (d + ek));
        }
    }
    // map Z -> ZZ and Li's test, update emin
    qmx = __ZERO;
    emn = __armas_get_at_unsafe(&se, 0);
    d   = __armas_get_at_unsafe(&sq, 0);
    for (k = 0; k < N-1; k++) {
        ek  = __armas_get_at_unsafe(&se, k);
        qk1 = __armas_get_at_unsafe(&sq, k+1);
        qk  = d + ek;
        if (ek < EPS2*d) {
            // Li's test eq.5 in (1), page 12
            //printf("..[li] ek=%13e < EPS2*d [%13e, %13e]\n", ek, d, EPS2*d);
            __armas_set_at_unsafe(&se, k, - __ZERO);
            qk = d;
            ek = __ZERO;
            d  = qk1;
#if 0   // this commented out as it seemed to true most of the time, even without any appearent underflow
        } else if (__SAFEMIN*qk1 <= qk && __SAFEMIN*qk <= qk1) {
            // test for underflow from (1) page 7
            //printf("..[li] underflow qk=%13e [%13e], qk1=%13e [%13e]\n", qk, __SAFEMIN*qk, qk1, __SAFEMIN*qk1);
            t  = qk1/qk;
            ek = ek*t;
            d  = d*t;
#endif
        } else {
            // standard dqd (dqds with zero shift)
            ek = qk1*(ek/qk);
            d  = qk1*(d/qk);
        }
        __armas_set_at_unsafe(&dq, k, qk);
        __armas_set_at_unsafe(&de, k, ek);
        emn = __MIN(emn, ek);
        qmx = __MAX(qmx, qk);
    }
    __armas_set_at_unsafe(&dq, N-1, d);
    *emin = emn;
    *qmax = qmx;
}


/*
 * \brief Top left of DQDS algorithm
 */
static
int __dqds_main(__armas_dense_t *Z, int N)
{
    __armas_dense_t z0, z1, sZ, sq, se;
    dmin_data_t dmind;
    DTYPE q0, q1, e0, e1, emin, emax, qmin, qmax, sigma, dmin, cterm;
    int ip, iq, niter, maxiter, ncnt;

    // test flipping 1.5*q(0) < q(N-1)
    q0 = __armas_get_unsafe(Z, 0, 0);
    q1 = __armas_get_unsafe(Z, 0, N-1);
    if (CFLIP*q0 < q1) {
        __dqds_flip(Z, N);
    }

#if 1
    // run Li's test and check for initial splitting;
    __dqds_li_test(PING, Z, N, &emin, &qmax);
    __dqds_li_test(PONG, Z, N, &emin, &qmax);
#endif
    __armas_row(&sq, Z, 0);
    __armas_row(&se, Z, 1);

    // find unreduce block and run dqds on the segment
    ip = 0; iq = N;
    maxiter = 10*N;
    dmind.cterm = __ZERO;
    sigma = __ZERO;

    for (niter = 0; niter < maxiter && iq > 0; niter++) {
        qmin = __armas_get_unsafe(Z, 0, iq-1);
        qmax = qmin;
        emax = __ZERO;
        emin = iq > 1 ? __armas_get_unsafe(Z, 2, iq-2) : __ZERO;

        for (ip = iq-1; ip > 0; ip--) {
            q0 = __armas_get_unsafe(Z, 0, ip-1);
            e0 = __armas_get_unsafe(Z, 2, ip-1);
            if (e0 <= __ZERO) {
                // e0 < 0.0 
                //printf("..[main] e0 <= 0.0 [%d], e0=%e\n", ip, e0);
                break;
            }
            if (qmin >= 4.0*emax) {
                emax = __MAX(emax, e0);
                qmin = __MIN(qmin, q0);
            }
            emin = __MIN(emin, e0);
            qmax = __MAX(qmax, q0+e0);
        }

        dmin = - __MAX(__ZERO, qmin - 2.0*__SQRT(qmin)*__SQRT(emax));
        dmind.dmin = dmin;
        dmind.qmax = qmax;
        dmind.emin = emin;

        // select subblock
        __armas_submatrix(&sZ, Z, 0, ip, Z->rows, iq-ip);
        cterm = dmind.cterm;
        if ((ncnt = __dqds_segment(&sZ, iq-ip, sigma, dmin, &dmind, 0)) < 0) {
            //printf("..[main] subblock [%d,%d] not converged\n", ip, iq);
            return -1;
        } 
        //printf("..[main] new iq=%d, [ip=%d]\n", iq - ncnt, ip);
        iq -= ncnt;
        dmind.cterm = cterm;
    }
    return niter == maxiter ? -1 : 0;
}

/*
 * \brief Scale matrix with from/to avoiding overflows/underflows.
 *
 * \param[in,out] A
 *      Matrix to scale
 * \param[in] from, to
 *      Scaling parameters, multiplied with from/to.
 * \param[in] flags
 *      Matrix type bits ARMAS_LOWER, ARMAS_UPPER, ARMAS_SYMM
 *
 * lapack.xLASCL
 */
int __armas_scale_to(__armas_dense_t *A, DTYPE from, DTYPE to, int flags)
{
    DTYPE cfrom1, cto1, mul;
    int ready = 0;

    if (A->rows == 0 || A->cols == 0)
        return 0;

    do {
        cfrom1 = from*__SAFEMIN;
        if (cfrom1 == from) {
            mul = to/from;
            ready = 1;
            cto1 = to;
        } else {
            cto1 = to/__SAFEMAX;
            if (cto1 == to) {
                mul = to;
                ready = 1;
                from = 1;
            } else if (__ABS(cfrom1) > __ABS(to) && to != __ZERO) {
                mul = __SAFEMIN;
                from = cfrom1;
            } else if (__ABS(cto1) > __ABS(from)) {
                mul = __SAFEMAX;
                to = cto1;
            } else {
                mul = to/from;
                ready = 1;
            }
        }
        __armas_mscale(A, mul, flags);
    } while (! ready);
    return 0;
}

/*
 * \brief Compute singular values of bidiagonal matrix using the DQDS algorithm.
 *
 * \param[in,out] D
 *      On entry, the diagonal elements. On exit singular values in descending order
 * \param[in,out] E
 *      On entry off-diagonal elements
 * \param[in]  W
 *      Workspace, required size is 4*N
 * \param[in]  conf
 *      Configuration block
 *
 * \return
 *      Zero for success, -1 for error and .error member in conf set.
 *
 * References:
 *  (1) Parlett, Marques;
 *      An Implementation of the DQDS Algorithm (Positive Case),
 *      Lapack working notes #121
 */
int __armas_dqds(__armas_dense_t *D, __armas_dense_t *E, __armas_dense_t* W, armas_conf_t *conf)
{
    __armas_dense_t sq, se, Z;
    int k, n, N, err, ok;
    DTYPE dmax, emax, di, ei, qmax, scalemax;

    N = __armas_size(D);
    if (!conf)
        conf = armas_conf_default();

    if (__armas_size(W) < 4*N) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    /*
     * Make a 4-by-N matrix where first row will hold q-values, 2nd qq-value
     * 3rd the e-values and 4th the ee-values as in (1) Z array. Each column
     * represents {q(k), qq(k), e(k), ee(k)} tuple.
     */
    __armas_make(&Z, 4, N, 4, __armas_data(W));
    __armas_scale(&Z, __ZERO, conf);
    __armas_row(&sq, &Z, 0);
    __armas_row(&se, &Z, 2);

    dmax = __ZERO; emax = __ZERO;
    for (k = 0; k < N-1; k++) {
        di = __ABS(__armas_get_at_unsafe(D, k));
        ei = __ABS(__armas_get_at_unsafe(E, k));
        emax = __MAX(emax, ei);
        dmax = __MAX(dmax, di);
        __armas_set_at_unsafe(&sq, k, di);
        __armas_set_at_unsafe(&se, k, ei);
    }
    di = __ABS(__armas_get_at_unsafe(D, N-1));
    dmax = __MAX(dmax, di);
    __armas_set_at_unsafe(&sq, N-1, di);
    __armas_set_at_unsafe(&se, N-1, __ZERO);

    if (emax == __ZERO) {
        // already diagonal
        return 0;
    }
    qmax     = __MAX(dmax, emax);
    scalemax = __SQRT(__EPS/__SAFEMIN);
    // scale values from QMAX to SCALEMAX
    /* missing */
    //__armas_scale_to(&sq, qmax, scalemax, ARMAS_ANY);
    //__armas_scale_to(&se, qmax, scalemax, ARMAS_ANY);

    // square scaled elements
    for (k = 0; k < N-1; k++) {
        di = __armas_get_at_unsafe(&sq, k);
        ei = __armas_get_at_unsafe(&se, k);
        __armas_set_at_unsafe(&sq, k, di*di);
        __armas_set_at_unsafe(&se, k, ei*ei);
    }
    di = __armas_get_at_unsafe(&sq, N-1);
    __armas_set_at_unsafe(&sq, N-1, di*di);

    // run DQDS algorithm on Z
    if ((err = __dqds_main(&Z, N)) == 0) {

        for (k = 0; k < N; k++) {
            di = __SQRT(__armas_get_at_unsafe(&sq, k));
            __armas_set_at_unsafe(D, k, di);
        }
        // rescale from SCALEMAX to QMAX
        //__armas_scale_to(D, scalemax, qmax, ARMAS_ANY);

        // and sort elements
        __sort_eigenvec(D, __nil, __nil, __nil, ARMAS_DESC);
    } else {
        conf->error = ARMAS_ECONVERGE;
        err = -1;
    }
    return err;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

