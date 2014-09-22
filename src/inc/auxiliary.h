
// Copyright (c) Harri Rautila, 2012-2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_AUXILIARY_H
#define __ARMAS_AUXILIARY_H 1

#include "internal_lapack.h"

#if defined(FLOAT64)
// include double precision optimized versions
#if defined(__AVX__) 
// for AVX instruction set
#include "aux_avx_f64.h"
#endif
#endif  // FLOAT64

// Auxiliary function for LAPACK functionalty

#if ! defined(__gvrotg_enhanced)
/*
 * \brief Compute Givens rotation
 */
static inline
void __gvrotg(DTYPE *c, DTYPE *s, DTYPE *r, DTYPE a, DTYPE b)
{
    DTYPE t, u;

    if (b == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = a;
    } else if (a == 0.0) {
        *c = 0.0;
        *s = 1.0;
        *r = b;
    } else if (__ABS(b) > __ABS(a)) {
        t = a / b;
        u = __SQRT(1.0 + t*t);
        if (__SIGN(b))
            u = -u;
        *s = 1.0 / u;
        *c = (*s) * t;
        *r = b * u;
    } else {
        t = b / a;
        u = __SQRT(1.0 + t*t);
        *r = a * u;
        *c = 1.0 / u;
        *s = (*c) * t;
    }
}
#endif  // __gvrotg_enhanced

/*
 * \brief Apply single plane rotation, internal version.
 *
 * \ingroup lapackaux internal
 */
#if ! defined(__gvrot_enhanced)
static inline
void __gvrot(DTYPE *v0, DTYPE *v1, DTYPE cos, DTYPE sin, DTYPE y0, DTYPE y1)
{
    *v0 = cos * y0 + sin * y1;
    *v1 = cos * y1 - sin * y0;
}
#endif

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

#if !defined(__bd_qrsweep_enhanced)
/*
 * \brief Bidiagonal top to bottom QR sweep.
 *
 * \param D [in,out]
 *      Diagonal elements
 * \param E [in,out]
 *      Off-diagonal elements
 * \param Cr, Sr, Cl, Sl [out]
 *      Saved plane rotations
 * \param f0, g0 [in]
 *      Initial plane rotation parameters
 * \param saves [in]
 *      If set then rotation are saved.
 * \return
 *      Number of rotations performed (N-2)
 *
 * Assume: len(D)-1 == len(E)
 *
 * \ingroup lapackaux internal
 */
static inline
int __bd_qrsweep(__armas_dense_t *D, __armas_dense_t *E,
                 __armas_dense_t *Cr, __armas_dense_t *Sr,
                 __armas_dense_t *Cl, __armas_dense_t *Sl, 
                 DTYPE f0, DTYPE g0, int saves)
{
    DTYPE d1, e1, d2, e2, f, g,  cosr, cosl, sinr, sinl, r;
    int k, N = __armas_size(D);

    d1 = __armas_get_at_unsafe(D, 0);
    e1 = __armas_get_at_unsafe(E, 0);
    f = f0;
    g = g0;
    for (k = 0; k < N-1; k++) {
        d2 = __armas_get_at_unsafe(D, k+1);

        __gvrotg(&cosr, &sinr, &r, f, g);
        if (k > 0) {
            // e[i-1] = r
            __armas_set_at_unsafe(E, k-1, r);
        }
        // f = cosr*d[i] + sinr*e[i]; e[i] = cosr*e[i] - sinr*d[i]
        __gvrot(&f, &e1, cosr, sinr, d1, e1);
        // g = sinr*d[i+1]; d[i+1] = cosr*d[i+];
        __gvrot(&g, &d2, cosr, sinr, 0.0, d2);
        __gvrotg(&cosl, &sinl, &r, f, g);
        // d[i] = r
        d1 = r;
        // f = cosl*e[i] + sinl*d[i+1]; d[i+1] = cosl*d[i+1] - sinl*e[i];
        __gvrot(&f, &d2, cosl, sinl, e1, d2);
        if (k < N-2) {
            e2 = __armas_get_at_unsafe(E, k+1);
            // g = sinl*e[i+1]; e[i+1] = cosl*e[i+1];
            __gvrot(&g, &e2, cosl, sinl, 0.0, e2);
            __armas_set_at_unsafe(E, k+1, e2);
        }

        __armas_set_at_unsafe(D, k, d1);
        __armas_set_at_unsafe(D, k+1, d2);
        d1 = d2; e1 = e2;

        // save rotations
        if (saves) {
            __armas_set_at_unsafe(Cr, k, cosr);
            __armas_set_at_unsafe(Sr, k, sinr);
            __armas_set_at_unsafe(Cl, k, cosl);
            __armas_set_at_unsafe(Sl, k, sinl);
        }
    }
    // e[-1] = f
    __armas_set_at_unsafe(E, N-2, f);
    //return k > 0 ? k+1 : 0;
    return N-1;
}
#endif // !defined(__bd_qrsweep_enhanced)

#if !defined(__bd_qrzero_enhanced)
/*
 * \brief Implicit zero shift QR sweep.
 *
 * As described in Demmel-Kahan, 1990.
 */
static inline
int __bd_qrzero(__armas_dense_t *D, __armas_dense_t *E,
                 __armas_dense_t *Cr, __armas_dense_t *Sr,
                 __armas_dense_t *Cl, __armas_dense_t *Sl, 
                 int saves)
{
    DTYPE d1, e1, d2, cosr, cosl, sinr, sinl, r;
    int k, N = __armas_size(D);

    d1 = __armas_get_at_unsafe(D, 0);
    cosr = __ONE;
    cosl = __ONE;
    for (k = 0; k < N-1; k++) {
        e1 = __armas_get_at_unsafe(E, k);
        d2 = __armas_get_at_unsafe(D, k+1);
        __gvrotg(&cosr, &sinr, &r, d1*cosr, e1);
        if (k > 0)
            __armas_set_at_unsafe(E, k-1, sinl*r);
        __gvrotg(&cosl, &sinl, &r, cosl*r, sinr*d2);
        __armas_set_at_unsafe(D, k, r);
        d1 = d2;
        if (saves) {
            __armas_set_at_unsafe(Cr, k, cosr);
            __armas_set_at_unsafe(Sr, k, sinr);
            __armas_set_at_unsafe(Cl, k, cosl);
            __armas_set_at_unsafe(Sl, k, sinl);
        }
    }
    d2 = cosr*d2;
    __armas_set_at_unsafe(D, N-1, d2*cosl);
    __armas_set_at_unsafe(E, N-2, d2*sinl);
    //return k > 0 ? k+1 : 0;
    return N-1;
}
#endif

#if ! defined(__bd_qlsweep_enhanced)
static inline
int __bd_qlsweep(__armas_dense_t *D, __armas_dense_t *E,
                 __armas_dense_t *Cr, __armas_dense_t *Sr,
                 __armas_dense_t *Cl, __armas_dense_t *Sl, DTYPE f0, DTYPE g0, int saves)
{
    DTYPE d1, e1, d2, e2, f, g,  cosr, cosl, sinr, sinl, r;
    int k, n, N = __armas_size(D);

    d1 = __armas_get_at_unsafe(D, N-1);
    e1 = __armas_get_at_unsafe(E, N-2);
    f = f0;
    g = g0;
    for (n = 0, k = N-1; k > 0; k--, n++) {
        d2 = __armas_get_at_unsafe(D, k-1);

        __gvrotg(&cosr, &sinr, &r, f, g);
        if (k < N-1) {
            __armas_set_at_unsafe(E, k, r);
        }
        //f  = cosr*d1 + sinr*e1;   e1 = cosr*e1 - sinr*d1;
        __gvrot(&f, &e1, cosr, sinr, d1, e1);
        //g  = sinr*d2;  d2 = cosr*d2;
        __gvrot(&g, &d2, cosr, sinr, 0.0, d2);

        __gvrotg(&cosl, &sinl, &r, f, g);
        d1 = r;
        //f  = cosl*e1 + sinl*d2;  d2 = cosl*d2 - sinl*e1;
        __gvrot(&f, &d2, cosl, sinl, e1, d2);
        if (k > 1) {
            e2 = __armas_get_at_unsafe(E, k-2);
            //g  = sinl*e2;  e2 = cosl*e2;
            __gvrot(&g, &e2, cosl, sinl, 0.0, e2);
            __armas_set_at_unsafe(E, k-2, e2);
        }

        __armas_set_at_unsafe(D, k, d1);
        __armas_set_at_unsafe(D, k-1, d2);
        // save values;
        d1 = d2; e1 = e2;
        // save rotations
        if (saves) {
            __armas_set_at_unsafe(Cr, k-1, cosr);
            __armas_set_at_unsafe(Sr, k-1, -sinr);
            __armas_set_at_unsafe(Cl, k-1, cosl);
            __armas_set_at_unsafe(Sl, k-1, -sinl);
        }
    }
    __armas_set_at_unsafe(E, 0, f);
    //return n > 0 ? N : 0;
    return N-1;
}
#endif


#if ! defined(__bd_qlzero_enhanced)
static inline
int __bd_qlzero(__armas_dense_t *D, __armas_dense_t *E,
                __armas_dense_t *Cr, __armas_dense_t *Sr,
                __armas_dense_t *Cl, __armas_dense_t *Sl, 
                int saves)
{
    DTYPE d1, e1, d2, cosr, cosl, sinr, sinl, r;
    int k, n, N = __armas_size(D);

    d1 = __armas_get_at_unsafe(D, N-1);
    cosr = __ONE;
    cosl = __ONE;
    for (n = 0, k = N-1; k > 0; k--, n++) {
        e1 = __armas_get_at_unsafe(E, k-1);
        d2 = __armas_get_at_unsafe(D, k-1);
        __gvrotg(&cosr, &sinr, &r, d1*cosr, e1);
        if (k < N-1)
            __armas_set_at_unsafe(E, k, sinl*r);
        __gvrotg(&cosl, &sinl, &r, cosl*r, sinr*d2);
        __armas_set_at_unsafe(D, k, r);
        d1 = d2;
        if (saves) {
            __armas_set_at_unsafe(Cr, k-1, cosr);
            __armas_set_at_unsafe(Sr, k-1, -sinr);
            __armas_set_at_unsafe(Cl, k-1, cosl);
            __armas_set_at_unsafe(Sl, k-1, -sinl);
        }
    }
    d2 = cosr*__armas_get_at_unsafe(D, 0);
    __armas_set_at_unsafe(D, 0, d2*cosl);
    __armas_set_at_unsafe(E, 0, d2*sinl);
    //return n > 0 ? N : 0;
    return N-1;
}


#endif

#if !defined(__trd_qlsweep_enhanced)
/*
 * \brief Implicit tridiagonal QL sweep from bottom to top.
 */
static inline
int __trd_qlsweep(__armas_dense_t *D, __armas_dense_t *E, __armas_dense_t *Cr,
                  __armas_dense_t *Sr, DTYPE f0, DTYPE g0, int saves)
{
    DTYPE cosr, sinr, r, d0, d1, e0, e1, e0r, e0c, w0, f, g; 
    int k;
    int N = __armas_size(D);

    f = f0; g = g0;
    d0 = __armas_get_at_unsafe(D, N-1);
    e0 = __armas_get_at_unsafe(E, N-2);

    for (k = N-1; k > 0; k--) {
        d1 = __armas_get_at_unsafe(D, k-1);
        __gvrotg(&cosr, &sinr, &r, f, g);
        if (k < N-1)
            __armas_set_at_unsafe(E, k, r);
        // rows
        __gvrot(&d0,  &e0c, cosr, sinr, d0, e0);
        __gvrot(&e0r, &d1,  cosr, sinr, e0, d1);
        __gvrot(&d0,  &e0r, cosr, sinr, d0, e0r);
        __gvrot(&e0c, &d1,  cosr, sinr, e0c, d1);
        if (k > 1) {
            e1 = __armas_get_at_unsafe(E, k-2);
            __gvrot(&w0, &e1,  cosr, sinr, 0.0, e1);
        }
        __armas_set_at_unsafe(D, k, d0);
        d0 = d1; e0 = e1; f = e0r; g = w0;
        if (saves) {
            __armas_set_at_unsafe(Cr, k-1, cosr);
            __armas_set_at_unsafe(Sr, k-1, -sinr);
        }
    }
    __armas_set_at_unsafe(D, 0, d0);
    __armas_set_at_unsafe(E, 0, e0c);
    return N-1;
}
#endif   // ! __trd_qlsweep_enhanced

#if !defined(__trd_qrsweep_enhanced)
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
static inline
int __trd_qrsweep(__armas_dense_t *D, __armas_dense_t *E, __armas_dense_t *Cr,
                  __armas_dense_t *Sr, DTYPE f0, DTYPE g0, int saves)
{
    DTYPE cosr, sinr, r, d0, d1, e0, e1, e0r, e0c, w0, f, g; 
    int k;
    int N = __armas_size(D);

    f = f0; g = g0;
    d0 = __armas_get_at_unsafe(D, 0);
    e0 = __armas_get_at_unsafe(E, 0);

    for (k = 0; k < N-1; k++) {
        d1 = __armas_get_at_unsafe(D, k+1);
        __gvrotg(&cosr, &sinr, &r, f, g);
        if (k > 0)
            __armas_set_at_unsafe(E, k-1, r);
        __gvrot(&d0,  &e0c, cosr, sinr, d0, e0);
        __gvrot(&e0r, &d1,  cosr, sinr, e0, d1);
        __gvrot(&d0,  &e0r, cosr, sinr, d0, e0r);
        __gvrot(&e0c, &d1,  cosr, sinr, e0c, d1);
        if (k < N-2) {
            e1 = __armas_get_at_unsafe(E, k+1);
            __gvrot(&w0, &e1,  cosr, sinr, 0.0, e1);
        }
        __armas_set_at_unsafe(D, k, d0);
        d0 = d1; e0 = e1; f = e0r; g = w0;
        if (saves) {
            __armas_set_at_unsafe(Cr, k, cosr);
            __armas_set_at_unsafe(Sr, k, sinr);
        }
    }
    __armas_set_at_unsafe(D, N-1, d0);
    __armas_set_at_unsafe(E, N-2, e0r);
    //return k > 0 ? N : 0;
    return N-1;
}
#endif // ! __trd_qrsweep_enhanced

/*
 * \brief Apply plane rotations from left.
 *
 * See armas_gvleft() for details.
 *
 * \ingroup lapackaux internal
 */
#if ! defined(__gvleft_enhanced)
static inline
void __gvleft(__armas_dense_t *A, DTYPE c, DTYPE s, int r1, int r2, int col, int ncol)
{
    double t0, *y0, *y1;
    int k, n;

    y0 = &A->elems[col*A->step+r1];
    y1 = &A->elems[col*A->step+r2];
    for (k = 0, n = 0; k < ncol; k++, n += A->step) {
        t0    = c * y0[n] + s * y1[n];
        y1[n] = c * y1[n] - s * y0[n];
        y0[n] = t0;
    }
}
#endif // ! __gvleft_enhanced


/*
 * \brief Apply plane rotations from right.
 *
 * See armas_gvright() for details.
 *
 * \ingroup lapackaux internal
 */
#if ! defined(__gvright_enhanced)
static inline
void __gvright(__armas_dense_t *A, DTYPE c, DTYPE s, int c1, int c2, int row, int nrow)
{
    double t0, *y0, *y1;
    int k;

    y0 = &A->elems[c1*A->step+row];
    y1 = &A->elems[c2*A->step+row];
    for (k = 0; k < nrow; k++) {
        t0    = c * y0[k] + s * y1[k];
        y1[k] = c * y1[k] - s * y0[k];
        y0[k] = t0;
    }
}
#endif // ! __gvright_enhanced


/*
 * \brief Compute Wilkinson shift from trailing 2-by-2 submatrix
 *
 * \verbatim
 *   ( tn1  tnn1 )
 *   ( tnn1 tn   )
 * \endverbatim
 *
 * Stable formula for the Wilkinson's shift (Hogben 2007, 42.3)
 *   d = (tn1 - tn)/2.0
 *   u = tn - sign(d)*tnn1^2/(abs(d) + sqrt(d^2 + tnn1^2))
 *
 */
static inline
DTYPE __wilkinson(DTYPE tn1, DTYPE tnn1, DTYPE tn)
{
    DTYPE d, tsq, sign;
    d = (tn1 - tn)/2.0;
    tsq = __HYPOT(d, tnn1);
    return tn - __COPYSIGN((tnn1/(__ABS(d) + tsq))*tnn1, d);
    //sign = __SIGN(d) ? -1.0 : 1.0;
    //return tn - sign*tnn1*tnn1/(__ABS(d) + tsq);
}


/*
 * \brief Compute eigenvalues of 2x2 matrix.
 *
 * \param e1, e2 [out]
 *      Computed eigenvalue
 * \param a, b, c, d [in]
 *      Matrix elements
 *
 *  Characteristic function for solving eigenvalues:
 * \verbatim
 *   det A-Ix = det ( a-x   b  )
 *                  (  c   d-x )
 * \endverbatim
 *      x^2 - (a + d)*x + (a*d - c*b) = 0 =>
 *      x^2 - 2*T*x + D = 0, T = (a + d)/2, D = a*d - c*b
 *
 *      e1 = T + sqrt(T*T - D)
 *      e2 = T - sqrt(T*T - D)
 */
static inline
void __eigen2x2(DTYPE *e1, DTYPE *e2, DTYPE a, DTYPE b, DTYPE c, DTYPE d)
{
    DTYPE T, D;

    T = (a + d)/2.0;
    D = a*d - b*c;
    __armas_qdroots(e1, e2, 1.0, T, D);
}

#endif

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
