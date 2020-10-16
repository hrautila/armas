
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armas_housemult) && defined(armas_housemult_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_blas1) && defined(armas_blas2)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "partition.h"

/*
 * Unblocked algorith for computing X = Q*X and X = Q.T*X
 *
 * Q = H(1)H(2)...H(m) where elementary reflectors H(i) are stored on
 * i'th column in P below diagonal (scaled) or on-and-below diagonal (unscaled)
 * and cols(P) == m.
 *
 * Progressing P from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces H(m)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(m) == Q
 * and C = Q*C
 *
 * If ARMAS_UNIT is set in flags then reflectors are scaled, otherwise
 * unscaled.
 * Requirements:
 *    1. rows(X) == rows(P) 
 *    2. ! isvector(X)             --> len(w) == len(X.row)  
 *    3. (flags & ARMAS_UNIT) != 0 --> len(tau) == len(P.col)
 */
static
void hmult_left(armas_dense_t * X, armas_dense_t * tau, armas_dense_t * P,
                armas_dense_t * w, int flags, armas_conf_t * cf)
{
    int pdir, xdir, xstart, pstart, xm, pm, pn;
    armas_dense_t PTL, PBR, P00, p11, P22, p21;
    armas_dense_t XT, XB, X0, x1, X2;
    armas_dense_t tT, tB, t0, t1, t2, t, v, *vp, *tp = __nil;
    DTYPE _tau, _v0val;

    EMPTY(tT);
    EMPTY(P00);
    EMPTY(PTL);
    EMPTY(X0);
    EMPTY(t0);

    if (!cf)
        cf = armas_conf_default();


    int unit = flags & ARMAS_UNIT;
    int m = P->cols;
    int isvector = armas_isvector(X);

    if ((flags & ARMAS_TRANS) != 0) {
        // compute x = H(m-1)*...H(1)*H(0)*x
        pm = pn = xm = 0;
        xstart = ARMAS_PTOP;
        pstart = ARMAS_PTOPLEFT;
        pdir = ARMAS_PBOTTOMRIGHT;
        xdir = ARMAS_PBOTTOM;
    } else {
        /*
         * Compute x = H(0)*H(1)*..H(m-1)*x; here pm,pn is size of the PBR block
         */
        pm = max(0, P->rows - P->cols);
        pn = max(0, P->cols - P->rows);
        xm = max(0, armas_size(tau) - P->cols);
        xstart = ARMAS_PBOTTOM;
        pstart = ARMAS_PBOTTOMRIGHT;
        pdir = ARMAS_PTOPLEFT;
        xdir = ARMAS_PTOP;
    }
    // setup
    mat_partition_2x2(&PTL, __nil, __nil, &PBR, /**/ P, pm, pn, pstart);
    mat_partition_2x1(&XT, &XB, /**/ X, pm, xstart);
    if (unit) {
        mat_partition_2x1(&tT, &tB, /**/ tau, xm, xstart);
        _v0val = ONE;
        armas_make(&v, 1, 1, 1, &_v0val);
        tp = &t1;
        vp = &v;
    } else {
        _tau = TWO;
        armas_make(&t, 1, 1, 1, &_tau);
        tp = &t;
        vp = &p11;
    }

    for (; m > 0; m--) {
        mat_repartition_2x2to3x3(
            &PTL,
            &P00, __nil, __nil,
            __nil, &p11, __nil,
            __nil, &p21, &P22, /**/ P, 1, pdir);
        mat_repartition_2x1to3x1(
            &XT, &X0, &x1, &X2, /**/ X, 1, xdir);
        if (unit)
            mat_repartition_2x1to3x1(&tT, &t0, &t1, &t2, /**/ tau, 1, xdir);
        // ---------------------------------------------------------------------
        DTYPE tval = armas_get_unsafe(tp, 0, 0);
        DTYPE v0val = armas_get_unsafe(vp, 0, 0);
        if (isvector) {
            // no workspace needed if X is vector
            DTYPE w1;
            // w  = v0*x1 + X2.T*p21
            w1 = armas_dot(&X2, &p21, cf);
            w1 += v0val * armas_get_unsafe(&x1, 0, 0);
            // x1 = x1 - tau*v0*w
            armas_set_unsafe(&x1, 0, 0,
                               armas_get_unsafe(&x1, 0,
                                                  0) - tval * v0val * w1);
            // X2 = X2 - tau*p21*w
            armas_axpy(&X2, -tval * w1, &p21, cf);
        } else {
            // w  = v0*x1
            armas_axpby(ZERO, w, v0val, &x1, cf);
            // w  = v0*x1 + X2.T*p21
            armas_mvmult(ONE, w, ONE, &X2, &p21, ARMAS_TRANSA, cf);
            // x1 = x1 - tau*v0*w
            armas_axpy(&x1, -tval * v0val, w, cf);
            // X2 = X2 - tau*p21*w
            armas_mvupdate(ONE, &X2, -tval, &p21, w, cf);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &PTL, __nil,
            __nil, &PBR, /**/ &P00, &p11, &P22, /**/ P, pdir);
        mat_continue_3x1to2x1(
            &XT, &XB, /**/ &X0, &x1, /**/ X, xdir);
        if (unit)
            mat_continue_3x1to2x1(&tT, &tB, /**/ &t0, &t1, /**/ tau, xdir);
    }
    return;
}

/*
 * Unblocked algorithm for computing X = X*Q and X = X*Q.T
 *
 * Q = H(1)H(2)...H(m) where elementary householder reflectors H(i) are stored on i'th column
 * in P below diagonal (scaled) or on-and-below diagonal (unscaled) and cols(P) == m.
 *
 * Requirements:
 *    1. cols(X) == rows(P) 
 *    2. ! isvector(X)             --> len(w) == len(X.col)
 *    3. (flags & ARMAS_UNIT) != 0 --> len(tau) == len(P.col)
 */
static
void hmult_right(armas_dense_t * X, armas_dense_t * tau,
                 armas_dense_t * P, armas_dense_t * w,
                 int flags, armas_conf_t * cf)
{
    int pdir, xdir, xstart, pstart, tstart, tdir, xm, pm, pn, tm;
    armas_dense_t PTL, PBR, P00, p11, P22, p21;
    armas_dense_t XL, XR, X0, x1, X2;
    armas_dense_t tT, tB, t0, t1, t2, t, v, *vp, *tp = __nil;
    DTYPE _tau, _v0val;

    EMPTY(tT);
    EMPTY(P00);
    EMPTY(PTL);
    EMPTY(XL);
    EMPTY(X0);
    EMPTY(t0);

    if (!cf)
        cf = armas_conf_default();


    int unit = flags & ARMAS_UNIT;
    int m = P->cols;
    int isvector = armas_isvector(X);

    if ((flags & ARMAS_TRANS) != 0) {
        // compute x = x*H(m-1)*...H(1)*H(0)
        pm = max(0, P->rows - P->cols);
        pn = max(0, P->cols - P->rows);
        xm = max(0, X->cols - P->cols);
        tm = max(0, armas_size(tau) - P->cols);
        xstart = ARMAS_PRIGHT;
        pstart = ARMAS_PBOTTOMRIGHT;
        pdir = ARMAS_PTOPLEFT;
        xdir = ARMAS_PLEFT;
        tstart = ARMAS_PBOTTOM;
        tdir = ARMAS_PTOP;
    } else {
        // compute x = x*H(0)*H(1)*...H(m-1) ; 
        pm = pn = xm = tm = 0;
        xstart = ARMAS_PLEFT;
        pstart = ARMAS_PTOPLEFT;
        pdir = ARMAS_PBOTTOMRIGHT;
        xdir = ARMAS_PRIGHT;
        tstart = ARMAS_PTOP;
        tdir = ARMAS_PBOTTOM;
    }
    // setup
    mat_partition_2x2(
        &PTL, __nil, __nil, &PBR, /**/ P, pm, pn, pstart);
    mat_partition_1x2(
        &XL, &XR, /**/ X, xm, xstart);
    if (unit) {
        mat_partition_2x1(&tT, &tB, /**/ tau, tm, tstart);
        _v0val = ONE;
        armas_make(&v, 1, 1, 1, &_v0val);
        tp = &t1;
        vp = &v;
    } else {
        _tau = TWO;
        armas_make(&t, 1, 1, 1, &_tau);
        tp = &t;
        vp = &p11;
    }

    for (; m > 0; m--) {
        mat_repartition_2x2to3x3(
            &PTL,
            &P00, __nil, __nil,
            __nil, &p11, __nil,
            __nil, &p21, &P22, /**/ P, 1, pdir);
        mat_repartition_1x2to1x3(
            &XL, &X0, &x1, &X2, /**/ X, 1, xdir);
        if (unit)
            mat_repartition_2x1to3x1(&tT, &t0, &t1, &t2, /**/ tau, 1, tdir);
        // ---------------------------------------------------------------------
        DTYPE tval = armas_get_unsafe(tp, 0, 0);
        DTYPE v0val = armas_get_unsafe(vp, 0, 0);
        if (isvector) {
            // no workspace needed if X is vector
            DTYPE w1, t;
            // w  = v0*x1 + X2*p21
            w1 = armas_dot(&X2, &p21, cf);
            w1 += v0val * armas_get_unsafe(&x1, 0, 0);
            // x1 = x1 - tau*v0*w
            t = armas_get_unsafe(&x1, 0, 0) - tval * v0val * w1;
            armas_set_unsafe(&x1, 0, 0, t);
            // X2 = X2 - tau*p21*w
            armas_axpy(&X2, -tval * w1, &p21, cf);
        } else {
            // w  = v0*x1
            armas_axpby(ZERO, w, v0val, &x1, cf);
            // w  = v0*x1 + X2*p21
            armas_mvmult(ONE, w, ONE, &X2, &p21, 0, cf);
            // x1 = x1 - tau*v0*w
            armas_axpy(&x1, -tval * v0val, w, cf);
            // X2 = X2 - tau*w*p21
            armas_mvupdate(ONE, &X2, -tval, w, &p21, cf);
        }
        // ---------------------------------------------------------------------
        mat_continue_3x3to2x2(
            &PTL, __nil,
            __nil, &PBR, /**/ &P00, &p11, &P22, /**/ P, pdir);
        mat_continue_1x3to1x2(
            &XL, &XR, /**/ &X0, &x1, /**/ X, xdir);
        if (unit)
            mat_continue_3x1to2x1(&tT, &tB, /**/ &t0, &t1, /**/ tau, tdir);
    }
}

/**
 * @brief Multiply by Householder reflectors.

 * Computes \f$ X = QX, X = Q^TX,  X = XQ or X = XQ^T \f$
 *
 * @param [in,out] X
 *     On entry the matrix X. On exit the result of the requested computation.
 * @param [in] tau
 *     Vector of scalar multipliers of unit scaled Householder reflectors stored
 *     in Q. If reflectors are not scaled ie. flag ARMAS_UNIT was not set when
 *     reflectors were generated by calls to 'armas_house' function, then
 *    `tau` is not used.
 * @param [in] Q
 *     Elementary householder reflectors store in columns of Q in the
 *     on-and-below diagonal entries, for unscaled reflectors, or in the below
 *     diagonal entries for unit scaled reflectors.
 * @param [in] flags
 *     Computation flags
 *       ARMAS_UNIT  reflectors are unit scaled (standard lapack like)
 *       ARMAS_TRANS use Q^T
 *       ARMAS_RIGHT multiply from right
 *       ARMAS_LEFT  multiply from left (default)
 * @param [in,out] wb
 *     Workspace buffer. If on entry wb.bytes == 0, workspace size in
 *     bytes is computed, saved to wb.bytes and function returns immediately
 *     with success.
 * @param [in,out] cf
 *     Configuration block, on error value of cf.error is set.
 *
 * @retval 0 OK
 * @retval <0 failure, cf.error is set
 * @ingroup lapack
 */
int armas_housemult_w(armas_dense_t * X,
                        armas_dense_t * tau,
                        armas_dense_t * Q,
                        int flags, armas_wbuf_t * wb, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();

    if (!wb && !armas_isvector(X)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (!armas_isvector(X) && wb->bytes == 0) {
        // compute workspace size
        int nelem = flags & ARMAS_LEFT ? X->cols : X->rows;
        wb->bytes = nelem * sizeof(DTYPE);
        return 0;
    }
    // check parameter sizes;
    if (flags & ARMAS_UNIT) {
        // this is unit scaled reflector; tau is needed
        if (armas_size(tau) != Q->cols) {
            cf->error = ARMAS_ESIZE;
            return -1;
        }
    }
    int ok = (flags & ARMAS_RIGHT) != 0
        ? Q->rows == X->cols : Q->rows == X->rows;
    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }

    size_t wpos = armas_wpos(wb);
    armas_dense_t w, *wptr;
    wptr = __nil;
    if (!armas_isvector(X)) {
        int nelem = flags & ARMAS_LEFT ? X->cols : X->rows;
        DTYPE *t = armas_wreserve(wb, nelem, sizeof(DTYPE));
        armas_make(&w, nelem, 1, nelem, t);
        wptr = &w;
    }

    if (flags & ARMAS_RIGHT) {
        hmult_right(X, tau, Q, wptr, flags, cf);
    } else {
        hmult_left(X, tau, Q, wptr, flags, cf);
    }
    armas_wsetpos(wb, wpos);
    return 0;
}

/**
 * @brief Multiply by Householder reflectors
 *
 * @see armas_housemult_w
 * @ingroup lapack
 */
int armas_housemult(armas_dense_t * X,
                      armas_dense_t * tau,
                      armas_dense_t * Q, int flags, armas_conf_t * cf)
{
    if (!cf)
        cf = armas_conf_default();
    if (armas_isvector(X)) {
        return armas_housemult_w(X, tau, Q, flags, (armas_wbuf_t *) 0, cf);
    }
    int err;
    armas_wbuf_t wb = ARMAS_WBNULL;
    if ((err = armas_housemult_w(X, tau, Q, flags, &wb, cf)) < 0)
        return err;

    if (!armas_walloc(&wb, wb.bytes)) {
        cf->error = ARMAS_EMEMORY;
        return -ARMAS_EMEMORY;
    }
    err = armas_housemult_w(X, tau, Q, flags, &wb, cf);
    armas_wrelease(&wb);
    return err;
}
#else
#warning "Missinged defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
