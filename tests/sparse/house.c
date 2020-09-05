
// Copyright (c) Harri Rautila, 2017

#include <stdio.h>
#include <unistd.h>
#include "testing.h"
#include "internal.h"
#include "partition.h"

#if FLOAT32
#define __ERROR 1e-6
#else
#define __ERROR 1e-14
#endif


/*
 * Generates a real elementary reflector H of order n, such that
 *
 *       H * ( alpha ) = ( +/-beta ),   H^T*H = H*H = I.
 *           (   x   )   (      0  )
 *
 * where alpha and beta are scalars, and x is an (n-1)-element real
 * vector. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v^T ) ,
 *                     ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element vector.
 *
 *       H = I - 2 * v * v^T ,
 *
 * if tau parameter is null;
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit matrix.
 *
 * Depending on flag bits generates H such that,
 *   flags == 0
 *     Hx = beta e_0, beta in R  
 *   flags == ARMAS_NONNEG
 *     Hx = beta e_0, beta >= 0.0
 *   flags == ARMAS_HHNEGATIVE
 *     Hx == -beta e_0, beta in R
 *   flags == ARMAS_HHNEGATIVE|ARMAS_NONNEG
 *     Hx == -beta e_0, beta >= 0.0
 *
 * Notes:
 *  Setting
 *    ( v1 ) = (alpha) - (beta)  => v1 = (alpha - beta)
 *    ( v2 )   (  x  )   ( 0  )     v2 = x / (alpha - beta)
 * 
 *  and 
 *    tau = (beta - alpha)/beta
 *    normx = ||x||_2
 *
 *  Selecting beta in classic version (always adding factors with same sign, no cancelation):
 *
 *    beta = - sign(alpha)*sqrt(alpha^2 + normx^2)
 *
 *  If always positive beta needed then:
 *
 *    beta = sqrt(alpha^2 + normx^2)
 *
 *    (alpha - beta) = (alpha - beta)(alpha + beta)/(alpha + beta)
 *                   = (alpha^2 - beta^2)/(alpha + beta)
 *                   = - normx^2/(alpha + beta)
 *                   = - normx*(normx/(alpha + beta))
 *   
 *     tau = (beta - alpha)/beta 
 *         = -(alpha - beta)/beta
 *         = normx^2/(alpha + beta)/beta
 *         = (normx/beta)*(normx/(alpha + beta))
 * 
 *    here: normx <= beta && alpha <= beta
 *
 * References: 
 * (1) Golub, Van Load: Matrix Computation, 4th edition, Section 5.1.2 and 5.1.3
 * (2) Demmel, Hoemmen, Hida & Riedy: Lapack Working Note #203, section 3
 *
 */

#if 0
/*
 * Compute the unscaled Householder reflector H = I - 2*v*v^T such that Hx = beta*e_0
 * 
 */
static 
int __hhcompute_unscaled(armas_x_dense_t *x0, armas_x_dense_t *x1, 
                         armas_x_dense_t *tau, int flags, armas_conf_t *conf)
{
    DTYPE normx, x0val, alpha, beta, delta, sign, safmin, rsafmin, t;
    int nscale = 0;

    normx = armas_x_nrm2(x, conf);
    if (normx == 0.0) {
        armas_x_set(tau, 0, 0, armas_x_get(a11, 0, 0));
        return 0;
    }

    x0val = armas_x_get_unsafe(x0, 0, 0);
    sign  = __SIGN(x0val) ? -1.0 : 1.0;
    
    beta = __HYPOT(x0val, normx);
    
    switch (flags & (ARMAS_NONNEG)) {
    case ARMAS_NONNEG:
        if (x0 <= 0.0) {
            alpha = x0val - beta;
            delta = __HYPOT(normx, alpha);
        } else {
            alpha = - normx * (normx/(x0val + beta));
            delta = __HYPOT(normx, alpha);
        }
        break;
    default:
        beta = -sign*beta;
        delta = __HYPOT(normx, x0val - beta);
        alpha = x0val - beta;
        break;
    }

    armas_x_set_unsafe(x0, 0, 0, alpha/delta);
    armas_x_scale(x1, 1.0/delta, conf);
    armas_x_set_unsafe(tau, 0, 0, beta);
    return 0;
}

int __hhapply(armas_x_dense_t *x, armas_x_dense_t *w, armas_x_dense_t *tau, int flags, armas_conf_t *cf)
{
    DTYPE alpha = armas_x_dot(x, w, cf);
    printf("alpha: %e\n", alpha);
    if (tau) {
        alpha *= armas_x_get(tau, 0, 0);
    }
    else {
        alpha *= 2.0;
    }      
    armas_x_axpy(x, -alpha, w, cf);
}
#endif

/*
 * Compute x = H*x =  x - tau*v*v^T*x = x - tau*(x^T*v)*v
 */
int __hhleft2x1(armas_x_dense_t *tau, armas_x_dense_t *v0, armas_x_dense_t *v1,
                armas_x_dense_t *a1,  armas_x_dense_t *A2,
                armas_x_dense_t *w1,  int flags, armas_conf_t *conf)
{
    DTYPE tval;
    tval = (flags & ARMAS_UNIT) != 0 ? armas_x_get_unsafe(tau, 0, 0) : __TWO;
    if (tval == 0.0) {
        return 0;
    }

    DTYPE v0val = (flags & ARMAS_UNIT) != 0 ? __ONE : armas_x_get_unsafe(v0, 0, 0);
    // w1 = v0*a1
    armas_x_axpby(__ZERO, w1, v0val, a1, conf);
    // w1 = v0*a1 + A2.T*v1
    armas_x_mvmult(__ONE, w1, __ONE, A2, v1, ARMAS_TRANSA, conf);
    // a1 = a1 - tau*v0*w1
    armas_x_axpy(a1, -tval*v0val, w1, conf);
    // A2 = A2 - tau*v1*w1
    armas_x_mvupdate(A2, -tval, v1, w1, conf);
    return 0;
}

/*
 * Unblocked algorith for computing X = Q*X and X = Q.T*X
 *
 * Q = H(1)H(2)...H(m) where elementary reflectors H(i) are stored on i'th column
 * in P below diagonal (scaled) or on-and-below diagonal (unscaled) and cols(P) == m.
 *
 * Progressing P from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces H(m)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(m) == Q and C = Q*C
 * 
 * If ARMAS_UNIT is set in flags then reflectors are scaled,  otherwise unscaled.
 */
int __hmult_left(armas_x_dense_t *X, armas_x_dense_t *tau, armas_x_dense_t *P, armas_x_dense_t *w, int flags, armas_conf_t *cf)
{
    int pdir, xdir, xstart, pstart, xm, pm, pn;
    armas_x_dense_t PTL, PBR, P00, p11, P22, p21;
    armas_x_dense_t XT, XB, X0, x1, X2;
    armas_x_dense_t tT, tB, t0, t1, t2, t, v, *vp, *tp = __nil;
    DTYPE _tau, _v0val;
    
    if (!cf)
        cf = armas_conf_default();


    int unit = flags & ARMAS_UNIT;
    int m = P->cols;
    int isvector = armas_x_isvector(X);
    
    if ((flags & ARMAS_TRANS) != 0) {
        // compute x = H(m-1)*...H(1)*H(0)*x
        pm = pn = xm = 0;
        xstart = ARMAS_PTOP;
        pstart = ARMAS_PTOPLEFT;
        pdir = ARMAS_PBOTTOMRIGHT;
        xdir = ARMAS_PBOTTOM;
    } else {
        // compute x = H(0)*H(1)*...H(m-1)*x ; here pm,pn is size of the PBR block
        pm = max(0, P->rows - P->cols);
        pn = max(0, P->cols - P->rows);
        xm = max(0, armas_x_size(tau) - P->cols);
        xstart = ARMAS_PBOTTOM;
        pstart = ARMAS_PBOTTOMRIGHT;
        pdir = ARMAS_PTOPLEFT;
        xdir = ARMAS_PTOP;
    }
    // setup
    __partition_2x2(&PTL, __nil,
                    __nil, &PBR,  /**/ P, pm, pn, pstart);
    __partition_2x1(&XT,
                    &XB,          /**/ X, pm, xstart);
    if (unit) {
        __partition_2x1(&tT,
                        &tB,      /**/ tau, xm, xstart);
        _v0val = __ONE;
        armas_x_make(&v, 1, 1, 1, &_v0val);
        tp = &t1;
        vp = &v;
    } else {
        _tau = __TWO;
        armas_x_make(&t, 1, 1, 1, &_tau);
        tp = &t;
        vp = &p11;
    }        

    for (; m > 0; m--) {
        __repartition_2x2to3x3(&PTL,
                               &P00,  __nil, __nil,
                               __nil,  &p11, __nil,
                               __nil,  &p21,  &P22,  /**/ P, 1, pdir);
        __repartition_2x1to3x1(&XT,
                               &X0,  &x1, &X2,      /**/ X, 1, xdir);
        if (unit) {
            __repartition_2x1to3x1(&tT,
                                   &t0,  &t1, &t2,  /**/ tau, 1, xdir);
        }
        // -----------------------------------------------------------------------------
        DTYPE tval  = armas_x_get_unsafe(tp, 0, 0);
        DTYPE v0val = armas_x_get_unsafe(vp, 0, 0);
        if (isvector) {
            // no workspace needed if X is vector
            DTYPE w1;           
            // w  = v0*x1 + X2.T*p21
            w1  = armas_x_dot(&X2, &p21, cf);
            w1 += v0val*armas_x_get_unsafe(&x1, 0, 0);
            // x1 = x1 - tau*v0*w
            armas_x_set_unsafe(&x1, 0, 0, armas_x_get_unsafe(&x1, 0, 0)-tval*v0val*w1);
            // X2 = X2 - tau*p21*w
            armas_x_axpy(&X2, -tval*w1, &p21, cf);
        } else {
            // w  = v0*x1
            armas_x_axpby(__ZERO, w, v0val, &x1, cf);
            // w  = v0*x1 + X2.T*p21
            armas_x_mvmult(__ONE, w, __ONE, &X2, &p21, ARMAS_TRANSA, cf);
            // x1 = x1 - tau*v0*w
            armas_x_axpy(&x1, -tval*v0val, w, cf);
            // X2 = X2 - tau*p21*w
            armas_x_mvupdate(&X2, -tval, &p21, w, cf);
        }
        // -----------------------------------------------------------------------------
        __continue_3x3to2x2(&PTL, __nil,
                            __nil, &PBR, /**/  &P00, &p11, &P22, /**/ P, pdir);
        __continue_3x1to2x1(&XT,
                            &XB,         /**/  &X0, &x1,         /**/ X, xdir);
        if (unit) {
            __continue_3x1to2x1(&tT,
                                &tB,     /**/  &t0, &t1,         /**/ tau, xdir);
        }
    }
    return 0;
}


void __hmult_right(armas_x_dense_t *X, armas_x_dense_t *tau, armas_x_dense_t *P, armas_x_dense_t *w, int flags, armas_conf_t *cf)
{
    int pdir, xdir, xstart, pstart, tstart, tdir, xm, pm, pn, tm;
    armas_x_dense_t PTL, PBR, P00, p11, P22, p21;
    armas_x_dense_t XL, XR, X0, x1, X2;
    armas_x_dense_t tT, tB, t0, t1, t2, t, v, *vp, *tp = __nil;
    DTYPE _tau, _v0val;
    
    if (!cf)
        cf = armas_conf_default();


    int unit = flags & ARMAS_UNIT;
    int m = P->cols;
    int isvector = armas_x_isvector(X);
    
    if ((flags & ARMAS_TRANS) != 0) {
        // compute x = x*H(m-1)*...H(1)*H(0)
        pm = max(0, P->rows - P->cols);
        pn = max(0, P->cols - P->rows);
        xm = max(0, X->cols - P->cols);
        tm = max(0, armas_x_size(tau) - P->cols);
        xstart = ARMAS_PRIGHT;
        pstart = ARMAS_PBOTTOMRIGHT;
        pdir = ARMAS_PTOPLEFT;
        xdir = ARMAS_PLEFT;
        tstart = ARMAS_PBOTTOM;
        tdir = ARMAS_PTOP;
    } else {
        // compute x = x*H(0)*H(1)*...H(m-1) ; here pm,pn is size of the PBR block
        pm = pn = xm = tm = 0;
        xstart = ARMAS_PLEFT;
        pstart = ARMAS_PTOPLEFT;
        pdir = ARMAS_PBOTTOMRIGHT;
        xdir = ARMAS_PRIGHT;
        tstart = ARMAS_PTOP;
        tdir = ARMAS_PBOTTOM;
   }
    // setup
    __partition_2x2(&PTL, __nil,
                    __nil, &PBR,  /**/ P, pm, pn, pstart);
    __partition_1x2(&XL,   &XR,   /**/ X, xm, xstart);
    if (unit) {
        __partition_2x1(&tT,
                        &tB,      /**/ tau, tm, tstart);
        _v0val = __ONE;
        armas_x_make(&v, 1, 1, 1, &_v0val);
        tp = &t1;
        vp = &v;
    } else {
        _tau = __TWO;
        armas_x_make(&t, 1, 1, 1, &_tau);
        tp = &t;
        vp = &p11;
    }        

    for (; m > 0; m--) {
        __repartition_2x2to3x3(&PTL,
                               &P00,  __nil, __nil,
                               __nil,  &p11, __nil,
                               __nil,  &p21,  &P22,  /**/ P, 1, pdir);
        __repartition_1x2to1x3(&XL,
                               &X0,  &x1, &X2,      /**/ X, 1, xdir);
        if (unit) {
            __repartition_2x1to3x1(&tT,
                                   &t0,  &t1, &t2,  /**/ tau, 1, tdir);
        }
        // -----------------------------------------------------------------------------
        DTYPE tval = armas_x_get_unsafe(tp, 0, 0);
        DTYPE v0val = armas_x_get_unsafe(vp, 0, 0);
        if (isvector) {
            // no workspace needed if X is vector
            DTYPE w1;           
            // w  = v0*x1 + X2*p21
            w1  = armas_x_dot(&X2, &p21, cf);
            w1 += v0val*armas_x_get_unsafe(&x1, 0, 0);
            // x1 = x1 - tau*v0*w
            armas_x_set_unsafe(&x1, 0, 0, armas_x_get_unsafe(&x1, 0, 0)-tval*v0val*w1);
            // X2 = X2 - tau*p21*w
            armas_x_axpy(&X2, -tval*w1, &p21, cf);
        } else {
            // w  = v0*x1
            armas_x_axpby(__ZERO, w, v0val, &x1, cf);
            // w  = v0*x1 + X2*p21
            armas_x_mvmult(__ONE, w, __ONE, &X2, &p21, 0, cf);
            // x1 = x1 - tau*v0*w
            armas_x_axpy(&x1, -tval*v0val, w, cf);
            // X2 = X2 - tau*w*p21
            armas_x_mvupdate(&X2, -tval, w, &p21, cf);
        }
        // -----------------------------------------------------------------------------
        __continue_3x3to2x2(&PTL, __nil,
                            __nil, &PBR, /**/  &P00, &p11, &P22, /**/ P, pdir);
        __continue_1x3to1x2(&XL,    &XR, /**/  &X0, &x1,         /**/ X, xdir);
        if (unit) {
            __continue_3x1to2x1(&tT,
                                &tB,     /**/  &t0, &t1,         /**/ tau, tdir);
        }
    }
}

int armas_x_housemult_w(armas_x_dense_t *X,
                        armas_x_dense_t *tau,
                        armas_x_dense_t *Q,
                        int flags,
                        armas_wbuf_t *wb,
                        armas_conf_t *cf)
{
    if (!cf)
        cf = armas_conf_default();

    if (!wb && ! armas_x_isvector(X)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (! armas_x_isvector(X) && wb->bytes == 0) {
        // compute workspace size
        int nelem = flags & ARMAS_LEFT ? X->cols : X->rows;
        wb->bytes = nelem*sizeof(DTYPE);
        return 0;
    }
    // check parameter sizes;
    if (flags & ARMAS_UNIT) {
        // this is unit scaled reflector; tau is needed
        if (armas_x_size(tau) != Q->cols) {
            cf->error = ARMAS_ESIZE;
            return -1;
        }
    }
    int ok = (flags & ARMAS_RIGHT) != 0
        ? Q->rows == X->cols
        : Q->rows == X->rows;
    if (!ok) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }

    size_t wpos = armas_wpos(wb);
    armas_x_dense_t w, *wptr;
    wptr = __nil;
    if (! armas_x_isvector(X)) {
        int nelem = flags & ARMAS_LEFT ? X->cols : X->rows;
        DTYPE *t = armas_wreserve(wb, nelem, sizeof(DTYPE));
        armas_x_make(&w, nelem, 1, nelem, t);
        wptr = &w;
    } 

    if (flags & ARMAS_RIGHT) {
        __hmult_right(X, tau, Q, wptr, flags, cf);
    } else {
        __hmult_left(X, tau, Q, wptr, flags, cf);
    }
    armas_wsetpos(wb, wpos);
    return 0;
}

int armas_x_housemult(armas_x_dense_t *X,
                      armas_x_dense_t *tau,
                      armas_x_dense_t *Q,
                      int flags,
                      armas_conf_t *cf)
{
    if (!cf)
        cf = armas_conf_default();
    if (armas_x_isvector(X)) {
        return armas_x_housemult_w(X, tau, Q, flags, (armas_wbuf_t *)0, cf);
    }
    armas_wbuf_t wb = ARMAS_WBNULL;
    if (armas_x_housemult_w(X, tau, Q, flags, &wb, cf) < 0)
        return -1;

    if (!armas_walloc(&wb, wb.bytes)) {
        cf->error = ARMAS_EMEMORY;
        return -1;
    }
    int stat = armas_x_housemult_w(X, tau, Q, flags, &wb, cf);
    armas_wrelease(&wb);
    return stat;
}

// compute: ||x - Q^T*(Q*x)||
int test_left(int flags, int verbose, int m, int n, int p)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y, z, Cd, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    // x and y 
    armas_d_init(&x, m, 1);
    armas_d_init(&y, m, p);
    armas_d_init(&z, m, p);
    armas_d_init(&tau, n, 1);
    armas_d_init(&w, p, 1);
    armas_d_init(&P, m, n);
    armas_d_set_values(&P, unitrand, 0);
    armas_d_set_values(&y, unitrand, 0);
    armas_d_mcopy(&z, &y);
    

    // generate Householder reflectors
    for (int k = 0; k < n; k++) {
        armas_x_submatrix(&x0, &P, k, k, 1, 1);
        armas_x_submatrix(&x1, &P, k+1, k, m-k-1, 1);
        //armas_x_set(&x0, 0, 0, -armas_x_get(&x0, 0, 0));
        armas_x_submatrix(&t, &tau, k, 0, 1, 1);
        armas_x_house(&x0, &x1, &t,  flags, &cf);
    }

    
    // Q^T*(Q*y) == y
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_LEFT|ARMAS_TRANS, &cf) < 0)
        printf("left.1: error %d\n", cf.error);
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_LEFT, &cf) < 0)
        printf("left.2: error %d\n", cf.error);

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_ONE, 0, &cf);
    int ok = isFINE(relerr, m*__ERROR);
    char *s = (flags & ARMAS_UNIT) != 0 ? "unit scaled" : "   unscaled";

    printf("%s: (%s) y - Q.T*(Qy) == y\n", PASS(ok), s);
    if (verbose > 0)
        printf("   ||y - Q^T*(Qy)||: %e [%d]\n", relerr, ndigits(relerr));

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&tau);
    armas_d_release(&P);
    armas_d_release(&w);

    return ok;
}

// compute: ||x - (x*Q)*Q^T||
int test_right(int flags, int verbose, int m, int n, int p)
{
    armas_d_dense_t x, y, z, Cd, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    // x and y 
    armas_d_init(&x, m, 1);
    armas_d_init(&y, p, m);
    armas_d_init(&z, p, m);
    armas_d_init(&tau, n, 1);
    armas_d_init(&w, p, 1);
    armas_d_init(&P, m, n);
    armas_d_set_values(&P, unitrand, 0);
    armas_d_set_values(&y, one, 0);
    armas_d_mcopy(&z, &y);
    

    for (int k = 0; k < n; k++) {
        armas_x_submatrix(&x0, &P, k, k, 1, 1);
        armas_x_submatrix(&x1, &P, k+1, k, m-k-1, 1);
        //armas_x_set(&x0, 0, 0, -armas_x_get(&x0, 0, 0));
        armas_x_submatrix(&t, &tau, k, 0, 1, 1);
        armas_x_house(&x0, &x1, &t,  flags, &cf);
    }

    printf("P:\n");      armas_x_printf(stdout, "%9.2e", &P);
    printf("tau:\n");    armas_x_printf(stdout, "%9.2e", &tau);
    printf("y:\n");      armas_x_printf(stdout, "%9.2e", &y);

    // (y*Q^T)*Q == y
    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_RIGHT|ARMAS_TRANS, &cf) < 0)
        printf("right.1: error %d\n", cf.error);
    printf("yQ:\n");      armas_x_printf(stdout, "%9.2e", &y);

    if (armas_x_housemult(&y, &tau, &P, flags|ARMAS_RIGHT, &cf) < 0)
        printf("right.2: error %d\n", cf.error);
    printf("yQ*Q^T:\n");      armas_x_printf(stdout, "%9.2e", &y);

    relerr = rel_error(&nrm, &z, &y, ARMAS_NORM_INF, 0, &cf);
    int ok = isFINE(relerr, m*__ERROR);
    char *s = (flags & ARMAS_UNIT) != 0 ? "unit scaled" : "   unscaled";

    printf("%s: (%s) y - (yQ.T)*Q == y\n", PASS(ok), s);
    if (verbose > 0)
        printf("   ||y - (yQ^T)*Q||: %e [%d]\n", relerr, ndigits(relerr));

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&tau);
    armas_d_release(&P);
    armas_d_release(&w);

    return stat;
}


// compute: ||x - A^-1*(A*x)||
int test_hh(armas_d_sparse_t *A, int flags, int verbose, armassp_type_enum kind, int m)
{
    armas_d_sparse_t *C, *Cu;
    armas_d_dense_t x, y, z, Cd, x0, x1, t, tau;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;

    // x and y 
    armas_d_init(&x, m, 1);
    armas_d_init(&y, m, 1);
    armas_d_init(&z, m, 1);
    armas_d_init(&tau, 1, 1);
    armas_d_set_values(&x, unitrand, 0);
    //armas_d_set_values(&y, unitrand, 0);
    
    armas_x_submatrix(&x0, &x, 0, 0, 1, 1);
    armas_x_set(&x0, 0, 0, -armas_x_get(&x0, 0, 0));
    armas_x_submatrix(&x1, &x, 1, 0, m-1, 1);
    armas_d_mcopy(&y, &x);

    armas_x_house(&x0, &x1, &tau,  ARMAS_UNIT|ARMAS_NONNEG, &cf);
    printf("x : "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &x));
    printf("y : "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &y));
    printf("z : "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &z));
    printf("t : "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &tau));

    cf.error = 0;
    if (armas_x_houseapply(&y, &tau, &x1,  __nil, ARMAS_UNIT, &cf) < 0)
        printf("error in apply\n");

    //__hhapply(&y, &x,  __nil, 0, &cf);
    printf("Hy: "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &y));
    //relerr = rel_error(&nrm, &z, &x, ARMAS_NORM_INF, 0, &cf);
    

    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&tau);
    return stat;
}

int hh_mult(int flags, int verbose, int m, int n, int p)
{
    armas_d_dense_t x, y, z, Cd, x0, x1, t, tau, P, w;
    double relerr, nrm;
    armas_conf_t cf = *armas_conf_default();
    int stat = 0;
    // x and y 
    armas_d_init(&x, m, 1);
    armas_d_init(&y, m, 1);
    armas_d_init(&z, m, 1);
    armas_d_init(&tau, n, 1);
    armas_d_init(&P, m, n);
    armas_d_set_values(&P, unitrand, 0);
    //armas_d_set_values(&y, one, 0);
    //armas_d_mcopy(&z, &y);
    

    // make householders
    for (int k = 0; k < n; k++) {
        armas_x_submatrix(&x0, &P, k, k, 1, 1);
        armas_x_submatrix(&x1, &P, k+1, k, m-k-1, 1);
        //armas_x_set(&x0, 0, 0, -armas_x_get(&x0, 0, 0));
        armas_x_submatrix(&t, &tau, k, 0, 1, 1);
        armas_x_house(&x0, &x1, &t,  flags, &cf);
    }

    armas_d_set(&x, p, 0, 1.0);
    printf("x  : "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &x));
    
    armas_x_housemult(&x, &tau, &P, ARMAS_LEFT, &cf);
    printf("Qx : "); armas_x_printf(stdout, "%9.2e", armas_x_col_as_row(&t, &x));
    
    armas_d_release(&x);
    armas_d_release(&y);
    armas_d_release(&z);
    armas_d_release(&tau);
    armas_d_release(&P);
}

int main(int argc, char **argv)
{
    int opt;
    int verbose = 0;
    char *path = (char *)0;
    FILE *fp;
    int tc;
    int m = 6;
    int n = 3;
    int p = 1;
    armas_d_sparse_t *A;

    while ((opt = getopt(argc, argv, "vf:n:m:p:")) != -1) {
        switch (opt) {
        case 'v':
            verbose += 1;
            break;
        case 'f':
            path = optarg;
            break;
        case 'm':
            m = atoi(optarg);
            break;
        case 'n':
            n = atoi(optarg);
            break;
        case 'p':
            p = atoi(optarg);
            break;
        default:
            fprintf(stderr, "usage: tstmul [-v -f path] \n");
            exit(1);
        }
    }
    
    if (path) {
        if (verbose > 0)
            fprintf(stderr, "opening '%s'...\n", path);

        if (! (fp = fopen(path, "r"))) {
            perror(path);
            exit(1);
        }
        A = armassp_d_mmload(&tc, fp);
        if (!A) {
            fprintf(stderr, "reading of '%s' failed\n", path);
            exit(1);      
        }
    }
    //test_hh(A, 0, verbose, ARMASSP_CSR, gmres_m);
    //test_left(A, 0, verbose, ARMASSP_CSR, m, n, p);
    //test_right(0, verbose, m, n, p);
    //test_right(ARMAS_UNIT, verbose, m, n, p);
    //test_left(0, verbose, m, n, p);
    //test_left(ARMAS_UNIT, verbose, m, n, p);
    hh_mult(0, verbose, m, n, p);
    return 0;
}

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
