
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_cgrad_w) && defined(armassp_x_cgrad)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_mvmult_sym)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <float.h>
#include <math.h>
#include <stdio.h>
#include "matrix.h"
#include "sparse.h"

#include "splocal.h"

// References;
// (1) Yousef Saad, Iterative Methods for Sparse Linear System, 2nd Edition

/* 
 * r0 = b - A*x0; z0 = A^T*r0; p0 = z0
 * for i ... until convergense
 *   w_j     = Ap_j
 *   alpha_j = (z_j, z_j)/(w_j, w_j)
 *   x_jp1   = x_j + alpha_j*p_j
 *   r_jp1   = r_j - alpha_j*w_j
 *   z_jp1   = A^T*r_jp1
 *   beta_j   = (z_jp1, z_jp1)/(z_j, z_j)
 *   p_jp1    = z_jp1  + beta_j*p_j
 * endfor
 *
 *  A     = [m,n] m > n
 *  x,z,p = [n,1]
 *  b,r,w = [m,1]
 */

#if 0
static
int __x_cgnr(armas_x_dense_t *x,
             const armas_x_sparse_t *A,
             armas_x_dense_t *b,
             int maxiter,
             DTYPE rstop,
             DTYPE rmult,
             DTYPE *res,
             armas_wbuf_t *W,
             armas_conf_t *cf)
{
    armas_x_dense_t p, Ap, r, z;
    int m = A->rows;
    int n = A->cols;
    DTYPE dot_z, dot_p, alpha, beta, dot_z1;

    // r0 = b - A*x
    DTYPE *t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&r, m, 1, m, t);
    armas_x_mcopy(&r, b);
    armassp_x_mvmult(__ONE, &r, -__ONE, A, x, 0, cf);

    if (rstop == __ZERO)
        rstop = (rmult > __ZERO ? rmult : __SQRT(__EPSILON)) * armas_x_nrm2(&r, cf);

    // z0 = A^T*r0
    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&z, n, 1, n, t);
    armassp_x_mvmult(__ZERO, &z, __ONE, A, &r, ARMAS_TRANS, cf);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&p, n, 1, n, t);
    armas_x_copy(&p, &z, cf);

    // w = Ap
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&Ap, m, 1, m, t);
    
    if (maxiter == 0)
        maxiter = 4*m;
    
    int niter;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_x_mvmult(__ZERO, &Ap, __ONE, A, &p, 0, cf);
        dot_z = armas_x_dot(&z, &z, cf);
        dot_p = armas_x_dot(&Ap, &Ap, cf);
        alpha = dot_z / dot_p;

        // x = x + alpha*p
        armas_x_axpy(x, alpha, &p, cf);
        // r = r - alpha*w = r - alpha*Ap
        armas_x_axpy(&r, -alpha, &Ap, cf);
        // z = A^T*r
        armassp_x_mvmult(__ZERO, &z, __ONE, A, &r, ARMAS_TRANS, cf);

        dot_z1 = armas_x_dot(&z, &z, cf);
        //printf("%3d: dot_z %e, dot_z1 %e, dot_p %e\n", i, dot_z, dot_z1, dot_p);
        if (__SQRT(dot_z1) < rstop) {
            if (res)
                *res = __SQRT(dot_z1);
            break;
        }
        beta = dot_z1 / dot_z;
        // p = beta*p + z;
        armas_x_scale_plus(beta, &p, __ONE, &z, 0, cf);
    }
    return niter;
}
#endif

#define CGNR_WSIZE(m, n)  ((2*(n) + (m))*sizeof(DTYPE))


static inline
int __check_parms(const armas_x_dense_t *x,
                  const armas_x_sparse_t *A, const armas_x_dense_t *b)
{
    if (!A)       return 0;
    if (!armas_x_isvector(x)) return 0;
    if (!armas_x_isvector(b)) return 0;
    if (armas_x_size(b) != A->rows) return 0;
    if (armas_x_size(x) != A->cols) return 0;
    return 1;
}


/**
 * \brief Solve least-squares problem \$ min ||b - Ax||_2
 *
 *
 * For details see (1) 8.1 and 8.3
 */
int armassp_x_cgnr_w(armas_x_dense_t *x,
                     const armas_x_sparse_t *A,
                     const armas_x_dense_t *b,
                     armas_wbuf_t *W,
                     armas_conf_t *cf)
{
    armas_x_dense_t p, Ap, r, z;
    DTYPE dot_z, dot_p, alpha, beta, dot_z1, *t, rstop, rmult;
    int m, n, maxiter, niter;

    if (!cf)
        cf = armas_conf_default();
    
    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (W && W->bytes == 0) {
        // get working size;
        W->bytes = CGNR_WSIZE(A->rows, A->cols);
        return 0;
    }
    if (A->rows == 0 || A->cols == 0)
        return 0;

    if (__check_parms(x, A, b) == 0) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (armas_wbytes(W) < CGNR_WSIZE(A->rows, A->cols)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }

    size_t pos = armas_wpos(W);
    // --------------------------------------------------------------------
    maxiter = cf->maxiter > 0 ? cf->maxiter : 2*A->rows;
    rstop   = cf->stop > 0.0D ? (DTYPE)cf->stop : __ZERO;
    rmult   = cf->smult > 0.0D ? (DTYPE)cf->smult : __EPSILON;

    m = A->rows;
    n = A->cols;

    // r0 = b - A*x
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&r, m, 1, m, t);
    armas_x_mcopy(&r, b);
    armassp_x_mvmult(__ONE, &r, -__ONE, A, x, 0, cf);

    // no absolute stopping criteria; make it epsilon^2*(r0, r0) 
    if (rstop == __ZERO)
        rstop = rmult * (rmult * armas_x_dot(&r, &r, cf));
    else
        // absolute stopping relative to epsilon*nrm2(r0)
        rstop *= rstop; 

    // z0 = A^T*r0
    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&z, n, 1, n, t);
    armassp_x_mvmult(__ZERO, &z, __ONE, A, &r, ARMAS_TRANS, cf);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&p, n, 1, n, t);
    armas_x_copy(&p, &z, cf);

    // w = Ap
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&Ap, m, 1, m, t);    
  
    // --------------------------------------------------------------------
    dot_z1 = __ZERO;   
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_x_mvmult(__ZERO, &Ap, __ONE, A, &p, 0, cf);
        dot_z = armas_x_dot(&z, &z, cf);
        dot_p = armas_x_dot(&Ap, &Ap, cf);
        alpha = dot_z / dot_p;

        // x = x + alpha*p
        armas_x_axpy(x, alpha, &p, cf);
        // r = r - alpha*w = r - alpha*Ap
        armas_x_axpy(&r, -alpha, &Ap, cf);
        // z = A^T*r
        armassp_x_mvmult(__ZERO, &z, __ONE, A, &r, ARMAS_TRANS, cf);

        dot_z1 = armas_x_dot(&z, &z, cf);
        //printf("%3d: dot_z %e, dot_z1 %e, dot_p %e\n", i, dot_z, dot_z1, dot_p);
        if (dot_z1 < rstop) {
            break;
        }
        beta = dot_z1 / dot_z;
        // p = beta*p + z;
        armas_x_scale_plus(beta, &p, __ONE, &z, 0, cf);
    }

    cf->numiters = n;
    cf->residual = (double)__SQRT(dot_z1);

    armas_wsetpos(W, pos);
    return 0;
}

int armassp_x_cgnr(armas_x_dense_t *x,
                   const armas_x_sparse_t *A,
                   const armas_x_dense_t *b,
                   armas_conf_t *cf)
{
    int stat;
    armas_wbuf_t W = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();
    
    if (!__check_parms(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }

    if (A->rows == 0 || A->cols == 0)
        return 0;
    
    if (armassp_x_cgnr_w(x, A, b, &W, cf) < 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }

    if (!armas_walloc(&W, W.bytes)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    stat = armassp_x_cgnr_w(x, A, b, &W, cf);

    armas_wrelease(&W);
    return stat;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End: