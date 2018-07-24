
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_pcgrad_w) && defined(armassp_x_pcgrad)
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
//
// References;
// (1) Yousef Saad, Iterative Methods for Sparse Linear System, 2nd Edition
// (2) Golub, 

/*
 * split preconditioned; Algorithm 9.2 in (1)
 * r0 = b - A*x0; r0 = L^-1*r0; p0 = L^-T*r0
 * for i ... until convergence
 *   alpha_j = (r_j, r_j)/(Ap_j, p_j)
 *   x_jp1   = x_j + alpha_j*p_j
 *   r_jp1   = r_j - alpha_j*L^-1*Ap_j
 *   beta_j   = (r_jp1, z_jp1)/(r_j, z_j)
 *   p_jp1    = L^-T*r_jp1  + beta_j*p_j
 * endfor

 */

#define PCGRAD_WSIZE(n) ((n)*4*sizeof(DTYPE))
#ifndef __EPS2
#define __EPS2 (__EPSILON*__EPSILON)
#endif

/*
 * Preconditioned conjugate gradient [Algorithm 9.1 in (1)]
 *
 * r0 = b - A*x0; z0 = M^-1*r0; p0 = z0
 * for i ... until convergence
 *   alpha_j = (r_j, z_j)/(Ap_j, p_j)
 *   x_jp1   = x_j + alpha_j*p_j
 *   r_jp1   = r_j - alpha_j*Ap_j
 *   z_jp1   = M^-1*r_jp1
 *   beta_j  = (r_jp1, z_jp1)/(r_j, z_j)
 *   p_jp1   = z_jp1  + beta_j*p_j
 * endfor
 */

static inline
int __check_parms(const armas_x_dense_t *x,
                  const armas_x_sparse_t *A, const armas_x_dense_t *b)
{
    if (!A)       return 0;
    if (!armas_x_isvector(x)) return 0;
    if (!armas_x_isvector(b)) return 0;
    if (armas_x_size(x) != armas_x_size(b)) return 0;
    if (A->rows != A->cols) return 0;
    if (armas_x_size(x) != A->rows) return 0;
    return 1;
}


/**
 * \brief Solve x = A^-1*b by conjugate gradient method
 */
int armassp_x_pcgrad_w(armas_x_dense_t *x,
                       const armas_x_sparse_t *A,
                       const armas_x_dense_t *b,
                       armassp_x_precond_t *M,
                       int flags,
                       armas_wbuf_t *W,
                       armas_conf_t *cf)
{
    int m, niter, maxiter;
    DTYPE dot_rz, dot_p, dot_rz1, *t, rstop, alpha, beta, rmult;
    armas_x_dense_t Ap, p, z, r;
    
    if (!cf)
        cf = armas_conf_default();   
    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
        
    m = A->rows;
    if (W && W->bytes == 0) {
        // get working size;
        W->bytes = PCGRAD_WSIZE(m);
        return 0;
    }
    if (__check_parms(x, A, b) == 0) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (armas_wbytes(W) < PCGRAD_WSIZE(m)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    if (A->rows == 0 || A->cols == 0)
        return 0;

    size_t wpos = armas_wpos(W);
    
    // --------------------------------------------------------------------------
    // r0 = b - A*x
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&r, m, 1, m, t);
    armas_x_mcopy(&r, b);
    armassp_x_mvmult_sym(__ONE, &r, -__ONE, A, x, flags, cf);

    rstop   = cf->stop > __ZERO ? cf->stop : __ZERO;
    maxiter = cf->maxiter > 0 ? cf->maxiter : 2*m;
    rmult   = cf->smult > __ZERO ? cf->smult : __EPSILON;
    if (rstop == __ZERO) 
        rstop = rmult * (rmult * armas_x_dot(&r, &r, cf));
    else
        rstop *= rstop;

    // z0 = M^-1*r0
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&z, m, 1, m, t);
    //armas_x_mcopy(&z, &r);
    M->precond(&z, M, &r, cf);
    
    // p0 = z0
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&p, m, 1, m, t);
    armas_x_mcopy(&p, &z);

    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&Ap, m, 1, m, t);

    // --------------------------------------------------------------------------
    dot_rz1 = __ZERO;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_x_mvmult_sym(__ZERO, &Ap, __ONE, A, &p, flags, cf);
        dot_p  = armas_x_dot(&Ap, &p, cf);
        dot_rz = armas_x_dot(&r, &z, cf);
        alpha = dot_rz / dot_p;

        // x = x + alpha*p
        armas_x_axpy(x, alpha, &p, cf);
        // r = r - alpha*Ap
        armas_x_axpy(&r, -alpha, &Ap, cf);
        // z = M^-1*r
        M->precond(&z, M, &r, cf);
        
        dot_rz1 = armas_x_dot(&r, &z, cf);
        if (dot_rz1 < rstop) {
            break;
        }
        beta = dot_rz1 / dot_rz;
        // p = beta*p + z;
        armas_x_scale_plus(beta, &p, __ONE, &z, 0, cf);
    }
    // --------------------------------------------------------------------------

    cf->numiters = niter;
    cf->residual = __SQRT(dot_rz1);

    armas_wsetpos(W, wpos);
    return 0;
}


int armassp_x_pcgrad(armas_x_dense_t *x,
                     const armas_x_sparse_t *A,
                     const armas_x_dense_t *b,
                     armassp_x_precond_t *M,
                     int flags,
                     armas_conf_t *cf)
{
    int stat;
    armas_wbuf_t W = ARMAS_WBNULL;

    if (!cf) {
        cf = armas_conf_default();
    }
    if (!__check_parms(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    if (A->rows == 0 || A->cols == 0) 
        return 0;
    
    // get working size
    if (armassp_x_pcgrad_w(x, A, b, M, flags, &W, cf) < 0) {
        return -1;
    }    
    if (!armas_walloc(&W, W.bytes)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    stat = armassp_x_pcgrad_w(x, A, b, M, flags, &W, cf);
    armas_wrelease(&W);
    return stat;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
