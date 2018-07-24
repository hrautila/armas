
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_cgne_w) && defined(armassp_x_cgne)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_mvmult)
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
 * Saad (1) algorithm 8.5 CGNE (Craig's Method)
 *
 * r0 = b - A*x0; p0 = A^T*r0;
 * for i ... until convergense
 *   alpha_j = (r_j, r_j)/(p_j, p_j)
 *   x_jp1   = x_j + alpha_j*p_j
 *   r_jp1   = r_j - alpha_j*Ap_j
 *   beta_j   = (r_jp1, r_jp1)/(r_j, r_j)
 *   p_jp1    = A^T*r_jp1  + beta_j*p_j
 * endfor
 *
 *  A      = [m,n] m < n
 *  x,p    = [n,1]
 *  b,r,Ap = [m,1]
 */


#define CGNE_WSIZE(m, n)  ((2*(m) + (n))*sizeof(DTYPE))


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
 * \brief Solve under-determined system Ax = b
 *
 * System
 *   \$   x = A^Tu, AA^Tu = b \$
 *
 * of equations can be used to solve under-determined systems, i.e., those systems 
 * involving rectangular matrices of size m × n, with m < n. Assume that m ≤ n and that A has full rank. 
 * If x_∗ is some solution to the underdetermined system Ax = b. Then \$ AA^Tu = b \$ represents 
 * the normal equations for the least-squares problem
 *
 *    minimize || x_* - A^Tu ||_2
 *
 * For details see (1) 8.1 and 8.3
 */
int armassp_x_cgne_w(armas_x_dense_t *x,
                     const armas_x_sparse_t *A,
                     const armas_x_dense_t *b,
                     armas_wbuf_t *W,
                     armas_conf_t *cf)
{
    DTYPE dot_r, dot_p, alpha, beta, dot_r1, *t, rstop, rmult;
    armas_x_dense_t p, Ap, r;
    int niter, m, n, maxiter;

    if (!cf)
        cf = armas_conf_default();
    
    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }
    
    if (W && W->bytes == 0) {
        // get working size;
        W->bytes = CGNE_WSIZE(A->rows, A->cols);
        return 0;
    }

    if (A->rows == 0 || A->cols == 0)
        return 0;

    if (__check_parms(x, A, b) == 0) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }

    if (armas_wbytes(W) < CGNE_WSIZE(A->rows, A->cols)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }

    // ------------------------------------------------------------------------------
    size_t pos = armas_wpos(W);

    maxiter = cf->maxiter > 0 ? cf->maxiter : 2*A->rows;
    rstop   = cf->stop != 0.0D ? (DTYPE)cf->stop : __ZERO;
    rmult   = cf->smult != 0.0D ? (DTYPE)cf->smult : __EPSILON;

    m = A->rows; n = A->cols;

    // r0 = b - A*x
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&r, m, 1, m, t);
    armas_x_mcopy(&r, b);
    armassp_x_mvmult(__ONE, &r, -__ONE, A, x, 0, cf);

    if (rstop == __ZERO)
        rstop = rmult * (rmult * armas_x_dot(&r, &r, cf));
    else
        rstop *= rstop;

    // p0 = A^T*r0
    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&p, n, 1, n, t);
    armassp_x_mvmult(__ZERO, &p, __ONE, A, &r, ARMAS_TRANS, cf);

    // Ap
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&Ap, m, 1, m, t);
    
    // ------------------------------------------------------------------------------
    dot_r1 = __ZERO;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_x_mvmult(__ZERO, &Ap, __ONE, A, &p, 0, cf);
        dot_r = armas_x_dot(&r, &r, cf);
        dot_p = armas_x_dot(&p, &p, cf);
        alpha = dot_r / dot_p;

        // x = x + alpha*p
        armas_x_axpy(x, alpha, &p, cf);
        // r = r - alpha*Ap
        armas_x_axpy(&r, -alpha, &Ap, cf);

        dot_r1 = armas_x_dot(&r, &r, cf);
        if (dot_r1 < rstop) {
            break;
        }
        beta = dot_r1 / dot_r;
        // p = beta*p + A^T*r;
        armassp_x_mvmult(beta, &p, __ONE, A, &r, ARMAS_TRANS, cf);
    }

    // ------------------------------------------------------------------------------
    cf->numiters = niter;
    cf->residual = (double)__SQRT(dot_r1);

    armas_wsetpos(W, pos);
    return 0;
}

int armassp_x_cgne(armas_x_dense_t *x,
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
    
    if (armassp_x_cgne_w(x, A, b, &W, cf) < 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }

    if (!armas_walloc(&W, W.bytes)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    stat = armassp_x_cgne_w(x, A, b, &W, cf);

    armas_wrelease(&W);
    return stat;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
