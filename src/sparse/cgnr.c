
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_cgrad_w) && defined(armassp_x_cgrad)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_mvmult_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
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

#define CGNR_WSIZE(m, n)  ((2*(n) + (m))*sizeof(DTYPE))

static inline
int check_parms(const armas_x_dense_t * x,
                const armas_x_sparse_t * A, const armas_x_dense_t * b)
{
    if (!A)
        return 0;
    if (!armas_x_isvector(x))
        return 0;
    if (!armas_x_isvector(b))
        return 0;
    if (armas_x_size(b) != A->rows)
        return 0;
    if (armas_x_size(x) != A->cols)
        return 0;
    return 1;
}

/**
 * @brief Solve least-squares problem \f$ min {\Vert {b - Ax}}_2 \f$
 *
 * For details see: Yousef Saad, *Iterative Methods for Sparse Linear System*, 2nd Edition
 * sections 8.1 and 8.3
 *
 * @param[out] x
 *   On exit solution to the system.
 * @param[in]  A
 *   Sparse matrix.
 * @param[in]  b
 *   Dense vector.
 * @param[in,out] W
 *   Workspace. If *wb.bytes* is zero then size of required workspace is calculated
 *   and returned immediately.
 * @param[in,out] cf
 *   Configuration block. On exit *cf.numiters* holds number of iterations and
 *   *cf.residual* the final error residual.
 *
 * Interation stops when maximum iteration count is reseached or residual
 * error goes below stopping criteria. Absolute stopping criteria is used
 * if *cf.stop* is non-zero positive number otherwise relative error is used.
 * If *cf.smult* is non-zero positive number the stopping criteria is
 * \f$ smult * {\Vert r_0 \Vert}_2 \f$ otherwise value
 * \f$ \epsilon {\Vert r_0 \Vert}_2 \f$ is used.
 *
 * Residual on iterattion k is \f$ r_k = b - A x_k \f$ and residual error
 * \f$ \sqrt {\Vert r_0 \Vert}_2 \f$.
 *
 * The maximum iteration count is *cf.maxiter* or \f$ 2 M(A) \f$ where M(A) is
 * row count of A matrix.
 *
 * On exit *cf.numiters* holds the number of iterations and *cf.residual* holds
 * the final residual error.
 *
 * @retval  0  Succes
 * @retval <0  Failure
 * @ingroup sparse
 */
int armassp_x_cgnr_w(armas_x_dense_t * x,
                     const armas_x_sparse_t * A,
                     const armas_x_dense_t * b,
                     armas_wbuf_t * W, armas_conf_t * cf)
{
    armas_x_dense_t p, Ap, r, z;
    DTYPE dot_z, dot_p, alpha, beta, dot_z1, *t, rstop, rmult;
    int m, n, maxiter, niter;

    if (!cf)
        cf = armas_conf_default();

    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (W && W->bytes == 0) {
        // get working size;
        W->bytes = CGNR_WSIZE(A->rows, A->cols);
        return 0;
    }
    if (A->rows == 0 || A->cols == 0)
        return 0;

    if (check_parms(x, A, b) == 0) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (armas_wbytes(W) < CGNR_WSIZE(A->rows, A->cols)) {
        cf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }

    size_t pos = armas_wpos(W);
    // --------------------------------------------------------------------
    maxiter = cf->maxiter > 0 ? cf->maxiter : 2 * A->rows;
    rstop = cf->stop > ZERO ? (DTYPE) cf->stop : ZERO;
    rmult = cf->smult > ZERO ? (DTYPE) cf->smult : EPS;

    m = A->rows;
    n = A->cols;

    // r0 = b - A*x
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&r, m, 1, m, t);
    armas_x_mcopy(&r, b, 0, cf);
    armassp_x_mvmult(ONE, &r, -ONE, A, x, 0, cf);

    // no absolute stopping criteria; make it epsilon^2*(r0, r0) 
    if (rstop == ZERO)
        rstop = rmult * (rmult * armas_x_dot(&r, &r, cf));
    else
        // absolute stopping relative to epsilon*nrm2(r0)
        rstop *= rstop;

    // z0 = A^T*r0
    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&z, n, 1, n, t);
    armassp_x_mvmult(ZERO, &z, ONE, A, &r, ARMAS_TRANS, cf);

    t = armas_wreserve(W, n, sizeof(DTYPE));
    armas_x_make(&p, n, 1, n, t);
    armas_x_copy(&p, &z, cf);

    // w = Ap
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_x_make(&Ap, m, 1, m, t);

    // --------------------------------------------------------------------
    dot_z1 = ZERO;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_x_mvmult(ZERO, &Ap, ONE, A, &p, 0, cf);
        dot_z = armas_x_dot(&z, &z, cf);
        dot_p = armas_x_dot(&Ap, &Ap, cf);
        alpha = dot_z / dot_p;

        // x = x + alpha*p
        armas_x_axpy(x, alpha, &p, cf);
        // r = r - alpha*w = r - alpha*Ap
        armas_x_axpy(&r, -alpha, &Ap, cf);
        // z = A^T*r
        armassp_x_mvmult(ZERO, &z, ONE, A, &r, ARMAS_TRANS, cf);

        dot_z1 = armas_x_dot(&z, &z, cf);
        if (dot_z1 < rstop) {
            break;
        }
        beta = dot_z1 / dot_z;
        // p = beta*p + z;
        armas_x_axpby(beta, &p, ONE, &z, cf);
    }

    cf->numiters = niter;
    cf->residual = (double) SQRT(dot_z1);

    armas_wsetpos(W, pos);
    return niter >= maxiter ? -ARMAS_ECONVERGE : 0;
}

/**
 * @brief Solve least-squares problem \f$ min ||b - Ax||_2 \f$
 * @see armassp_x_cgnr_w
 * @ingroup sparse
 */
int armassp_x_cgnr(armas_x_dense_t * x,
                   const armas_x_sparse_t * A,
                   const armas_x_dense_t * b, armas_conf_t * cf)
{
    int stat;
    armas_wbuf_t W = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!check_parms(x, A, b)) {
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
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
