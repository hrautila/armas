
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_cgrad_w) && defined(armassp_cgrad)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_mvmult_sym)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "matrix.h"
#include "sparse.h"
#include "splocal.h"

// References;
// (1) Yousef Saad, Iterative Methods for Sparse Linear System, 2nd Edition
// (2) Golub, 

/*
 * r0 = b - A*x0; p0 = r0
 * for i ... until convergense
 *   alpha_j = (r_j, r_j)/(Ap_j, p_j)
 *   x_jp1   = x_j + alpha_j*p_j
 *   r_jp1   = r_j - alpha_j*Ap_j
 *   beta_j   = (r_jp1, r_jp1)/(r_j, r_j)
 *   p_jp1    = r_jp1  + beta_j*p_j
 * endfor
 *
 */

#define CGRAD_WSIZE(n)  ((n)*3*sizeof(DTYPE))

static inline
int check_parms(const armas_dense_t * x,
                const armas_sparse_t * A, const armas_dense_t * b)
{
    if (!A)
        return 0;
    if (!armas_isvector(x))
        return 0;
    if (!armas_isvector(b))
        return 0;
    if (armas_size(x) != armas_size(b))
        return 0;
    if (A->rows != A->cols)
        return 0;
    if (armas_size(x) != A->rows)
        return 0;
    return 1;
}

/**
 * @brief Solve  \f$ x = A^-1*b \f$ with conjugate gradient method.
 *
 * @param [in,out] x
 *    On entry, initial value of x. On exit solution to linear system.
 * @param [in] A
 *    Sparse symmetric and positive semi-definite matrix in CSR or CSC storage
 *    format. Only lower or upper triangular elements are access.
 * @param [in] b
 *    Initial vector b
 * @param [in] flags
 *    Indicator bits. If ARMAS_LOWER (ARMAS_UPPER) is set then A is a lowet
 *    (upper) triangular matrix.
 * @param [in] W
 *    Workspace for intermediate results. If *W.bytes* is zero then workspace
 *    size is computed and control returned immediately to caller.
 * @param [in,out] cf
 *    Configuration parameters. See below for discussion.
 *
 * Solves a symmetric positive definite system with conjugate gradient method.
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
 * The maximum iteration count is *cf.maxiter* or \f$ 4 N(A) \f$ where N(A) is
 * dimension of A matrix.
 *
 * On exit *cf.numiters* holds the number of iterations and *cf.residual* holds
 * the final residual error.
 *
 * @retval  0  Success
 * @retval <0  Failure
 *
 * @ingroup sparse
 */
int armassp_cgrad_w(armas_dense_t * x,
                      const armas_sparse_t * A,
                      const armas_dense_t * b,
                      int flags, armas_wbuf_t * W, armas_conf_t * cf)
{
    int maxiter, niter, m;
    DTYPE dot_r, dot_p, alpha, beta, dot_r1, *t, rstop, rmult;
    armas_dense_t p, Ap, r;

    if (!cf)
        cf = armas_conf_default();

    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (W && W->bytes == 0) {
        // get working size;
        W->bytes = CGRAD_WSIZE(A->rows);
        return 0;
    }
    if (A->rows == 0 || A->cols == 0)
        return 0;

    if (check_parms(x, A, b) == 0) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (armas_wbytes(W) < CGRAD_WSIZE(A->rows)) {
        cf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    // -------------------------------------------------------------------------
    size_t pos = armas_wpos(W);

    maxiter = cf->maxiter > 0 ? cf->maxiter : 4 * A->rows;
    rstop = cf->stop > ZERO ? (DTYPE) cf->stop : ZERO;
    rmult = cf->smult > ZERO ? (DTYPE) cf->smult : EPS;

    m = A->rows;

    // r0 = b - A*x
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&r, m, 1, m, t);
    armas_mcopy(&r, b, 0, cf);
    armassp_mvmult_sym(ONE, &r, -ONE, A, x, flags, cf);

    if (rstop == ZERO)
        rstop = rmult * (rmult * armas_dot(&r, &r, cf));
    else
        rstop *= rstop;

    // p0 = r0
    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&p, m, 1, m, t);
    armas_mcopy(&p, &r, 0, cf);

    t = armas_wreserve(W, m, sizeof(DTYPE));
    armas_make(&Ap, m, 1, m, t);

    // -------------------------------------------------------------------------
    dot_r1 = ZERO;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_mvmult_sym(ZERO, &Ap, ONE, A, &p, flags, cf);
        dot_r = armas_dot(&r, &r, cf);
        dot_p = armas_dot(&Ap, &p, cf);
        alpha = dot_r / dot_p;

        // x = x + alpha*p
        armas_axpy(x, alpha, &p, cf);
        // r = r - alpha*Ap
        armas_axpy(&r, -alpha, &Ap, cf);

        dot_r1 = armas_dot(&r, &r, cf);
        if (dot_r1 < rstop) {
            break;
        }
        beta = dot_r1 / dot_r;
        // p = beta*p + r;
        armas_axpby(beta, &p, ONE, &r, cf);
    }

    // -------------------------------------------------------------------------
    cf->numiters = niter;
    cf->residual = (double) SQRT(dot_r1);

    armas_wsetpos(W, pos);
    return niter >= maxiter ? -ARMAS_ECONVERGE : 0;
}

/**
 * @brief Solve PSD system \$ x = A^-1*b \$ with conjugate gradient method.
 *
 * @see armassp_cgrad_w
 * @ingroup sparse
 */
int armassp_cgrad(armas_dense_t * x,
                    const armas_sparse_t * A,
                    const armas_dense_t * b, int flags, armas_conf_t * cf)
{
    int stat;
    armas_wbuf_t W = ARMAS_WBNULL;

    if (!cf)
        cf = armas_conf_default();

    if (!check_parms(x, A, b)) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }
    if (A->rows == 0 || A->cols == 0)
        return 0;

    if ((stat = armassp_cgrad_w(x, A, b, flags, &W, cf)) < 0) {
        return stat;
    }
    if (!armas_walloc(&W, W.bytes)) {
        cf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    stat = armassp_cgrad_w(x, A, b, flags, &W, cf);

    armas_wrelease(&W);
    return stat;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
