
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_cgne_w) && defined(armassp_cgne)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_mvmult)
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
int check_parms(const armas_dense_t * x,
                const armas_sparse_t * A, const armas_dense_t * b)
{
    if (!A)
        return 0;
    if (!armas_isvector(x))
        return 0;
    if (!armas_isvector(b))
        return 0;
    if (armas_size(b) != A->rows)
        return 0;
    if (armas_size(x) != A->cols)
        return 0;
    return 1;
}


/**
 * @brief Solve under-determined system \f$ Ax = b \f$
 *
 * System
 *   \f$   x = A^Tu, AA^Tu = b \f$
 *
 * of equations can be used to solve under-determined systems, i.e.,
 * those systems  involving rectangular matrices of size m × n, with m < n.
 * Assume that m ≤ n and that A has full rank. If x_∗ is some solution to
 * the underdetermined system Ax = b. Then \f$ AA^Tu = b \f$ represents
 * the normal equations for the least-squares problem
 *
 *   \f$ minimize {\Vert {x_* - A^T u} \Vert}_2 \f$
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
 * @param[in,out] wb
 *   Workspace. If *W.bytes* is zero then size of required workspace is calculated
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
int armassp_cgne_w(armas_dense_t * x,
                     const armas_sparse_t * A,
                     const armas_dense_t * b,
                     armas_wbuf_t * wb, armas_conf_t * cf)
{
    DTYPE dot_r, dot_p, alpha, beta, dot_r1, *t, rstop, rmult;
    armas_dense_t p, Ap, r;
    int niter, m, n, maxiter;

    if (!cf)
        cf = armas_conf_default();

    if (!A) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    if (wb && wb->bytes == 0) {
        // get working size;
        wb->bytes = CGNE_WSIZE(A->rows, A->cols);
        return 0;
    }

    if (A->rows == 0 || A->cols == 0)
        return 0;

    if (check_parms(x, A, b) == 0) {
        cf->error = ARMAS_EINVAL;
        return -ARMAS_EINVAL;
    }

    if (armas_wbytes(wb) < CGNE_WSIZE(A->rows, A->cols)) {
        cf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    // -------------------------------------------------------------------------
    size_t pos = armas_wpos(wb);

    maxiter = cf->maxiter > 0 ? cf->maxiter : 2 * A->rows;
    rstop = cf->stop != ZERO ? (DTYPE) cf->stop : ZERO;
    rmult = cf->smult != ZERO ? (DTYPE) cf->smult : EPS;

    m = A->rows;
    n = A->cols;

    // r0 = b - A*x
    t = armas_wreserve(wb, m, sizeof(DTYPE));
    armas_make(&r, m, 1, m, t);
    armas_mcopy(&r, b, 0, cf);
    armassp_mvmult(ONE, &r, -ONE, A, x, 0, cf);

    if (rstop == ZERO)
        rstop = rmult * (rmult * armas_dot(&r, &r, cf));
    else
        rstop *= rstop;

    // p0 = A^T*r0
    t = armas_wreserve(wb, n, sizeof(DTYPE));
    armas_make(&p, n, 1, n, t);
    armassp_mvmult(ZERO, &p, ONE, A, &r, ARMAS_TRANS, cf);

    // Ap
    t = armas_wreserve(wb, m, sizeof(DTYPE));
    armas_make(&Ap, m, 1, m, t);

    // -------------------------------------------------------------------------
    dot_r1 = ZERO;
    for (niter = 0; niter < maxiter; niter++) {
        // Ap = A*p
        armassp_mvmult(ZERO, &Ap, ONE, A, &p, 0, cf);
        dot_r = armas_dot(&r, &r, cf);
        dot_p = armas_dot(&p, &p, cf);
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
        // p = beta*p + A^T*r;
        armassp_mvmult(beta, &p, ONE, A, &r, ARMAS_TRANS, cf);
    }

    // -------------------------------------------------------------------------
    cf->numiters = niter;
    cf->residual = (double) SQRT(dot_r1);

    armas_wsetpos(wb, pos);
    return niter >= maxiter ? -ARMAS_ECONVERGE : 0;
}

/**
 * @brief Solve under-determined system \f$ Ax = b \f$
 * @see armassp_cgne_w
 * @ingroup sparse
 */
int armassp_cgne(armas_dense_t * x,
                   const armas_sparse_t * A,
                   const armas_dense_t * b, armas_conf_t * cf)
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

    if (armassp_cgne_w(x, A, b, &W, cf) < 0) {
        cf->error = ARMAS_EWORK;
        return -1;
    }

    if (!armas_walloc(&W, W.bytes)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    stat = armassp_cgne_w(x, A, b, &W, cf);

    armas_wrelease(&W);
    return stat;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
