
// Copyright (c) Harri Rautila, 2013-2020

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Bidiagonal SVD

#include "dtype.h"
#include "dlpack.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_bdsvd) && defined(armas_x_bdsvd_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_gvcompute) && defined(armas_x_gvupdate) \
    && defined(armas_x_bdsvd2x2_vec)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
#include "auxiliary.h"
//! \endcond

/*
 * \brief Rotate lower bidiagonal matrix to upper bidiagonal matrix.
 */
static
void bdmake_upper(armas_x_dense_t * D, armas_x_dense_t * E,
                  armas_x_dense_t * U, armas_x_dense_t * C,
                  armas_x_dense_t * CS)
{
    armas_x_dense_t Cl, Sl;
    DTYPE cosl, sinl, r, d0, e0, d1;
    int saves, k, N = armas_x_size(D);

    saves = 0;
    if (U || C) {
        armas_x_subvector(&Cl, CS, 0, N);
        armas_x_subvector(&Sl, CS, N, N);
        saves = 1;
    }
    d0 = armas_x_get_at_unsafe(D, 0);
    for (k = 0; k < N - 1; k++) {
        e0 = armas_x_get_at_unsafe(E, k);
        d1 = armas_x_get_at_unsafe(D, k + 1);
        armas_x_gvcompute(&cosl, &sinl, &r, d0, e0);
        armas_x_set_at_unsafe(D, k, r);
        armas_x_set_at_unsafe(E, k, sinl * d1);
        d0 = cosl * d1;
        armas_x_set_at_unsafe(D, k + 1, d0);
        if (saves) {
            armas_x_set_at_unsafe(&Cl, k, cosl);
            armas_x_set_at_unsafe(&Sl, k, sinl);
        }
    }

    if (U && k > 0) {
        armas_x_gvupdate(U, 0, &Cl, &Sl, N - 1, ARMAS_RIGHT);
    }
    if (C && k > 0) {
        armas_x_gvupdate(C, 0, &Cl, &Sl, N - 1, ARMAS_LEFT);
    }
}

/**
 * \brief Compute SVD of bidiagonal matrix.
 *
 * Computes the singular values and, optionially, the left and/or right
 * singular vectors from the SVD of a N-by-N upper or lower bidiagonal
 * matrix. The SVD of B has the form
 *
 *   \f$ B = U S V^T \f$
 *
 * where S is the diagonal matrix with singular values, U is an orthogonal
 * matrix of left singular vectors, and \f$ V^T \f$ is an orthogonal matrix of
 * right singular vectors. If singular vectors are requested they must be
 * initialized either to unit diagonal matrix or some other orthogonal matrices.
 *
 * \param[in,out] D
 *      On entry, the diagonal elements of B. On exit, the singular values
 *      of B in decreasing order.
 * \param[in] E
 *      On entry, the offdiagonal elements of B. On exit, E is destroyed.
 * \param[in,out] U
 *      On entry, initial orthogonal matrix of left singular vectors. On exit,
 *      updated left singular vectors.
 * \param[in,out] V
 *      On entry, initial orthogonal matrix of right singular vectors. On exit,
 *      updated right singular vectors.
 * \param[in] flags
 *      Indicators, *ARMAS_WANTU*, *ARMAS_WANTV*. Use *ARMAS_FORWARD* to force
 *      implicit QR-iteration only in forward direction from top to bottom.
 * \param[in,out] conf
 *      Configuration block.
 *
 * Singular values are computed with Demmel-Kahan implicit QR algorithm to
 * high relative accuracy. Tolerance used is conf.tolmult*EPSILON. If absolute
 * tolerance is needed, conf.optflags bit ARMAS_ABSTOL flag must be set.
 *
 * Corresponds to lapack.xBDSQR
 * \ingroup lapack
 */
int armas_x_bdsvd(armas_x_dense_t * D, armas_x_dense_t * E,
                  armas_x_dense_t * U, armas_x_dense_t * V,
                  int flags, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_bdsvd_w(D, E, U, V, flags, &wb, conf) < 0)
        return -1;

    wbs = &wb;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else {
        wbs = ARMAS_NOWORK;
    }

    err = armas_x_bdsvd_w(D, E, U, V, flags, wbs, conf);
    armas_wrelease(&wb);

    return err;
}

/**
 * @brief Workspace to compute SVD of bidiagonal or bidiagonalizable matrix S.
 * @ingroup lapack
 */
int armas_x_bdsvd_work(armas_x_dense_t * S, armas_conf_t * conf)
{
    if (armas_x_isvector(S)) {
        return 4 * armas_x_size(S);
    }
    return 4 * (S->rows < S->cols ? S->rows : S->cols);
}

/**
 * @brief Compute SVD of bidiagonal matrix.
 *
 * Computes the singular values and, optionially, the left and/or right
 * singular vectors from the SVD of a N-by-N upper or lower bidiagonal
 * matrix. The SVD of B has the form
 *
 *   \f$ B = U S V^T \f$
 *
 * where S is the diagonal matrix with singular values, U is an orthogonal
 * matrix of left singular vectors, and \f$ V^T \f$ is an orthogonal matrix of
 * right singular vectors. If singular vectors are requested they must be
 * initialized either to unit diagonal matrix or some other orthogonal matrices.
 *
 * @param[in,out] D
 *      On entry, the diagonal elements of B. On exit, the singular values
 *      of B in decreasing order.
 * @param[in] E
 *      On entry, the offdiagonal elements of B. On exit, E is destroyed.
 * @param[in,out] U
 *      On entry, initial orthogonal matrix of left singular vectors. On exit,
 *      updated left singular vectors.
 * @param[in,out] V
 *      On entry, initial orthogonal matrix of right singular vectors. On exit,
 *      updated right singular vectors.
 * @param[in] flags
 *      Indicators, *ARMAS_WANTU*, *ARMAS_WANTV*. Use *ARMAS_FORWARD* to force
 *      implicit QR-iteration only in forward direction from top to bottom.
 * @param[out] W
 *      Workspace of size 4*N elements if eigenvectors needed.
 * @param[in,out] conf
 *      Configuration block.
 *
 * Singular values are computed with Demmel-Kahan implicit QR algorithm to
 * high relative accuracy. Tolerance used is conf.tolmult*EPSILON. If absolute
 * tolerance is needed, conf.optflags bit ARMAS_ABSTOL flag must be set.
 *
 * Corresponds to lapack.xBDSQR
 * \ingroup lapack
 */
int armas_x_bdsvd_w(armas_x_dense_t * D,
                    armas_x_dense_t * E,
                    armas_x_dense_t * U,
                    armas_x_dense_t * V,
                    int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t CS, *uu, *vv;
    int uuvv, err, N = armas_x_size(D);
    ABSTYPE tol = 8.0;

    if (!conf)
        conf = armas_conf_default();

    if (!D) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    uuvv = (flags & (ARMAS_WANTU | ARMAS_WANTV)) != 0;
    if (wb && wb->bytes == 0) {
        if (uuvv)
            wb->bytes = 4 * N * sizeof(DTYPE);
        return 0;
    }

    uu = (armas_x_dense_t *) 0;
    vv = (armas_x_dense_t *) 0;
    // check for sizes
    if (!(armas_x_isvector(D) && armas_x_isvector(E))) {
        conf->error = ARMAS_ENEED_VECTOR;
        return -1;
    }
    if (flags & ARMAS_WANTU) {
        if (!U) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        // U columns need to be at least N
        if (U->cols < N) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        uu = U;
    }
    if (flags & ARMAS_WANTV) {
        if (!V) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        // V rows need to be at least N
        if (V->rows < N) {
            conf->error = ARMAS_ESIZE;
            return -1;
        }
        vv = V;
    }
    if (armas_x_size(E) != N - 1) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if ((uu || vv) && armas_wbytes(wb) < 4 * N * sizeof(DTYPE)) {
        // if eigenvectors needed then must have workspace
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (uu || vv) {
        armas_x_make(&CS, 4 * N, 1, 4 * N, (DTYPE *) armas_wptr(wb));
    } else {
        armas_x_make(&CS, 0, 0, 1, (DTYPE *) 0);
    }
    if (flags & ARMAS_LOWER) {
        // rotate to UPPER bidiagonal
        bdmake_upper(D, E, uu, __nil, &CS);
    }

    tol = tol * EPS;
    if (conf->tolmult != ZERO) {
        tol = ((ABSTYPE) conf->tolmult) * EPS;
    }
    if (conf->optflags & ARMAS_OBSVD_GOLUB) {
        err = armas_x_bdsvd_golub(D, E, uu, vv, &CS, tol, conf);
    } else {
        err = armas_x_bdsvd_demmel(D, E, uu, vv, &CS, tol, flags, conf);
    }
    if (err == 0) {
        armas_x_sort_eigenvec(D, uu, vv, __nil, -1);
    } else {
        conf->error = ARMAS_ECONVERGE;
    }
    return err;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
