
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_eigen_sym)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_trdeigen) && defined(armas_x_trdreduce) && defined(armas_x_trdbuild)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "matrix.h"
#include "internal.h"
#include "internal_lapack.h"
//! \endcond

#ifndef ARMAS_NIL
#define ARMAS_NIL (armas_x_dense_t *)0
#endif

/*
 * \brief Eigenvalues and vectors of 2x2 matrix
 */
static
int eigen_sym_small(armas_x_dense_t * D, armas_x_dense_t * A,
                    armas_x_dense_t * W, int flags, armas_conf_t * conf)
{
    DTYPE d0, b0, d1, e0, e1, cs, sn;

    d0 = armas_x_get_unsafe(A, 0, 0);
    d1 = armas_x_get_unsafe(A, 1, 1);
    if (flags & ARMAS_UPPER) {
        b0 = armas_x_get_unsafe(A, 0, 1);
    } else {
        b0 = armas_x_get_unsafe(A, 1, 0);
    }
    armas_x_sym_eigen2x2vec(&e0, &e1, &cs, &sn, d0, b0, d1);
    armas_x_set_at_unsafe(D, 0, e0);
    armas_x_set_at_unsafe(D, 1, e1);

    if (flags & ARMAS_WANTV) {
        armas_x_set_unsafe(A, 0, 0, ONE);
        armas_x_set_unsafe(A, 1, 1, ONE);
        armas_x_set_unsafe(A, 0, 1, ZERO);
        armas_x_set_unsafe(A, 1, 0, ZERO);
        armas_x_gvright(A, cs, sn, 0, 1, 0, A->rows);
    }
    return 0;
}

/**
 * \brief Compute eigenvalue decomposition of symmetric N-by-N matrix
 *
 * Compute eigenvalue decomposition of symmetric N-by-N matrix
 *
 * \param[out] D
 *      Eigenvalues of A in increasing order
 * \param[in,out] A
 *      On entry, symmetric matrix stored in lower or upper triangular part.
 *      On exit, eigenvector if requested, otherwise contents are destroyd.
 * \param[in] flags
 *      Flag bits, set ARMAS_UPPER (ARMAS_LOWER) if upper (lower) triangular storage
 *      is used. If eigenvectors wanted, set ARMAS_WANTV.
 * \param[in] conf
 *      Optional configuration block, if NULL then default configuration used. 
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 *
 * \ingroup lapack
 */
int armas_x_eigen_sym(armas_x_dense_t * D,
                      armas_x_dense_t * A, int flags, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(A) == 0)
        return 0;

    wbs = &wb;
    if (armas_x_eigen_sym_w(D, A, flags, &wb, conf) < 0)
        return -1;

    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;

    err = armas_x_eigen_sym_w(D, A, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;
}


/**
 * \brief Compute eigenvalue decomposition of symmetric N-by-N matrix
 *
 * Compute eigenvalue decomposition of symmetric N-by-N matrix
 *
 * \param[out] D
 *      On exit, eigenvalues of A in increasing order
 * \param[in,out] A
 *      On entry, symmetric matrix stored in lower or upper triangular part.
 *      On exit, eigenvectors of A if requested, otherwise contents are destroyd.
 * \param[in] flags
 *      Flag bits, set ARMAS_UPPER (ARMAS_LOWER) if upper (lower) triangular storage
 *      is used. If eigenvectors wanted, set ARMAS_WANTV.
 * \param[in] wb
 *      Workspace, size at least 3*N
 * \param[in] conf
 *      Optional configuration block, if NULL then default configuration used. 
 *
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 *
 * \ingroup lapack
 */
int armas_x_eigen_sym_w(armas_x_dense_t * D,
                        armas_x_dense_t * A,
                        int flags, armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t sD, sE, E, tau, *vv;
    size_t wsz, wpos, wpt;
    DTYPE *buf;
    int ioff, N;
    vv = __nil;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    N = A->rows;

    if (wb && wb->bytes == 0) {
        if (N > 2) {
            if (armas_x_trdreduce_w(A, ARMAS_NIL, flags, wb, conf) < 0)
                return -1;
            wb->bytes += 2 * N * sizeof(DTYPE);
        }
        return 0;
    }

    if (A->rows != A->cols && armas_x_size(D) != A->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (N == 1) {
        armas_x_set_at_unsafe(D, 0, armas_x_get_unsafe(A, 0, 0));
        if (flags & ARMAS_WANTV) {
            armas_x_set_unsafe(A, 0, 0, ONE);
        }
        return 0;
    }
    if (N == 2) {
        eigen_sym_small(D, A, ARMAS_NIL, flags, conf);
        return 0;
    }

    if (!wb || (wsz = armas_wbytes(wb)) < 3 * N * sizeof(DTYPE)) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // default to lower triangular storage
    if (!(flags & (ARMAS_UPPER | ARMAS_LOWER)))
        flags |= ARMAS_LOWER;

    wsz /= sizeof(DTYPE);
    wpos = armas_wpos(wb);

    ioff = flags & ARMAS_LOWER ? -1 : 1;
    // sub/super diagonal reservation
    buf = (DTYPE *) armas_wreserve(wb, N - 1, sizeof(DTYPE));
    armas_x_make(&E, N - 1, 1, N - 1, buf);

    // tau vector reservation; save work buffer position
    wpt = armas_wpos(wb);
    buf = (DTYPE *) armas_wreserve(wb, N, sizeof(DTYPE));
    armas_x_make(&tau, N, 1, N, buf);

    // reduce to tridiagonal form
    if (armas_x_trdreduce_w(A, &tau, flags, wb, conf) != 0)
        return -2;

    // copy diagonals
    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, A, ioff);
    armas_x_copy(D, &sD, conf);
    armas_x_copy(&E, &sE, conf);

    // if vectors required, build in A
    if (flags & ARMAS_WANTV) {
        if (armas_x_trdbuild_w(A, &tau, N, flags, wb, conf) != 0)
            return -3;
        vv = A;
    }
    // reset reservations workspace
    armas_wsetpos(wb, wpt);

    // compute eigenvalues/vectors of tridiagonal matrix
    if (armas_x_trdeigen_w(D, &E, vv, flags, wb, conf) != 0)
        return -4;

    armas_wsetpos(wb, wpos);
    return 0;
}

/**
 * \brief Compute eigenvalue decomposition of symmetric N-by-N matrix
 *
 * Compute selected eigenvalue of symmetric N-by-N matrix with bisection algorightm.
 *
 * \param[out] D
 *      Requested eigenvalues in increasing order
 * \param[in,out] A
 *      On entry, symmetric matrix stored in lower or upper triangular part.
 *      On exit, matrix reduced to tridiagonal form.
 * \param[in] flags
 *      Flag bits, set ARMAS_UPPER (ARMAS_LOWER) if upper (lower) triangular storage
 *      is used. If eigenvectors wanted, set ARMAS_WANTV.
 * \param[in] params
 *      Requested eigenvalue intervals, defined with macros ARMAS_EIGEN_INT and ARMAS_EIGEN_VAL.
 * \param[in] conf
 *      Optional configuration block, if NULL then default configuration used. 
 *
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 * \retval -2 Tridiagonal reduction failed.
 * \retval -3 Not all eigenvalues for interval returned
 *
 * \ingroup lapack
 */
int armas_x_eigen_sym_selected(armas_x_dense_t * D,
                               armas_x_dense_t * A,
                               const armas_x_eigen_parameter_t * params,
                               int flags, armas_conf_t * conf)
{
    int err;
    armas_wbuf_t *wbs, wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();

    wbs = &wb;
    if (armas_x_eigen_sym_selected_w(D, A, params, flags, &wb, conf) < 0)
        return -1;
    if (wb.bytes > 0) {
        if (!armas_walloc(&wb, wb.bytes)) {
            conf->error = ARMAS_EMEMORY;
            return -1;
        }
    } else
        wbs = ARMAS_NOWORK;
    err = armas_x_eigen_sym_selected_w(D, A, params, flags, wbs, conf);
    armas_wrelease(&wb);
    return err;
}

int armas_x_eigen_sym_selected_w(armas_x_dense_t * D,
                                 armas_x_dense_t * A,
                                 const armas_x_eigen_parameter_t * params,
                                 int flags,
                                 armas_wbuf_t * wb, armas_conf_t * conf)
{
    armas_x_dense_t sD, sE, tau;
    size_t wsz, wpos;
    DTYPE *buf;
    int ioff, N;

    if (!conf)
        conf = armas_conf_default();

    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }
    N = A->rows;
    if (wb && wb->bytes == 0) {
        if (armas_x_trdreduce_w(A, ARMAS_NIL, flags, wb, conf) < 0)
            return -1;
        wb->bytes += N * sizeof(DTYPE);
        return 0;
    }

    if (A->rows != A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    if (!wb && (wsz = armas_wbytes(wb)) < 2 * N * sizeof(DTYPE)) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    // default to lower triangular storage
    if (!(flags & (ARMAS_UPPER | ARMAS_LOWER)))
        flags |= ARMAS_LOWER;

    wpos = armas_wpos(wb);

    ioff = flags & ARMAS_LOWER ? -1 : 1;
    if (N > 2) {
        buf = (DTYPE *) armas_wreserve(wb, N, sizeof(DTYPE));
        armas_x_make(&tau, N, 1, N, buf);

        // reduce to tridiagonal form
        if (armas_x_trdreduce_w(A, &tau, flags, wb, conf) < 0) {
            armas_wsetpos(wb, wpos);
            return -2;
        }

    }
    armas_wsetpos(wb, wpos);

    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, A, ioff);

    // compute selected eigenvalues
    if (armas_x_trdbisect(D, &sD, &sE, params, conf) < 0)
        return -3;

    return 0;
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
