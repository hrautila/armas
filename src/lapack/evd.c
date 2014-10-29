
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_eigen_sym) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(__armas_trdeigen) && defined(__armas_trdreduce) && defined(__armas_trdbuild)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

/*
 * \brief Eigenvalues and vectors of 2x2 matrix
 */
int __eigen_sym_small(__armas_dense_t *D, __armas_dense_t *A,
                      __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    DTYPE d0, b0, d1, e0, e1, cs, sn;

    d0 = __armas_get_unsafe(A, 0, 0);
    d1 = __armas_get_unsafe(A, 1, 1);
    if (flags & ARMAS_UPPER) {
        b0 = __armas_get_unsafe(A, 0, 1);
    } else {
        b0 = __armas_get_unsafe(A, 1, 0);
    }
    __sym_eigen2x2vec(&e0, &e1, &cs, &sn, d0, b0, d1);
    __armas_set_at_unsafe(D, 0, e0);
    __armas_set_at_unsafe(D, 1, e1);

    if (flags & ARMAS_WANTV) {
        __armas_set_unsafe(A, 0, 0, __ONE);
        __armas_set_unsafe(A, 1, 1, __ONE);
        __armas_set_unsafe(A, 0, 1, __ZERO);
        __armas_set_unsafe(A, 1, 0, __ZERO);
        __armas_gvright(A, cs, sn, 0, 1, 0, A->rows);
    }
    return 0;
}

/*
 * \brief Compute eigenvalue decomposition of symmetric N-by-N matrix
 *
 * \param[out] D
 *      Eigenvalues of A in increasing order
 * \param[in,out] A
 *      On entry, symmetric matrix stored in lower or upper triangular part.
 *      On exit, eigenvector if requested, otherwise contents are destroyd.
 * \param[in] W
 *      Workspace, size at least 3*N
 * \param[in] flags
 *      Flag bits, set ARMAS_UPPER (ARMAS_LOWER) if upper (lower) triangular storage
 *      is used. If eigenvectors wanted, set ARMAS_WANTV.
 * \param[in] conf
 *      Optional configuration block, if NULL then default configuration used. 
 * \return
 *      Zero for success, negative error for failure and .error field in configuration
 *      block is set.
 */
int __armas_eigen_sym(__armas_dense_t *D, __armas_dense_t *A,
                      __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    __armas_dense_t sD, sE, E, tau, Wred, *vv;
    int wrl, ioff, N = A->rows;
    vv = __nil;

    if (!conf)
        conf = armas_conf_default();

    if (A->rows != A->cols && __armas_size(D) != A->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (N == 1) {
        __armas_set_at_unsafe(D, 0, __armas_get_unsafe(A, 0, 0));
        if (flags & ARMAS_WANTV) {
            __armas_set_unsafe(A, 0, 0, __ONE);
        }
        return 0;
    }
    if (N == 2) {
        __eigen_sym_small(D, A, W, flags, conf);
        return 0;
    }

    if ((flags & ARMAS_WANTV) && __armas_size(W) < 3*N) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    // default to lower triangular storage
    if (!(flags & (ARMAS_UPPER|ARMAS_LOWER)))
        flags |= ARMAS_LOWER;

    ioff = flags & ARMAS_LOWER ? -1 : 1;
    __armas_make(&E, N-1, 1, N-1, __armas_data(W));
    __armas_make(&tau, N, 1, N, &__armas_data(W)[N-1]);
    wrl = __armas_size(W) - 2*N - 1;
    __armas_make(&Wred, wrl, 1, wrl, &__armas_data(W)[2*N-1]);

    // reduce to tridiagonal form
    if (__armas_trdreduce(A, &tau, &Wred, flags, conf) != 0)
        return -2;

    // copy diagonals
    __armas_diag(&sD, A, 0);
    __armas_diag(&sE, A, ioff);
    __armas_copy(D, &sD, conf);
    __armas_copy(&E, &sE, conf);

    // if vectors required, build in A
    if (flags & ARMAS_WANTV) {
        if (__armas_trdbuild(A, &tau, &Wred, N, flags, conf) != 0)
            return -3;
        vv = A;
    }

    // reszie workspace
    wrl = __armas_size(W) - N - 1;
    __armas_make(&Wred, wrl, 1, wrl, &__armas_data(W)[N-1]);

    // compute eigenvalues/vectors of tridiagonal matrix
    if (__armas_trdeigen(D, &E, vv, &Wred, flags, conf) != 0)
        return -4;

    return 0;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

