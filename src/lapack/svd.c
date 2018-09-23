
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Singular value decomposition of general matrix

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_svd) && defined(armas_x_svd_w)
#define __ARMAS_PROVIDES 1
#endif
// this file requires at least following external public functions
#if defined(armas_x_bdsvd) && defined(armas_x_bdreduce) && defined(armas_x_bdmult)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
//! \endcond

static inline
int crossover(int N1, int N2) {
    return N1 > (int)(1.6*(double)N2);
}

// zero value function
static
DTYPE zeros(int i, int j) {
    return __ZERO;
}

#ifndef ARMAS_MAX_ONSTACK_WSPACE
#define ARMAS_MAX_ONSTACK_WSPACE  384
#endif

static
int armas_x_svd_small(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V,
                      armas_x_dense_t *A, int flags, armas_wbuf_t *wb, armas_conf_t *conf)
{
    armas_x_dense_t tau;
    int M, N, K;
    DTYPE d0, d1, e0, smin, smax, cosr, sinr, cosl, sinl;
    DTYPE buf[2];

    M = A->rows; N = A->cols;
    K = __IMIN(M, N);
    armas_x_make(&tau, K, 1, K, buf);
    if (M >= N) {
        if (armas_x_qrfactor_w(A, &tau, wb, conf) != 0)
            return -2;
    } else {
        if (armas_x_lqfactor_w(A, &tau, wb, conf) != 0)
            return -2;
    }

    if (M == 1 || N == 1) {
        // either tall M-by-1 or wide 1-by-N
        armas_x_set_at_unsafe(S, 0, __ABS(armas_x_get_unsafe(A, 0, 0)));
        if (! (flags & (ARMAS_WANTU|ARMAS_WANTV)))
            return 0;

        if (flags & ARMAS_WANTU) {
            if (N == 1) {
                if (U->cols == A->cols) {
                    armas_x_copy(U, A, conf);
                    armas_x_qrbuild_w(U, &tau, N, wb, conf);
                } else {
                    // U is M-by-M
                    armas_x_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
                    if (armas_x_qrmult_w(U, A, &tau, ARMAS_RIGHT, wb, conf) != 0)
                        return -3;
                }
            } else {
                // U is 1-by-1
                armas_x_set_unsafe(U,  0, 0, -1.0);
            }
        }
        if (flags & ARMAS_WANTV) {
            if (M == 1) {
                if (V->rows == A->rows) {
                    armas_x_copy(V, A, conf);
                    armas_x_lqbuild_w(V, &tau, M, wb, conf);
                } else {
                    // V is N-by-N
                    armas_x_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
                    if (armas_x_lqmult_w(V, A, &tau, ARMAS_RIGHT, wb, conf) != 0)
                        return -3;
                }
            } else {
                // V is 1-by-1
                armas_x_set_unsafe(V, 0, 0, -1.0);
            }
        }
        return 0;
    }

    // use __bdsvd2x2 functions 
    d0 = armas_x_get_unsafe(A, 0, 0);
    d1 = armas_x_get_unsafe(A, 1, 1);
    // upper bidiagonal if M >= N and get [0, 1] otherwise get [1, 0]
    e0 = armas_x_get_unsafe(A, (M >= N ? 0 : 1), (M >= N ? 1 : 0));

    if (!(flags & (ARMAS_WANTU|ARMAS_WANTV))) {
        // no vectors wanted
        __bdsvd2x2(&smin, &smax, d0, e0, d1);
        armas_x_set_at_unsafe(S, 0, __ABS(smax));
        armas_x_set_at_unsafe(S, 1, __ABS(smin));
        return 0;
    }

    // want some vectors
    
    __bdsvd2x2_vec(&smin, &smax, &cosl, &sinl, &cosr, &sinr, d0, e0, d1);

    armas_x_set_at_unsafe(S, 0, __ABS(smax));
    armas_x_set_at_unsafe(S, 1, __ABS(smin));

    if (flags & ARMAS_WANTU) {
        // generate Q matrix, tall M-by-2 
        if (M >= N) {
            if (U->cols == A->cols) {
                armas_x_mcopy(U, A);
                if (armas_x_qrbuild_w(U, &tau, N, wb, conf) != 0)
                    return -3;
            } else {
                // U is M-by-M
                armas_x_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
                if (armas_x_qrmult_w(U, A, &tau, ARMAS_RIGHT, wb, conf) != 0)
                    return -3;
            }
            armas_x_gvright(U, cosl, sinl, 0, 1, 0, M);
        } else {
            // otherwise V is 2-by-2
            armas_x_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
            armas_x_gvright(U, cosr, sinr, 0, 1, 0, M);
        }
    }

    if (flags & ARMAS_WANTV) {
        if (N > M) {
            if (V->rows == A->rows) {
                // generate Q matrix, wide 2-by-N
                armas_x_mcopy(V, A);
                if (armas_x_lqbuild_w(V, &tau, M, wb, conf) != 0)
                    return -4;
            } else {
                // V is N-by-N
                armas_x_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
                if (armas_x_lqmult_w(V, A, &tau, ARMAS_RIGHT, wb, conf) != 0)
                    return -3;
            }
            armas_x_gvleft(V, cosl, sinl, 0, 1, 0, N);
        } else {
            // otherwise V is 2-by-2
            armas_x_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
            armas_x_gvleft(V, cosr, sinr, 0, 1, 0, N);
        }

        if (smax < __ZERO) {
            armas_x_row(&tau, V, 0);
            armas_x_scale(&tau, -1.0, conf);
        }
        if (smin < __ZERO) {
            armas_x_row(&tau, V, 1);
            armas_x_scale(&tau, -1.0, conf);
        }
    }
    return 0;
}


/*
 * Compute for a matrix with M >= N
 *
 * crossover point to first use QR factorization and then bidiagonal reduction
 * on R is when m(A) > 1.6*n(A)
 *
 * Error returns (< 0)
 *  -1  Generic error, size or such
 *  -2  Error on factorization or bidiagonal reduction
 *  -3  Error on left eigenvector (U) computation
 *  -4  Error on right eigenvector (V) computation
 *  -5  Error bidiagonal eigenvalue computation
 */
static 
int armas_x_svd_tall(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V,
                     armas_x_dense_t *A, int flags, armas_wbuf_t *wb, armas_conf_t *conf)
{
    armas_x_dense_t tauq, taup, sD, sE, R, Un, *uu, *vv;
    int M, N;
    DTYPE *buf;
    size_t wsz;

    M = A->rows; N = A->cols;
    uu = __nil;
    vv = __nil;
    
    if (flags & (ARMAS_WANTU | ARMAS_WANTV)) {
        if (armas_wbytes(wb) < 4*N*sizeof(DTYPE)) {
            conf->error = ARMAS_EWORK;
            return -1;
        }
    }
       
    wsz = armas_wpos(wb);

    // 1. divide workspace
    buf = (DTYPE *)armas_wreserve(wb, N, sizeof(DTYPE));
    armas_x_make(&tauq, N, 1, N, buf);

    buf = (DTYPE *)armas_wreserve(wb, N-1, sizeof(DTYPE));
    armas_x_make(&taup, N-1, 1, N-1, buf);

    if (crossover(M, N))
        goto do_m_much_bigger;

    // standard case, M > N but not too much
    if (armas_x_bdreduce_w(A, &tauq, &taup, wb,  conf) != 0)
        return -2;

    // setup diagonals
    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, A, 1);
    armas_x_copy(S, &sD, conf);
    
    // generate left vectors
    if (flags & ARMAS_WANTU) {
        if (U->cols == A->cols) {
            // U is M-by-N
            armas_x_mcopy(U, A);
            armas_x_make_trm(U, ARMAS_LOWER);
            if (armas_x_bdbuild_w(U, &tauq, U->cols, ARMAS_WANTQ, wb, conf) != 0)
                return -3;
        } else {
            // U is M-by-M; 
            armas_x_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (armas_x_bdmult_w(U, A, &tauq, ARMAS_MULTQ|ARMAS_RIGHT, wb, conf) != 0)
                return -3;
        }
        uu = U;
    }

    // generate right vectors
    if (flags & ARMAS_WANTV) {
        // V is N-by-N
        armas_x_submatrix(&R, A, 0, 0, N, N);
        armas_x_mcopy(V, &R);
        armas_x_make_trm(V, ARMAS_UPPER);
        if (armas_x_bdbuild_w(V, &taup, V->rows, ARMAS_WANTP, wb, conf) != 0)
            return -4;
        vv = V;
    }

    // run bidiagonal SVD
    if (armas_x_bdsvd_w(S, &sE, uu, vv, flags|ARMAS_UPPER, wb, conf) != 0)
        return -5;

    // we are done
    return 0;

 do_m_much_bigger:
    // here M >> N, first QR factor 
    if (armas_x_qrfactor_w(A, &tauq, wb,  conf) != 0)
        return -2;

    if (flags & ARMAS_WANTU) {
        if (U->cols == A->cols) {
            armas_x_mcopy(U, A);
            if (armas_x_qrbuild_w(U, &tauq, U->cols, wb, conf) != 0)
                return -3;
        } else {
            // U is M-by-M
            armas_x_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (armas_x_qrmult_w(U, A, &tauq, ARMAS_RIGHT, wb, conf) != 0)
                return -3;
        }
        uu = U;
    }
    
    // zero out all but R
    armas_x_make_trm(A, ARMAS_UPPER);
    armas_x_submatrix(&R, A, 0, 0, N, N);
    
    // run bidiagonal reduce on R
    if (armas_x_bdreduce_w(&R, &tauq, &taup, wb, conf) != 0)
        return -2;

    if (flags & ARMAS_WANTU) {
        armas_x_submatrix(&Un, U, 0, 0, M, N);
        if (armas_x_bdmult_w(&Un, &R, &tauq, ARMAS_MULTQ|ARMAS_RIGHT, wb, conf) != 0)
            return -3;
    }

    if (flags & ARMAS_WANTV) {
        armas_x_mcopy(V, &R);
        armas_x_make_trm(V, ARMAS_UPPER);
        if (armas_x_bdbuild_w(V, &taup, V->rows, ARMAS_WANTP, wb, conf) != 0)
            return -4;
        vv = V;
    }

    // setup diagonals
    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, A, 1);
    armas_x_copy(S, &sD, conf);
    
    armas_wsetpos(wb, wsz);
    
    // run bidiagonal SVD
    if (armas_x_bdsvd_w(S, &sE, uu, vv, flags|ARMAS_UPPER, wb, conf) != 0)
        return -5;

    return 0;
}

static 
int armas_x_svd_wide(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V,
                     armas_x_dense_t *A, int flags, armas_wbuf_t *wb, armas_conf_t *conf)
{
    armas_x_dense_t tauq, taup, sD, sE, Vm, L, *uu, *vv;
    int M, N;
    size_t wsz;
    DTYPE *buf;

    M = A->rows; N = A->cols;
    uu = __nil;
    vv = __nil;

    if (flags & (ARMAS_WANTU | ARMAS_WANTV)) {
        if (armas_wbytes(wb) < 4*M*sizeof(DTYPE)) {
            conf->error = ARMAS_EWORK;
            return -1;
        }
    }

    wsz = armas_wpos(wb);
    // 1. divide workspace
    buf = (DTYPE *)armas_wreserve(wb, M-1, sizeof(DTYPE));
    armas_x_make(&tauq, M-1, 1, M-1, buf);

    buf = (DTYPE *)armas_wreserve(wb, M, sizeof(DTYPE));
    armas_x_make(&taup, M, 1, M, buf);

    if (crossover(N, M))
        goto do_n_much_bigger;

    // standard case, M < N but not too much
    if (armas_x_bdreduce_w(A, &tauq, &taup, wb, conf) != 0)
        return -2;

    // copy diagonals
    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, A, -1);
    armas_x_copy(S, &sD, conf);

    // generate left vectors
    if (flags & ARMAS_WANTU) {
        armas_x_submatrix(&L, A, 0, 0, M, M);
        armas_x_mcopy(U, &L);
        armas_x_make_trm(U, ARMAS_LOWER);
        if (armas_x_bdbuild_w(U, &tauq, U->rows, ARMAS_WANTQ|ARMAS_LOWER, wb, conf) != 0)
            return -3;
        uu = U;
    }

    // generate right vectors
    if (flags & ARMAS_WANTV) {
        if (V->rows == A->rows) {
            // V is M-by-N
            armas_x_mcopy(V, A);
            if (armas_x_bdbuild_w(V, &taup, V->rows, ARMAS_WANTP, wb, conf) != 0)
                return -4;
        } else {
            // V is N-by-N
            armas_x_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (armas_x_bdmult_w(V, A, &taup, ARMAS_MULTP|ARMAS_LEFT|ARMAS_TRANS, wb, conf) != 0)
                return -4;
        }
        vv = V;
    }

    armas_wsetpos(wb, wsz);
    // run bidiagonal SVD
    if (armas_x_bdsvd_w(S, &sE, uu, vv, flags|ARMAS_LOWER, wb, conf) != 0)
        return -5;

    // we are done
    return 0;

 do_n_much_bigger:
    // here M << N, first LQ factor 
    if (armas_x_lqfactor_w(A, &taup, wb, conf) != 0)
        return -2;

    // generate right vectors
    if (flags & ARMAS_WANTV) {
        if (V->rows == A->rows) {
            // V is M-by-N
            armas_x_mcopy(V, A);
            if (armas_x_lqbuild_w(V, &taup, M, wb, conf) != 0)
                return -4;
        } else {
            // V is N-by-N
            armas_x_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (armas_x_lqmult_w(V, A, &taup, ARMAS_RIGHT, wb, conf) != 0)
                return -4;
        }
        vv = V;
    }

    // zero out all but L
    armas_x_make_trm(A, ARMAS_LOWER);
    armas_x_submatrix(&L, A, 0, 0, M, M);

    armas_wsetpos(wb, wsz);
    // resize tauq/taup for UPPER bidiagonal reduction
    buf = (DTYPE *)armas_wreserve(wb, M, sizeof(DTYPE));
    armas_x_make(&tauq, M, 1, M, buf);

    buf = (DTYPE *)armas_wreserve(wb, M-1, sizeof(DTYPE));
    armas_x_make(&taup, M-1, 1, M-1, buf);

    // run bidiagonal reduce on L; n(L) == m(L) then UPPER bidiagonal reduction
    if (armas_x_bdreduce_w(&L, &tauq, &taup, wb, conf) != 0)
        return -2;

    if (flags & ARMAS_WANTV) {
        armas_x_submatrix(&Vm, V, 0, 0, M, N);
        if (armas_x_bdmult_w(&Vm, &L, &taup, ARMAS_MULTP|ARMAS_LEFT|ARMAS_TRANS, wb, conf) != 0)
            return -4;
    }
    // generate left vectors
    if (flags & ARMAS_WANTU) {
        armas_x_mcopy(U, &L);
        if (armas_x_bdbuild_w(U, &tauq, U->rows, ARMAS_WANTQ, wb, conf) != 0)
            return -3;
        uu = U;
    }

    armas_wsetpos(wb, wsz);

    // setup diagonals
    armas_x_diag(&sD, A, 0);
    armas_x_diag(&sE, &L, 1);
    armas_x_copy(S, &sD, conf);

    // run bidiagonal SVD
    if (armas_x_bdsvd_w(S, &sE, uu, vv, flags|ARMAS_UPPER, wb, conf) != 0)
        return -5;

    return 0;
}

/**
 * \brief Compute singular value and optionally singular vectors of a general matrix.
 *
 * \param[out] S
 *      Singular values
 * \param[out] U
 *      Left singular vectors, M-by-N or M-by-M matrix. Not used if ARMAS_WANTU is not set.
 * \param[out] V
 *      Right singular vectors, M-by-N or N-by-N matrix. Not used if ARMAS_WANTV is not set.
 * \param[in,out] A
 *      On entry, general M-by-N matrix which singular values are computed. On exit, contents
 *      are destroyed.
 * \param[in] flags
 *      Option bits, left singular vectors computed if ARMAS_WANTU set. Right vectors
 *      computed if ARMAS_WANTV is set.
 * \param[in] conf
 *      Configuration block. Member .error set if if error returned.
 *
 * If M >= N then left singular vector matrix U is either M-by-N or M-by-M and right singular
 * vector matrix is always N-by-N matrix. If M < N matrix U is always M-by-M and right singular
 * vector matrix is either M-by-N or N-by-N matrix.
 *
 * \return
 *      Zero for success, negative error indicator if failure.
 *
 * Error returns (< 0)
 *  -1  Generic error, size or such
 *  -2  Error on factorization or bidiagonal reduction
 *  -3  Error on left eigenvector (U) computation
 *  -4  Error on right eigenvector (V) computation
 *  -5  Error on bidiagonal singular value computation
 */
int armas_x_svd(armas_x_dense_t *S, armas_x_dense_t *U, armas_x_dense_t *V,
                armas_x_dense_t *A, int flags, armas_conf_t *conf)
{
    int err;
    armas_wbuf_t wb = ARMAS_WBNULL;

    if (!conf)
        conf = armas_conf_default();
    
    if (armas_x_svd_w(S, U, V, A, flags, &wb, conf) < 0) {
        conf->error = ARMAS_EWORK;
        return -1;
    }
    if (wb.bytes > 0 && wb.bytes <= ARMAS_MAX_ONSTACK_WSPACE) {
        char b[ARMAS_MAX_ONSTACK_WSPACE];
        armas_wbuf_t wbt = (armas_wbuf_t){
            .buf = b,
            .offset = 0,
            .bytes = ARMAS_MAX_ONSTACK_WSPACE
        };
        err = armas_x_svd_w(S, U, V, A, flags, &wbt, conf);        
    } else {
        armas_wbuf_t *wbs = &wb;
        if (wb.bytes > 0) {
            if (!armas_walloc(&wb, wb.bytes)) {
                conf->error = ARMAS_EMEMORY;
                return -1;
            }
        }
        else
            wbs = ARMAS_NOWORK;

        err = armas_x_svd_w(S, U, V, A, flags, wbs, conf);
        armas_wrelease(&wb);
    }

    return err;
}


/**
 * \brief Compute singular value decomposition and optionally singular vectors of a general matrix.
 *
 * \param[out] S
 *      Singular values
 * \param[out] U
 *      Left singular vectors, M-by-N or M-by-M matrix. Not used if ARMAS_WANTU is not set.
 * \param[out] V
 *      Right singular vectors, M-by-N or N-by-N matrix. Not used if ARMAS_WANTV is not set.
 * \param[in,out] A
 *      On entry, general M-by-N matrix which singular values are computed. On exit, contents
 *      are destroyed.
 * \param[in] flags
 *      Option bits, left singular vectors computed if ARMAS_WANTU set. Right vectors
 *      computed if ARMAS_WANTV is set.
 * \param[in] wb
 *      Workspace.
 * \param[in] conf
 *      Configuration block. Member .error set if if error returned.
 *
 * If M >= N then left singular vector matrix U is either M-by-N or M-by-M and right singular
 * vector matrix is always N-by-N matrix. If M < N matrix U is always M-by-M and right singular
 * vector matrix is either M-by-N or N-by-N matrix.
 *
 * \return
 *      Zero for success, negative error indicator if failure.
 *
 * Error returns (< 0)
 *  -1  Generic error, size or such
 *  -2  Error on factorization or bidiagonal reduction
 *  -3  Error on left eigenvector (U) computation
 *  -4  Error on right eigenvector (V) computation
 *  -5  Error on bidiagonal singular value computation
 */
int armas_x_svd_w(armas_x_dense_t *S,
                  armas_x_dense_t *U,
                  armas_x_dense_t *V,
                  armas_x_dense_t *A,
                  int flags,
                  armas_wbuf_t *wb,
                  armas_conf_t *conf)
{
    int err, tall;
    armas_x_dense_t W;
    
    if (!conf)
        conf = armas_conf_default();
    
    if (!A) {
        conf->error = ARMAS_EINVAL;
        return -1;
    }

    tall = A->rows >= A->cols;

    if (wb && wb->bytes == 0) {
        int P = A->rows >= A->cols ? A->cols : A->rows;
        int K = A->rows >= A->cols ? A->rows : A->cols;
        if (P <= conf->lb) 
            wb->bytes = 4 * P * sizeof(DTYPE);
        else
            wb->bytes = (K + conf->lb) * conf->lb * sizeof(DTYPE);
        return 0;
    }

    if (tall && armas_x_size(S) < A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    } else if (!tall && armas_x_size(S) < A->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    if (!wb) {
        conf->error = ARMAS_EWORK;
        return -1;
    }

    if (flags & ARMAS_WANTU) {
        if (!U) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        if (tall) {
            // if M >= N then U be either M-by-N or M-by-M
            if (U->rows != A->rows || (U->cols != A->rows && U->cols != A->cols)) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
        } else {
            if (U->rows != A->rows) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
            // U is square if M < N
            if (U->rows != U->cols) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
        }
            
    }
    if (flags & ARMAS_WANTV) {
        if (!V) {
            conf->error = ARMAS_EINVAL;
            return -1;
        }
        if (tall) {
            if (V->cols != A->cols) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
            // V is square if M > N
            if (V->rows != V->cols) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
        } else {
            // if wide then V must be either M-by-N or N-by-N
            if (V->cols != A->cols || (V->rows != A->cols && V->rows != A->rows)) {
                conf->error = ARMAS_ESIZE;
                return -1;
            }
        }
    }

    int K = armas_wbytes(wb)/sizeof(DTYPE);
    armas_x_make(&W, K, 1, K, armas_wptr(wb));
    
    if (tall) {
        if (A->cols <= 2) {
            err = armas_x_svd_small(S, U, V, A, flags, wb, conf);
        } else {
            err = armas_x_svd_tall(S, U, V, A, flags, wb, conf);
        }
    } else {
        if (A->rows <= 2) {
            err = armas_x_svd_small(S, U, V, A, flags, wb, conf);
        } else {
            err = armas_x_svd_wide(S, U, V, A, flags, wb, conf);
        }
    }
    return err;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

