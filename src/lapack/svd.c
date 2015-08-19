
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(__armas_svd) && defined(__armas_svd_work)
#define __ARMAS_PROVIDES 1
#endif
// this file requires at least following external public functions
#if defined(__armas_bdsvd) && defined(__armas_bdreduce) && defined(__armas_bdmult)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------


#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"

static inline
int crossover(int N1, int N2) {
    return N1 > (int)(1.6*(double)N2);
}

// zero value function
static
DTYPE zeros(int i, int j) {
    return __ZERO;
}


static
int __armas_svd_small(__armas_dense_t *S, __armas_dense_t *U, __armas_dense_t *V,
                      __armas_dense_t *A, __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    __armas_dense_t tau;
    int M, N, K;
    DTYPE d0, d1, e0, smin, smax, cosr, sinr, cosl, sinl;
    DTYPE buf[2];

    M = A->rows; N = A->cols;
    K = __IMIN(M, N);
    __armas_make(&tau, K, 1, K, buf);
    if (M >= N) {
        if (__armas_qrfactor(A, &tau, W, conf) != 0)
            return -2;
    } else {
        if (__armas_lqfactor(A, &tau, W, conf) != 0)
            return -2;
    }

    if (M == 1 || N == 1) {
        // either tall M-by-1 or wide 1-by-N
        __armas_set_at_unsafe(S, 0, __ABS(__armas_get_unsafe(A, 0, 0)));
        if (! (flags & (ARMAS_WANTU|ARMAS_WANTV)))
            return 0;

        if (flags & ARMAS_WANTU) {
            if (N == 1) {
                if (U->cols == A->cols) {
                    __armas_copy(U, A, conf);
                    __armas_qrbuild(U, &tau, W, N, conf);
                } else {
                    // U is M-by-M
                    __armas_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
                    if (__armas_qrmult(U, A, &tau, W, ARMAS_RIGHT, conf) != 0)
                        return -3;
                }
            } else {
                // U is 1-by-1
                __armas_set_unsafe(U,  0, 0, -1.0);
            }
        }
        if (flags & ARMAS_WANTV) {
            if (M == 1) {
                if (V->rows == A->rows) {
                    __armas_copy(V, A, conf);
                    __armas_lqbuild(V, &tau, W, M, conf);
                } else {
                    // V is N-by-N
                    __armas_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
                    if (__armas_lqmult(V, A, &tau, W, ARMAS_RIGHT, conf) != 0)
                        return -3;
                }
            } else {
                // V is 1-by-1
                __armas_set_unsafe(V, 0, 0, -1.0);
            }
        }
        return 0;
    }

    // use __bdsvd2x2 functions 
    d0 = __armas_get_unsafe(A, 0, 0);
    d1 = __armas_get_unsafe(A, 1, 1);
    // upper bidiagonal if M >= N and get [0, 1] otherwise get [1, 0]
    e0 = __armas_get_unsafe(A, (M >= N ? 0 : 1), (M >= N ? 1 : 0));

    if (!(flags & (ARMAS_WANTU|ARMAS_WANTV))) {
        // no vectors wanted
        __bdsvd2x2(&smin, &smax, d0, e0, d1);
        __armas_set_at_unsafe(S, 0, __ABS(smax));
        __armas_set_at_unsafe(S, 1, __ABS(smin));
        return 0;
    }

    // want some vectors
    
    __bdsvd2x2_vec(&smin, &smax, &cosl, &sinl, &cosr, &sinr, d0, e0, d1);

    __armas_set_at_unsafe(S, 0, __ABS(smax));
    __armas_set_at_unsafe(S, 1, __ABS(smin));

    if (flags & ARMAS_WANTU) {
        // generate Q matrix, tall M-by-2 
        if (M >= N) {
            if (U->cols == A->cols) {
                __armas_mcopy(U, A);
                if (__armas_qrbuild(U, &tau, W, N, conf) != 0)
                    return -3;
            } else {
                // U is M-by-M
                __armas_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
                if (__armas_qrmult(U, A, &tau, W, ARMAS_RIGHT, conf) != 0)
                    return -3;
            }
            __armas_gvright(U, cosl, sinl, 0, 1, 0, M);
        } else {
            // otherwise V is 2-by-2
            __armas_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
            __armas_gvright(U, cosr, sinr, 0, 1, 0, M);
        }
    }

    if (flags & ARMAS_WANTV) {
        if (N > M) {
            if (V->rows == A->rows) {
                // generate Q matrix, wide 2-by-N
                __armas_mcopy(V, A);
                if (__armas_lqbuild(V, &tau, W, M, conf) != 0)
                    return -4;
            } else {
                // V is N-by-N
                __armas_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
                if (__armas_lqmult(V, A, &tau, W, ARMAS_RIGHT, conf) != 0)
                    return -3;
            }
            __armas_gvleft(V, cosl, sinl, 0, 1, 0, N);
        } else {
            // otherwise V is 2-by-2
            __armas_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
            __armas_gvleft(V, cosr, sinr, 0, 1, 0, N);
        }

        if (smax < __ZERO) {
            __armas_row(&tau, V, 0);
            __armas_scale(&tau, -1.0, conf);
        }
        if (smin < __ZERO) {
            __armas_row(&tau, V, 1);
            __armas_scale(&tau, -1.0, conf);
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
int __armas_svd_tall(__armas_dense_t *S, __armas_dense_t *U, __armas_dense_t *V,
                     __armas_dense_t *A, __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    __armas_dense_t tauq, taup, Wred, sD, sE, R, Un, *uu, *vv;
    int M, N, wr_len;
    
    M = A->rows; N = A->cols;
    uu = __nil;
    vv = __nil;
    
    if (flags & (ARMAS_WANTU | ARMAS_WANTV)) {
        if (__armas_size(W) < 4*N) {
            conf->error = ARMAS_EWORK;
            return -1;
        }
    }
    
    // 1. divide workspace
    __armas_make(&tauq, N, 1, N, &__armas_data(W)[0]);
    __armas_make(&taup, N-1, 1, N-1, &__armas_data(W)[N]);
    wr_len = __armas_size(W) - 2*N - 1;
    __armas_make(&Wred, wr_len, 1, wr_len, &__armas_data(W)[2*N-1]);
    
    if (crossover(M, N))
        goto do_m_much_bigger;

    // standard case, M > N but not too much
    if (__armas_bdreduce(A, &tauq, &taup, &Wred, conf) != 0)
        return -2;

    // setup diagonals
    __armas_diag(&sD, A, 0);
    __armas_diag(&sE, A, 1);
    __armas_copy(S, &sD, conf);
    
    // generate left vectors
    if (flags & ARMAS_WANTU) {
        if (U->cols == A->cols) {
            // U is M-by-N
            __armas_mcopy(U, A);
            __armas_make_trm(U, ARMAS_LOWER);
            if (__armas_bdbuild(U, &tauq, &Wred, U->cols, ARMAS_WANTQ, conf) != 0)
                return -3;
        } else {
            // U is M-by-M; 
            __armas_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (__armas_bdmult(U, A, &tauq, &Wred, ARMAS_MULTQ|ARMAS_RIGHT, conf) != 0)
                return -3;
        }
        uu = U;
    }

    // generate right vectors
    if (flags & ARMAS_WANTV) {
        // V is N-by-N
        __armas_submatrix(&R, A, 0, 0, N, N);
        __armas_mcopy(V, &R);
        __armas_make_trm(V, ARMAS_UPPER);
        if (__armas_bdbuild(V, &taup, &Wred, V->rows, ARMAS_WANTP, conf) != 0)
            return -4;
        vv = V;
    }

    // run bidiagonal SVD
    if (__armas_bdsvd(S, &sE, uu, vv, W, flags|ARMAS_UPPER, conf) != 0)
        return -5;

    // we are done
    return 0;

 do_m_much_bigger:
    // here M >> N, first QR factor 
    if (__armas_qrfactor(A, &tauq, &Wred, conf) != 0)
        return -2;

    if (flags & ARMAS_WANTU) {
        if (U->cols == A->cols) {
            __armas_mcopy(U, A);
            if (__armas_qrbuild(U, &tauq, &Wred, U->cols, conf) != 0)
                return -3;
        } else {
            // U is M-by-M
            __armas_set_values(U, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (__armas_qrmult(U, A, &tauq, &Wred, ARMAS_RIGHT, conf) != 0)
                return -3;
        }
        uu = U;
    }
    
    // zero out all but R
    __armas_make_trm(A, ARMAS_UPPER);
    __armas_submatrix(&R, A, 0, 0, N, N);
    
    // run bidiagonal reduce on R
    if (__armas_bdreduce(&R, &tauq, &taup, &Wred, conf) != 0)
        return -2;

    if (flags & ARMAS_WANTU) {
        __armas_submatrix(&Un, U, 0, 0, M, N);
        if (__armas_bdmult(&Un, &R, &tauq, &Wred, ARMAS_MULTQ|ARMAS_RIGHT, conf) != 0)
            return -3;
    }

    if (flags & ARMAS_WANTV) {
        __armas_mcopy(V, &R);
        __armas_make_trm(V, ARMAS_UPPER);
        if (__armas_bdbuild(V, &taup, &Wred, V->rows, ARMAS_WANTP, conf) != 0)
            return -4;
        vv = V;
    }

    // setup diagonals
    __armas_diag(&sD, A, 0);
    __armas_diag(&sE, A, 1);
    __armas_copy(S, &sD, conf);
    
    // run bidiagonal SVD
    if (__armas_bdsvd(S, &sE, uu, vv, W, flags|ARMAS_UPPER, conf) != 0)
        return -5;

    return 0;
}

static 
int __armas_svd_wide(__armas_dense_t *S, __armas_dense_t *U, __armas_dense_t *V,
                     __armas_dense_t *A, __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    __armas_dense_t tauq, taup, Wred, sD, sE, Vm, L, *uu, *vv;
    int M, N, wr_len;

    M = A->rows; N = A->cols;
    uu = __nil;
    vv = __nil;

    if (flags & (ARMAS_WANTU | ARMAS_WANTV)) {
        if (__armas_size(W) < 4*M) {
            conf->error = ARMAS_EWORK;
            return -1;
        }
    }

    // 1. divide workspace
    __armas_make(&tauq, M-1, 1, M-1, &__armas_data(W)[0]);
    __armas_make(&taup, M, 1, M, &__armas_data(W)[M-1]);
    wr_len = __armas_size(W) - 2*M - 1;
    __armas_make(&Wred, wr_len, 1, wr_len, &__armas_data(W)[2*M-1]);

    if (crossover(N, M))
        goto do_n_much_bigger;

    // standard case, M < N but not too much
    if (__armas_bdreduce(A, &tauq, &taup, &Wred, conf) != 0)
        return -2;

    // copy diagonals
    __armas_diag(&sD, A, 0);
    __armas_diag(&sE, A, -1);
    __armas_copy(S, &sD, conf);

    // generate left vectors
    if (flags & ARMAS_WANTU) {
        __armas_submatrix(&L, A, 0, 0, M, M);
        __armas_mcopy(U, &L);
        __armas_make_trm(U, ARMAS_LOWER);
        if (__armas_bdbuild(U, &tauq, &Wred, U->rows, ARMAS_WANTQ|ARMAS_LOWER, conf) != 0)
            return -3;
        uu = U;
    }

    // generate right vectors
    if (flags & ARMAS_WANTV) {
        if (V->rows == A->rows) {
            // V is M-by-N
            __armas_mcopy(V, A);
            if (__armas_bdbuild(V, &taup, &Wred, V->rows, ARMAS_WANTP, conf) != 0)
                return -4;
        } else {
            // V is N-by-N
            __armas_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (__armas_bdmult(V, A, &taup, &Wred, ARMAS_MULTP|ARMAS_LEFT|ARMAS_TRANS, conf) != 0)
                return -4;
        }
        vv = V;
    }

    // run bidiagonal SVD
    if (__armas_bdsvd(S, &sE, uu, vv, W, flags|ARMAS_LOWER, conf) != 0)
        return -5;

    // we are done
    return 0;

 do_n_much_bigger:
    // here M << N, first LQ factor 
    if (__armas_lqfactor(A, &taup, &Wred, conf) != 0)
        return -2;

    // generate right vectors
    if (flags & ARMAS_WANTV) {
        if (V->rows == A->rows) {
            // V is M-by-N
            __armas_mcopy(V, A);
            if (__armas_lqbuild(V, &taup, &Wred, M, conf) != 0)
                return -4;
        } else {
            // V is N-by-N
            __armas_set_values(V, zeros, ARMAS_SYMM|ARMAS_UNIT);
            if (__armas_lqmult(V, A, &taup, &Wred, ARMAS_RIGHT, conf) != 0)
                return -4;
        }
        vv = V;
    }

    // zero out all but L
    __armas_make_trm(A, ARMAS_LOWER);
    __armas_submatrix(&L, A, 0, 0, M, M);

    // resize tauq/taup for UPPER bidiagonal reduction
    __armas_make(&tauq, M, 1, M, &__armas_data(W)[0]);
    __armas_make(&taup, M-1, 1, M-1, &__armas_data(W)[M]);

    // run bidiagonal reduce on L; n(L) == m(L) then UPPER bidiagonal reduction
    if (__armas_bdreduce(&L, &tauq, &taup, &Wred, conf) != 0)
        return -2;

    if (flags & ARMAS_WANTV) {
        __armas_submatrix(&Vm, V, 0, 0, M, N);
        if (__armas_bdmult(&Vm, &L, &taup, &Wred, ARMAS_MULTP|ARMAS_LEFT|ARMAS_TRANS, conf) != 0)
            return -4;
    }
    // generate left vectors
    if (flags & ARMAS_WANTU) {
        __armas_mcopy(U, &L);
        if (__armas_bdbuild(U, &tauq, &Wred, U->rows, ARMAS_WANTQ, conf) != 0)
            return -3;
        uu = U;
    }

    // setup diagonals
    __armas_diag(&sD, A, 0);
    __armas_diag(&sE, &L, 1);
    __armas_copy(S, &sD, conf);

    // run bidiagonal SVD
    if (__armas_bdsvd(S, &sE, uu, vv, W, flags|ARMAS_UPPER, conf) != 0)
        return -5;

    return 0;
}

/*
 * \brief Compute singular value decomposition and optionally singular vectors of a general matrix.
 *
 * \param[out] S
 *      Singular values
 * \param[out] U
 *      Left singular vectors, M-by-N or M-by-M matrix
 * \param[out] V
 *      Right singular vectors, M-by-N or N-by-N matrix
 * \param[in,out] A
 *      On entry, general M-by-N matrix which singular values are computed. On exit, contents
 *      are destroyed.
 * \param[in] W
 *      Workspace
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
 *  -5  Error bidiagonal eigenvalue computation
 */
int __armas_svd(__armas_dense_t *S, __armas_dense_t *U, __armas_dense_t *V,
                __armas_dense_t *A, __armas_dense_t *W, int flags, armas_conf_t *conf)
{
    int err; //, tq_len, tp_len;
    int tall = A->rows >= A->cols;

    if (tall && __armas_size(S) < A->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    } else if (!tall && __armas_size(S) < A->rows) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    //tq_len = tall ? A->cols : A->rows - 1;
    //tp_len = tall ? A->cols - 1 : A->cols;

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

    if (tall) {
        if (A->cols <= 2) {
            err = __armas_svd_small(S, U, V, A, W, flags, conf);
        } else {
            err = __armas_svd_tall(S, U, V, A, W, flags, conf);
        }
    } else {
        if (A->rows <= 2) {
            err = __armas_svd_small(S, U, V, A, W, flags, conf);
        } else {
            err = __armas_svd_wide(S, U, V, A, W, flags, conf);
        }
    }
    return err;
}

int __armas_svd_work(__armas_dense_t *A, int flags, armas_conf_t *conf)
{
    __armas_dense_t Tmp;
    int ws, s0;
    ws = __armas_bdreduce_work(A, conf);
    if (!(flags & (ARMAS_WANTU|ARMAS_WANTV)))
        return ws;
    
    // singular vectors wanted
    s0 = __armas_bdsvd_work(A, conf);
    if (s0 > ws)
        ws = s0;

    // tall matrix
    if (A->rows >= A->cols) {
        if (flags & ARMAS_WANTU) {
            ws = __IMAX(ws, __armas_qrfactor_work(A, conf));
            __armas_make(&Tmp, A->rows, A->rows, A->rows, (DTYPE *)0);
            ws = __IMAX(ws, __armas_qrmult_work(&Tmp, ARMAS_LEFT, conf));
        }
        if (flags & ARMAS_WANTV) {
            __armas_make(&Tmp, A->cols, A->cols, A->cols, (DTYPE *)0);
            ws = __IMAX(ws, __armas_qrbuild_work(A, conf));
        }
        return ws;
    }
    // wide matrix
    if (flags & ARMAS_WANTV) {
        ws = __IMAX(ws, __armas_lqfactor_work(A, conf));
        __armas_make(&Tmp, A->rows, A->rows, A->rows, (DTYPE *)0);
        ws = __IMAX(ws, __armas_lqmult_work(&Tmp, ARMAS_LEFT, conf));
    }
    if (flags & ARMAS_WANTU) {
        __armas_make(&Tmp, A->cols, A->cols, A->cols, (DTYPE *)0);
        ws = __IMAX(ws, __armas_lqbuild_work(A, conf));
    }
    return ws;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

