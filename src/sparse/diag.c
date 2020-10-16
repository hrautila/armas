
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_mult_diag) && defined(armassp_add_diag)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "armas.h"
#include "sparse.h"

// A = alpha*diag(x)*A or A = alpha*A*diag(x)

/**
 * \brief Compute A = alpha*diag(x)*A or A = alpha*A*diag(x)
 */
int armassp_mult_diag(armas_sparse_t * A, DTYPE alpha,
                        const armas_dense_t * x, int flags)
{
    int p;
    DTYPE *xp = armas_data(x);
    DTYPE *Ae = A->elems.v;

    if (A->kind != ARMASSP_CSR && A->kind != ARMASSP_CSC)
        return -1;

    switch (flags & (ARMAS_LEFT | ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        if (A->kind == ARMASSP_CSC) {
            for (int i = 0; i < A->cols; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i + 1); p++) {
                    Ae[p] *= alpha * xp[i];
                }
            }
        } else {
            for (int i = 0; i < A->rows; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i + 1); p++) {
                    Ae[p] *= alpha * xp[sp_at(A, p)];
                }
            }
        }
        break;
    case ARMAS_LEFT:
    default:
        if (A->kind == ARMASSP_CSR) {
            for (int i = 0; i < A->rows; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i + 1); p++) {
                    Ae[p] *= alpha * xp[i];
                }
            }
        } else {
            for (int i = 0; i < A->cols; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i + 1); p++) {
                    Ae[p] *= alpha * xp[sp_at(A, p)];
                }
            }
        }
        break;
    }
    return 0;
}

/**
 * \brief Compute A = A + mu*I
 */
int armassp_add_diag(armas_sparse_t * A, DTYPE mu)
{
    int p;
    DTYPE *Ae;
    switch (A->kind) {
    case ARMASSP_CSR:
        Ae = A->elems.v;
        for (int i = 0; i < A->rows; i++) {
            for (p = sp_index(A, i); sp_at(A, p) < i && p < sp_index(A, i + 1);
                 p++);
            if (sp_at(A, p) == i)
                Ae[p] += mu;
        }
        break;
    case ARMASSP_CSC:
        Ae = A->elems.v;
        for (int i = 0; i < A->cols; i++) {
            for (p = sp_index(A, i); sp_at(A, p) < i && p < sp_index(A, i + 1);
                 p++);
            if (sp_at(A, p) == i)
                Ae[p] += mu;
        }
        break;
    default:
        return -1;
    }
    return 0;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
