
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_mult_diag) && defined(armassp_x_add_diag)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <armas/armas.h>
#include "sparse.h"

// A = alpha*diag(x)*A or A = alpha*A*diag(x)

/**
 * \brief Compute A = alpha*diag(x)*A or A = alpha*A*diag(x)
 */
int armassp_x_mult_diag(armas_x_sparse_t *A, DTYPE alpha, const armas_x_dense_t *x, int flags)
{
    int p;
    DTYPE *xp = armas_x_data(x);
    DTYPE *Ae =  A->elems.v;
    
    if (A->kind != ARMASSP_CSR && A->kind != ARMASSP_CSC)
        return -1;
    
    switch (flags & (ARMAS_LEFT|ARMAS_RIGHT)) {
    case ARMAS_RIGHT:
        if (A->kind == ARMASSP_CSC) {
            for (int i = 0; i < A->cols; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i+1); p++) {
                    Ae[p] *= alpha*xp[i];
                }
            }
        } else {
            for (int i = 0; i < A->rows; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i+1); p++) {
                    Ae[p] *= alpha*xp[sp_at(A, p)];
                }
            }
        }
        break;
    case ARMAS_LEFT:
    default:
        if (A->kind == ARMASSP_CSR) {
            for (int i = 0; i < A->rows; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i+1); p++) {
                    Ae[p] *= alpha*xp[i];
                }
            }
        } else {
            for (int i = 0; i < A->cols; i++) {
                for (p = sp_index(A, i); p < sp_index(A, i+1); p++) {
                    Ae[p] *= alpha*xp[sp_at(A, p)];
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
int armassp_x_add_diag(armas_x_sparse_t *A, DTYPE mu)
{
    int p;
    DTYPE *Ae;
    switch (A->kind) {
    case ARMASSP_CSR:
        Ae = A->elems.v;
        for (int i = 0; i < A->rows; i++) {
            for (p = sp_index(A, i); sp_at(A, p) < i && p < sp_index(A, i+1); p++);
            if (sp_at(A, p) == i)
                Ae[p] += mu;
        }
        break;
    case ARMASSP_CSC:
        Ae = A->elems.v;
        for (int i = 0; i < A->cols; i++) {
            for (p = sp_index(A, i); sp_at(A, p) < i && p < sp_index(A, i+1); p++);
            if (sp_at(A, p) == i)
                Ae[p] += mu;
        }
        break;
    default:
        return -1;
    }
    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
