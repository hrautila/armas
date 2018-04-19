
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_todense) 
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <armas/armas.h>
#include "matrix.h"
#include "sparse.h"

int armassp_x_todense(armas_x_dense_t *A, const armas_x_sparse_t *B, armas_conf_t *cf)
{
    if (!cf)
        cf = armas_conf_default();
    
    if (A->rows != B->rows || A->cols != B->cols) {
        cf->error = ARMAS_ESIZE;
        return -1;
    }
        
    int stat = 0;
    switch (B->kind) {
    case ARMASSP_CSC:
        for (int j = 0; j < B->nptr; j++) {
            for (int k = B->ptr[j]; k < B->ptr[j+1]; k++) {
                armas_x_set(A, B->ix[k], j, B->elems.v[k]);
            }
        }
        break;
    case ARMASSP_CSR:
        for (int i = 0; i < B->nptr; i++) {
            for (int k = B->ptr[i]; k < B->ptr[i+1]; k++) {
                armas_x_set(A, i, B->ix[k], B->elems.v[k]);
            }
        }
        break;
    case ARMASSP_COO:
        for (int j = 0; j < B->nnz; j++) {
            armas_x_set(A, B->elems.ep[j].i, B->elems.ep[j].j, B->elems.ep[j].val);
        }
        break;
    default:
        stat = -1;
        break;
    }
    return stat;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
