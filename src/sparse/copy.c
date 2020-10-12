
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_mkcopy) && defined(armassp_x_copy_to)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "sparse.h"

/**
 * @brief Make A = B
 * @return Pointer to A if success otherwise null pointer.
 * @ingroup sparse
 */
armas_x_sparse_t *armassp_x_copy_to(armas_x_sparse_t * A,
                                    const armas_x_sparse_t * B)
{
    if (!B || !A)
        return (armas_x_sparse_t *) 0;

    if (armassp_x_nbytes(A) < armassp_x_bytes_for(B)) {
        return (armas_x_sparse_t *) 0;
    }

    DTYPE *Ae, *Be;
    switch (B->kind) {
    case ARMASSP_CSC:
    case ARMASSP_CSR:
        Ae = A->elems.v;
        Be = B->elems.v;
        for (int k = 0; k < B->nnz; k++) {
            Ae[k] = Be[k];
            A->ix[k] = B->ix[k];
        }
        for (int k = 0; k <= B->nptr; k++) {
            A->ptr[k] = B->ptr[k];
        }
        A->nptr = B->nptr;
        A->nnz = B->nnz;
        break;
    case ARMASSP_COO:
        for (int k = 0; k < B->nnz; k++) {
            A->elems.ep[k] = B->elems.ep[k];
        }
        break;
    }
    return A;
}

/**
 * @brief Create copy of matrix.
 * @return Pointer to new matrix if success otherwise null pointer.
 * @ingroup sparse
 */
armas_x_sparse_t *armassp_x_mkcopy(const armas_x_sparse_t * B)
{
    armas_x_sparse_t *A = armassp_x_new(B->rows, B->cols, B->nnz, B->kind);
    return armassp_x_copy_to(A, B);
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
