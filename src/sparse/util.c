
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_make) && defined(armassp_x_init) && defined(armassp_x_new)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_bytes_needed)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "matrix.h"
#include "sparse.h"

/**
 * \brief Initializes sparse matrix using provided memory block
 *
 * \param[in,out] 
 *     On entry uninitialized matrix, on exit initialized sparse matrix.
 * \param[in] rows, cols
 *     Matrix dimensions.
 * \param[in] nnz
 *     Number of non-zero entries matrix will hold.
 * \param[in] storage
 *     Sparse storage scheme, SPARSE_CSC for column compressed, SPARSE_CSR for row compressed
 *     or SPARSE_COO for coordinate list storage.
 * \param[in] data
 *     Pointer to memory block
 * \param[in] dlen
 *     Length of memory block in bytes
 * 
 * \return
 *     0 if ok, otherwise -1
 */
int armassp_x_make(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum storage, void *data, size_t dlen)
{
    size_t nbytes = armassp_x_bytes_needed(rows, cols, nnz, storage);
    if (dlen < nbytes)
        return -1;
    switch (storage) {
    case ARMASSP_COO:
        A->elems.ep = (coo_elem_t *)data;
        A->nptr = 0;
        A->ix = A->ptr = (int *)0;
        break;
    default:
        A->elems.v = (double *)data;
        A->ptr = (int *)&A->elems.v[nnz];
        A->nptr = (storage == ARMASSP_CSR ? rows : cols);
        A->ix = &A->ptr[A->nptr+1];
        break;
    }
    A->__nbytes = nbytes;
    A->rows = rows;
    A->cols = cols;
    A->nnz = 0;
    A->size = nnz;
    A->kind = storage;
    return 0;
}

/**
 * \brief Initializes sparse matrix
 *
 * \param[in,out] 
 *     On entry uninitialized matrix, on exit initialized matrix with proper space allocations.
 * \param[in] rows, cols
 *     Matrix dimensions.
 * \param[in] nnz
 *     Number of non-zero entries matrix will hold.
 * \param[in] storage
 *     Sparse storage scheme, SPARSE_CSC for column compressed, SPARSE_CSR for row compressed
 *     or SPARSE_COO for coordinate list storage.
 * 
 * \return
 *     Pointer to initialized matrix. Null if initialization failed.
 */
armas_x_sparse_t *armassp_x_init(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum storage)
{
    if (rows == 0 || cols == 0 || nnz == 0) {
        armassp_x_clear(A);
        return A;
    }
    size_t nbytes = armassp_x_bytes_needed(rows, cols, nnz, storage);
    void *data = calloc(nbytes, 1);
    if (!data)
        return (armas_x_sparse_t *)0;
    armassp_x_make(A, rows, cols, nnz, storage, data, nbytes);
    return A;
}

armas_x_sparse_t *armassp_x_new(int rows, int cols, int nnz, armassp_type_enum kind)
{
    armas_x_sparse_t *A = (armas_x_sparse_t *)calloc(1, sizeof(armas_x_sparse_t));
    if (!A)
        return A;
    return armassp_x_init(A, rows, cols, nnz, kind);
}

/**
 * \brief Resize sparse matrix.
 */
int armassp_x_resize(armas_x_sparse_t *A, int newsize)
{
    if (A->kind == ARMASSP_COO) {
        coo_elem_t *nb = (coo_elem_t *)calloc(newsize, sizeof(coo_elem_t));
        if (!nb)
            return -1;
        memcpy(nb, A->elems.ep, A->nnz*sizeof(coo_elem_t));
        free(A->elems.ep);
        A->elems.ep = nb;
        A->size = newsize;
        return 0;
    }
    // others not yet supported
    return -1;
}

/**
 * \brief Append element to sparse coo matrix.
 */
int armassp_x_append(armas_x_sparse_t *A, int m, int n, DTYPE v)
{
    if (A->kind != ARMASSP_COO)
        return -1;

    if (A->nnz == A->size) {
        if (armassp_x_resize(A, A->size + 64) < 0)
            return -1;
    }
    A->elems.ep[A->nnz].i = m;
    A->elems.ep[A->nnz].j = n;
    A->elems.ep[A->nnz].val = v;
    A->nnz ++;
    return 0;
}

#if defined(armassp_x_hasdiag)
/**
 */
int armassp_x_hasdiag(const armas_x_sparse_t *A, int diag)
{
    int i, j, start, end, p, roff, coff; 
    int nc = 0;
    
    if (!A || A->rows != A->cols)
        return 0;
    if (diag < 0 && A->rows < -diag)
        return 0;
    if (diag > 0 && A->cols < diag)
        return 0;
    
    switch (A->kind) {
    case ARMASSP_COO: 
        roff = diag < 0 ? 0 : diag;
        coff = diag > 0 ? 0 : diag;
        nc = diag < 0 ? A->rows + diag : A->cols - diag;
        for (i = 0; i < A->nnz; i++) {
            if (A->elems.ep[i].i + coff == A->elems.ep[i].j + roff)
                nc--;
        }
        break;
    case ARMASSP_CSC:
        start = diag > 0 ? diag : 0;
        end   = diag < 0 ? A->cols - diag : A->cols;
        nc    = end - start;
        for (i = 0, j = start; j < end; i++, j++) {
            for (p = armassp_x_index(A, j); p < armassp_x_index(A, j+1); p++) {
                if (armassp_x_at(A, p) == i) {
                    nc --;
                    break;
                }
            }
        }
        break;
    case ARMASSP_CSR:
        start = diag < 0 ? -diag : 0;
        end   = diag > 0 ? A->rows - diag : A->rows;
        nc    = end - start;
        for (j = 0, i = start; i < end; i++, j++) {
            for (p = armassp_x_index(A, i); p < armassp_x_index(A, i+1); p++) {
                if (armassp_x_at(A, p) == j) {
                    nc --;
                    break;
                }
            }
        }
        break;
    default:
        nc = -1;
        break;
    }
    return nc == 0;
}
#endif // defined(armassp_x_hasdiag)

#if 0
/**
 * \brief Allocate space for sparse accumulator
 *
 * \param[out] spa
 *    On entry uninitialized sparse structure, on exit initialized sparse accumulator.
 *
 * \param[in] nnz
 *   Allocation size for accumulator.
 *
 * \return pointer to initialized accumulator or null.
 */
armas_x_sparse_t *armassp_x_accum_alloc(armas_x_sparse_t *spa, int nnz)
{
    return armassp_x_init(spa, nnz, 1, nnz, ARMASSP_CSC);
}

#endif


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */



// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
