
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_sort_to) && defined(armassp_sort)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include <stdio.h>
#include "armas.h"
#include "sparse.h"

// \return <0 if a < b, =0 if a==b and >0 if a > b
static
int coo_rowcmp(const void *ap, const void *bp)
{
    const coo_elem_t *a = (const coo_elem_t *) ap;
    const coo_elem_t *b = (const coo_elem_t *) bp;
    // a < b
    if (a->i < b->i)
        return -1;
    // a > b
    if (a->i > b->i)
        return 1;
    // here a->i == b->i
    return a->j < b->j ? -1 : (a->j == b->j ? 0 : 1);
}

static
int coo_colcmp(const void *ap, const void *bp)
{
    const coo_elem_t *a = (const coo_elem_t *) ap;
    const coo_elem_t *b = (const coo_elem_t *) bp;
    // a < b
    if (a->j < b->j)
        return -1;
    // a > b
    if (a->j > b->j)
        return 1;
    // here a->j == b->j
    return a->i < b->i ? -1 : (a->i == b->i ? 0 : 1);
}

// Simple bubble sort to sort (index, value) pairs to ascending order
static
void simple_sort(int *index, double *elems, int n)
{
    int ix, k, j;
    double ex;
    for (k = 1; k < n; k++) {
        for (j = k; j > 0 && index[j] < index[j - 1]; j--) {
            ix = index[j];
            ex = elems[j];
            index[j] = index[j - 1];
            elems[j] = elems[j - 1];
            index[j - 1] = ix;
            elems[j - 1] = ex;
        }
    }
}

static
void compressed_sort(armas_sparse_t * A)
{
    double *Ae = A->elems.v;
    int k, *Ax = A->ix;
    for (k = 0; k < A->nptr; k++) {
        simple_sort(&Ax[A->ptr[k]], &Ae[A->ptr[k]],
                    A->ptr[k + 1] - A->ptr[k]);
    }
}

/**
 *  \brief Sort elements to asceding ROW or COLUMN order
 *
 *  \param[in,out] A
 *     On entry unsorted matrix, on exit sorted matrix. Compressed
 *     matrix is sorted to ascending row (column) index order.
 *
 *  \param[in] order
 *     Sort order, SPARSE_ROW_ORDER or ARMASSP_COL_ORDER
 *
 *  \retval
 *     =0 Ok, matrix sorted
 *     -1 not sorted
 */
int armassp_sort_to(armas_sparse_t * A, armassp_order_t order)
{
    switch (A->kind) {
    case ARMASSP_COO:
        if (order == ARMASSP_COL_ORDER) {
            qsort(A->elems.ep, A->nnz, sizeof(coo_elem_t), coo_colcmp);
        } else {
            qsort(A->elems.ep, A->nnz, sizeof(coo_elem_t), coo_rowcmp);
        }
        break;
    default:
        compressed_sort(A);
        break;
    }
    return 0;
}

/**
 *  \brief Sort elements of compressed matrix to ascending index order
 *
 *  \param[in,out] A
 *     On entry column (row) compressed matrix, on exit matrix elements
 *     sorted in ascending row (column) index order.
 * 
 *  \retval
 *     0 OK
 *    <0 Matrix not changed
 */
int armassp_sort(armas_sparse_t * A)
{
    if (!A || (A->kind != ARMASSP_CSC && A->kind != ARMASSP_CSR))
        return -1;
    compressed_sort(A);
    return 0;
}
#else
#warning "Missing defines. NO code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
