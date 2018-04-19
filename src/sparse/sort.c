
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_sort_to) && defined(armassp_x_sort)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#define __ARMAS_REQUIRES 1

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <stdio.h>
#include <armas/armas.h>
#include "matrix.h"
#include "sparse.h"

// \return <0 if a < b, =0 if a==b and >0 if a > b
static
int __coo_rowcmp(const void *ap, const void *bp)
{
    const coo_elem_t *a = (const coo_elem_t *)ap;
    const coo_elem_t *b = (const coo_elem_t *)bp;
    if (a->i < b->i)
        return -1;        // a < b

    if (a->i > b->i)
        return 1;       // a > b

    // here a->i == b->i
    return a->j < b->j ? -1
        : (a->j == b->j ? 0 : 1);
}

static
int __coo_colcmp(const void *ap, const void *bp)
{
    const coo_elem_t *a = (const coo_elem_t *)ap;
    const coo_elem_t *b = (const coo_elem_t *)bp;
    if (a->j < b->j)
        return -1;        // a < b

    if (a->j > b->j)
        return 1;       // a > b

    // here a->j == b->j
    return a->i < b->i ? -1
        : (a->i == b->i ? 0 : 1);
}

// Simple bubble sort to sort (index, value) pairs to ascending order
static
void __simple_sort(int *index, double *elems, int n)
{
    int ix, k, j;
    double ex;
    for (k = 1; k < n; k++) {
        for (j = k; j > 0 && index[j] < index[j-1]; j--) {
            ix = index[j];
            ex = elems[j];
            index[j] = index[j-1];
            elems[j] = elems[j-1];
            index[j-1] = ix;
            elems[j-1] = ex;
        }
    }
}

static
void __compressed_sort(armas_x_sparse_t *A)
{
    double *Ae = A->elems.v;
    int k, *Ax = A->ix;
    for (k = 0; k < A->nptr; k++) {
        __simple_sort(&Ax[A->ptr[k]], &Ae[A->ptr[k]], A->ptr[k+1]-A->ptr[k]);
    }
}

/**
 *  \brief Sort elements to asceding ROW or COLUMN order
 *
 *  \param[in,out] A
 *     On entry unsorted matrix, on exit sorted matrix. Compressed matrix is sorted
 *     to ascending row (column) index order.
 * 
 *  \param[in] order
 *     Sort order, SPARSE_ROW_ORDER or ARMASSP_COL_ORDER
 *
 *  \retval 
 *     =0 Ok, matrix sorted
 *     -1 not sorted
 */
int armassp_x_sort_to(armas_x_sparse_t *A, armassp_order_t order)
{
    switch (A->kind) {
    case ARMASSP_COO:
        if (order == ARMASSP_COL_ORDER) {
            qsort(A->elems.ep, A->nnz, sizeof(coo_elem_t), __coo_colcmp);
        } else {
            qsort(A->elems.ep, A->nnz, sizeof(coo_elem_t), __coo_rowcmp);
        }
        break;
    default:
        __compressed_sort(A);
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
int armassp_x_sort(armas_x_sparse_t *A)
{
    if (!A || (A->kind != ARMASSP_CSC && A->kind != ARMASSP_CSR))
        return -1;
    __compressed_sort(A);
    return 0;
}


#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
