
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_convert_to) && defined(armassp_x_convert) && \
    defined(armassp_x_transpose_to) && defined(armassp_x_transpose)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armassp_x_sort)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)
// -----------------------------------------------------------------------------
#include <stdio.h>
#include "armas.h"
#include "sparse.h"
#include "splocal.h"

/**
 * \brief Convert from row compressed storage column compressed storage. 
 *
 * Runtime O(nnz).
 */
static
armas_x_sparse_t *csr_to_csc(armas_x_sparse_t * A,
                             const armas_x_sparse_t * B)
{
    double *Ae, *Be;

    for (int i = 0; i < A->cols; i++)
        A->ptr[i] = 0;

    // count column occurences;
    for (int k = 0; k < B->nnz; k++) {
        A->ptr[B->ix[k]]++;
    }
    // create A->ptr to 
    int csum = 0;
    for (int i = 0; i < A->cols; i++) {
        int t = A->ptr[i];
        A->ptr[i] = csum;
        csum += t;
    }
    A->ptr[A->cols] = csum;

    Ae = A->elems.v;
    Be = B->elems.v;
    for (int k = 0; k < B->rows; k++) {
        for (int p = B->ptr[k]; p < B->ptr[k + 1]; p++) {
            int col = B->ix[p];
            A->ix[A->ptr[col]] = k;
            Ae[A->ptr[col]] = Be[p];
            A->ptr[col]++;
        }
    }
    // now restore A->ptr
    csum = 0;
    for (int i = 0; i < A->cols; i++) {
        int t = A->ptr[i];
        A->ptr[i] = csum;
        csum = t;
    }
    A->ptr[A->cols] = B->nnz;
    A->nptr = A->cols;
    A->nnz = B->nnz;
    A->kind = ARMASSP_CSC;
    return A;
}

/**
 *
 * Runtime O(nnz).
 */
static
armas_x_sparse_t *csc_to_csr(armas_x_sparse_t * A,
                             const armas_x_sparse_t * B)
{
    double *Ae, *Be;

    for (int i = 0; i < A->rows; i++)
        A->ptr[i] = 0;

    // count row occurences;
    for (int k = 0; k < B->nnz; k++) {
        A->ptr[B->ix[k]]++;
    }
    // create A->ptr to 
    int csum = 0;
    for (int i = 0; i < A->rows; i++) {
        int t = A->ptr[i];
        A->ptr[i] = csum;
        csum += t;
    }
    A->ptr[A->rows] = csum;

    Ae = A->elems.v;
    Be = B->elems.v;
    for (int k = 0; k < B->cols; k++) {
        for (int p = B->ptr[k]; p < B->ptr[k + 1]; p++) {
            int row = B->ix[p];
            A->ix[A->ptr[row]] = k;
            Ae[A->ptr[row]] = Be[p];
            A->ptr[row]++;
        }
    }
    // now restore A->ptr
    csum = 0;
    for (int i = 0; i < A->rows; i++) {
        int t = A->ptr[i];
        A->ptr[i] = csum;
        csum = t;
    }
    A->ptr[A->rows] = B->nnz;
    A->nptr = A->rows;
    A->nnz = B->nnz;
    A->kind = ARMASSP_CSR;
    return A;
}


static
void csc_convert(armas_x_sparse_t * A, const armas_x_sparse_t * B)
{
    coo_elem_t *Be = B->elems.ep;
    double *Ae = A->elems.v;
    int cur, sum, col, *Ax = A->ix;

    // count rows in column
    for (int k = 0; k < B->nnz; k++) {
        A->ptr[Be[k].j]++;
    }
    // compute cumulative sums to get indexes
    cur = sum = 0;
    for (int k = 0; k < B->cols + 1; k++) {
        cur = A->ptr[k];
        A->ptr[k] = sum;
        sum += cur;
    }
    require(A->ptr[B->cols] == B->nnz);
    for (int k = 0; k < B->nnz; k++) {
        col = Be[k].j;
        Ae[A->ptr[col]] = Be[k].val;
        // save index to row; 
        Ax[A->ptr[col]] = Be[k].i;
        //advance ptr indexing for this column
        A->ptr[col]++;
    }
    // now restore A->ptr
    sum = 0;
    for (int i = 0; i < A->cols; i++) {
        int t = A->ptr[i];
        A->ptr[i] = sum;
        sum = t;
    }
    A->nnz = B->nnz;
    A->nptr = B->cols;
    A->kind = ARMASSP_CSC;
}

static
void csr_convert(armas_x_sparse_t * A, const armas_x_sparse_t * B)
{
    coo_elem_t *Be = B->elems.ep;
    double *Ae = A->elems.v;
    int cur, sum, row, *Ax = A->ix;

    // count rows in column
    for (int k = 0; k < B->nnz; k++) {
        A->ptr[Be[k].i]++;
    }
    // compute cumulative sums to get indexes
    cur = sum = 0;
    for (int k = 0; k < B->rows + 1; k++) {
        cur = A->ptr[k];
        A->ptr[k] = sum;
        sum += cur;
    }
    require(A->ptr[B->rows] == B->nnz);
    for (int k = 0; k < B->nnz; k++) {
        row = Be[k].i;
        Ae[A->ptr[row]] = Be[k].val;
        // save index to col; 
        Ax[A->ptr[row]] = Be[k].j;
        //advance ptr indexing for this column
        A->ptr[row]++;
    }
    // now restore A->ptr
    sum = 0;
    for (int i = 0; i < A->rows; i++) {
        int t = A->ptr[i];
        A->ptr[i] = sum;
        sum = t;
    }
    A->nnz = B->nnz;
    A->nptr = B->rows;
    A->kind = ARMASSP_CSR;
}

/**
 *  \brief Convert source matrix to defined compressed format
 * 
 *  \param[out] A
 *      Target matrix,  if nnz(A) is less than nnz(B) then error is returned. Otherwise
 *      source matrix is converted to defined format.
 *  \param[in] B
 *      Source matrix.
 *  \param[in] target
 *      Result matrix compressed storage format (column or row)
 *
 *  \return
 *      Pointer to target matrix or null if no conversion happened.
 */
armas_x_sparse_t *armassp_x_convert_to(armas_x_sparse_t * A,
                                       const armas_x_sparse_t * B,
                                       armassp_type_enum target)
{
    if (!B)
        return (armas_x_sparse_t *) 0;

    // only conversion to CSR or CSC supported
    if (target != ARMASSP_CSC && target != ARMASSP_CSR)
        return (armas_x_sparse_t *) 0;

    // test for space
    if (!sp_test_alloc(A, B->rows, B->cols, B->nnz, target)) {
        return (armas_x_sparse_t *) 0;
    }
    // test structure
    if (!sp_test_structure(A, B->rows, B->cols, B->nnz, target)) {
        return (armas_x_sparse_t *) 0;
    }
    // init internal arrays
    sp_init_arrays(A, B->rows, B->cols, B->nnz, target);

    switch (B->kind) {
    case ARMASSP_COO:
        if (target == ARMASSP_CSR)
            csr_convert(A, B);
        else
            csc_convert(A, B);
        armassp_x_sort(A);
        break;
    case ARMASSP_CSC:
        if (target == ARMASSP_CSR)
            csc_to_csr(A, B);
        // target CSC implies copying ...
        break;
    case ARMASSP_CSR:
    default:
        if (target == ARMASSP_CSC)
            csr_to_csc(A, B);
        // target CSR implies copying ...
        break;
    }
    return A;
}

/**
 * \brief Create new column compressed matrix from uncompressed matrix.
 *
 * \param[in] B
 *    Input matrix
 *
 * \return
 *    New matrix in request storage format
 */
armas_x_sparse_t *armassp_x_convert(const armas_x_sparse_t * B,
                                    armassp_type_enum target)
{
    armas_x_sparse_t *A = armassp_x_new(B->rows, B->cols, B->nnz, target);
    return armassp_x_convert_to(A, B, target);
}

/**
 * \brief Transpose compressed storage matrix
 *
 * \param[in,out] A
 *    On entry sparse matrix with allocated space. On exit transpose of B
 * \param[in] B
 *    Input matrix
 *
 * \return
 *    Pointer to target matrix or null if no transpose happened.
 */
armas_x_sparse_t *armassp_x_transpose_to(armas_x_sparse_t * A,
                                         const armas_x_sparse_t * B)
{
    if (!B || !A)
        return (armas_x_sparse_t *) 0;

    // transpose of CSR or CSC supported
    if (B->kind != ARMASSP_CSC && B->kind != ARMASSP_CSR)
        return (armas_x_sparse_t *) 0;

    armassp_type_enum t = B->kind == ARMASSP_CSC ? ARMASSP_CSR : ARMASSP_CSC;

    // make convertion to other compressed format
    if (!armassp_x_convert_to(A, B, t))
        return (armas_x_sparse_t *) 0;

    // fix result row,cols sizes and type
    A->rows = B->cols;
    A->cols = B->rows;
    A->kind = B->kind;
    return A;
}

armas_x_sparse_t *armassp_x_transpose(const armas_x_sparse_t * B)
{
    armassp_type_enum t = B->kind == ARMASSP_CSC ? ARMASSP_CSR : ARMASSP_CSC;
    // make space reservation as if converted to other storage format
    armas_x_sparse_t *A = armassp_x_new(B->rows, B->cols, B->nnz, t);
    return armassp_x_transpose_to(A, B);
}
#else
#warning "Missing defines. No code."
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
