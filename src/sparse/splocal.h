
// Copyright (c) Harri Rautila, 2018-2020

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_SPLOCAL_H
#define ARMAS_SPLOCAL_H 1

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>


static inline
int sp_test_alloc(const armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    return A->__nbytes >= armassp_x_bytes_needed(rows, cols, nnz, typ);
}

static inline
int sp_test_structure(const armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    // non-zero elements
    int nz = (DTYPE *)A->ptr - A->elems.v;
    // length of ptr array
    int np = A->ix - A->ptr;
    return n + 1 == np && nnz == nz;
}

static inline
int sp_init_structure(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    A->ptr = (int *)&A->elems.v[nnz];
    A->ix = &A->ptr[n+1];
    A->nptr = n;
    A->rows = rows;
    A->cols = cols;
    A->size = nnz;
    A->nnz = 0;
    A->kind = typ;
    return 0;
}

static inline
int sp_init_arrays(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    A->ptr = (int *)&A->elems.v[nnz];
    A->ix = &A->ptr[n+1];
    A->nptr = n;
    return 0;
}

#endif // ARMAS_SPLOCAL_H
