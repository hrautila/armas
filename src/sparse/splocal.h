
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef ARMAS_SPLOCAL_H
#define ARMAS_SPLOCAL_H 1

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>

#include "sparse.h"

// @brief Mark index in array
static inline
void armassp_imark(int *w, int j)
{
    w[j] = -w[j] - 1;
}
// @brief Test if index is marked in array
static inline
int armassp_imarked(int *w, int j)
{
    return w[j] < 0;
}
// @brief Unmark index in array
static inline
void armassp_iunmark(int *w, int j)
{
    if (w[j] < 0)
        armassp_imark(w, j);
}
// @brief Get index value as unmarked
static inline
int armassp_iunmarked(int *w, int j)
{
    return w[j] < 0 ? -w[j]-1 : w[j];
}

// ------------------------------------------------------------------------------
// marking the vertexes via graph

// @brief Mark vertex in graph.
static inline
void armassp_mark(armas_sparse_t *G, int j)
{
    G->ptr[j] = -G->ptr[j]-1;
}

// @brief Test if vertex is marked in graph.
static inline
int armassp_marked(const armas_sparse_t *G, int j)
{
    return G->ptr[j] < 0;
}

static inline
int armassp_get_unmarked(const armas_sparse_t *G, int j)
{
    return G->ptr[j] < 0 ? -G->ptr[j]-1 : G->ptr[j];
}

// @brief Unmark vertex in graph.
static inline
void armassp_unmark(armas_sparse_t *G, int j)
{
    if (G->ptr[j] < 0)
        armassp_mark(G, j);
}

// ------------------------------------------------------------------------------
// bits string vertex marking

typedef unsigned char armas_bits_t;

static inline
void armassp_bit_mark(armas_bits_t *w, int j)
{
    int    k = j >> 3;
    armas_bits_t m = 1 << (j & 0x7);
    w[k] |= m;
}

static inline
int armassp_bit_marked(armas_bits_t *w, int j)
{
    int    k = j >> 3;
    armas_bits_t m = 1 << (j & 0x7);
    return (w[k] & m) != 0;
}

static inline
void armassp_bit_unmark(armas_bits_t *w, int j)
{
    int    k = j >> 3;
    armas_bits_t m = 1 << (j & 0x7);
    w[k] &= ~m;
}

static inline
void armassp_bit_zero(armas_bits_t *w, int n)
{
    memset(w, 0, __nbits_aligned8(n));
}


static inline
int armassp_test_alloc(
    const armas_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    return A->__nbytes >= armassp_bytes_needed(rows, cols, nnz, typ);
}

static inline
int armassp_test_structure(
    const armas_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    // non-zero elements
    int nz = (DTYPE *)A->ptr - A->elems.v;
    // length of ptr array
    int np = A->ix - A->ptr;
    return n + 1 == np && nnz == nz;
}

static inline
int armassp_init_structure(
    armas_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
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
int armassp_init_arrays(
    armas_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    A->ptr = (int *)&A->elems.v[nnz];
    A->ix = &A->ptr[n+1];
    A->nptr = n;
    return 0;
}

#endif // ARMAS_SPLOCAL_H
