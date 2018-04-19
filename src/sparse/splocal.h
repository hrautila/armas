
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#ifndef __ARMAS_SPLOCAL_H
#define __ARMAS_SPLOCAL_H 1

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>

#ifdef COMPLEX128
/* ---------------------------------------------------------------------------
 * Definitions for double precision complex numbers.
 */

#elif COMPLEX64
/* ---------------------------------------------------------------------------
 * Definitions for single precision complex numbers.
 */

#elif FLOAT32
/* ---------------------------------------------------------------------------
 * Definitions for single precision floating type.
 */

#else  // default is double precision float (FLOAT64)

/* cgrad.c */
#define __x_cgrad               __d_cgrad

/* convert.c */
#define __x_csr_to_csc		__d_csr_to_csc
#define __x_csc_to_csr		__d_csc_to_csr
#define __x_csr_convert		__d_csr_convert
#define __x_csc_convert		__d_csc_convert

/* add.c */
#define __x_add_csr_nn          __d_add_csr_nn
#define __x_add_csr_nt          __d_add_csr_nt
#define __x_add_csr_tn          __d_add_csr_tn

/* mult.c */
#define __x_mult_csr_nn		__d_mult_csr_nn
#define __x_mult_csr_tn		__d_mult_csr_tn
#define __x_mult_csr_nt		__d_mult_csr_nt
#define __x_mult_csr_tt		__d_mult_csr_tt
#define __x_mult_csc_nn		__d_mult_csc_nn
#define __x_mult_csc_tn		__d_mult_csc_tn
#define __x_mult_csc_nt		__d_mult_csc_nt
#define __x_mult_csc_tt		__d_mult_csc_tt


#endif


static inline
int __x_test_alloc(const armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    return A->__nbytes >= armassp_x_bytes_needed(rows, cols, nnz, typ);
}

static inline
int __x_test_structure(const armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    // non-zero elements
    int nz = (DTYPE *)A->ptr - A->elems.v;
    // length of ptr array
    int np = A->ix - A->ptr;
    return n + 1 == np && nnz == nz;
}

static inline
int __x_init_structure(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
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
int __x_init_arrays(armas_x_sparse_t *A, int rows, int cols, int nnz, armassp_type_enum typ)
{
    int n = typ == ARMASSP_CSR ? rows : cols;
    A->ptr = (int *)&A->elems.v[nnz];
    A->ix = &A->ptr[n+1];
    A->nptr = n;
    return 0;
}



#endif // __ARMAS_SPLOCAL_H

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
