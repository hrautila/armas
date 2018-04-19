
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_mult) && defined(armassp_x_multto_w)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_accumulator)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include <assert.h>

#include "matrix.h"
#include "sparse.h"

// local mappings to type spesific names
#include "splocal.h"

// -------------------------------------------------------------------------------------------------
// sparse CSR multiply:  C = alpha*A*B
static
int __x_mult_csr_nn(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    armas_x_spvec_t r;
    int i, p;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->rows; i++) {
        // scatter 
        for (p = A->ptr[i]; p < A->ptr[i+1]; p++) {
            // elements contributed by B[k,:] 
            armassp_x_vector(&r, B, armassp_x_at(A, p));
            // accumulate updates A[i,k]*B[k,:]
            armas_x_accum_scatter(spa, &r, armassp_x_value(A, p), i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSR multiply:  C = alpha*A*B^T
static
int __x_mult_csr_nt(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    armas_x_spvec_t a, b;
    int i, k;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->rows; i++) {
        // scatter 
        armassp_x_vector(&a, A, i);
        for (k = 0; k < B->rows; k++) {
            armassp_x_vector(&b, B, k);
            armas_x_accum_dot(spa, k, &a, &b, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSR multiply:  C = alpha*A^T*B
static
int __x_mult_csr_tn(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, p;
    armas_x_spvec_t b;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->rows; i++) {
        // for i'th row find non-zero columns j
        for (j = 0; j < A->rows; j++ ) {
            if ((p = armassp_x_nz(A, j, i)) < 0)
                continue;
            armassp_x_vector(&b, B, j);
            armas_x_accum_scatter(spa, &b, armassp_x_value(A, p), i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSR multiply:  C = alpha*A^T*B^T
static
int __x_mult_csr_tt(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, k, p0, p1;
    DTYPE val;

    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < C->rows; i++) {
        for (j = 0; j < C->cols; j++ ) {
            // get column j from B;
            for (p1 = armassp_x_index(B, j);  p1 < armassp_x_index(B, j+1); p1++) {
                k = armassp_x_at(B, p1);
                // test if A[i,k] non-zero
                if ((p0 = armassp_x_nz(A, k, i)) < 0)
                    continue;           
                val = armassp_x_value(A, p0) * armassp_x_value(B, p1);
                armas_x_accum_addpos(spa, j, val, i);
            }
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSC multiply:  C = alpha*A*B
static
int __x_mult_csc_nn(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    armas_x_spvec_t r;
    int i, p;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < B->cols; i++) {
        // scatter 
        for (p = B->ptr[i]; p < B->ptr[i+1]; p++) {
            armassp_x_vector(&r, A, armassp_x_at(B, p));
            armas_x_accum_scatter(spa, &r, armassp_x_value(B, p), i);
        }
        // gather data to C[:,i] column
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[B->cols] = C->nnz;
    C->nptr = B->cols;
    return 0;
}

// sparse CSC multiply:  C = alpha*A^T*B
static
int __x_mult_csc_tn(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    armas_x_spvec_t a, b;
    int i, k;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < B->cols; i++) {
        // scatter 
        armassp_x_vector(&b, B, i);
        for (k = 0; k < A->cols; k++) {
            armassp_x_vector(&a, A, k);
            armas_x_accum_dot(spa, k, &a, &b, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[B->cols] = C->nnz;
    C->nptr = B->cols;
    return 0;
}

// sparse CSC multiply:  C = alpha*A*B^T
static
int __x_mult_csc_nt(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, p;
    armas_x_spvec_t a;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < B->cols; i++) {
        // for i'th row find non-zero columns j
        for (j = 0; j < B->cols; j++ ) {
            if ((p = armassp_x_nz(B, j, i)) < 0)
                continue;
            armassp_x_vector(&a, A, j);
            armas_x_accum_scatter(spa, &a, armassp_x_value(B, p), i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[C->cols] = C->nnz;
    C->nptr = C->cols;
    return 0;
}

// sparse CSR multiply:  C = alpha*A^T*B^T
static
int __x_mult_csc_tt(armas_x_sparse_t *C,
                    DTYPE alpha,
                    const armas_x_sparse_t *A,
                    const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, k, p0, p1;
    DTYPE val;

    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < C->rows; i++) {
        for (j = 0; j < C->cols; j++ ) {
            // get column j from A;
            for (p1 = armassp_x_index(A, j);  p1 < armassp_x_index(A, j+1); p1++) {
                k = armassp_x_at(A, p1);
                // test if A[i,k] non-zero
                if ((p0 = armassp_x_nz(B, k, i)) < 0)
                    continue;           
                val = armassp_x_value(B, p0) * armassp_x_value(A, p1);
                armas_x_accum_addpos(spa, j, val, i);
            }
        }
        // gather data to C[:,i] col
        armas_x_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[C->cols] = C->nnz;
    C->nptr = C->cols;
    return 0;
}



size_t armassp_x_mult_nnz(const armas_x_sparse_t *A, const armas_x_sparse_t *B, int flags)
{
    int n = (flags & ARMAS_TRANSB) == 0 ? B->cols : B->rows;
    return ((size_t)A->nnz)*((size_t)B->nnz)/n;
}

/**
 * \brief Compute \$ C = alpha*op(A)*op(B) \$
 */
int armassp_x_multto_w(armas_x_sparse_t *C,
                       DTYPE alpha,
                       const armas_x_sparse_t *A,
                       const armas_x_sparse_t *B,
                       int flags,
                       armas_wbuf_t *w,
                       armas_conf_t *cf)
{
    int m, n, ok;

    switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
    case ARMAS_TRANSA|ARMAS_TRANSB:
        m = A->cols; n = B->rows;
        ok = A->rows == B->cols && C->rows == m && C->cols == n;
        break;
    case ARMAS_TRANSB:
        m = A->rows; n = B->rows;
        ok = A->cols == B->cols && C->rows == m && C->cols == n;
        break;
    case ARMAS_TRANSA:
        m = A->cols; n = B->cols;
        ok = A->rows == B->rows && C->rows == m && C->cols == n;
        break;
    default:
        m = A->rows; n = B->cols;
        ok = A->cols == B->rows && C->rows == m && C->cols == n;
        break;
    }
    if (!ok || !w) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }


    // get estimate for needed work space
    if (w->bytes == 0) {
        w->bytes = armas_x_accum_need(C);
        return 0;
    }
    // 
    // size estimate;
    int cnz = A->nnz*B->nnz/n;
    size_t cbytes = armassp_x_bytes_needed(m, n, cnz, C->kind);
    if (armassp_x_nbytes(C) < cbytes) {
        cf->error = ARMAS_ESIZE;
        return -2;
    }

    size_t sz = armas_x_accum_need(C);
    if (armas_wbytes(w) < sz) {
        cf->error = ARMAS_EWORK;
        return -3;
    }

    armas_x_accum_t spa;
    armas_x_accum_make(&spa, armas_x_accum_dim(C), armas_wreserve_bytes(w, sz), armas_wbytes(w));
    
    int stat = 0;
    switch (flags & (ARMAS_TRANSA|ARMAS_TRANSB)) {
    case ARMAS_TRANSA|ARMAS_TRANSB:
        if (C->kind == ARMASSP_CSC)
            stat = __x_mult_csc_tt(C, alpha, A, B, &spa);
        else 
            stat = __x_mult_csr_tt(C, alpha, A, B, &spa);
        break;
    case ARMAS_TRANSB:
        if (C->kind == ARMASSP_CSC)
            stat = __x_mult_csc_nt(C, alpha, A, B, &spa);
        else 
            stat = __x_mult_csr_nt(C, alpha, A, B, &spa);
        break;
    case ARMAS_TRANSA:
        if (C->kind == ARMASSP_CSC)
            stat = __x_mult_csc_tn(C, alpha, A, B, &spa);
        else 
            stat = __x_mult_csr_tn(C, alpha, A, B, &spa);
        break;
    default:
        if (C->kind == ARMASSP_CSC)
            stat = __x_mult_csc_nn(C, alpha, A, B, &spa);
        else 
            stat = __x_mult_csr_nn(C, alpha, A, B, &spa);
        break;
    }
    return stat;
}

armas_x_sparse_t *armassp_x_mult(DTYPE alpha,
                                 const armas_x_sparse_t *A,
                                 const armas_x_sparse_t *B,
                                 int flags,
                                 armas_conf_t *cf)
{
    armas_x_sparse_t *C;
    //armas_x_accum_t spa;
    
    if (!A || !B || A->kind != B->kind || A->kind == ARMASSP_COO) {
        return (armas_x_sparse_t *)0;
    }

    // allocate target matrix; size estimate 
    int cnz = A->nnz*B->nnz/A->cols;
    C = armassp_x_new(A->rows, B->cols, cnz, A->kind);
    if (!C) {
        cf->error = ARMAS_EMEMORY;
        return (armas_x_sparse_t *)0;
    }

    return C;
}


#endif // __ARMAS_PROVIDES && __ARMAS_REQUIRES


// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
