
// Copyright (c) Harri Rautila, 2018

// This file is part of github.com/hrautila/armas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armassp_x_add) && defined(armassp_x_addto_w)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_accumulator)
#define __ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

#include "matrix.h"
#include "sparse.h"

#include "splocal.h"

static
void __x_add_csr_nn(armas_x_sparse_t *C,
                    DTYPE alpha, const armas_x_sparse_t *A,
                    DTYPE beta, const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, p;
    DTYPE val;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->rows; i++) {
        for (p = A->ptr[i]; p < A->ptr[i+1]; p++) {
            val = alpha*armassp_x_value(A, p);
            armas_x_accum_addpos(spa, armassp_x_at(A, p), val, i);
        }
        for (p = B->ptr[i]; p < B->ptr[i+1]; p++) {
            val = beta*armassp_x_value(B, p);
            armas_x_accum_addpos(spa, armassp_x_at(B, p), val, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
}

static
void __x_add_csr_nt(armas_x_sparse_t *C,
                    DTYPE alpha, const armas_x_sparse_t *A,
                    DTYPE beta, const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, p;
    DTYPE val;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->rows; i++) {
        for (p = A->ptr[i]; p < A->ptr[i+1]; p++) {
            val = alpha*armassp_x_value(A, p);
            armas_x_accum_addpos(spa, armassp_x_at(A, p), val, i);
        }
        
        for (j = 0; j < B->rows; j++) {
            if ((p = armassp_x_nz(B, j, i)) < 0)
                continue;
            val = beta*armassp_x_value(B, p);
            armas_x_accum_addpos(spa, j, val, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
}

static
void __x_add_csr_tn(armas_x_sparse_t *C,
                    DTYPE alpha, const armas_x_sparse_t *A,
                    DTYPE beta, const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, p;
    DTYPE val;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->rows; j++) {
            if ((p = armassp_x_nz(A, j, i)) < 0)
                continue;
            val = beta*armassp_x_value(A, p);
            armas_x_accum_addpos(spa, j, val, i);
        }
        for (p = B->ptr[i]; p < B->ptr[i+1]; p++) {
            val = beta*armassp_x_value(B, p);
            armas_x_accum_addpos(spa, armassp_x_at(B, p), val, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
}


static
void __x_add_csc_nn(armas_x_sparse_t *C,
                    DTYPE alpha, const armas_x_sparse_t *A,
                    DTYPE beta, const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, p;
    DTYPE val;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->cols; i++) {
        for (p = A->ptr[i]; p < A->ptr[i+1]; p++) {
            val = alpha*armassp_x_value(A, p);
            armas_x_accum_addpos(spa, armassp_x_at(A, p), val, i);
        }
        for (p = B->ptr[i]; p < B->ptr[i+1]; p++) {
            val = beta*armassp_x_value(B, p);
            armas_x_accum_addpos(spa, armassp_x_at(B, p), val, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->cols] = C->nnz;
    C->nptr = A->cols;
}

static
void __x_add_csc_nt(armas_x_sparse_t *C,
                    DTYPE alpha, const armas_x_sparse_t *A,
                    DTYPE beta, const armas_x_sparse_t *B,
                    armas_x_accum_t *spa)
{
    int i, j, p;
    DTYPE val;
    
    C->nnz = 0;
    armas_x_accum_clear(spa);
    
    for (i = 0; i < A->cols; i++) {
        for (p = A->ptr[i]; p < A->ptr[i+1]; p++) {
            val = alpha*armassp_x_value(A, p);
            armas_x_accum_addpos(spa, armassp_x_at(A, p), val, i);
        }
        
        for (j = 0; j < B->cols; j++) {
            if ((p = armassp_x_nz(B, j, i)) < 0)
                continue;
            val = beta*armassp_x_value(B, p);
            armas_x_accum_addpos(spa, j, val, i);
        }
        // gather data to C[i,:] row
        armas_x_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->cols] = C->nnz;
    C->nptr = A->cols;
}

/** 
 *
 */
int armassp_x_addto_w(armas_x_sparse_t *C,
                      DTYPE alpha,
                      const armas_x_sparse_t *A,
                      DTYPE beta,
                      const armas_x_sparse_t *B,
                      int bits,
                      armas_wbuf_t *work,
                      armas_conf_t *cf)
{
    if (!A || !B) 
        return 0;

    if (A->kind != B->kind) {
        return -1;
    }
    if (A->rows != B->rows || A->cols != B->cols) {
        return -1;
    }

    if (work && work->bytes == 0) {
        work->bytes = armas_x_accum_need(C);
        return 0;
    }
    if (armas_wbytes(work) < armas_x_accum_need(C)) {
        cf->error = ARMAS_EWORK;
        return -1;
    }
    armas_x_accum_t spa;
    // get current position in work space
    size_t pos = armas_wpos(work);
    // make sparse accumulator 
    armas_x_accum_make(&spa, armas_x_accum_dim(C),
                       armas_wptr(work), armas_wbytes(work));

    switch (bits & (ARMAS_TRANSA|ARMAS_TRANSB)) {
    case ARMAS_TRANSB:
        if (A->kind == ARMASSP_CSC) {
            __x_add_csc_nt(C, alpha, A, beta, B, &spa);
        } else {
            __x_add_csr_nt(C, alpha, A, beta, B, &spa);
        }
        break;
    case ARMAS_TRANSA:
        if (A->kind == ARMASSP_CSC) {
        } else {
            __x_add_csr_tn(C, alpha, A, beta, B, &spa);
        }
        break;
    default:
        if (A->kind == ARMASSP_CSC) {
            __x_add_csc_nn(C, alpha, A, beta, B, &spa);
        } else {
            __x_add_csr_nn(C, alpha, A, beta, B, &spa);
        }
        break;
    }
    // release used workspace
    armas_wsetpos(work, pos);
    return 0;
}

/** 
 *
 */
armas_x_sparse_t *armassp_x_add(DTYPE alpha,
                                const armas_x_sparse_t *A,
                                DTYPE beta,
                                const armas_x_sparse_t *B,
                                int bits,
                                armas_conf_t *cf)
{
    armas_x_sparse_t *C = (armas_x_sparse_t *)0;
    armas_wbuf_t work =  ARMAS_WBNULL;

    if (!A || !B) 
        return C;

    if (armassp_x_addto_w(C, alpha, A, beta, B, bits, &work, cf) < 0) 
        return C;
    
    if (!armas_walloc(&work, work.bytes)) {
        cf->error = ARMAS_EWORK;
        return C;
    }
    // allocate target matrix; size estimate 
    int cnz = A->nnz + B->nnz;
    C = armassp_x_new(A->rows, A->cols, cnz, A->kind);
    if (!C) {
        armas_wrelease(&work);
        return C;
    }

    armassp_x_addto_w(C, alpha, A, beta, B, bits, &work, cf);
    armas_wrelease(&work);
    return C;
}

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:
