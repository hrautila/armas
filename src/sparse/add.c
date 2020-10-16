
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_add) && defined(armassp_addto_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_accumulator)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include "matrix.h"
#include "sparse.h"

#include "splocal.h"

static
void add_csr_nn(armas_sparse_t * C,
                DTYPE alpha, const armas_sparse_t * A,
                DTYPE beta, const armas_sparse_t * B,
                armas_accum_t * spa)
{
    int i, p;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->rows; i++) {
        for (p = A->ptr[i]; p < A->ptr[i + 1]; p++) {
            val = alpha * armassp_value(A, p);
            armas_accum_addpos(spa, armassp_at(A, p), val, i);
        }
        for (p = B->ptr[i]; p < B->ptr[i + 1]; p++) {
            val = beta * armassp_value(B, p);
            armas_accum_addpos(spa, armassp_at(B, p), val, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
}

static
void add_csr_nt(armas_sparse_t * C,
                DTYPE alpha, const armas_sparse_t * A,
                DTYPE beta, const armas_sparse_t * B,
                armas_accum_t * spa)
{
    int i, j, p;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->rows; i++) {
        for (p = A->ptr[i]; p < A->ptr[i + 1]; p++) {
            val = alpha * armassp_value(A, p);
            armas_accum_addpos(spa, armassp_at(A, p), val, i);
        }

        for (j = 0; j < B->rows; j++) {
            if ((p = armassp_nz(B, j, i)) < 0)
                continue;
            val = beta * armassp_value(B, p);
            armas_accum_addpos(spa, j, val, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
}

static
void add_csr_tn(armas_sparse_t * C,
                DTYPE alpha, const armas_sparse_t * A,
                DTYPE beta, const armas_sparse_t * B,
                armas_accum_t * spa)
{
    int i, j, p;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->rows; j++) {
            if ((p = armassp_nz(A, j, i)) < 0)
                continue;
            val = beta * armassp_value(A, p);
            armas_accum_addpos(spa, j, val, i);
        }
        for (p = B->ptr[i]; p < B->ptr[i + 1]; p++) {
            val = beta * armassp_value(B, p);
            armas_accum_addpos(spa, armassp_at(B, p), val, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
}


static
void add_csc_nn(armas_sparse_t * C,
                DTYPE alpha, const armas_sparse_t * A,
                DTYPE beta, const armas_sparse_t * B,
                armas_accum_t * spa)
{
    int i, p;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->cols; i++) {
        for (p = A->ptr[i]; p < A->ptr[i + 1]; p++) {
            val = alpha * armassp_value(A, p);
            armas_accum_addpos(spa, armassp_at(A, p), val, i);
        }
        for (p = B->ptr[i]; p < B->ptr[i + 1]; p++) {
            val = beta * armassp_value(B, p);
            armas_accum_addpos(spa, armassp_at(B, p), val, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->cols] = C->nnz;
    C->nptr = A->cols;
}

static
void add_csc_nt(armas_sparse_t * C,
                DTYPE alpha, const armas_sparse_t * A,
                DTYPE beta, const armas_sparse_t * B,
                armas_accum_t * spa)
{
    int i, j, p;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->cols; i++) {
        for (p = A->ptr[i]; p < A->ptr[i + 1]; p++) {
            val = alpha * armassp_value(A, p);
            armas_accum_addpos(spa, armassp_at(A, p), val, i);
        }

        for (j = 0; j < B->cols; j++) {
            if ((p = armassp_nz(B, j, i)) < 0)
                continue;
            val = beta * armassp_value(B, p);
            armas_accum_addpos(spa, j, val, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->cols] = C->nnz;
    C->nptr = A->cols;
}

/**
 * @brief Compute \f$ C = \alpha * A + \beta * B \f$
 *
 * @param[out] C
 * @param[in]  alpha
 * @param[in]  A
 * @param[in]  beta
 * @param[in]  B
 * @param[in]  bits
 * @param[in,out] wb
 * @param[in]  cf
 *
 * @retval  0  Success
 * @retval <0  Failure
 * @ingroup sparse
 */
int armassp_addto_w(armas_sparse_t * C,
                      DTYPE alpha,
                      const armas_sparse_t * A,
                      DTYPE beta,
                      const armas_sparse_t * B,
                      int bits, armas_wbuf_t * wb, armas_conf_t * cf)
{
    if (!A || !B)
        return 0;

    if (A->kind != B->kind) {
        return -ARMAS_EINVAL;
    }
    if (A->rows != B->rows || A->cols != B->cols) {
        return -ARMAS_ESIZE;
    }

    if (wb && wb->bytes == 0) {
        wb->bytes = armas_accum_need(C);
        return 0;
    }
    if (armas_wbytes(wb) < armas_accum_need(C)) {
        cf->error = ARMAS_EWORK;
        return -ARMAS_EWORK;
    }
    armas_accum_t spa;
    // get current position in wb space
    size_t pos = armas_wpos(wb);
    // make sparse accumulator 
    armas_accum_make(&spa, armas_accum_dim(C),
                       armas_wptr(wb), armas_wbytes(wb));

    switch (bits & (ARMAS_TRANSA | ARMAS_TRANSB)) {
    case ARMAS_TRANSB:
        if (A->kind == ARMASSP_CSC) {
            add_csc_nt(C, alpha, A, beta, B, &spa);
        } else {
            add_csr_nt(C, alpha, A, beta, B, &spa);
        }
        break;
    case ARMAS_TRANSA:
        if (A->kind == ARMASSP_CSC) {
        } else {
            add_csr_tn(C, alpha, A, beta, B, &spa);
        }
        break;
    default:
        if (A->kind == ARMASSP_CSC) {
            add_csc_nn(C, alpha, A, beta, B, &spa);
        } else {
            add_csr_nn(C, alpha, A, beta, B, &spa);
        }
        break;
    }
    // release used wbspace
    armas_wsetpos(wb, pos);
    return 0;
}

/**
 * @brief Compute \f$ \alpha * A + \beta * B \f$
 *
 * @return Pointer to new sparse matrix or null pointer on error.
 * @ingroup sparse
 */
armas_sparse_t *armassp_add(DTYPE alpha,
                                const armas_sparse_t * A,
                                DTYPE beta,
                                const armas_sparse_t * B,
                                int bits, armas_conf_t * cf)
{
    armas_sparse_t *C = (armas_sparse_t *) 0;
    armas_wbuf_t work = ARMAS_WBNULL;

    if (!A || !B)
        return C;

    if (armassp_addto_w(C, alpha, A, beta, B, bits, &work, cf) < 0)
        return C;

    if (!armas_walloc(&work, work.bytes)) {
        cf->error = ARMAS_EWORK;
        return C;
    }
    // allocate target matrix; size estimate 
    int cnz = A->nnz + B->nnz;
    C = armassp_new(A->rows, A->cols, cnz, A->kind);
    if (!C) {
        armas_wrelease(&work);
        return C;
    }

    armassp_addto_w(C, alpha, A, beta, B, bits, &work, cf);
    armas_wrelease(&work);
    return C;
}
#else
#warning "Missing defines. No code!"
#endif /* ARMAS_PROVIDES && ARMAS_REQUIRES */
