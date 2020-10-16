
// Copyright by libARMAS authors. See AUTHORS file in this archive.

// This file is part of libARMAS package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

#include "spdefs.h"

// -----------------------------------------------------------------------------
// this file provides following type dependent functions
#if defined(armassp_mult) && defined(armassp_multto_w)
#define ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_accumulator)
#define ARMAS_REQUIRES 1
#endif

// compile if type dependent public function names defined
#if (defined(ARMAS_PROVIDES) && defined(ARMAS_REQUIRES)) || defined(CONFIG_NOTYPENAMES)
// -----------------------------------------------------------------------------

#include <assert.h>
#include "matrix.h"
#include "sparse.h"
// local mappings to type spesific names
#include "splocal.h"

// -----------------------------------------------------------------------------
// sparse CSR multiply:  C = alpha*A*B
static
int mult_csr_nn(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    armas_spvec_t r;
    int i, p;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->rows; i++) {
        // scatter 
        for (p = A->ptr[i]; p < A->ptr[i + 1]; p++) {
            // elements contributed by B[k,:] 
            armassp_vector(&r, B, armassp_at(A, p));
            // accumulate updates A[i,k]*B[k,:]
            armas_accum_scatter(spa, &r, armassp_value(A, p), i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, 1.0, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSR multiply:  C = alpha*A*B^T
static
int mult_csr_nt(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    armas_spvec_t a, b;
    int i, k;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->rows; i++) {
        // scatter 
        armassp_vector(&a, A, i);
        for (k = 0; k < B->rows; k++) {
            armassp_vector(&b, B, k);
            armas_accum_dot(spa, k, &a, &b, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSR multiply:  C = alpha*A^T*B
static
int mult_csr_tn(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    int i, j, p;
    armas_spvec_t b;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < A->rows; i++) {
        // for i'th row find non-zero columns j
        for (j = 0; j < A->rows; j++) {
            if ((p = armassp_nz(A, j, i)) < 0)
                continue;
            armassp_vector(&b, B, j);
            armas_accum_scatter(spa, &b, armassp_value(A, p), i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSR multiply:  C = alpha*A^T*B^T
static
int mult_csr_tt(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    int i, j, k, p0, p1;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < C->rows; i++) {
        for (j = 0; j < C->cols; j++) {
            // get column j from B;
            for (p1 = armassp_index(B, j); p1 < armassp_index(B, j + 1); p1++) {
                k = armassp_at(B, p1);
                // test if A[i,k] non-zero
                if ((p0 = armassp_nz(A, k, i)) < 0)
                    continue;
                val = armassp_value(A, p0) * armassp_value(B, p1);
                armas_accum_addpos(spa, j, val, i);
            }
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[A->rows] = C->nnz;
    C->nptr = A->rows;
    return 0;
}

// sparse CSC multiply:  C = alpha*A*B
static
int mult_csc_nn(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    armas_spvec_t r;
    int i, p;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < B->cols; i++) {
        // scatter 
        for (p = B->ptr[i]; p < B->ptr[i + 1]; p++) {
            armassp_vector(&r, A, armassp_at(B, p));
            armas_accum_scatter(spa, &r, armassp_value(B, p), i);
        }
        // gather data to C[:,i] column
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[B->cols] = C->nnz;
    C->nptr = B->cols;
    return 0;
}

// sparse CSC multiply:  C = alpha*A^T*B
static
int mult_csc_tn(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    armas_spvec_t a, b;
    int i, k;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < B->cols; i++) {
        // scatter 
        armassp_vector(&b, B, i);
        for (k = 0; k < A->cols; k++) {
            armassp_vector(&a, A, k);
            armas_accum_dot(spa, k, &a, &b, i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[B->cols] = C->nnz;
    C->nptr = B->cols;
    return 0;
}

// sparse CSC multiply:  C = alpha*A*B^T
static
int mult_csc_nt(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    int i, j, p;
    armas_spvec_t a;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < B->cols; i++) {
        // for i'th row find non-zero columns j
        for (j = 0; j < B->cols; j++) {
            if ((p = armassp_nz(B, j, i)) < 0)
                continue;
            armassp_vector(&a, A, j);
            armas_accum_scatter(spa, &a, armassp_value(B, p), i);
        }
        // gather data to C[i,:] row
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[C->cols] = C->nnz;
    C->nptr = C->cols;
    return 0;
}

// sparse CSR multiply:  C = alpha*A^T*B^T
static
int mult_csc_tt(armas_sparse_t * C,
                DTYPE alpha,
                const armas_sparse_t * A,
                const armas_sparse_t * B, armas_accum_t * spa)
{
    int i, j, k, p0, p1;
    DTYPE val;

    C->nnz = 0;
    armas_accum_clear(spa);

    for (i = 0; i < C->rows; i++) {
        for (j = 0; j < C->cols; j++) {
            // get column j from A;
            for (p1 = armassp_index(A, j); p1 < armassp_index(A, j + 1); p1++) {
                k = armassp_at(A, p1);
                // test if A[i,k] non-zero
                if ((p0 = armassp_nz(B, k, i)) < 0)
                    continue;
                val = armassp_value(B, p0) * armassp_value(A, p1);
                armas_accum_addpos(spa, j, val, i);
            }
        }
        // gather data to C[:,i] col
        armas_accum_gather(C, alpha, spa, i, C->size);
        spa->tail = 0;
    }
    C->ptr[C->cols] = C->nnz;
    C->nptr = C->cols;
    return 0;
}

size_t armassp_mult_nnz(const armas_sparse_t * A,
                          const armas_sparse_t * B, int flags)
{
    int n = (flags & ARMAS_TRANSB) == 0 ? B->cols : B->rows;
    return ((size_t) A->nnz) * ((size_t) B->nnz) / n;
}

/**
 * @brief Compute \f$ C = alpha*op(A)*op(B) \f$
 *
 * @param[in,out]  C
 *   On entry result matrix with proper space allocation. On exit the
 *   result matrix.
 * @param[in] alpha
 *   Scalar
 * @param[in] A
 *   First operand. Same sparse type as B, CSC or CSR.
 * @param[in] B
 *   Second operand. Same sparse type as A, CSC or CSR.
 * @param[in] flags
 *   Operator flags, *ARMAS_TRANSA*, *ARMAS_TRANSB*
 * @param[in] w
 *   Workspace. If wb.bytes is zero then required workspace is
 *   calculated and immediately returned.
 * @param[in,out] cf
 *   Configuration block.
 *
 * @retval  0  Success
 * @retval <0  Failure
 * @ingroup sparse
 */
int armassp_multto_w(armas_sparse_t * C,
                       DTYPE alpha,
                       const armas_sparse_t * A,
                       const armas_sparse_t * B,
                       int flags, armas_wbuf_t * w, armas_conf_t * cf)
{
    int m, n, ok;

    switch (flags & (ARMAS_TRANSA | ARMAS_TRANSB)) {
    case ARMAS_TRANSA | ARMAS_TRANSB:
        m = A->cols;
        n = B->rows;
        ok = A->rows == B->cols && C->rows == m && C->cols == n;
        break;
    case ARMAS_TRANSB:
        m = A->rows;
        n = B->rows;
        ok = A->cols == B->cols && C->rows == m && C->cols == n;
        break;
    case ARMAS_TRANSA:
        m = A->cols;
        n = B->cols;
        ok = A->rows == B->rows && C->rows == m && C->cols == n;
        break;
    default:
        m = A->rows;
        n = B->cols;
        ok = A->cols == B->rows && C->rows == m && C->cols == n;
        break;
    }
    if (!ok || !w) {
        cf->error = ARMAS_EINVAL;
        return -1;
    }

    // get estimate for needed work space
    if (w->bytes == 0) {
        w->bytes = armas_accum_need(C);
        return 0;
    }
    // 
    // size estimate;
    int cnz = A->nnz * B->nnz / n;
    size_t cbytes = armassp_bytes_needed(m, n, cnz, C->kind);
    if (armassp_nbytes(C) < cbytes) {
        cf->error = ARMAS_ESIZE;
        return -2;
    }

    size_t sz = armas_accum_need(C);
    if (armas_wbytes(w) < sz) {
        cf->error = ARMAS_EWORK;
        return -3;
    }

    armas_accum_t spa;
    armas_accum_make(&spa, armas_accum_dim(C), armas_wreserve_bytes(w, sz),
                       armas_wbytes(w));

    int stat = 0;
    switch (flags & (ARMAS_TRANSA | ARMAS_TRANSB)) {
    case ARMAS_TRANSA | ARMAS_TRANSB:
        if (C->kind == ARMASSP_CSC)
            stat = mult_csc_tt(C, alpha, A, B, &spa);
        else
            stat = mult_csr_tt(C, alpha, A, B, &spa);
        break;
    case ARMAS_TRANSB:
        if (C->kind == ARMASSP_CSC)
            stat = mult_csc_nt(C, alpha, A, B, &spa);
        else
            stat = mult_csr_nt(C, alpha, A, B, &spa);
        break;
    case ARMAS_TRANSA:
        if (C->kind == ARMASSP_CSC)
            stat = mult_csc_tn(C, alpha, A, B, &spa);
        else
            stat = mult_csr_tn(C, alpha, A, B, &spa);
        break;
    default:
        if (C->kind == ARMASSP_CSC)
            stat = mult_csc_nn(C, alpha, A, B, &spa);
        else
            stat = mult_csr_nn(C, alpha, A, B, &spa);
        break;
    }
    return stat;
}

/**
 * @brief Compute \f$ \alpha * op(A) * op(B) \f$
 *
 * @param[in] alpha
 *   Scalar
 * @param[in] A
 *   First operand. Same sparse type as B, CSC or CSR.
 * @param[in] B
 *   Second operand. Same sparse type as A, CSC or CSR.
 * @param[in] flags
 *   Operator flags, *ARMAS_TRANSA*, *ARMAS_TRANSB*
 * @param[in,out] cf
 *   Configuration block.
 *
 * @return  Allocated new result matrix or null pointer.
 * @ingroup sparse
 *
 */
armas_sparse_t *armassp_mult(DTYPE alpha,
                                 const armas_sparse_t * A,
                                 const armas_sparse_t * B,
                                 int flags, armas_conf_t * cf)
{
    armas_sparse_t *C;
    //armas_accum_t spa;

    if (!A || !B || A->kind != B->kind || A->kind == ARMASSP_COO) {
        return (armas_sparse_t *) 0;
    }
    // allocate target matrix; size estimate 
    int cnz = A->nnz * B->nnz / A->cols;
    C = armassp_new(A->rows, B->cols, cnz, A->kind);
    if (!C) {
        cf->error = ARMAS_EMEMORY;
        return (armas_sparse_t *) 0;
    }

    return C;
}
#else
#warning "Missing defines. No code!"
#endif // ARMAS_PROVIDES && ARMAS_REQUIRES
